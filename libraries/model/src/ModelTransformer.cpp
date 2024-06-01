////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ModelTransformer.cpp (model)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ModelTransformer.h"
#include "InputNode.h"
#include "Node.h"
#include "OutputNode.h"
#include "RefineTransformation.h"

#include <utilities/include/Exception.h>
#include <utilities/include/Logger.h>
#include <utilities/include/StringUtil.h>

#include <algorithm>

using namespace ell::logging;

namespace ell
{
namespace model
{
    // Represents a correspondence between two ports, as when a port is transformed or copied into another model
    struct PortCorrespondence
    {
        const InputPortBase* source;
        const OutputPortBase* destination;
    };

    // Represents a set of correspondences between ports, for use when making copies of submodels onto a new location
    class PortCorrespondences
    {
    public:
        PortCorrespondences() = default;
        PortCorrespondences(const std::vector<const InputPortBase*>& sources, const std::vector<const OutputPortBase*>& destinations);

        void Add(const PortCorrespondence& correspondence);
        PortCorrespondence operator[](int index) const;

        using iterator = std::vector<PortCorrespondence>::iterator;
        using const_iterator = std::vector<PortCorrespondence>::const_iterator;

    private:
        friend PortCorrespondences::iterator begin(PortCorrespondences& correspondences);
        friend PortCorrespondences::iterator end(PortCorrespondences& correspondences);
        friend PortCorrespondences::const_iterator begin(const PortCorrespondences& correspondences);
        friend PortCorrespondences::const_iterator end(const PortCorrespondences& correspondences);

        std::vector<PortCorrespondence> _correspondences;
    };

    //
    // NullNode -- used for deleting nodes
    //
    template <typename ValueType>
    class NullNode : public model::Node
    {
    public:
        const model::OutputPort<ValueType>& output = _output;

        NullNode(size_t size) :
            Node({}, { &_output }),
            _output(this, defaultOutputPortName, size){};
        static std::string GetTypeName() { return utilities::GetCompositeTypeName<ValueType>("NullNode"); }
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

    protected:
        void Compute() const override{};
        void WriteToArchive(utilities::Archiver& archiver) const override{};
        void ReadFromArchive(utilities::Unarchiver& archiver) override{};

    private:
        void Copy(model::ModelTransformer& transformer) const override
        {
            auto newNode = transformer.AddNode<NullNode<ValueType>>(_output.Size());
            transformer.MapNodeOutput(output, newNode->output);
        }

        model::OutputPort<ValueType> _output;
    };

    //
    // PortCorrespondences
    //
    void VerifyOntoModel(const Model& destModel, const std::vector<const OutputPortBase*>& onto);
    void VerifyOntoCorrespondences(const std::vector<const InputPortBase*>& srcInputs, const std::vector<const OutputPortBase*>& onto);

    PortCorrespondences::PortCorrespondences(const std::vector<const InputPortBase*>& sources, const std::vector<const OutputPortBase*>& destinations)
    {
        VerifyOntoCorrespondences(sources, destinations);
        for (size_t i = 0, end = sources.size(); i < end; ++i)
        {
            _correspondences.push_back({ sources[i], destinations[i] });
        }
    }

    void PortCorrespondences::Add(const PortCorrespondence& correspondence)
    {
        _correspondences.push_back(correspondence);
    }

    PortCorrespondence PortCorrespondences::operator[](int index) const
    {
        return _correspondences[index];
    }

    PortCorrespondences::iterator begin(PortCorrespondences& correspondences)
    {
        return begin(correspondences._correspondences);
    }

    PortCorrespondences::iterator end(PortCorrespondences& correspondences)
    {
        return end(correspondences._correspondences);
    }

    PortCorrespondences::const_iterator begin(const PortCorrespondences& correspondences)
    {
        return begin(correspondences._correspondences);
    }

    PortCorrespondences::const_iterator end(const PortCorrespondences& correspondences)
    {
        return end(correspondences._correspondences);
    }

    //
    // PortOutputsMap
    //
    void ModelTransformer::PortOutputsMap::Clear()
    {
        _outputPortMap.clear();
    }

    bool ModelTransformer::PortOutputsMap::IsEmpty() const
    {
        return _outputPortMap.empty();
    }

    bool ModelTransformer::PortOutputsMap::IsOutputMapped(const OutputPortBase& queryPort) const
    {
        auto queryPortPtr = &queryPort;
        return (_outputPortMap.find(queryPortPtr) != _outputPortMap.end());
    }

    const OutputPortBase& ModelTransformer::PortOutputsMap::GetCorrespondingPort(const OutputPortBase& queryPort, bool isInPlace) const
    {
        auto queryPortPtr = &queryPort;
        if (!IsOutputMapped(queryPort))
        {
            // If we don't have a mapping from old->new for this port, it may be because we elided a copy
            // during an in-place transformation.
            // TODO: find a way to tell the differerence between a port that doesn't have a mapping because
            //   of an in-place (elided) copy, and one that doesn't have a mapping because the query is in error
            return queryPort;
        }

        auto targetPort = _outputPortMap.at(queryPortPtr);

        if (targetPort->Size() != queryPort.Size())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::sizeMismatch,
                                            utilities::FormatString("Model transformation resulted in a mismatching port size, expecting %lld, but found %lld", queryPort.Size(), targetPort->Size()));
        }
        return *targetPort;
    }

    void ModelTransformer::PortOutputsMap::MapNodeOutput(const OutputPortBase* oldPort, const OutputPortBase* newPort)
    {
        if (oldPort->Size() != newPort->Size())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::sizeMismatch,
                                            utilities::FormatString("Trying to map port %s to output of different size, expecting %lld, but found %lld", oldPort->GetName().c_str(), oldPort->Size(), newPort->Size()));
        }
        _outputPortMap[oldPort] = newPort;
    }

    ModelTransformer::PortOutputsMap ModelTransformer::PortOutputsMap::ConcatenateMaps(const PortOutputsMap& prevMap, const PortOutputsMap& newMap, bool isInPlace)
    {
        PortOutputsMap result;
        for (const auto& entry : prevMap._outputPortMap)
        {
            const auto& newMappedValue = newMap.GetCorrespondingPort(*entry.second, isInPlace);
            result.MapNodeOutput(entry.first, &newMappedValue);
        }

        return result;
    }

    //
    // ModelTransformer implementation
    //
    const OutputPortBase& ModelTransformer::SimplifyOutputs(const PortElementsBase& elements)
    {
        return _model.SimplifyOutputs(elements);
    }

    Model ModelTransformer::CopyModel(const Model& oldModel)
    {
        TransformContext context;
        return CopyModel(oldModel, context);
    }

    Model ModelTransformer::CopyModel(const Model& oldModel, const TransformContext& context)
    {
        Submodel m(oldModel);
        auto result = CopySubmodel(m, context);
        return std::move(result.GetModel());
    }

    Submodel ModelTransformer::CopySubmodel(const Submodel& submodel, const TransformContext& context)
    {
        Model destModel;
        destModel.GetMetadata() = submodel.GetModel().GetMetadata();
        auto result = TransformSubmodelOnto(submodel, destModel, {}, context, [](const Node& node, ModelTransformer& transformer) {
            transformer.CopyNode(node);
        });

        ResetContext();
        return result;
    }

    Submodel ModelTransformer::CopySubmodelOnto(const Submodel& submodel, const std::vector<const OutputPortBase*>& onto, const TransformContext& context)
    {
        if (onto.empty())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Location to transform submodel onto is empty");
        }
        auto destModel = onto[0]->GetNode()->GetModel()->ShallowCopy();
        return CopySubmodelOnto(submodel, destModel, onto, context);
    }

    Submodel ModelTransformer::CopySubmodelOnto(const Submodel& submodel, Model& destModel, const std::vector<const OutputPortBase*>& onto, const TransformContext& context)
    {
        auto result = TransformSubmodelOnto(submodel, destModel, onto, context, [](const Node& node, ModelTransformer& transformer) {
            transformer.CopyNode(node);
        });

        ResetContext();
        return result;
    }

    bool ModelTransformer::IsModelCompilable(const Model& model) const
    {
        auto iter = model.GetNodeIterator();
        while (iter.IsValid())
        {
            if (!_context.IsNodeCompilable(*iter.Get()))
            {
                return false;
            }
            iter.Next();
        }
        return true;
    }

    bool ModelTransformer::IsInPlace() const
    {
        return _isInPlace;
    }

    bool ModelTransformer::ShouldCopyNode(const Node& node) const
    {
        // Always copy if we're not in in-place mode
        if (!IsInPlace())
            return true;

        const auto& inputs = node.GetInputPorts();
        for (auto in : inputs)
        {
            if (HasNontrivialInputMapping(*in))
            {
                return true;
            }
        }

        return false;
    }

    void ModelTransformer::MapNodeOutput(const OutputPortBase& oldPort, const OutputPortBase& newPort)
    {
        _elementsMap.MapNodeOutput(&oldPort, &newPort);
    }

    bool ModelTransformer::IsInputMapped(const InputPortBase& input) const
    {
        return _elementsMap.IsOutputMapped(input.GetReferencedPort());
    }

    bool ModelTransformer::HasNontrivialInputMapping(const InputPortBase& input) const
    {
        if (!_elementsMap.IsOutputMapped(input.GetReferencedPort()))
        {
            return false;
        }

        const auto& referencedPort = input.GetReferencedPort();
        return &GetCorrespondingOutputs(referencedPort) != &referencedPort;
    }

    bool ModelTransformer::IsOutputMapped(const OutputPortBase& output) const
    {
        return _elementsMap.IsOutputMapped(output);
    }

    bool ModelTransformer::IsInputNode(const Node& node) const
    {
        return dynamic_cast<const InputNodeBase*>(&node) != nullptr;
    }

    bool ModelTransformer::ArePortsCompatible(const InputPortBase* source, const OutputPortBase* dest)
    {
        return (source->Size() == dest->Size()) && (source->GetType() == dest->GetType());
    }

    void ModelTransformer::MapCorrespondingInputs(const std::vector<const InputPortBase*>& sources, const std::vector<const OutputPortBase*>& destinations)
    {
        // Set up port mapping between the given source inputs and their corresponding destination inputs
        // throw an exception if the inputs are ill-formed
        PortCorrespondences correspondences(sources, destinations);
        for (auto correspondence : correspondences)
        {
            auto source = correspondence.source;
            auto dest = correspondence.destination;
            if (!ArePortsCompatible(source, dest))
            {
                throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Incompatible source and destination inputs");
            }
            MapNodeOutput(source->GetReferencedPort(), *dest);
        }
    }

    Model ModelTransformer::TransformModel(const Model& model, const TransformContext& context, const NodeTransformFunction& transformFunction)
    {
        Model newModel;
        Submodel submodel(model);
        auto result = TransformSubmodelOnto(submodel, newModel, {}, context, transformFunction);
        return result.GetModel().ShallowCopy();
    }

    // Transforms the submodel by visiting its nodes in execution order (visit all inputs to a node before visiting that node)
    Submodel ModelTransformer::TransformSubmodelOnto(const Submodel& submodel, const Model& destModel, const std::vector<const OutputPortBase*>& onto, const TransformContext& context, const NodeTransformFunction& transformFunction)
    {
        _context = context;
        _model = destModel.ShallowCopy();
        _isInPlace = (submodel.GetModel() == destModel);
        auto previousElementMap = std::move(_elementsMap);
        _elementsMap.Clear();

        VerifyOntoModel(destModel, onto);
        VerifyOntoCorrespondences(submodel.GetInputs(), onto);

        std::vector<const InputPortBase*> sourceInputs = submodel.GetInputs();
        MapCorrespondingInputs(sourceInputs, onto);

        // This call to VisitSubmodel does the actual transformation (creating new nodes, etc.)
        submodel.GetModel().VisitSubmodel(sourceInputs, submodel.GetOutputs(), [this, transformFunction](const Node& node) {
            transformFunction(node, *this);
            AssignNodeAncestor(node);
        });

        if (!previousElementMap.IsEmpty())
        {
            // Now we have 2 maps, the previous one mapping A->B, and a new one mapping B->C (in _elementsMap).
            // Concatenate them to get a map A->C, and keep it.
            auto newElementsMap = PortOutputsMap::ConcatenateMaps(previousElementMap, _elementsMap, IsInPlace());
            _elementsMap = newElementsMap;
        }

        ResetContext();

        auto newOutputs = GetCorrespondingOutputs(submodel.GetOutputs());

        // This part of the code discovers the corresponding outputs in the transformed submodel for each of the outputs in the original submodel

        // Now visit the new outputs until we encounter an input port referencing a port from the "onto" list. That will be one of the new inputs. We should encounter one for each of the "onto" items.
        std::vector<const InputPortBase*> newInputs(submodel.GetInputs().size(), nullptr);
        std::unordered_map<const OutputPortBase*, std::vector<int>> ontoPortToIndexMap;
        for (int i = 0, size = (int)onto.size(); i < size; ++i)
        {
            ontoPortToIndexMap[onto[i]].push_back(i);
        }
        destModel.VisitSubmodel(newOutputs, [&ontoPortToIndexMap, &newInputs](const Node& node) {
            // check node's input ports -- if one of them references one of the onto ports, set it in the newInputs vector at the corresponding location
            for (auto input : node.GetInputPorts())
            {
                auto referencedPort = &input->GetReferencedPort();
                if (ontoPortToIndexMap.find(referencedPort) != ontoPortToIndexMap.end())
                {
                    for (auto index : ontoPortToIndexMap[referencedPort])
                    {
                        newInputs[index] = input;
                    }
                }
            }
        });

        for (auto input : newInputs)
        {
            if (input == nullptr)
            {
                throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState, "Found a null input in copied submodel");
            }
        }

        if (newInputs.empty() && newOutputs.empty())
        {
            return { destModel };
        }
        return { newInputs, newOutputs };
    }

    void VerifyOntoModel(const Model& destModel, const std::vector<const OutputPortBase*>& destinations)
    {
        for (auto destination : destinations)
        {
            if (*destination->GetNode()->GetModel() != destModel)
            {
                throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Destination model and destination outputs don't match");
            }
        }
    }

    void VerifyOntoCorrespondences(const std::vector<const InputPortBase*>& sources, const std::vector<const OutputPortBase*>& destinations)
    {
        if (sources.size() != destinations.size())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Trying to graft a submodel with " + std::to_string(sources.size()) + " outputs onto a destination with " + std::to_string(destinations.size()) + " outputs");
        }

        bool ok = std::equal(sources.begin(), sources.end(), destinations.begin(), [](const InputPortBase* s, const OutputPortBase* d) {
            return s->GetType() == d->GetType();
        });

        if (!ok)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Trying to graft a submodel onto a destination with an incompatible port type");
        }
    }

    Submodel ModelTransformer::TransformSubmodelOnto(const Submodel& submodel, const std::vector<const OutputPortBase*>& onto, const TransformContext& context, const NodeTransformFunction& transformFunction)
    {
        if (onto.empty())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Location to transform submodel onto is empty");
        }
        auto destModel = onto[0]->GetNode()->GetModel()->ShallowCopy();
        return TransformSubmodelOnto(submodel, destModel, onto, context, transformFunction);
    }

    Submodel ModelTransformer::TransformSubmodel(const Submodel& submodel, const TransformContext& context, const NodeTransformFunction& transformFunction)
    {
        return TransformSubmodelOnto(submodel, submodel.GetModel(), {}, context, transformFunction);
    }

    void ModelTransformer::ResetContext()
    {
        _context = TransformContext();
    }

    const OutputPortBase& ModelTransformer::GetCorrespondingInputs(const InputPortBase& port) const
    {
        return GetCorrespondingOutputs(port);
    }

    const OutputPortBase& ModelTransformer::GetCorrespondingOutputs(const InputPortBase& port) const
    {
        return GetCorrespondingOutputs(port.GetReferencedPort());
    }

    const OutputPortBase& ModelTransformer::GetCorrespondingOutputs(const OutputPortBase& port) const
    {
        return _elementsMap.GetCorrespondingPort(port, IsInPlace());
    }

    std::vector<const OutputPortBase*> ModelTransformer::GetCorrespondingOutputs(const std::vector<const OutputPortBase*>& ports) const
    {
        std::vector<const OutputPortBase*> result;
        for (auto output : ports)
        {
            result.push_back(&GetCorrespondingOutputs(*output));
        }
        return result;
    }

    const OutputPortBase& ModelTransformer::GetCorrespondingOutputs(const PortElementsBase& elements) const
    {
        return GetCorrespondingOutputs(*elements.GetRanges()[0].ReferencedPort());
    }

    InputNodeBase* ModelTransformer::GetCorrespondingInputNode(const InputNodeBase* inputNode) const
    {
        return GetCorrespondingInputNodeAs(inputNode);
    }

    void ModelTransformer::DeleteNode(const Node& node)
    {
        using PortType = Port::PortType;

        const auto& outputPorts = node.GetOutputPorts();
        for (auto outputPort : outputPorts)
        {
            auto layout = outputPort->GetMemoryLayout().GetExtent();
            switch (outputPort->GetType())
            {
            case PortType::boolean: {
                auto outputNode = AddNode<NullNode<bool>>(outputPort->Size());
                MapNodeOutput(*static_cast<const OutputPort<bool>*>(outputPort), outputNode->output);
                break;
            }
            case PortType::integer: {
                auto outputNode = AddNode<NullNode<int>>(outputPort->Size());
                MapNodeOutput(*static_cast<const OutputPort<int>*>(outputPort), outputNode->output);
                break;
            }
            case PortType::bigInt: {
                auto outputNode = AddNode<NullNode<std::int64_t>>(outputPort->Size());
                MapNodeOutput(*static_cast<const OutputPort<std::int64_t>*>(outputPort), outputNode->output);
                break;
            }
            case PortType::smallReal: {
                auto outputNode = AddNode<NullNode<float>>(outputPort->Size());
                MapNodeOutput(*static_cast<const OutputPort<float>*>(outputPort), outputNode->output);
                break;
            }
            case PortType::real: {
                auto outputNode = AddNode<NullNode<double>>(outputPort->Size());
                MapNodeOutput(*static_cast<const OutputPort<double>*>(outputPort), outputNode->output);
                break;
            }
            default:
                throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Unknown port type");
            }
        }
    }

    void ModelTransformer::CopyNode(const Node& node)
    {
        CopyNodeWithMetadata(node, {});
    }

    void ModelTransformer::CopyNodeWithMetadata(const Node& node, const utilities::PropertyBag& metadata)
    {
        if (ShouldCopyNode(node) || !metadata.IsEmpty())
        {
            node.Copy(*this);

            if (IsOutputMapped(*node.GetOutputPort(0)))
            {
                const auto& port = GetCorrespondingOutputs(*node.GetOutputPort(0));
                auto newNode = const_cast<Node*>(port.GetNode());
                if (newNode != (&node))
                {
                    // copy metadata
                    newNode->GetMetadata() = node.GetMetadata();
                }

                // append supplied metadata
                for (const auto& entry : metadata)
                {
                    newNode->GetMetadata()[entry.first] = entry.second;
                }
            }
        }
        else
        {
            // If we don't make a copy of the node, record the self->self mapping
            // so that subsequent queries will find it
            for (auto output : node.GetOutputPorts())
            {
                MapNodeOutput(*output, *output);
            }
        }
    }

    bool ModelTransformer::RefineNode(const Node& node)
    {
        // If the node action is "refine" or the default, try to refine the node, otherwise leave it alone
        auto action = GetContext().GetNodeAction(node);
        if (action == NodeAction::refine || action == NodeAction::abstain)
        {
            Log() << "Attempting to refine " << node.GetRuntimeTypeName() << " [id = " << node.GetId().ToString() << "]" << EOL;

            auto didRefineNode = node.Refine(*this);
            AssignNodeAncestor(node);
            return didRefineNode;
        }
        else
        {
            CopyNode(node);
        }

        return false;
    }

    void ModelTransformer::AssignNodeAncestor(const Node& ancestorNode)
    {
        auto iter = _model.GetReverseNodeIterator();
        while (iter.IsValid())
        {
            auto node = const_cast<Node*>(iter.Get());
            if (node->GetMetadata().HasEntry("ancestor"))
            {
                break;
            }
            else
            {
                if (ancestorNode.GetMetadata().HasEntry("ancestor"))
                {
                    node->GetMetadata().SetEntry("ancestor", ancestorNode.GetMetadata().GetEntry<std::string>("ancestor"));
                }
                else
                {
                    node->GetMetadata().SetEntry("ancestor", ancestorNode.GetId().ToString());
                }
            }
            iter.Next();
        }
    }
} // namespace model
} // namespace ell
