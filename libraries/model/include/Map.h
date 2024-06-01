////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Map.h (model)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "InputNode.h"
#include "Node.h"
#include "PortElements.h"
#include "Submodel.h"
#include "Transformation.h"

#include <data/include/DataVector.h>
#include <data/include/StlIndexValueIterator.h>

#include <utilities/include/Exception.h>
#include <utilities/include/IArchivable.h>
#include <utilities/include/PropertyBag.h>
#include <utilities/include/StlVectorUtil.h>
#include <utilities/include/TypeName.h>
#include <utilities/include/TypeTraits.h>

#include <value/include/ComputeContext.h>
#include <value/include/EmitterContext.h>

#include <algorithm>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ell
{
namespace model
{
    class ModelTransformer;
    class OutputNodeBase;
    class TransformContext;

    enum class InputMethod
    {
        InputNode,
        SourceNode
    };

    enum class OutputMethod
    {
        OutputNode,
        SinkNode
    };

    /// <summary> Class that wraps a model and its designated outputs </summary>
    class Map : public utilities::IArchivable
    {
    public:
        Map() = default;

        /// <summary> Copy constructor </summary>
        ///
        /// <param name="other"> The other map. </param>
        Map(const Map& other);

        /// <summary> Assignment operator. </summary>
        ///
        /// <param name="other"> The other map. </param>
        /// <returns> A reference to this map. </returns>
        Map& operator=(Map other);

        /// <summary> Constructor </summary>
        ///
        /// <param name="model"> The model to wrap. A deep copy of this model is stored inside the map. </param>
        /// <param name="inputs"> A vector of name/value pairs for the inputs this map uses </param>
        /// <param name="outputs"> A vector of name/value pairs for the outputs this map generates </param>
        /// @{
        Map(const Model& model, const std::vector<std::pair<std::string, InputNodeBase*>>& inputs, const std::vector<std::pair<std::string, const OutputPortBase&>>& outputs);
        Map(Model&& model, const std::vector<std::pair<std::string, InputNodeBase*>>& inputs, const std::vector<std::pair<std::string, const OutputPortBase&>>& outputs);
        /// @}

        /// <summary> Gets the model wrapped by this map </summary>
        ///
        /// <returns> The `Model` </returns>
        const Model& GetModel() const { return _model; }

        /// <summary> Gets the model wrapped by this map </summary>
        ///
        /// <returns> The `Model` </returns>
        Model& GetModel() { return _model; }

        /// <summary> Gets the submodel wrapped by this map </summary>
        Submodel GetSubmodel();

        /// <summary> Computes the map's output from input values </summary>
        ///
        /// <param name="inputValues"> The input to the map </param>
        /// <returns> A vector of output values </returns>
        template <typename OutputType, typename InputType, utilities::IsFundamental<OutputType> OutputConcept = true, utilities::IsFundamental<InputType> InputConcept = true>
        std::vector<OutputType> Compute(const std::vector<InputType>& inputValues);

        /// <summary> Computes the map's output from input values </summary>
        ///
        /// <param name="inputValues"> The input to the map </param>
        /// <returns> A vector of output values </returns>
        template <typename OutputVectorType, typename InputVectorType, data::IsDataVector<OutputVectorType> OutputConcept = true, data::IsDataVector<InputVectorType> InputConcept = true>
        OutputVectorType Compute(const InputVectorType& inputValues);

        /// <summary> Reset the state of the model </summary>
        void Reset();

        /// <summary> Returns the number of inputs to the map </summary>
        ///
        /// <returns> The number of inputs to the map </returns>
        size_t NumInputs() const;

        /// <summary> Returns the size of the given map's input </summary>
        ///
        /// <param name="index"> The index of the input node (in case there is more than one)</param>
        /// <returns> The dimensionality of the map's input port </returns>
        size_t GetInputSize(size_t index = 0) const;

        /// <summary> Returns the shape of the map's input </summary>
        ///
        /// <param name="index"> The index of the input node (in case there is more than one)</param>
        /// <returns> The shape of the map's input </returns>
        MemoryShape GetInputShape(size_t index = 0) const;

        /// <summary> Returns the type of the map's input </summary>
        ///
        /// <param name="index"> The index of the input node (in case there is more than one)</param>
        /// <reutrns> The type of the map's input </summary>
        Port::PortType GetInputType(size_t index = 0) const;

        /// <summary> Returns the number of outputs from the map </summary>
        ///
        /// <returns> The number of outputs from the map </returns>
        size_t NumOutputs() const;

        /// <summary> Returns the size of the given map's output </summary>
        ///
        /// <param name="index"> The index of the output node (in case there is more than one)</param>
        /// <returns> The dimensionality of the map's output port </returns>
        size_t GetOutputSize(size_t index = 0) const;

        /// <summary> Returns the shape of the map's output </summary>
        ///
        /// <param name="index"> The index of the output node (in case there is more than one)</param>
        /// <returns> The dimensionality of the map's output port </returns>
        MemoryShape GetOutputShape(size_t index = 0) const;

        /// <summary> Returns the type of the map's output </summary>
        ///
        /// <param name="index"> The index of the output node (in case there is more than one)</param>
        /// <reutrns> The type of the map's output </summary>
        Port::PortType GetOutputType(size_t index = 0) const;

        /// <summary> Returns the number of sink nodes in the map </summary>
        ///
        /// <returns> The number of sink nodes in the map </returns>
        size_t GetNumSinkNodes() const;

        /// <summary> Returns the size of the given map's sink output </summary>
        ///
        /// <param name="index"> The index of the sink node (in case there is more than one)</param>
        /// <returns> The dimensionality of the map's sink node port </returns>
        size_t GetSinkOutputSize(size_t index = 0) const;

        /// <summary> Returns the shape of the map's sink output </summary>
        ///
        /// <param name="index"> The index of the sink node (in case there is more than one)</param>
        /// <returns> The dimensionality of the map's sink node port </returns>
        MemoryShape GetSinkOutputShape(size_t index = 0) const;

        /// <summary> Returns the type of the map's sink node output </summary>
        ///
        /// <param name="index"> The index of the sink node  (in case there is more than one)</param>
        /// <reutrns> The type of the map's sink node output </summary>
        Port::PortType GetSinkOutputType(size_t index = 0) const;

        /// <summary> Refines the model wrapped by this map. </summary>
        ///
        /// <param name="maxIterations"> The maximum number of refinement iterations. </param>
        void Refine(int maxIterations = 10);

        /// <summary> Refines the model wrapped by this map. </summary>
        ///
        /// <param name="context"> The TransformContext to use during refinement. </param>
        /// <param name="maxIterations"> The maximum number of refinement iterations. </param>
        void Refine(const TransformContext& context, int maxIterations = 10);

        /// <summary> Transforms the model wrapped by this map by applying a transformation function to each node </summary>
        ///
        /// <param name="transformFunction"> The function to apply on each node </param>
        void Transform(const std::function<void(const Node&, ModelTransformer&)>& transformFunction);

        /// <summary> Transforms the model wrapped by this map by applying a transformation function to each node </summary>
        ///
        /// <param name="context"> The TransformContext to use during the transformation </param>
        /// <param name="transformFunction"> The function to apply on each node </param>
        void Transform(const TransformContext& context, const std::function<void(const Node&, ModelTransformer&)>& transformFunction);

        /// <summary> Transforms the model wrapped by this map by applying a Transformation </summary>
        ///
        /// <param name="transformFunction"> The Transformation to apply to the model </param>
        void Transform(Transformation& transformation);

        /// <summary> Transforms the model wrapped by this map by applying a Transformation </summary>
        ///
        /// <param name="transformFunction"> The Transformation to apply to the model </param>
        /// <param name="context"> The TransformContext to use during the transformation </param>
        void Transform(Transformation& transformation, const TransformContext& context);

        /// <summary> Renames the model callbacks in this map. </summary>
        ///
        /// <param name="sourceCallbackName"> The new source callback name. </param>
        /// <param name="sinkCallbackName"> The new sink callback name. </param>
        void RenameCallbacks(const std::string& sourceCallbackName, const std::string& sinkCallbackName);

        //
        // ELL-Internal routines for getting information about inputs / outputs of the map
        // and doing type-safe operations.
        //

        /// <summary> Returns an input node </summary>
        ///
        /// <param name="index"> The index of the input </param>
        /// <returns> The input node </returns>
        InputNodeBase* GetInput(size_t index = 0) const;

        /// <summary> Returns an input node </summary>
        ///
        /// <param name="inputName"> The name of the input </param>
        /// <returns> The input node </returns>
        InputNodeBase* GetInput(const std::string& inputName) const;

        /// <summary> Returns the name of a map input </summary>
        ///
        /// <param name="index"> The index of the input </param>
        /// <returns> The name the map uses for this input </returns>
        std::string GetInputName(size_t index = 0) const;

        /// <summary> Returns the input nodes </summary>
        ///
        /// <returns> The input nodes </returns>
        const std::vector<InputNodeBase*>& GetInputs() { return _inputNodes; }

        /// <summary> Returns the input nodes </summary>
        ///
        /// <returns> The input nodes </returns>
        std::vector<const InputNodeBase*> GetInputNodes() const;

        /// <summary> Returns the source nodes </summary>
        ///
        /// <returns> The source nodes </returns>
        std::vector<const SourceNodeBase*> GetSourceNodes() const;

        /// <summary> Returns the output nodes </summary>
        ///
        /// <returns> The output nodes </returns>
        std::vector<const OutputNodeBase*> GetOutputNodes() const;

        /// <summary> Returns the sink nodes </summary>
        ///
        /// <returns> The sink nodes </returns>
        std::vector<const Node*> GetSinkNodes() const;

        /// <summary> Returns an output </summary>
        ///
        /// <param name="index"> The index of the output </param>
        /// <returns> The output </returns>
        const OutputPortBase& GetOutput(size_t index = 0) const;

        /// <summary> Returns an output </summary>
        ///
        /// <param name="outputName"> The name of the output </param>
        /// <returns> The output </returns>
        const OutputPortBase& GetOutput(const std::string& outputName) const;

        /// <summary> Returns the name of an output </summary>
        ///
        /// <param name="index"> The index of the output </param>
        /// <returns> The name the map uses for this output </returns>
        std::string GetOutputName(size_t index = 0) const;

        /// <summary> Returns an sink node </summary>
        ///
        /// <param name="index"> The index of the sink node </param>
        /// <returns> The output </returns>
        const Node* GetSinkNode(size_t index = 0) const;

        /// <summary> Returns the outputs </summary>
        ///
        /// <returns> The outputs </returns>
        const std::vector<const OutputPortBase*>& GetOutputs() const { return _outputs; }

        //
        // Routines for computing output (processing data)
        //

        /// <summary> Set a single InputNode's input </summary>
        ///
        /// <typeparam name="ValueType"> The datatype of the input node </typeparam>
        /// <param name="index"> index of the input node </param>
        /// <param name="inputValues"> The values to set on the input node </param>
        template <typename ValueType>
        void SetInputValue(int index, const std::vector<ValueType>& inputValues);

        /// <summary> Set a single InputNode's input </summary>
        ///
        /// <typeparam name="ValueType"> The datatype of the input node </typeparam>
        /// <param name="inputName"> The name assigned to the input node </param>
        /// <param name="inputValues"> The values to set on the input node </param>
        template <typename ValueType>
        void SetInputValue(const std::string& inputName, const std::vector<ValueType>& inputValues);

        /// <summary> Set a single InputNode's input </summary>
        ///
        /// <typeparam name="ValueType"> The datatype of the input node </typeparam>
        /// <param name="index"> The index of the input node </param>
        /// <param name="inputValues"> The values to set on the input node </param>
        template <typename DataVectorType, data::IsDataVector<DataVectorType> Concept = true>
        void SetInputValue(int index, const DataVectorType& inputValues);

        /// <summary> Set a single InputNode's input </summary>
        ///
        /// <typeparam name="ValueType"> The datatype of the input node </typeparam>
        /// <param name="index"> The index of the input node </param>
        /// <param name="inputValues"> The values to set on the input node </param>
        template <typename DataVectorType, data::IsDataVector<DataVectorType> Concept = true>
        void SetInputValue(const std::string& inputName, const DataVectorType& inputValues);

        /// <summary> Computes one of the map's outputs from its current input values </summary>
        ///
        /// <param name="index"> The index of the output </param>
        /// <returns> A vector of output values </returns>
        template <typename ValueType, utilities::IsFundamental<ValueType> = 0>
        std::vector<ValueType> ComputeOutput(int index);

        /// <summary> Computes one of the map's outputs from its current input values </summary>
        ///
        /// <param name="index"> The index of the output </param>
        /// <returns> A vector of output values </returns>
        template <typename DataVectorType, data::IsDataVector<DataVectorType> Concept = true>
        DataVectorType ComputeOutput(int index);

        /// <summary> Computes one of the map's outputs from its current input values </summary>
        ///
        /// <param name="outputName"> The name of the output </param>
        /// <returns> A vector of output values </returns>
        template <typename ValueType, utilities::IsFundamental<ValueType> = 0>
        std::vector<ValueType> ComputeOutput(const std::string& outputName);

        /// <summary> Computes one of the map's outputs from its current input values </summary>
        ///
        /// <param name="outputName"> The name of the output </param>
        /// <returns> A vector of output values </returns>
        template <typename DataVectorType, data::IsDataVector<DataVectorType> Concept = true>
        DataVectorType ComputeOutput(const std::string& outputName);

        /// <summary> Returns a `OutputPort` for the indicated map output </summary>
        ///
        /// <param name="outputIndex"> The zero-based index of the map output </param>
        /// <returns> The `OutputPort` for the indicated outputs </returns>
        template <typename ValueType>
        const OutputPort<ValueType>& GetOutput(size_t outputIndex) const;

        /// <summary> Returns a `OutputPort` for the indicated map output </summary>
        ///
        /// <param name="outputName"> The name of the map output </param>
        /// <returns> The `OutputPort` for the indicated outputs </returns>
        template <typename ValueType>
        const OutputPort<ValueType>& GetOutput(std::string outputName) const;

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "Map"; }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

        /// <summary> Get this object's metadata object. </summary>
        ///
        /// <returns> A reference to the PropertyBag containing the metadata for this object. </returns>
        utilities::PropertyBag& GetMetadata() { return _metadata; }

        /// <summary> Get this object's metadata object. </summary>
        ///
        /// <returns> A const reference to the PropertyBag containing the metadata for this object. </returns>
        const utilities::PropertyBag& GetMetadata() const { return _metadata; }

        /// <summary> Swaps the contents of two maps. </summary>
        ///
        /// <param name="a"> One of the maps to swap. </param>
        /// <param name="b"> The other map to swap. </param>
        friend void swap(Map& a, Map& b);

        /// <summary>Prune away unused parts of internal model. </summary>
        void Prune();

        /// <summary> Adds the given input node to the map. </summary>
        ///
        /// <param name="inputName"> Name of the input. </param>
        /// <param name="inputNode"> The input node. </param>
        void AddInput(const std::string& inputName, InputNodeBase* inputNode);

        /// <summary> Removes the inputs from the map. </summary>
        void RemoveInputs();

        /// <summary> Adds an output to the map. </summary>
        ///
        /// <param name="outputName"> Name of the output. </param>
        /// <param name="inputNode"> The output elements. </param>
        void AddOutput(const std::string& outputName, PortElementsBase outputElements);

        /// <summary> This exposes another way to call Compute when the model has multiple inputs and outputs
        /// These void* pointers must point to pre-allocated buffers of the right type (float*, double*, etc)
        /// and size. </summary>
        ///
        /// <param name="inputs"> A vector containing all the input buffers. </param>
        /// <param name="outputs"> A vector containing all the output buffers. </param>
        virtual void ComputeMultiple(const std::vector<void*>& inputs, const std::vector<void*>& outputs);

    protected:
        template <typename DataVectorType, typename ElementsType, data::IsDataVector<DataVectorType> Concept = true>
        void SetInputValue(InputNodeBase* node, const DataVectorType& inputValues);

        template <typename DataVectorType, data::IsDataVector<DataVectorType> Concept = true>
        void SetInputValue(InputNodeBase* node, const DataVectorType& inputValues);

        template <typename DataVectorType, typename ElementsType, data::IsDataVector<DataVectorType> Concept = true>
        DataVectorType ComputeOutput(const PortElementsBase& elements);

        template <typename ValueType, utilities::IsFundamental<ValueType> = 1>
        std::vector<ValueType> ComputeOutput(const PortElementsBase& elements);

        template <typename DataVectorType, data::IsDataVector<DataVectorType> Concept = true>
        DataVectorType ComputeOutput(const PortElementsBase& elements);

        utilities::ArchiveVersion GetArchiveVersion() const override;
        bool CanReadArchiveVersion(const utilities::ArchiveVersion& version) const override;
        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;

        virtual void SetNodeInput(InputNode<bool>* node, const std::vector<bool>& inputValues);
        virtual void SetNodeInput(InputNode<int>* node, const std::vector<int>& inputValues);
        virtual void SetNodeInput(InputNode<int64_t>* node, const std::vector<int64_t>& inputValues);
        virtual void SetNodeInput(InputNode<float>* node, const std::vector<float>& inputValues);
        virtual void SetNodeInput(InputNode<double>* node, const std::vector<double>& inputValues);

        virtual std::vector<bool> ComputeBoolOutput(const PortElementsBase& outputs);
        virtual std::vector<int> ComputeIntOutput(const PortElementsBase& outputs);
        virtual std::vector<int64_t> ComputeInt64Output(const PortElementsBase& outputs);
        virtual std::vector<float> ComputeFloatOutput(const PortElementsBase& outputs);
        virtual std::vector<double> ComputeDoubleOutput(const PortElementsBase& outputs);

    private:
        std::vector<const Node*> GetAllOutputNodes() const;
        std::vector<const Node*> GetDebugSinkNodes() const;
        std::vector<const Node*> GetMatchingNodesByType(const std::string name) const;
        void FixTransformedIO(ModelTransformer& transformer);

        Model _model;

        std::vector<InputNodeBase*> _inputNodes;
        std::vector<std::string> _inputNames;
        std::unordered_map<std::string, InputNodeBase*> _inputNodeMap;

        std::vector<const OutputPortBase*> _outputs;
        std::vector<std::string> _outputNames;
        std::unordered_map<std::string, const OutputPortBase*> _outputsMap;
        utilities::PropertyBag _metadata;

        value::ComputeContext _computeContext{ "map_compute" };
    };

    /// <summary> A serialization context used during Map deserialization. Wraps an existing `ModelSerializationContext` </summary>
    class MapSerializationContext : public ModelSerializationContext
    {
    public:
        /// <summary> Constructor </summary>
        ///
        /// <param name="previousContext"> The `SerializationContext` to wrap </param>
        MapSerializationContext(utilities::SerializationContext& previousContext);
    };
} // namespace model
} // namespace ell

#pragma region implementation

namespace ell
{
namespace model
{
    namespace MapImpl
    {
        template <typename T>
        T FromDouble(double x)
        {
            return static_cast<T>(x);
        }

        template <>
        inline bool FromDouble<bool>(double x)
        {
            return x != 0;
        }
    } // namespace MapImpl

    template <typename OutputType, typename InputType, utilities::IsFundamental<OutputType>, utilities::IsFundamental<InputType>>
    std::vector<OutputType> Map::Compute(const std::vector<InputType>& inputValues)
    {
        if (NumInputs() != 1)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Map::Compute can only be called on maps with a single input");
        }
        SetInputValue(0, inputValues);
        return ComputeOutput<OutputType>(GetOutput(0));
    }

    template <typename OutputVectorType, typename InputVectorType, data::IsDataVector<OutputVectorType>, data::IsDataVector<InputVectorType>>
    OutputVectorType Map::Compute(const InputVectorType& inputValues)
    {
        if (NumInputs() != 1)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Map::Compute can only be called on maps with a single input");
        }
        SetInputValue(0, inputValues);
        return ComputeOutput<OutputVectorType>(GetOutput(0));
    }

    //
    // SetInput
    //
    template <typename DataVectorType, typename ElementsType, data::IsDataVector<DataVectorType>>
    void Map::SetInputValue(InputNodeBase* node, const DataVectorType& inputValues)
    {
        auto inputSize = node->GetOutputPort().Size();
        auto inputArray = inputValues.ToArray(inputSize);
        std::vector<ElementsType> array = utilities::TransformVector(inputArray.begin(), inputArray.end(), [](auto x) { return MapImpl::FromDouble<ElementsType>(x); });
        auto typedNode = static_cast<InputNode<ElementsType>*>(node);

        SetNodeInput(typedNode, array);
    }

    template <typename DataVectorType, data::IsDataVector<DataVectorType>>
    void Map::SetInputValue(InputNodeBase* inputNode, const DataVectorType& inputValues)
    {
        switch (inputNode->GetOutputPort().GetType())
        {
        case Port::PortType::none:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument);
            break;
        case Port::PortType::smallReal:
            SetInputValue<DataVectorType, float>(inputNode, inputValues);
            break;
        case Port::PortType::real:
            SetInputValue<DataVectorType, double>(inputNode, inputValues);
            break;
        case Port::PortType::integer:
            SetInputValue<DataVectorType, int>(inputNode, inputValues);
            break;
        case Port::PortType::bigInt:
            SetInputValue<DataVectorType, int64_t>(inputNode, inputValues);
            break;
        case Port::PortType::categorical:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument);
            break;
        case Port::PortType::boolean:
            SetInputValue<DataVectorType, bool>(inputNode, inputValues);
            break;
        default:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument);
        }
    }

    // By name
    template <typename ValueType>
    void Map::SetInputValue(const std::string& inputName, const std::vector<ValueType>& inputValues)
    {
        auto node = dynamic_cast<InputNode<ValueType>*>(GetInput(inputName));
        if (node == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::typeMismatch);
        }

        SetNodeInput(node, inputValues);
    }

    template <typename DataVectorType, data::IsDataVector<DataVectorType>>
    void Map::SetInputValue(const std::string& inputName, const DataVectorType& inputValues)
    {
        auto node = GetInput(inputName);
        SetInputValue(node, inputValues);
    }

    // By index
    template <typename ValueType>
    void Map::SetInputValue(int index, const std::vector<ValueType>& inputValues)
    {
        auto node = dynamic_cast<InputNode<ValueType>*>(GetInput(index));
        if (node == nullptr)
        {
            std::string nodeType = "missing InputNode<";
            throw utilities::InputException(utilities::InputExceptionErrors::typeMismatch, nodeType + utilities::TypeName<ValueType>::GetName() + ">");
        }

        SetNodeInput(node, inputValues);
    }

    template <typename DataVectorType, data::IsDataVector<DataVectorType>>
    void Map::SetInputValue(int index, const DataVectorType& inputValues)
    {
        auto node = GetInput(index);
        SetInputValue(node, inputValues);
    }

    //
    // ComputeOutput
    //

    template <typename OutputDataVectorType, typename ElementsValueType, data::IsDataVector<OutputDataVectorType>>
    OutputDataVectorType Map::ComputeOutput(const PortElementsBase& elements)
    {
        value::ContextGuard<> guard(_computeContext);

        auto resultVector = ComputeOutput<ElementsValueType>(elements);
        auto resultVectorIterator = data::MakeVectorIndexValueIterator<data::IterationPolicy::skipZeros>(resultVector);
        return { resultVectorIterator };
    }

    template <typename DataVectorType, data::IsDataVector<DataVectorType>>
    DataVectorType Map::ComputeOutput(const PortElementsBase& elements)
    {
        value::ContextGuard<> guard(_computeContext);

        switch (elements.GetPortType())
        {
        case Port::PortType::none:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument);
            break;
        case Port::PortType::smallReal:
            return ComputeOutput<DataVectorType, float>(elements);
            break;
        case Port::PortType::real:
            return ComputeOutput<DataVectorType, double>(elements);
            break;
        case Port::PortType::integer:
            return ComputeOutput<DataVectorType, int>(elements);
            break;
        case Port::PortType::bigInt:
            return ComputeOutput<DataVectorType, int64_t>(elements);
            break;
        case Port::PortType::categorical:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument);
            break;
        case Port::PortType::boolean:
            return ComputeOutput<DataVectorType, bool>(elements);
            break;
        default:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument);
        }
    }

    // By index
    template <typename ValueType, utilities::IsFundamental<ValueType>>
    std::vector<ValueType> Map::ComputeOutput(int index)
    {
        value::ContextGuard<> guard(_computeContext);

        return ComputeOutput<ValueType>(GetOutput(index));
    }

    template <typename DataVectorType, data::IsDataVector<DataVectorType>>
    DataVectorType Map::ComputeOutput(int index)
    {
        value::ContextGuard<> guard(_computeContext);

        return ComputeOutput<DataVectorType>(GetOutput(index));
    }

    // By name
    template <typename ValueType, utilities::IsFundamental<ValueType>>
    std::vector<ValueType> Map::ComputeOutput(const std::string& outputName)
    {
        value::ContextGuard<> guard(_computeContext);

        return ComputeOutput<ValueType>(GetOutput(outputName));
    }

    template <typename DataVectorType, data::IsDataVector<DataVectorType>>
    DataVectorType Map::ComputeOutput(const std::string& outputName)
    {
        value::ContextGuard<> guard(_computeContext);

        return ComputeOutput<DataVectorType>(GetOutput(outputName));
    }

    template <typename ValueType>
    const OutputPort<ValueType>& Map::GetOutput(size_t outputIndex) const
    {
        return static_cast<const OutputPort<ValueType>&>(GetOutput(outputIndex));
    }

    template <typename ValueType>
    const OutputPort<ValueType>& Map::GetOutput(std::string outputName) const
    {
        return static_cast<const OutputPort<ValueType>&>(GetOutput(outputName));
    }

} // namespace model
} // namespace ell

#pragma endregion implementation
