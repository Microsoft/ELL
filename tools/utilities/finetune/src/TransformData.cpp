////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     TransformData.cpp (utilities)
//  Authors:  Byron Changuion, Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TransformData.h"
#include "DataUtils.h"
#include "ModelUtils.h"

#include <model/include/Map.h>

#include <passes/include/StandardTransformations.h>

#include <utilities/include/Logger.h>
#include <utilities/include/MemoryLayout.h>

#include <memory>
#include <vector>

#include <iostream>

namespace ell
{
using namespace model;
namespace
{
    template <typename T1, typename T2>
    std::vector<T1> CastVector(const std::vector<T2>& v)
    {
        auto size = v.size();
        std::vector<T1> result(size);
        for (size_t i = 0; i < size; ++i)
        {
            result[i] = static_cast<T1>(v[i]);
        }
        return result;
    }

    template <typename DataVector, typename LabelType>
    void AddExample(UnlabeledDataContainer& dataset, const DataVector& example, const LabelType& label)
    {
        dataset.Add(CastVector<float>(example));
    }

    template <typename DataVector, typename LabelType>
    void AddExample(BinaryLabelDataContainer& dataset, const DataVector& data, const LabelType& label)
    {
        dataset.push_back({ CastVector<float>(data), label });
    }

    template <typename DataVector, typename LabelType>
    void AddExample(MultiClassDataContainer& dataset, const DataVector& data, const LabelType& label)
    {
        dataset.Add({ CastVector<float>(data), label });
    }

    template <typename DataVector, typename LabelType>
    void AddExample(VectorLabelDataContainer& dataset, const DataVector& data, const LabelType& label)
    {
        dataset.push_back({ CastVector<float>(data), label });
    }

    template <typename ExampleType>
    auto GetInput(const ExampleType& example)
    {
        return example.input;
    }

    template <typename ExampleType>
    auto GetOutput(const ExampleType& example)
    {
        return example.output;
    }

    template <>
    auto GetInput(const UnlabeledExample& example)
    {
        return example;
    }

    template <>
    auto GetOutput(const UnlabeledExample& example)
    {
        return 0;
    }

    template <typename DataVectorType, typename Visitor>
    void Visit(const DataVectorType& data, const utilities::MemoryLayout& layout, int currentDimension, int currentOffset, Visitor visitor)
    {
        auto offset = layout.GetOffset(currentDimension);
        auto length = layout.GetActiveSize(currentDimension);
        bool isInnerLoop = currentDimension == layout.NumDimensions() - 1;
        for (int i = 0; i < length; ++i)
        {
            auto index = (i + offset) * layout.GetCumulativeIncrement(currentDimension);
            if (isInnerLoop)
            {
                visitor(data[index + currentOffset]);
            }
            else
            {
                Visit(data, layout, currentDimension + 1, currentOffset + index, visitor);
            }
        }
    }

    template <typename DataVectorType, typename Visitor>
    void Visit(const DataVectorType& data, const utilities::MemoryLayout& layout, Visitor visitor)
    {
        Visit(data, layout, 0, 0, visitor);
    }

    const OutputPortBase& AppendInputNode(Model& model, Port::PortType type, const PortMemoryLayout& layout)
    {
        switch (type)
        {
        case Port::PortType::smallReal: {
            return model.AddNode<InputNode<float>>(layout)->output;
            break;
        }
        case Port::PortType::real: {
            return model.AddNode<InputNode<double>>(layout)->output;
            break;
        }
        default:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Unexpected input type for model. Should be double or float.");
            break;
        };
    }

    // Check if we can use cached data at the given port to compute submodel's output
    bool IsValidCachePort(const OutputPortBase& begin, const OutputPortBase& end, const OutputPortBase& port)
    {
        if (&begin == &port || &end == &port)
        {
            return true;
        }
        if (&begin == &end)
        {
            return false;
        }
        auto node = end.GetNode();
        for (auto input : node->GetInputPorts())
        {
            if (!IsValidCachePort(begin, input->GetReferencedPort(), port))
            {
                return false;
            }
        }
        return true;
    }
} // namespace

// prototypes
template <typename ElementType>
UnlabeledDataContainer TransformDataWithSubmodelImpl(const UnlabeledDataContainer& dataset, ell::model::Submodel& submodel);
template <typename DataContainerType, typename ElementType>
DataContainerType TransformDataInputsWithModelImpl(const DataContainerType& dataset, const OutputPort<ElementType>& output);
template <typename DatasetType>
double GetModelAccuracyImpl(const OutputPortBase& output, const DatasetType& testDataset);
template <typename ElementType, typename DatasetType>
double GetModelAccuracyImpl(const OutputPort<ElementType>& output, const DatasetType& testDataset);

std::unique_ptr<Map> GetMapForModel(const OutputPortBase& output, bool compile, model::IRMapCompiler& compiler);
std::unique_ptr<Map> GetMapForSubmodel(Submodel& submodel, bool compile, model::IRMapCompiler& compiler);

// implementation
UnlabeledDataContainer TransformDataWithSubmodel(const UnlabeledDataContainer& dataset, ell::model::Submodel& submodel, ModelOutputDataCache& dataCache, bool cacheResult)
{
    using namespace logging;

    auto submodelInputs = submodel.GetInputs();
    auto submodelOutputs = submodel.GetOutputs();

    // submodels with multiple inputs/outputs are too hairy for this function to deal with currently
    if (submodelInputs.size() > 1 || submodelOutputs.size() > 1)
    {
        Log() << "Skipping caching for complex submodel" << EOL;
        return TransformDataWithSubmodel(dataset, submodel);
    }

    auto submodelOutput = submodelOutputs[0];
    const OutputPortBase* submodelInput = nullptr;
    if (submodelInputs.size() > 0)
    {
        submodelInput = &(submodelInputs[0]->GetReferencedPort());
    }
    else
    {
        auto inputNode = GetInputNode(*submodelOutput);
        submodelInput = &(inputNode->GetOutputPort());
    }

    // direct cache hit
    if (dataCache.HasCachedData(submodelOutput))
    {
        Log() << "Direct cache hit for port " << submodelOutput->GetFullName() << EOL;
        return dataCache.GetCachedData(submodelOutput);
    }

    auto transformDataset = std::cref(dataset);
    auto transformSubmodel = submodel;
    const OutputPortBase* cachedOutput = nullptr;

    submodel.Visit([&dataCache, &submodelInput, &submodelOutput, &cachedOutput](const Node& node) {
        for (auto output : node.GetOutputPorts())
        {
            // Check if we have cached data here, and if it's valid to use it
            if (dataCache.HasCachedData(output))
            {
                if ((GetInputFanIn(*output, *submodelOutput) == 1) && (IsValidCachePort(*submodelInput, *submodelOutput, *output)))
                {
                    Log() << "Considering cached output " << output->GetFullName() << EOL;
                    cachedOutput = output;
                }
                else
                {
                    Log() << "Cached output " << output->GetFullName() << " is invalid" << EOL;
                }
            }
        }
    });

    if (cachedOutput != nullptr)
    {
        ModelTransformer transformer;
        TransformContext context;
        // auto model = submodel.GetModel().ShallowCopy();

        // We want the input port along the search path that the input node is connected to
        auto originalInputReferences = GetInputReferences(*cachedOutput, *submodelOutput);
        Submodel submodelMinusInput = { originalInputReferences, { submodelOutput } };

        // Create a new model to copy the submodel to evaluate into it
        // First add an input node we can feed the cached dataset into
        Model transformModel;
        const auto& newInput = AppendInputNode(transformModel, cachedOutput->GetType(), cachedOutput->GetMemoryLayout());
        auto onto = std::vector<const OutputPortBase*>(originalInputReferences.size(), &newInput);
        auto newSubmodel = transformer.CopySubmodelOnto(submodelMinusInput, transformModel, onto, context);
        transformSubmodel = newSubmodel;
        transformDataset = dataCache.GetCachedData(cachedOutput);
    }

    auto result = TransformDataWithSubmodel(transformDataset, transformSubmodel);

    if (cacheResult && transformSubmodel.GetOutputs().size() == 1)
    {
        dataCache.SetCachedData(submodelOutput, result);
    }
    return result;
}

UnlabeledDataContainer TransformDataWithSubmodel(const UnlabeledDataContainer& dataset, ell::model::Submodel& submodel)
{
    switch (submodel.GetOutputs()[0]->GetType())
    {
    case Port::PortType::smallReal: {
        return TransformDataWithSubmodelImpl<float>(dataset, submodel);
        break;
    }
    case Port::PortType::real: {
        return TransformDataWithSubmodelImpl<double>(dataset, submodel);
        break;
    }
    default:
        throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Unexpected output type for model. Should be double or float.");
        break;
    };
}

template <typename ElementType>
UnlabeledDataContainer TransformDataWithSubmodelImpl(const UnlabeledDataContainer& dataset, ell::model::Submodel& submodel)
{
    return TransformDataWithModel(dataset, static_cast<const OutputPort<ElementType>&>(*submodel.GetOutputs()[0]));
}

template <typename ElementType>
UnlabeledDataContainer TransformDataWithModel(const UnlabeledDataContainer& dataset, const OutputPort<ElementType>& output)
{
    return TransformDataInputsWithModelImpl(dataset, output);
}

template <typename ElementType>
BinaryLabelDataContainer TransformDataInputsWithModel(const BinaryLabelDataContainer& dataset, const OutputPort<ElementType>& output)
{
    return TransformDataInputsWithModelImpl(dataset, output);
}

template <typename ElementType>
MultiClassDataContainer TransformDataInputsWithModel(const MultiClassDataContainer& dataset, const OutputPort<ElementType>& output)
{
    return TransformDataInputsWithModelImpl(dataset, output);
}

template <typename ElementType>
VectorLabelDataContainer TransformDataInputsWithModel(const VectorLabelDataContainer& dataset, const OutputPort<ElementType>& output)
{
    return TransformDataInputsWithModelImpl(dataset, output);
}

template <typename DataContainerType, typename ElementType>
DataContainerType TransformDataInputsWithModelImpl(const DataContainerType& dataset, const OutputPort<ElementType>& output)
{
    const bool compile = true;

    // This compiler stuff is here because compiled maps contain a reference to the module emitter inside the compiler.
    // If we were to just return the compiled map from `GetMapForModel`, the compiler object within that function would
    // disappear and we'd have a dangling reference.
    passes::AddStandardTransformationsToRegistry();
    MapCompilerOptions settings;
    settings.compilerSettings.targetDevice.deviceName = "host";
    settings.compilerSettings.allowVectorInstructions = true;
    settings.compilerSettings.optimize = true;
    model::ModelOptimizerOptions optimizerOptions;
    optimizerOptions["fuseLinearFunctionNodes"] = true;
    model::IRMapCompiler compiler(settings, optimizerOptions);

    auto map = GetMapForModel(output, compile, compiler);

    auto inputSize = map->GetInputSize(0);
    DataContainerType result;
    for (size_t i = 0; i < dataset.Size(); ++i)
    {
        auto example = dataset[i];
        const auto& exampleData = GetInput(example);
        std::vector<ElementType> input = CastVector<ElementType>(exampleData.ToArray());
        if (input.size() != inputSize)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "dataset has wrong number of elements -- expected " + std::to_string(inputSize) + ", but got " + std::to_string(input.size()));
        }
        auto pred = map->template Compute<ElementType>(input);
        AddExample(result, pred, GetOutput(example));
    }
    return result;
}

std::unique_ptr<Map> GetMapForModel(const OutputPortBase& output, bool compile, model::IRMapCompiler& compiler)
{
    Submodel submodel({ &output });
    return GetMapForSubmodel(submodel, compile, compiler);
}

std::unique_ptr<Map> GetMapForSubmodel(Submodel& inputSubmodel, bool compile, model::IRMapCompiler& compiler)
{
    TransformContext context;
    ModelTransformer transformer;
    const auto& output = *inputSubmodel.GetOutputs()[0];
    auto newSubmodel = transformer.CopySubmodel(inputSubmodel, context);
    const auto& newOutput = transformer.GetCorrespondingOutputs(output);
    const auto& newNewOutput = RemoveSourceAndSinkNodes(newOutput);

    Submodel submodel({ &newNewOutput });
    auto prunedSubmodel = transformer.CopySubmodel(submodel, context);
    const auto& prunedOutput = *prunedSubmodel.GetOutputs()[0];

    auto inputNode = GetInputNode(prunedOutput);
    if (!inputNode)
    {
        throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Couldn't find input nodes in model");
    }

    Map map(prunedSubmodel.GetModel(), { { "input", inputNode } }, { { "output", prunedOutput } });
    if (!compile)
    {
        return std::make_unique<Map>(map);
    }

    // Compile and optimize the model
    // passes::AddStandardTransformationsToRegistry();
    // MapCompilerOptions settings;
    // settings.compilerSettings.targetDevice.deviceName = "host";
    // settings.compilerSettings.allowVectorInstructions = true;
    // settings.compilerSettings.optimize = true;
    // model::ModelOptimizerOptions optimizerOptions;
    // optimizerOptions["fuseLinearFunctionNodes"] = true;
    // model::IRMapCompiler compiler(settings, optimizerOptions);

    auto compiledMap = compiler.Compile(map);
    compiledMap.FinishJitting();
    return std::make_unique<IRCompiledMap>(std::move(compiledMap));
}

double GetModelAccuracy(const OutputPortBase& output, const BinaryLabelDataContainer& testDataset)
{
    return GetModelAccuracyImpl(output, testDataset);
}

double GetModelAccuracy(const OutputPortBase& output, const MultiClassDataContainer& testDataset)
{
    return GetModelAccuracyImpl(output, testDataset);
}

template <typename DatasetType>
double GetModelAccuracyImpl(const OutputPortBase& output, const DatasetType& testDataset)
{
    switch (output.GetType())
    {
    case Port::PortType::smallReal: {
        return GetModelAccuracyImpl<float>(static_cast<const OutputPort<float>&>(output), testDataset);
        break;
    }
    case Port::PortType::real: {
        return GetModelAccuracyImpl<double>(static_cast<const OutputPort<double>&>(output), testDataset);
        break;
    }
    default:
        throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Unexpected output type for model. Should be double or float.");
        break;
    };
}

template <typename ElementType, typename DatasetType>
double GetModelAccuracyImpl(const OutputPort<ElementType>& output, const DatasetType& testDataset)
{
    auto predictions = GetDatasetInputs(TransformDataInputsWithModel(testDataset, output));
    return GetModelAccuracy(testDataset, predictions);
}

template <typename DataVectorType>
DataVectorType RemovePadding(const DataVectorType& data, const utilities::MemoryLayout& layout)
{
    DataVectorType result(layout.NumElements());
    int index = 0;
    Visit(data, layout, [&result, &index](auto value) {
        result[index] = value;
        ++index;
    });

    return result;
}

// Explicit instantiation definitions
template UnlabeledDataContainer TransformDataWithModel(const UnlabeledDataContainer& dataset, const OutputPort<float>& output);
template BinaryLabelDataContainer TransformDataInputsWithModel(const BinaryLabelDataContainer& dataset, const OutputPort<float>& output);
template MultiClassDataContainer TransformDataInputsWithModel(const MultiClassDataContainer& dataset, const OutputPort<float>& output);
template VectorLabelDataContainer TransformDataInputsWithModel(const VectorLabelDataContainer& dataset, const OutputPort<float>& output);

template UnlabeledDataContainer TransformDataWithModel(const UnlabeledDataContainer& dataset, const OutputPort<double>& output);
template BinaryLabelDataContainer TransformDataInputsWithModel(const BinaryLabelDataContainer& dataset, const OutputPort<double>& output);
template MultiClassDataContainer TransformDataInputsWithModel(const MultiClassDataContainer& dataset, const OutputPort<double>& output);
template VectorLabelDataContainer TransformDataInputsWithModel(const VectorLabelDataContainer& dataset, const OutputPort<double>& output);

template std::vector<float> RemovePadding(const std::vector<float>& data, const utilities::MemoryLayout& layout);
template std::vector<double> RemovePadding(const std::vector<double>& data, const utilities::MemoryLayout& layout);
} // namespace ell
