////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ModelBuilderInterface.cpp (interfaces)
//  Authors:  Chuck Jacobs, Kirk Olynyk
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ModelBuilderInterface.h"
#include "DatasetInterface.h"
#include "DatasetInterfaceImpl.h"
#include "NeuralLayersInterface.h"

#include <common/include/RegisterNodeCreators.h>

#include <dsp/include/FilterBank.h>

#include <math/include/Tensor.h>

#include <model/include/InputNode.h>
#include <model/include/InputNodeBase.h>
#include <model/include/OutputNode.h>
#include <model/include/SpliceNode.h>

#include <nodes/include/BinaryOperationNode.h>
#include <nodes/include/BroadcastOperationNodes.h>
#include <nodes/include/BufferNode.h>
#include <nodes/include/ClockNode.h>
#include <nodes/include/ConcatenationNode.h>
#include <nodes/include/DCTNode.h>
#include <nodes/include/DTWDistanceNode.h>
#include <nodes/include/DotProductNode.h>
#include <nodes/include/FFTNode.h>
#include <nodes/include/FastGRNNNode.h>
#include <nodes/include/FilterBankNode.h>
#include <nodes/include/GRUNode.h>
#include <nodes/include/HammingWindowNode.h>
#include <nodes/include/HannWindowNode.h>
#include <nodes/include/IIRFilterNode.h>
#include <nodes/include/LSTMNode.h>
#include <nodes/include/MatrixMatrixMultiplyCodeNode.h>
#include <nodes/include/MatrixMatrixMultiplyNode.h>
#include <nodes/include/MatrixVectorMultiplyNode.h>
#include <nodes/include/NeuralNetworkPredictorNode.h>
#include <nodes/include/NodeOperations.h>
#include <nodes/include/ReinterpretLayoutNode.h>
#include <nodes/include/ReorderDataCodeNode.h>
#include <nodes/include/ScalingNode.h>
#include <nodes/include/TypeCastNode.h>
#include <nodes/include/UnaryOperationNode.h>
#include <nodes/include/VoiceActivityDetectorNode.h>

#include <predictors/neural/include/ActivationLayer.h>
#include <predictors/neural/include/BatchNormalizationLayer.h>
#include <predictors/neural/include/BiasLayer.h>
#include <predictors/neural/include/BinaryConvolutionalLayer.h>
#include <predictors/neural/include/ConvolutionalLayer.h>
#include <predictors/neural/include/FullyConnectedLayer.h>
#include <predictors/neural/include/InputLayer.h>
#include <predictors/neural/include/PoolingLayer.h>
#include <predictors/neural/include/RegionDetectionLayer.h>
#include <predictors/neural/include/ScalingLayer.h>

#include <utilities/include/Boolean.h>
#include <utilities/include/Exception.h>

#include <algorithm>

namespace ELL_API
{

using namespace ell::utilities;

template <typename OutputType, typename InputType>
std::vector<OutputType> CastVector(const std::vector<InputType>& vector)
{
    std::vector<OutputType> result(vector.size());
    std::transform(vector.begin(), vector.end(), result.begin(), [](InputType x) { return static_cast<OutputType>(x); });
    return result;
}

namespace
{
    template <typename OutputType, typename InputType>
    std::vector<std::vector<OutputType>> CastVector(const std::vector<std::vector<InputType>>& vector)
    {
        std::vector<std::vector<OutputType>> result;
        result.reserve(vector.size());
        for (const auto& row : vector)
        {
            std::vector<OutputType> outRow(row.size());
            std::transform(row.begin(), row.end(), outRow.begin(), [](InputType x) { return static_cast<OutputType>(x); });
            result.push_back(outRow);
        }
        return result;
    }
} // namespace

//
// ModelBuilder
//
ModelBuilder::ModelBuilder()
{
    ell::common::RegisterNodeCreators(_modelBuilder);
}

Node ModelBuilder::AddNode(Model model, const std::string& nodeType, const std::vector<std::string>& args)
{
    ell::model::Node* newNode = _modelBuilder.AddNode(*(model.GetModel()), nodeType, args);
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddNeuralNetworkPredictorNode(Model model, PortElements input, ell::api::predictors::NeuralNetworkPredictor predictor)
{
    if (predictor.GetDataType() == PortType::real)
    {
        return AddNeuralNetworkPredictorNode<double>(model, input, predictor.GetUnderlyingPredictor<double>());
    }
    else if (predictor.GetDataType() == PortType::smallReal)
    {
        return AddNeuralNetworkPredictorNode<float>(model, input, predictor.GetUnderlyingPredictor<float>());
    }
    throw std::invalid_argument("Error: AddNeuralNetworkPredictorNode only supports PortType::real or PortType::smallReal ");
}

template <typename ElementType>
Node ModelBuilder::AddNeuralNetworkPredictorNode(Model model, PortElements input, const ell::predictors::NeuralNetworkPredictor<ElementType>& predictor)
{
    auto elements = ell::model::PortElements<ElementType>(input.GetPortElements());
    auto newNode = model.GetModel()->AddNode<ell::nodes::NeuralNetworkPredictorNode<ElementType>>(elements, predictor);
    return Node(newNode, model.GetModel());
}

InputNode ModelBuilder::AddInputNode(Model model, const PortMemoryLayout& memoryLayout, PortType type)
{
    auto outputLayout = memoryLayout.Get();
    using namespace std::string_literals;
    ell::model::InputNodeBase* newNode = nullptr;
    switch (type)
    {
    case PortType::boolean:
        newNode = model.GetModel()->AddNode<ell::model::InputNode<bool>>(outputLayout);
        break;
    case PortType::integer:
        newNode = model.GetModel()->AddNode<ell::model::InputNode<int>>(outputLayout);
        break;
    case PortType::bigInt:
        newNode = model.GetModel()->AddNode<ell::model::InputNode<int64_t>>(outputLayout);
        break;
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::model::InputNode<double>>(outputLayout);
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::model::InputNode<float>>(outputLayout);
        break;
    default:
        throw std::invalid_argument("Error: could not create InputNode of the requested type");
    }
    return InputNode(newNode, model.GetModel());
}

OutputNode ModelBuilder::AddOutputNode(Model model, const PortMemoryLayout& memoryLayout, PortElements input)
{
    auto outputLayout = memoryLayout.Get();
    auto outputShape = outputLayout.GetActiveSize();
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    ell::model::OutputNodeBase* newNode = nullptr;
    switch (type)
    {
    case PortType::boolean:
        newNode = model.GetModel()->AddNode<ell::model::OutputNode<bool>>(ell::model::PortElements<bool>(elements), outputShape);
        break;
    case PortType::integer:
        newNode = model.GetModel()->AddNode<ell::model::OutputNode<int>>(ell::model::PortElements<int>(elements), outputShape);
        break;
    case PortType::bigInt:
        newNode = model.GetModel()->AddNode<ell::model::OutputNode<int64_t>>(ell::model::PortElements<int64_t>(elements), outputShape);
        break;
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::model::OutputNode<double>>(ell::model::PortElements<double>(elements), outputShape);
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::model::OutputNode<float>>(ell::model::PortElements<float>(elements), outputShape);
        break;
    default:
        throw std::invalid_argument(std::string("Error: could not create OutputNode of the requested type") + typeid(type).name());
    }
    return OutputNode(newNode, model.GetModel());
}

Node ModelBuilder::AddClockNode(Model model, PortElements input, double interval, double lagThreshold, const std::string& lagNotificationName)
{
    auto elements = input.GetPortElements();
    auto newNode = model.GetModel()->AddNode<ell::nodes::ClockNode>(
        ell::model::PortElements<ell::nodes::TimeTickType>(elements),
        ell::nodes::TimeTickType(interval),
        ell::nodes::TimeTickType(lagThreshold),
        lagNotificationName);
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddTypeCastNode(Model model, PortElements input, PortType outputType)
{
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;
    switch (input.GetType())
    {
    case PortType::integer:
        switch (outputType)
        {
        case PortType::integer:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<int, int>>(ell::model::PortElements<int>(elements));
            break;
        case PortType::bigInt:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<int, int64_t>>(ell::model::PortElements<int>(elements));
            break;
        case PortType::real:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<int, double>>(ell::model::PortElements<int>(elements));
            break;
        case PortType::smallReal:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<int, float>>(ell::model::PortElements<int>(elements));
            break;
        case PortType::boolean:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<int, bool>>(ell::model::PortElements<int>(elements));
            break;
        default:
            throw std::invalid_argument("Error: CastNode unknown output type");
        }
        break;
    case PortType::bigInt:
        switch (outputType)
        {
        case PortType::integer:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<int64_t, int>>(ell::model::PortElements<int64_t>(elements));
            break;
        case PortType::bigInt:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<int64_t, int64_t>>(ell::model::PortElements<int64_t>(elements));
            break;
        case PortType::real:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<int64_t, double>>(ell::model::PortElements<int64_t>(elements));
            break;
        case PortType::smallReal:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<int64_t, float>>(ell::model::PortElements<int64_t>(elements));
            break;
        case PortType::boolean:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<int64_t, bool>>(ell::model::PortElements<int64_t>(elements));
            break;
        default:
            throw std::invalid_argument("Error: CastNode unknown output type");
        }
        break;
    case PortType::real:
        switch (outputType)
        {
        case PortType::integer:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<double, int>>(ell::model::PortElements<double>(elements));
            break;
        case PortType::bigInt:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<double, int64_t>>(ell::model::PortElements<double>(elements));
            break;
        case PortType::real:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<double, double>>(ell::model::PortElements<double>(elements));
            break;
        case PortType::smallReal:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<double, float>>(ell::model::PortElements<double>(elements));
            break;
        case PortType::boolean:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<double, bool>>(ell::model::PortElements<double>(elements));
            break;
        default:
            throw std::invalid_argument("Error: CastNode unknown output type");
        }
        break;
    case PortType::smallReal:
        switch (outputType)
        {
        case PortType::integer:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<float, int>>(ell::model::PortElements<float>(elements));
            break;
        case PortType::bigInt:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<float, int64_t>>(ell::model::PortElements<float>(elements));
            break;
        case PortType::real:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<float, double>>(ell::model::PortElements<float>(elements));
            break;
        case PortType::smallReal:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<float, float>>(ell::model::PortElements<float>(elements));
            break;
        case PortType::boolean:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<float, bool>>(ell::model::PortElements<float>(elements));
            break;
        default:
            throw std::invalid_argument("Error: CastNode unknown output type");
        }
        break;
    case PortType::boolean:
        switch (outputType)
        {
        case PortType::integer:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<bool, int>>(ell::model::PortElements<bool>(elements));
            break;
        case PortType::bigInt:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<bool, int64_t>>(ell::model::PortElements<bool>(elements));
            break;
        case PortType::real:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<bool, double>>(ell::model::PortElements<bool>(elements));
            break;
        case PortType::smallReal:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<bool, float>>(ell::model::PortElements<bool>(elements));
            break;
        case PortType::boolean:
            newNode = model.GetModel()->AddNode<ell::nodes::TypeCastNode<bool, bool>>(ell::model::PortElements<bool>(elements));
            break;
        default:
            throw std::invalid_argument("Error: CastNode unknown output type");
        }
        break;
    default:
        throw std::invalid_argument("Error: CastNode unknown input type");
    }
    return Node(newNode, model.GetModel());
}

template <typename ElementType>
ell::model::PortElements<ElementType> GetPortElementsFromList(const std::vector<PortElements*>& inputs)
{
    auto elements_list = std::vector<ell::model::PortElements<ElementType>>{};
    for (const auto input : inputs)
    {
        const auto& elements = input->GetPortElements();
        elements_list.push_back(ell::model::PortElements<ElementType>(elements));
    }
    return ell::model::PortElements<ElementType>(elements_list);
}

Node ModelBuilder::AddConcatenationNode(Model model, const PortMemoryLayout& outputMemoryLayout, const std::vector<PortElements*>& inputs)
{
    if (inputs.size() < 1)
    {
        throw std::invalid_argument("Error: expected at least one input port element for AddConcatenationNode");
    }

    auto type = inputs[0]->GetType();
    ell::model::Node* newNode = nullptr;

    switch (type)
    {
    case PortType::boolean:
        newNode = model.GetModel()->AddNode<ell::nodes::ConcatenationNode<bool>>(GetPortElementsFromList<bool>(inputs), outputMemoryLayout.Get());
        break;
    case PortType::integer:
        newNode = model.GetModel()->AddNode<ell::nodes::ConcatenationNode<int>>(GetPortElementsFromList<int>(inputs), outputMemoryLayout.Get());
        break;
    case PortType::bigInt:
        newNode = model.GetModel()->AddNode<ell::nodes::ConcatenationNode<int64_t>>(GetPortElementsFromList<int64_t>(inputs), outputMemoryLayout.Get());
        break;
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::ConcatenationNode<double>>(GetPortElementsFromList<double>(inputs), outputMemoryLayout.Get());
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::ConcatenationNode<float>>(GetPortElementsFromList<float>(inputs), outputMemoryLayout.Get());
        break;
    default:
        throw std::invalid_argument("Error: could not create ConcatenationNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

std::vector<const ell::model::OutputPortBase*> GeOutputPortElementsFromList(const std::vector<PortElements*>& inputs)
{
    auto port_list = std::vector<const ell::model::OutputPortBase*>{};
    for (const auto& input : inputs)
    {
        const auto outputPortBase = input->GetPortElements().GetElement(0).ReferencedPort();
        port_list.push_back(outputPortBase);
    }
    return port_list;
}

Node ModelBuilder::AddSpliceNode(Model model, const std::vector<PortElements*>& inputs)
{
    if (inputs.size() < 1)
    {
        throw std::invalid_argument("Error: expected at least one input port element for AddSpliceNode");
    }
    auto type = inputs[0]->GetType();
    ell::model::Node* newNode = nullptr;

    switch (type)
    {
    case PortType::boolean:
        newNode = model.GetModel()->AddNode<ell::model::SpliceNode<bool>>(GeOutputPortElementsFromList(inputs));
        break;
    case PortType::integer:
        newNode = model.GetModel()->AddNode<ell::model::SpliceNode<int>>(GeOutputPortElementsFromList(inputs));
        break;
    case PortType::bigInt:
        newNode = model.GetModel()->AddNode<ell::model::SpliceNode<int64_t>>(GeOutputPortElementsFromList(inputs));
        break;
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::model::SpliceNode<double>>(GeOutputPortElementsFromList(inputs));
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::model::SpliceNode<float>>(GeOutputPortElementsFromList(inputs));
        break;
    default:
        throw std::invalid_argument("Error: could not create SpliceNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddReinterpretLayoutNode(Model model, PortElements input, PortMemoryLayout outputMemoryLayout)
{
    auto elements = input.GetPortElements();
    auto type = input.GetType();

    ell::utilities::MemoryLayout outputLayout = outputMemoryLayout.Get();
    if (elements.GetMemoryLayout().GetActiveSize().NumElements() != outputLayout.GetActiveSize().NumElements())
    {
        throw std::invalid_argument("Error: ReinterpretLayoutNode can only reinterpret shapes that have the same number of elements");
    }

    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::boolean:
        newNode = model.GetModel()->AddNode<ell::nodes::ReinterpretLayoutNode<bool>>(ell::model::PortElements<bool>(elements), outputLayout);
        break;
    case PortType::integer:
        newNode = model.GetModel()->AddNode<ell::nodes::ReinterpretLayoutNode<int>>(ell::model::PortElements<int>(elements), outputLayout);
        break;
    case PortType::bigInt:
        newNode = model.GetModel()->AddNode<ell::nodes::ReinterpretLayoutNode<int64_t>>(ell::model::PortElements<int64_t>(elements), outputLayout);
        break;
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::ReinterpretLayoutNode<double>>(ell::model::PortElements<double>(elements), outputLayout);
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::ReinterpretLayoutNode<float>>(ell::model::PortElements<float>(elements), outputLayout);
        break;
    default:
        throw std::invalid_argument("Error: could not create ReinterpretLayoutNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddReorderDataNode(Model model, PortElements input, PortMemoryLayout inputMemoryLayout, PortMemoryLayout outputMemoryLayout, std::vector<int> order, double outputPaddingValue)
{
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;
    if (order.size() == 0)
    {
        // provide default sequential ordering
        order = std::vector<int>(outputMemoryLayout.size.size());
        std::iota(order.begin(), order.end(), 0);
    }
    switch (type)
    {
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::ReorderDataCodeNode<double>>(
            ell::model::PortElements<double>(elements),
            inputMemoryLayout.Get(),
            outputMemoryLayout.Get(),
            order,
            outputPaddingValue);
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::ReorderDataCodeNode<float>>(
            ell::model::PortElements<float>(elements),
            inputMemoryLayout.Get(),
            outputMemoryLayout.Get(),
            order,
            static_cast<float>(outputPaddingValue));
        break;
    default:
        throw std::invalid_argument("Error: could not create ReorderDataNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddReorderDataNode(Model model, PortElements input, std::vector<int> order)
{
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::ReorderDataCodeNode<double>>(
            ell::model::PortElements<double>(elements),
            order);
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::ReorderDataCodeNode<float>>(
            ell::model::PortElements<float>(elements),
            order);
        break;
    default:
        throw std::invalid_argument("Error: could not create ReorderDataNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddScalingNode(Model model, PortElements input, double scale)
{
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::ScalingNode<double>>(
            ell::model::PortElements<double>(elements),
            scale);
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::ScalingNode<float>>(
            ell::model::PortElements<float>(elements),
            scale);
        break;
    default:
        throw std::invalid_argument("Error: could not create ReorderDataNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

SinkNode ModelBuilder::AddSinkNode(Model model, PortElements input, const PortMemoryLayout& memoryLayout, const std::string& sinkFunctionName, PortElements trigger)
{
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    auto triggerElements = trigger.GetPortElements();
    ell::model::SinkNodeBase* newNode = nullptr;
    auto layout = memoryLayout.Get();

    if (triggerElements.Size() == 0)
    {
        // no trigger specified, so add a default, always trigger condition
        auto condition_node = model.GetModel()->AddNode<ell::nodes::ConstantNode<bool>>(true);
        triggerElements = *(condition_node->GetOutputPort("output"));
    }

    switch (type)
    {
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::SinkNode<double>>(
            ell::model::PortElements<double>(elements),
            ell::model::PortElements<bool>(triggerElements),
            layout.GetActiveSize(),
            sinkFunctionName);
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::SinkNode<float>>(
            ell::model::PortElements<float>(elements),
            ell::model::PortElements<bool>(triggerElements),
            layout.GetActiveSize(),
            sinkFunctionName);
        break;
    case PortType::integer:
        newNode = model.GetModel()->AddNode<ell::nodes::SinkNode<int>>(
            ell::model::PortElements<int>(elements),
            ell::model::PortElements<bool>(triggerElements),
            layout.GetActiveSize(),
            sinkFunctionName);
        break;
    case PortType::bigInt:
        newNode = model.GetModel()->AddNode<ell::nodes::SinkNode<int64_t>>(
            ell::model::PortElements<int64_t>(elements),
            ell::model::PortElements<bool>(triggerElements),
            layout.GetActiveSize(),
            sinkFunctionName);
        break;
    case PortType::boolean:
        newNode = model.GetModel()->AddNode<ell::nodes::SinkNode<bool>>(
            ell::model::PortElements<bool>(elements),
            ell::model::PortElements<bool>(triggerElements),
            layout.GetActiveSize(),
            sinkFunctionName);
        break;
    default:
        throw std::invalid_argument("Error: could not create SinkNode of the requested type");
    }
    return SinkNode(newNode, model.GetModel());
}

SourceNode ModelBuilder::AddSourceNode(Model model, PortElements input, PortType outputType, const PortMemoryLayout& memoryLayout, const std::string& sourceFunctionName)
{
    auto inputType = input.GetType();
    if (inputType != PortType::real)
    {
        throw std::invalid_argument("Only PortType::real is supported for time signal input");
    }

    using TimeTickType = double;
    auto inputElements = input.GetPortElements();
    ell::model::SourceNodeBase* newNode = nullptr;
    auto layout = memoryLayout.Get();

    switch (outputType)
    {
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::SourceNode<double>>(
            ell::model::PortElements<TimeTickType>(inputElements), layout, sourceFunctionName);
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::SourceNode<float>>(
            ell::model::PortElements<TimeTickType>(inputElements), layout, sourceFunctionName);
        break;
    case PortType::integer:
        newNode = model.GetModel()->AddNode<ell::nodes::SourceNode<int>>(
            ell::model::PortElements<TimeTickType>(inputElements), layout, sourceFunctionName);
        break;
    case PortType::bigInt:
        newNode = model.GetModel()->AddNode<ell::nodes::SourceNode<int64_t>>(
            ell::model::PortElements<TimeTickType>(inputElements), layout, sourceFunctionName);
        break;
    case PortType::boolean:
        newNode = model.GetModel()->AddNode<ell::nodes::SourceNode<bool>>(
            ell::model::PortElements<TimeTickType>(inputElements), layout, sourceFunctionName);
        break;
    default:
        throw std::invalid_argument("Error: could not create SourceNode of the requested type");
    }
    return SourceNode(newNode, model.GetModel());
}

Node ModelBuilder::AddConstantNode(Model model, std::vector<double> values, PortType type)
{
    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::boolean:
        newNode = model.GetModel()->AddNode<ell::nodes::ConstantNode<bool>>(CastVector<bool>(values));
        break;
    case PortType::integer:
        newNode = model.GetModel()->AddNode<ell::nodes::ConstantNode<int>>(CastVector<int>(values));
        break;
    case PortType::bigInt:
        newNode = model.GetModel()->AddNode<ell::nodes::ConstantNode<int64_t>>(CastVector<int64_t>(values));
        break;
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::ConstantNode<double>>(CastVector<double>(values));
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::ConstantNode<float>>(CastVector<float>(values));
        break;
    default:
        throw std::invalid_argument("Error: could not create ConstantNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddConstantNode(Model model, std::vector<double> values, const PortMemoryLayout& outputMemoryLayout, PortType type)
{
    auto outputLayout = outputMemoryLayout.Get();
    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::boolean:
        newNode = model.GetModel()->AddNode<ell::nodes::ConstantNode<bool>>(CastVector<bool>(values), outputLayout);
        break;
    case PortType::integer:
        newNode = model.GetModel()->AddNode<ell::nodes::ConstantNode<int>>(CastVector<int>(values), outputLayout);
        break;
    case PortType::bigInt:
        newNode = model.GetModel()->AddNode<ell::nodes::ConstantNode<int64_t>>(CastVector<int64_t>(values), outputLayout);
        break;
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::ConstantNode<double>>(values, outputLayout);
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::ConstantNode<float>>(CastVector<float>(values), outputLayout);
        break;
    default:
        throw std::invalid_argument("Error: could not create ConstantNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddUnaryOperationNode(Model model, PortElements input, UnaryOperationType op)
{
    auto operation = static_cast<ell::nodes::UnaryOperationType>(op);

    auto type = input.GetType();
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::boolean:
        newNode = model.GetModel()->AddNode<ell::nodes::UnaryOperationNode<bool>>(ell::model::PortElements<bool>(elements), operation);
        break;
    case PortType::integer:
        newNode = model.GetModel()->AddNode<ell::nodes::UnaryOperationNode<int>>(ell::model::PortElements<int>(elements), operation);
        break;
    case PortType::bigInt:
        newNode = model.GetModel()->AddNode<ell::nodes::UnaryOperationNode<int64_t>>(ell::model::PortElements<int64_t>(elements), operation);
        break;
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::UnaryOperationNode<double>>(ell::model::PortElements<double>(elements), operation);
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::UnaryOperationNode<float>>(ell::model::PortElements<float>(elements), operation);
        break;
    default:
        throw std::invalid_argument("Error: could not create UnaryOperationNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddBinaryOperationNode(Model model, PortElements input1, PortElements input2, BinaryOperationType op)
{
    auto operation = static_cast<ell::nodes::BinaryOperationType>(op);

    auto type = input1.GetType();
    if (type != input2.GetType())
    {
        throw std::invalid_argument("Error: BinaryOperationNode requires both arguments to be of the same type");
    }
    auto elements1 = input1.GetPortElements();
    auto elements2 = input2.GetPortElements();

    ell::model::Node* newNode = nullptr;
    if (elements1.GetMemoryLayout().GetActiveSize().NumElements() != elements2.GetMemoryLayout().GetActiveSize().NumElements())
    {
        // then we need to do some broadcasting...
        switch (type)
        {
        case PortType::integer:
            newNode = model.GetModel()->AddNode<ell::nodes::BroadcastBinaryOperationNode<int>>(ell::model::PortElements<int>(elements1), ell::model::PortElements<int>(elements2), operation);
            break;
        case PortType::bigInt:
            newNode = model.GetModel()->AddNode<ell::nodes::BroadcastBinaryOperationNode<int64_t>>(ell::model::PortElements<int64_t>(elements1), ell::model::PortElements<int64_t>(elements2), operation);
            break;
        case PortType::real:
            newNode = model.GetModel()->AddNode<ell::nodes::BroadcastBinaryOperationNode<double>>(ell::model::PortElements<double>(elements1), ell::model::PortElements<double>(elements2), operation);
            break;
        case PortType::smallReal:
            newNode = model.GetModel()->AddNode<ell::nodes::BroadcastBinaryOperationNode<float>>(ell::model::PortElements<float>(elements1), ell::model::PortElements<float>(elements2), operation);
            break;
        default:
            throw std::invalid_argument("Error: could not create BroadcastBinaryOperationNode of the requested type");
        }
    }
    else
    {
        switch (type)
        {
        case PortType::boolean:
            newNode = model.GetModel()->AddNode<ell::nodes::BinaryOperationNode<bool>>(ell::model::PortElements<bool>(elements1), ell::model::PortElements<bool>(elements2), operation);
            break;
        case PortType::integer:
            newNode = model.GetModel()->AddNode<ell::nodes::BinaryOperationNode<int>>(ell::model::PortElements<int>(elements1), ell::model::PortElements<int>(elements2), operation);
            break;
        case PortType::bigInt:
            newNode = model.GetModel()->AddNode<ell::nodes::BinaryOperationNode<int64_t>>(ell::model::PortElements<int64_t>(elements1), ell::model::PortElements<int64_t>(elements2), operation);
            break;
        case PortType::real:
            newNode = model.GetModel()->AddNode<ell::nodes::BinaryOperationNode<double>>(ell::model::PortElements<double>(elements1), ell::model::PortElements<double>(elements2), operation);
            break;
        case PortType::smallReal:
            newNode = model.GetModel()->AddNode<ell::nodes::BinaryOperationNode<float>>(ell::model::PortElements<float>(elements1), ell::model::PortElements<float>(elements2), operation);
            break;
        default:
            throw std::invalid_argument("Error: could not create BinaryOperationNode of the requested type");
        }
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddIIRFilterNode(Model model, PortElements input, std::vector<double> bCoeffs, std::vector<double> aCoeffs)
{
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::IIRFilterNode<double>>(ell::model::PortElements<double>(elements), bCoeffs, aCoeffs);
        break;
    case PortType::smallReal: {
        std::vector<float> bFloatCoeffs(bCoeffs.begin(), bCoeffs.end());
        std::vector<float> aFloatCoeffs(aCoeffs.begin(), aCoeffs.end());
        newNode = model.GetModel()->AddNode<ell::nodes::IIRFilterNode<float>>(ell::model::PortElements<float>(elements), bFloatCoeffs, aFloatCoeffs);
    }
    break;
    default:
        throw std::invalid_argument("Error: could not create IIRFilterNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddBufferNode(Model model, PortElements input, int windowSize)
{
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::boolean:
        newNode = model.GetModel()->AddNode<ell::nodes::BufferNode<Boolean>>(ell::model::PortElements<Boolean>(elements), windowSize);
        break;
    case PortType::integer:
        newNode = model.GetModel()->AddNode<ell::nodes::BufferNode<int>>(ell::model::PortElements<int>(elements), windowSize);
        break;
    case PortType::bigInt:
        newNode = model.GetModel()->AddNode<ell::nodes::BufferNode<int64_t>>(ell::model::PortElements<int64_t>(elements), windowSize);
        break;
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::BufferNode<double>>(ell::model::PortElements<double>(elements), windowSize);
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::BufferNode<float>>(ell::model::PortElements<float>(elements), windowSize);
        break;
    default:
        throw std::invalid_argument("Error: could not create BufferNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddHammingWindowNode(Model model, PortElements input)
{
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::HammingWindowNode<double>>(ell::model::PortElements<double>(elements));
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::HammingWindowNode<float>>(ell::model::PortElements<float>(elements));
        break;
    default:
        throw std::invalid_argument("Error: could not create HammingWindowNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddHannWindowNode(Model model, PortElements input)
{
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::HannWindowNode<double>>(ell::model::PortElements<double>(elements));
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::HannWindowNode<float>>(ell::model::PortElements<float>(elements));
        break;
    default:
        throw std::invalid_argument("Error: could not create HannWindowNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddFFTNode(Model model, PortElements input, int nfft)
{
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::real:
        if (nfft == 0)
        {
            newNode = model.GetModel()->AddNode<ell::nodes::FFTNode<double>>(ell::model::PortElements<double>(elements));
        }
        else
        {
            newNode = model.GetModel()->AddNode<ell::nodes::FFTNode<double>>(ell::model::PortElements<double>(elements), nfft);
        }
        break;
    case PortType::smallReal:
        if (nfft == 0)
        {
            newNode = model.GetModel()->AddNode<ell::nodes::FFTNode<float>>(ell::model::PortElements<float>(elements));
        }
        else
        {
            newNode = model.GetModel()->AddNode<ell::nodes::FFTNode<float>>(ell::model::PortElements<float>(elements), nfft);
        }
        break;
    default:
        throw std::invalid_argument("Error: could not create FFTNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddLinearFilterBankNode(Model model, PortElements input, double sampleRate, int numFilters, int numFiltersToUse, double offset)
{
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    auto windowSize = elements.Size();
    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::LinearFilterBankNode<double>>(ell::model::PortElements<double>(elements), ell::dsp::LinearFilterBank(windowSize, sampleRate, numFilters, numFiltersToUse, offset));
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::LinearFilterBankNode<float>>(ell::model::PortElements<float>(elements), ell::dsp::LinearFilterBank(windowSize, sampleRate, numFilters, numFiltersToUse, offset));
        break;
    default:
        throw std::invalid_argument("Error: could not create LinearFilterBankNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddMelFilterBankNode(Model model, PortElements input, double sampleRate, int fftSize, int numFilters, int numFiltersToUse, double offset)
{
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    auto windowSize = elements.Size();
    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::MelFilterBankNode<double>>(ell::model::PortElements<double>(elements), ell::dsp::MelFilterBank(windowSize, sampleRate, fftSize, numFilters, numFiltersToUse, offset));
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::MelFilterBankNode<float>>(ell::model::PortElements<float>(elements), ell::dsp::MelFilterBank(windowSize, sampleRate, fftSize, numFilters, numFiltersToUse, offset));
        break;
    default:
        throw std::invalid_argument("Error: could not create MelFilterBankNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddDCTNode(Model model, PortElements input, int numFilters)
{
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::DCTNode<double>>(ell::model::PortElements<double>(elements), numFilters);
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::DCTNode<float>>(ell::model::PortElements<float>(elements), numFilters);
        break;
    default:
        throw std::invalid_argument("Error: could not create DCTNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddMatrixMultiplyNode(Model model, PortElements input1, PortElements input2)
{
    auto type = input1.GetType();
    auto t2 = input2.GetType();
    if (type != t2)
    {
        throw std::invalid_argument("Error: input1 has different element types from input2");
    }
    auto elements1 = input1.GetPortElements();
    auto elements2 = input2.GetPortElements();

    auto layout1 = elements1.GetMemoryLayout();
    auto layout2 = elements2.GetMemoryLayout();

    auto s1 = layout1.GetActiveSize().NumElements();
    auto s2 = layout2.GetActiveSize().NumElements();
    ell::model::Node* newNode = nullptr;

    if (layout1.NumDimensions() == 1 && layout2.NumDimensions() == 2)
    {
        // then this is a MatrixVectorMultiplyNode, but MatrixVectorMultiplyNode requires matrix first so we
        // have to transpose input2 in this case.
        // TODO: fix MatrixVectorMultiplyNode so it exposes the transpose options supported by BLAS GEMV functions.
        // In the meantime we can handle the constant case.
        throw std::invalid_argument("Error: input sizes not yet supported");
    }
    else if (layout1.NumDimensions() == 2 && layout2.NumDimensions() == 1)
    {
        auto m = layout1.GetActiveSize()[0];
        auto n = layout1.GetActiveSize()[1];

        // then this is a MatrixVectorMultiplyNode
        switch (type)
        {
        case PortType::real:
            newNode = model.GetModel()->AddNode<ell::nodes::MatrixVectorMultiplyNode<double>>(ell::model::PortElements<double>(elements1), m, n, n, ell::model::PortElements<double>(elements2));
            break;
        case PortType::smallReal:
            newNode = model.GetModel()->AddNode<ell::nodes::MatrixVectorMultiplyNode<float>>(ell::model::PortElements<float>(elements1), m, n, n, ell::model::PortElements<float>(elements2));
            break;
        default:
            throw std::invalid_argument("Error: could not create MatrixVectorMultiplyNode of the requested type");
        }
    }
    else if (layout1.NumDimensions() == 1 && layout2.NumDimensions() == 1)
    {
        if (s1 != s2)
        {
            throw std::invalid_argument("Error: cannot multiply vectors of different sizes");
        }
        else
        {
            return AddDotProductNode(model, input1, input2);
        }
    }
    else
    {
        // todo: figure out numpy rules here,  https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
        // e.g. if layout 1 is 2-D and layout 2 is 1-D then turn this into a MatrixVectorMultiplyNode
        // if both are 2D then use MatrixMatrixMultiplyNode.
        // if both are 1D then use DotProductNode.
        // etc.
        throw std::invalid_argument("Error: input sizes not yet supported");
    }

    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddMatrixMatrixMultiplyNode(Model model, PortElements input1, PortElements input2)
{
    auto type = input1.GetType();
    auto t2 = input2.GetType();
    if (type != t2)
    {
        throw std::invalid_argument("Error: input1 has different element types from input2");
    }
    auto elements1 = input1.GetPortElements();
    auto elements2 = input2.GetPortElements();

    auto layout1 = elements1.GetMemoryLayout();
    auto layout2 = elements2.GetMemoryLayout();

    ell::model::Node* newNode = nullptr;

    if (layout1.NumDimensions() != 2 && layout2.NumDimensions() != 2)
    {
        throw std::invalid_argument("Error: input sizes invalid");
    }
    else
    {
        switch (type)
        {
        case PortType::real:
            newNode = model.GetModel()->AddNode<ell::nodes::MatrixMatrixMultiplyNode<double>>(ell::model::PortElements<double>(elements1), ell::model::PortElements<double>(elements2));
            break;
        case PortType::smallReal:
            newNode = model.GetModel()->AddNode<ell::nodes::MatrixMatrixMultiplyNode<float>>(ell::model::PortElements<float>(elements1), ell::model::PortElements<float>(elements2));
            break;
        default:
            throw std::invalid_argument("Error: could not create MatrixMatrixMultiplyNode of the requested type");
        }
    }

    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddMatrixMatrixMultiplyCodeNode(Model model, PortElements input1, PortElements input2, int gemmImpl)
{
    auto type = input1.GetType();
    auto t2 = input2.GetType();
    if (type != t2)
    {
        throw std::invalid_argument("Error: input1 has different element types from input2");
    }
    auto elements1 = input1.GetPortElements();
    auto elements2 = input2.GetPortElements();

    auto layout1 = elements1.GetMemoryLayout();
    auto layout2 = elements2.GetMemoryLayout();

    ell::model::Node* newNode = nullptr;

    if (layout1.NumDimensions() != 2 && layout2.NumDimensions() != 2)
    {
        throw std::invalid_argument("Error: input sizes invalid");
    }
    else
    {
        switch (type)
        {
        case PortType::real:
            newNode = model.GetModel()->AddNode<ell::nodes::MatrixMatrixMultiplyCodeNode<double>>(ell::model::PortElements<double>(elements1), ell::model::PortElements<double>(elements2), static_cast<ell::nodes::MatrixMatrixMultiplyImplementation>(gemmImpl));
            break;
        case PortType::smallReal:
            newNode = model.GetModel()->AddNode<ell::nodes::MatrixMatrixMultiplyCodeNode<float>>(ell::model::PortElements<float>(elements1), ell::model::PortElements<float>(elements2), static_cast<ell::nodes::MatrixMatrixMultiplyImplementation>(gemmImpl));
            break;
        default:
            throw std::invalid_argument("Error: could not create MatrixMatrixMultiplyCodeNode of the requested type");
        }
    }

    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddMatrixMatrixMultiplyCodeNode(Model model, PortElements input1, PortElements input2, int panelM, int panelN, int panelK, int kernelM, int kernelN, int kernelK, int gemmImpl)
{
    auto type = input1.GetType();
    auto t2 = input2.GetType();
    if (type != t2)
    {
        throw std::invalid_argument("Error: input1 has different element types from input2");
    }
    auto elements1 = input1.GetPortElements();
    auto elements2 = input2.GetPortElements();

    auto layout1 = elements1.GetMemoryLayout();
    auto layout2 = elements2.GetMemoryLayout();

    ell::model::Node* newNode = nullptr;

    if (layout1.NumDimensions() != 2 && layout2.NumDimensions() != 2)
    {
        throw std::invalid_argument("Error: input sizes invalid");
    }
    else
    {
        switch (type)
        {
        case PortType::real:
            newNode = model.GetModel()->AddNode<ell::nodes::MatrixMatrixMultiplyCodeNode<double>>(ell::model::PortElements<double>(elements1), ell::model::PortElements<double>(elements2), panelM, panelN, panelK, kernelM, kernelN, kernelK, static_cast<ell::nodes::MatrixMatrixMultiplyImplementation>(gemmImpl));
            break;
        case PortType::smallReal:
            newNode = model.GetModel()->AddNode<ell::nodes::MatrixMatrixMultiplyCodeNode<float>>(ell::model::PortElements<float>(elements1), ell::model::PortElements<float>(elements2), panelM, panelN, panelK, kernelM, kernelN, kernelK, static_cast<ell::nodes::MatrixMatrixMultiplyImplementation>(gemmImpl));
            break;
        default:
            throw std::invalid_argument("Error: could not create MatrixMatrixMultiplyCodeNode of the requested type");
        }
    }

    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddDotProductNode(Model model, PortElements input1, PortElements input2)
{
    ell::model::Node* newNode = nullptr;
    auto type = input1.GetType();
    auto t2 = input2.GetType();
    if (type != t2)
    {
        throw std::invalid_argument("Error: input1 has different element types from input2");
    }
    auto elements1 = input1.GetPortElements();
    auto elements2 = input2.GetPortElements();

    switch (type)
    {
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::DotProductNode<double>>(ell::model::PortElements<double>(elements1), ell::model::PortElements<double>(elements2));
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::DotProductNode<float>>(ell::model::PortElements<float>(elements1), ell::model::PortElements<float>(elements2));
        break;
    default:
        throw std::invalid_argument("Error: could not create DotProductNode of the requested type");
    }

    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddVoiceActivityDetectorNode(Model model, PortElements input, double sampleRate, double frameDuration, double tauUp, double tauDown, double largeInput, double gainAtt, double thresholdUp, double thresholdDown, double levelThreshold)
{
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;

    switch (type)
    {
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::VoiceActivityDetectorNode>(ell::model::PortElements<double>(elements), sampleRate, frameDuration, tauUp, tauDown, largeInput, gainAtt, thresholdUp, thresholdDown, levelThreshold);
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::VoiceActivityDetectorNode>(ell::model::PortElements<float>(elements), sampleRate, frameDuration, tauUp, tauDown, largeInput, gainAtt, thresholdUp, thresholdDown, levelThreshold);
        break;
    default:
        throw std::invalid_argument("Error: could not create BufferNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

template <typename ElementType>
typename ell::predictors::neural::Layer<ElementType>::LayerParameters GetLayerParametersForLayerNode(const ell::api::predictors::neural::Layer& layer)
{
    using UnderlyingLayerParameters = typename ell::predictors::neural::Layer<ElementType>::LayerParameters;
    using TensorType = typename ell::predictors::neural::Layer<ElementType>::TensorType;
    return UnderlyingLayerParameters{
        TensorType(static_cast<size_t>(layer.parameters.inputShape.rows), static_cast<size_t>(layer.parameters.inputShape.columns), static_cast<size_t>(layer.parameters.inputShape.channels)),
        layer.parameters.inputPaddingParameters,
        { static_cast<size_t>(layer.parameters.outputShape.rows), static_cast<size_t>(layer.parameters.outputShape.columns), static_cast<size_t>(layer.parameters.outputShape.channels) },
        layer.parameters.outputPaddingParameters,
    };
}

Node ModelBuilder::AddActivationLayerNode(Model model, PortElements input, const ell::api::predictors::neural::ActivationLayer& layer)
{
    auto type = layer.parameters.dataType;
    switch (type)
    {
    case PortType::real:
        return AddActivationLayerNode<double>(model, input, layer);
        break;
    case PortType::smallReal:
        return AddActivationLayerNode<float>(model, input, layer);
        break;
    default:
        throw std::invalid_argument("Error: could not create ActivationLayerNode of the requested type");
    }
}

template <typename ElementType>
Node ModelBuilder::AddActivationLayerNode(Model model, PortElements input, const ell::api::predictors::neural::ActivationLayer& layer)
{
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;

    using UnderlyingLayerParameters = typename ell::predictors::neural::Layer<ElementType>::LayerParameters;
    using TensorType = typename ell::predictors::neural::Layer<ElementType>::TensorType;
    using ActivationImplType = ell::predictors::neural::ActivationImpl<ElementType>;
    using ActivationType = ell::api::predictors::neural::ActivationType;
    using namespace ell::predictors;

    // Set the layer parameters. Note the the input tensor reference will be immediately replaced inside the
    // layer node's constructor.
    UnderlyingLayerParameters parameters = GetLayerParametersForLayerNode<ElementType>(layer);

    switch (layer.activation)
    {
    case ActivationType::relu:
    case ActivationType::hardSigmoid:
    case ActivationType::hardTanh:
    case ActivationType::sigmoid:
    case ActivationType::tanh: {
        auto activationLayer = ell::predictors::neural::ActivationLayer<ElementType>(parameters, ell::api::predictors::neural::ActivationLayer::CreateActivation<ElementType>(layer.activation));
        newNode = model.GetModel()->AddNode<ell::nodes::ActivationLayerNode<ElementType>>(ell::model::PortElements<ElementType>(elements), activationLayer);
        break;
    }
    case ActivationType::leaky: {
        // can't use the ell::api::predictors::CreateActivation helper method in this case because the neural::LeakyReLUActivation requires the alpha value parameter.
        using ApiLeakyReLUActivationLayer = ell::api::predictors::neural::LeakyReLUActivationLayer;
        auto* activationlayer = const_cast<ell::api::predictors::neural::ActivationLayer*>(&layer);
        ActivationImplType* implementation = nullptr;
        if (activationlayer->template Is<ApiLeakyReLUActivationLayer>())
        {
            auto& apiLeakyLayer = activationlayer->template As<ApiLeakyReLUActivationLayer>();
            implementation = new ell::predictors::neural::LeakyReLUActivation<ElementType>(static_cast<ElementType>(apiLeakyLayer._alpha));
        }
        else
        {
            implementation = new ell::predictors::neural::LeakyReLUActivation<ElementType>();
        }
        newNode = model.GetModel()->AddNode<ell::nodes::ActivationLayerNode<ElementType>>(ell::model::PortElements<ElementType>(elements),
                                                                                          ell::predictors::neural::ActivationLayer<ElementType>(parameters, implementation));
        break;
    }
    case ActivationType::prelu: {
        // can't use the ell::api::predictors::CreateActivation helper method in this case because the neural::PReLUActivationLayer requires the alpha value parameter.
        using ApiPReLUActivationLayer = ell::api::predictors::neural::PReLUActivationLayer;
        auto* activationlayer = const_cast<ell::api::predictors::neural::ActivationLayer*>(&layer);
        auto& preluApiLayer = activationlayer->template As<ApiPReLUActivationLayer>();
        TensorType alpha(preluApiLayer.alpha.shape.rows, preluApiLayer.alpha.shape.columns, preluApiLayer.alpha.shape.channels, CastVector<ElementType>(preluApiLayer.alpha.data));
        auto activationLayer = ell::predictors::neural::ActivationLayer<ElementType>(parameters, new ell::predictors::neural::ParametricReLUActivation<ElementType>(alpha));
        newNode = model.GetModel()->AddNode<ell::nodes::ParametricReLUActivationLayerNode<ElementType>>(ell::model::PortElements<ElementType>(elements), activationLayer);
        break;
    }
    default:
        throw InputException(InputExceptionErrors::invalidArgument,
                             FormatString("Encountered unsupported activation type in neural network predictor: %d", static_cast<int>(layer.activation)));
    }

    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddBatchNormalizationLayerNode(Model model, PortElements input, const ell::api::predictors::neural::BatchNormalizationLayer& layer)
{
    auto type = layer.parameters.dataType;
    switch (type)
    {
    case PortType::real:
        return AddBatchNormalizationLayerNode<double>(model, input, layer);
        break;
    case PortType::smallReal:
        return AddBatchNormalizationLayerNode<float>(model, input, layer);
        break;
    default:
        throw std::invalid_argument("Error: could not create BatchNormalizationLayerNode of the requested type");
    }
}

template <typename ElementType>
Node ModelBuilder::AddBatchNormalizationLayerNode(Model model, PortElements input, const ell::api::predictors::neural::BatchNormalizationLayer& layer)
{
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;

    using UnderlyingLayerParameters = typename ell::predictors::neural::Layer<ElementType>::LayerParameters;
    using namespace ell::predictors;

    // Set the layer parameters. Note the the input tensor reference will be immediately replaced inside the
    // layer node's constructor.
    UnderlyingLayerParameters parameters = GetLayerParametersForLayerNode<ElementType>(layer);
    auto epsilonSummand = (layer.epsilonSummand == ell::api::predictors::neural::EpsilonSummand::variance) ? ell::predictors::neural::EpsilonSummand::Variance : ell::predictors::neural::EpsilonSummand::SqrtVariance;

    ell::predictors::neural::BatchNormalizationLayer<ElementType> batchNormalizationLayer(parameters, CastVector<ElementType>(layer.mean), CastVector<ElementType>(layer.variance), static_cast<ElementType>(layer.epsilon), epsilonSummand);

    newNode = model.GetModel()->AddNode<ell::nodes::BatchNormalizationLayerNode<ElementType>>(ell::model::PortElements<ElementType>(elements), batchNormalizationLayer);
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddBiasLayerNode(Model model, PortElements input, const ell::api::predictors::neural::BiasLayer& layer)
{
    auto type = layer.parameters.dataType;
    switch (type)
    {
    case PortType::real:
        return AddBiasLayerNode<double>(model, input, layer);
        break;
    case PortType::smallReal:
        return AddBiasLayerNode<float>(model, input, layer);
        break;
    default:
        throw std::invalid_argument("Error: could not create BiasLayerNode of the requested type");
    }
}

template <typename ElementType>
Node ModelBuilder::AddBiasLayerNode(Model model, PortElements input, const ell::api::predictors::neural::BiasLayer& layer)
{
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;

    using UnderlyingLayerParameters = typename ell::predictors::neural::Layer<ElementType>::LayerParameters;

    // Set the layer parameters. Note the the input tensor reference will be immediately replaced inside the
    // layer node's constructor.
    UnderlyingLayerParameters parameters = GetLayerParametersForLayerNode<ElementType>(layer);
    ell::predictors::neural::BiasLayer<ElementType> biasLayer(parameters, CastVector<ElementType>(layer.bias));

    newNode = model.GetModel()->AddNode<ell::nodes::BiasLayerNode<ElementType>>(ell::model::PortElements<ElementType>(elements), biasLayer);
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddBinaryConvolutionalLayerNode(Model model, PortElements input, const ell::api::predictors::neural::BinaryConvolutionalLayer& layer)
{
    auto type = layer.parameters.dataType;
    switch (type)
    {
    case PortType::real:
        return AddBinaryConvolutionalLayerNode<double>(model, input, layer);
        break;
    case PortType::smallReal:
        return AddBinaryConvolutionalLayerNode<float>(model, input, layer);
        break;
    default:
        throw std::invalid_argument("Error: could not create BinaryConvolutionalLayerNode of the requested type");
    }
}

template <typename ElementType>
Node ModelBuilder::AddBinaryConvolutionalLayerNode(Model model, PortElements input, const ell::api::predictors::neural::BinaryConvolutionalLayer& layer)
{
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;

    using UnderlyingLayerParameters = typename ell::predictors::neural::Layer<ElementType>::LayerParameters;
    using TensorType = typename ell::predictors::neural::Layer<ElementType>::TensorType;

    // Set the layer parameters. Note the the input tensor reference will be immediately replaced inside the
    // layer node's constructor.
    UnderlyingLayerParameters parameters = GetLayerParametersForLayerNode<ElementType>(layer);

    TensorType weights(layer.weights.shape.rows, layer.weights.shape.columns, layer.weights.shape.channels, CastVector<ElementType>(layer.weights.data));
    ell::predictors::neural::BinaryConvolutionalLayer<ElementType> convolutionalLayer(parameters, layer.convolutionalParameters, weights);

    newNode = model.GetModel()->AddNode<ell::nodes::BinaryConvolutionalLayerNode<ElementType>>(ell::model::PortElements<ElementType>(elements), convolutionalLayer);
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddConvolutionalLayerNode(Model model, PortElements input, const ell::api::predictors::neural::ConvolutionalLayer& layer)
{
    auto type = layer.parameters.dataType;
    switch (type)
    {
    case PortType::real:
        return AddConvolutionalLayerNode<double>(model, input, layer);
        break;
    case PortType::smallReal:
        return AddConvolutionalLayerNode<float>(model, input, layer);
        break;
    default:
        throw std::invalid_argument("Error: could not create ConvolutionalLayerNode of the requested type");
    }
}

template <typename ElementType>
Node ModelBuilder::AddConvolutionalLayerNode(Model model, PortElements input, const ell::api::predictors::neural::ConvolutionalLayer& layer)
{
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;

    using UnderlyingLayerParameters = typename ell::predictors::neural::Layer<ElementType>::LayerParameters;
    using TensorType = typename ell::predictors::neural::Layer<ElementType>::TensorType;

    // Set the layer parameters. Note the the input tensor reference will be immediately replaced inside the
    // layer node's constructor.
    UnderlyingLayerParameters parameters = GetLayerParametersForLayerNode<ElementType>(layer);

    TensorType weights(layer.weights.shape.rows, layer.weights.shape.columns, layer.weights.shape.channels, CastVector<ElementType>(layer.weights.data));
    ell::predictors::neural::ConvolutionalLayer<ElementType> convolutionalLayer(parameters, layer.convolutionalParameters, weights);

    newNode = model.GetModel()->AddNode<ell::nodes::ConvolutionalLayerNode<ElementType>>(ell::model::PortElements<ElementType>(elements), convolutionalLayer);
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddFullyConnectedLayerNode(Model model, PortElements input, const ell::api::predictors::neural::FullyConnectedLayer& layer)
{
    auto type = layer.parameters.dataType;
    switch (type)
    {
    case PortType::real:
        return AddFullyConnectedLayerNode<double>(model, input, layer);
        break;
    case PortType::smallReal:
        return AddFullyConnectedLayerNode<float>(model, input, layer);
        break;
    default:
        throw std::invalid_argument("Error: could not create RegionDetectionLayerNode of the requested type");
    }
}

template <typename ElementType>
Node ModelBuilder::AddFullyConnectedLayerNode(Model model, PortElements input, const ell::api::predictors::neural::FullyConnectedLayer& layer)
{
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;

    using UnderlyingLayerParameters = typename ell::predictors::neural::Layer<ElementType>::LayerParameters;
    using TensorType = typename ell::predictors::neural::Layer<ElementType>::TensorType;

    // Set the layer parameters. Note the the input tensor reference will be immediately replaced inside the
    // layer node's constructor.
    UnderlyingLayerParameters parameters = GetLayerParametersForLayerNode<ElementType>(layer);
    TensorType weights(layer.weights.shape.rows, layer.weights.shape.columns, layer.weights.shape.channels, CastVector<ElementType>(layer.weights.data));
    ell::predictors::neural::FullyConnectedLayer<ElementType> fullyConnectedLayer(parameters, weights);

    newNode = model.GetModel()->AddNode<ell::nodes::FullyConnectedLayerNode<ElementType>>(ell::model::PortElements<ElementType>(elements), fullyConnectedLayer);
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddRegionDetectionLayerNode(Model model, PortElements input, const ell::api::predictors::neural::RegionDetectionLayer& layer)
{
    auto type = layer.parameters.dataType;
    switch (type)
    {
    case PortType::real:
        return AddRegionDetectionLayerNode<double>(model, input, layer);
        break;
    case PortType::smallReal:
        return AddRegionDetectionLayerNode<float>(model, input, layer);
        break;
    default:
        throw std::invalid_argument("Error: could not create RegionDetectionLayerNode of the requested type");
    }
}

template <typename ElementType>
Node ModelBuilder::AddRegionDetectionLayerNode(Model model, PortElements input, const ell::api::predictors::neural::RegionDetectionLayer& layer)
{
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;

    auto parameters = GetLayerParametersForLayerNode<ElementType>(layer);

    ell::predictors::neural::RegionDetectionLayer<ElementType> regionDetectionLayer(parameters, layer.detectionParameters);

    newNode = model.GetModel()->AddNode<ell::nodes::RegionDetectionLayerNode<ElementType>>(ell::model::PortElements<ElementType>(elements), regionDetectionLayer);

    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddPoolingLayerNode(Model model, PortElements input, const ell::api::predictors::neural::PoolingLayer& layer)
{
    auto type = layer.parameters.dataType;
    switch (type)
    {
    case PortType::real:
        return AddPoolingLayerNode<double>(model, input, layer);
        break;
    case PortType::smallReal:
        return AddPoolingLayerNode<float>(model, input, layer);
        break;
    default:
        throw std::invalid_argument("Error: could not create PoolingLayerNode of the requested type");
    }
}

template <typename ElementType>
Node ModelBuilder::AddPoolingLayerNode(Model model, PortElements input, const ell::api::predictors::neural::PoolingLayer& layer)
{
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;

    using UnderlyingLayerParameters = typename ell::predictors::neural::Layer<ElementType>::LayerParameters;
    using namespace ell::predictors;

    // Set the layer parameters. Note the the input tensor reference will be immediately replaced inside the
    // layer node's constructor.
    UnderlyingLayerParameters parameters = GetLayerParametersForLayerNode<ElementType>(layer);
    if (layer.poolingType == ell::api::predictors::neural::PoolingType::max)
    {
        ell::predictors::neural::PoolingLayer<ElementType, ell::predictors::neural::MaxPoolingFunction> poolingLayer(parameters, layer.poolingParameters);
        newNode = model.GetModel()->AddNode<ell::nodes::PoolingLayerNode<ElementType, ell::predictors::neural::MaxPoolingFunction>>(ell::model::PortElements<ElementType>(elements), poolingLayer);
    }
    else
    {
        ell::predictors::neural::PoolingLayer<ElementType, ell::predictors::neural::MeanPoolingFunction> poolingLayer(parameters, layer.poolingParameters);
        newNode = model.GetModel()->AddNode<ell::nodes::PoolingLayerNode<ElementType, ell::predictors::neural::MeanPoolingFunction>>(ell::model::PortElements<ElementType>(elements), poolingLayer);
    }

    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddScalingLayerNode(Model model, PortElements input, const ell::api::predictors::neural::ScalingLayer& layer)
{
    auto type = layer.parameters.dataType;
    switch (type)
    {
    case PortType::real:
        return AddScalingLayerNode<double>(model, input, layer);
        break;
    case PortType::smallReal:
        return AddScalingLayerNode<float>(model, input, layer);
        break;
    default:
        throw std::invalid_argument("Error: could not create ScalingLayerNode of the requested type");
    }
}

template <typename ElementType>
Node ModelBuilder::AddScalingLayerNode(Model model, PortElements input, const ell::api::predictors::neural::ScalingLayer& layer)
{
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;

    using UnderlyingLayerParameters = typename ell::predictors::neural::Layer<ElementType>::LayerParameters;

    // Set the layer parameters. Note the the input tensor reference will be immediately replaced inside the
    // layer node's constructor.
    UnderlyingLayerParameters parameters = GetLayerParametersForLayerNode<ElementType>(layer);
    ell::predictors::neural::ScalingLayer<ElementType> scalingLayer(parameters, CastVector<ElementType>(layer.scales));

    newNode = model.GetModel()->AddNode<ell::nodes::ScalingLayerNode<ElementType>>(ell::model::PortElements<ElementType>(elements), scalingLayer);
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddSoftmaxLayerNode(Model model, PortElements input, const ell::api::predictors::neural::SoftmaxLayer& layer)
{
    auto type = layer.parameters.dataType;
    switch (type)
    {
    case PortType::real:
        return AddSoftmaxLayerNode<double>(model, input, layer);
        break;
    case PortType::smallReal:
        return AddSoftmaxLayerNode<float>(model, input, layer);
        break;
    default:
        throw std::invalid_argument("Error: could not create SoftmaxLayerNode of the requested type");
    }
}

template <typename ElementType>
Node ModelBuilder::AddSoftmaxLayerNode(Model model, PortElements input, const ell::api::predictors::neural::SoftmaxLayer& layer)
{
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;

    using UnderlyingLayerParameters = typename ell::predictors::neural::Layer<ElementType>::LayerParameters;

    // Set the layer parameters. Note the the input tensor reference will be immediately replaced inside the
    // layer node's constructor.
    UnderlyingLayerParameters parameters = GetLayerParametersForLayerNode<ElementType>(layer);
    ell::predictors::neural::SoftmaxLayer<ElementType> softmaxLayer(parameters);

    newNode = model.GetModel()->AddNode<ell::nodes::SoftmaxLayerNode<ElementType>>(ell::model::PortElements<ElementType>(elements), softmaxLayer);
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddRNNNode(Model model, PortElements input, PortElements reset, size_t hiddenUnits, PortElements inputWeights, PortElements hiddenWeights, PortElements inputBias, PortElements hiddenBias, ell::api::predictors::neural::ActivationType activation)
{
    auto type = input.GetType();
    switch (type)
    {
    case PortType::real:
        return AddRNNNode<double>(model, input, reset, hiddenUnits, inputWeights, hiddenWeights, inputBias, hiddenBias, activation);
        break;
    case PortType::smallReal:
        return AddRNNNode<float>(model, input, reset, hiddenUnits, inputWeights, hiddenWeights, inputBias, hiddenBias, activation);
        break;
    default:
        throw std::invalid_argument("Error: could not create RNNNode of the requested type");
    }
}

template <typename ElementType>
Node ModelBuilder::AddRNNNode(Model model, PortElements input, PortElements reset, size_t hiddenUnits, PortElements inputWeights, PortElements hiddenWeights, PortElements inputBias, PortElements hiddenBias, ell::api::predictors::neural::ActivationType activation)
{
    using namespace ell::predictors::neural;
    using namespace ell::nodes;

    ell::model::Node* newNode = model.GetModel()->AddNode<ell::nodes::RNNNode<ElementType>>(
        ell::model::PortElements<ElementType>(input.GetPortElements()),
        reset.GetPortElements(),
        hiddenUnits,
        ell::model::PortElements<ElementType>(inputWeights.GetPortElements()),
        ell::model::PortElements<ElementType>(hiddenWeights.GetPortElements()),
        ell::model::PortElements<ElementType>(inputBias.GetPortElements()),
        ell::model::PortElements<ElementType>(hiddenBias.GetPortElements()),
        ell::api::predictors::neural::ActivationLayer::CreateActivation<ElementType>(activation));
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddGRUNode(Model model, PortElements input, PortElements reset, size_t hiddenUnits, PortElements inputWeights, PortElements hiddenWeights, PortElements inputBias, PortElements hiddenBias, ell::api::predictors::neural::ActivationType activation, ell::api::predictors::neural::ActivationType recurrentActivation)
{
    auto type = input.GetType();
    switch (type)
    {
    case PortType::real:
        return AddGRUNode<double>(model, input, reset, hiddenUnits, inputWeights, hiddenWeights, inputBias, hiddenBias, activation, recurrentActivation);
        break;
    case PortType::smallReal:
        return AddGRUNode<float>(model, input, reset, hiddenUnits, inputWeights, hiddenWeights, inputBias, hiddenBias, activation, recurrentActivation);
        break;
    default:
        throw std::invalid_argument("Error: could not create GRUNode of the requested type");
    }
}

template <typename ElementType>
Node ModelBuilder::AddGRUNode(Model model, PortElements input, PortElements reset, size_t hiddenUnits, PortElements inputWeights, PortElements hiddenWeights, PortElements inputBias, PortElements hiddenBias, ell::api::predictors::neural::ActivationType activation, ell::api::predictors::neural::ActivationType recurrentActivation)
{
    using namespace ell::predictors::neural;
    using namespace ell::nodes;

    ell::model::Node* newNode = model.GetModel()->AddNode<ell::nodes::GRUNode<ElementType>>(
        ell::model::PortElements<ElementType>(input.GetPortElements()),
        reset.GetPortElements(),
        hiddenUnits,
        ell::model::PortElements<ElementType>(inputWeights.GetPortElements()),
        ell::model::PortElements<ElementType>(hiddenWeights.GetPortElements()),
        ell::model::PortElements<ElementType>(inputBias.GetPortElements()),
        ell::model::PortElements<ElementType>(hiddenBias.GetPortElements()),
        ell::api::predictors::neural::ActivationLayer::CreateActivation<ElementType>(activation),
        ell::api::predictors::neural::ActivationLayer::CreateActivation<ElementType>(recurrentActivation));
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddFastGRNNNode(Model model, PortElements input, PortElements reset, size_t hiddenUnits, size_t wRank, size_t uRank, PortElements W1, PortElements W2, PortElements U1, PortElements U2, PortElements biasGate, PortElements biasUpdate, PortElements zeta, PortElements nu, ell::api::predictors::neural::ActivationType gateNonlinearity, ell::api::predictors::neural::ActivationType updateNonlinearity)
{
    auto type = input.GetType();
    switch (type)
    {
    case PortType::real:
        return AddFastGRNNNode<double>(model, input, reset, hiddenUnits, wRank, uRank, W1, W2, U1, U2, biasGate, biasUpdate, zeta, nu, gateNonlinearity, updateNonlinearity);
        break;
    case PortType::smallReal:
        return AddFastGRNNNode<float>(model, input, reset, hiddenUnits, wRank, uRank, W1, W2, U1, U2, biasGate, biasUpdate, zeta, nu, gateNonlinearity, updateNonlinearity);
        break;
    default:
        throw std::invalid_argument("Error: could not create FastGRNNNode of the requested type");
    }
}

template <typename ElementType>
Node ModelBuilder::AddFastGRNNNode(Model model, PortElements input, PortElements reset, size_t hiddenUnits, size_t wRank, size_t uRank, PortElements W1, PortElements W2, PortElements U1, PortElements U2, PortElements biasGate, PortElements biasUpdate, PortElements zeta, PortElements nu, ell::api::predictors::neural::ActivationType gateNonlinearity, ell::api::predictors::neural::ActivationType updateNonlinearity)
{
    using namespace ell::predictors::neural;
    using namespace ell::nodes;

    ell::model::Node* newNode = model.GetModel()->AddNode<ell::nodes::FastGRNNNode<ElementType>>(
        ell::model::PortElements<ElementType>(input.GetPortElements()),
        reset.GetPortElements(),
        hiddenUnits,
        wRank,
        uRank,
        ell::model::PortElements<ElementType>(W1.GetPortElements()),
        ell::model::PortElements<ElementType>(W2.GetPortElements()),
        ell::model::PortElements<ElementType>(U1.GetPortElements()),
        ell::model::PortElements<ElementType>(U2.GetPortElements()),
        ell::model::PortElements<ElementType>(biasGate.GetPortElements()),
        ell::model::PortElements<ElementType>(biasUpdate.GetPortElements()),
        ell::model::PortElements<ElementType>(zeta.GetPortElements()),
        ell::model::PortElements<ElementType>(nu.GetPortElements()),
        ell::api::predictors::neural::ActivationLayer::CreateActivation<ElementType>(gateNonlinearity),
        ell::api::predictors::neural::ActivationLayer::CreateActivation<ElementType>(updateNonlinearity));
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddLSTMNode(Model model, PortElements input, PortElements reset, size_t hiddenUnits, PortElements inputWeights, PortElements hiddenWeights, PortElements inputBias, PortElements hiddenBias, ell::api::predictors::neural::ActivationType activation, ell::api::predictors::neural::ActivationType recurrentActivation)
{
    auto type = input.GetType();
    switch (type)
    {
    case PortType::real:
        return AddLSTMNode<double>(model, input, reset, hiddenUnits, inputWeights, hiddenWeights, inputBias, hiddenBias, activation, recurrentActivation);
        break;
    case PortType::smallReal:
        return AddLSTMNode<float>(model, input, reset, hiddenUnits, inputWeights, hiddenWeights, inputBias, hiddenBias, activation, recurrentActivation);
        break;
    default:
        throw std::invalid_argument("Error: could not create LSTMNode of the requested type");
    }
}

template <typename ElementType>
Node ModelBuilder::AddLSTMNode(Model model, PortElements input, PortElements reset, size_t hiddenUnits, PortElements inputWeights, PortElements hiddenWeights, PortElements inputBias, PortElements hiddenBias, ell::api::predictors::neural::ActivationType activation, ell::api::predictors::neural::ActivationType recurrentActivation)
{
    using namespace ell::predictors::neural;
    using namespace ell::nodes;

    ell::model::Node* newNode = model.GetModel()->AddNode<ell::nodes::LSTMNode<ElementType>>(
        ell::model::PortElements<ElementType>(input.GetPortElements()),
        reset.GetPortElements(),
        hiddenUnits,
        ell::model::PortElements<ElementType>(inputWeights.GetPortElements()),
        ell::model::PortElements<ElementType>(hiddenWeights.GetPortElements()),
        ell::model::PortElements<ElementType>(inputBias.GetPortElements()),
        ell::model::PortElements<ElementType>(hiddenBias.GetPortElements()),
        ell::api::predictors::neural::ActivationLayer::CreateActivation<ElementType>(activation),
        ell::api::predictors::neural::ActivationLayer::CreateActivation<ElementType>(recurrentActivation));
    return Node(newNode, model.GetModel());
}

Node ModelBuilder::AddDTWNode(Model model, std::vector<std::vector<double>> prototype, PortElements input)
{
    auto type = input.GetType();
    auto elements = input.GetPortElements();
    ell::model::Node* newNode = nullptr;
    switch (type)
    {
    case PortType::real:
        newNode = model.GetModel()->AddNode<ell::nodes::DTWDistanceNode<double>>(ell::model::PortElements<double>(elements), prototype);
        break;
    case PortType::smallReal:
        newNode = model.GetModel()->AddNode<ell::nodes::DTWDistanceNode<float>>(ell::model::PortElements<float>(elements), CastVector<float>(prototype));
        break;
    default:
        throw std::invalid_argument("Error: could not create DCTNode of the requested type");
    }
    return Node(newNode, model.GetModel());
}

// explicit instantiations

template std::vector<float> CastVector<float>(const std::vector<double>&);
template std::vector<float> CastVector<float>(const std::vector<float>&);
template std::vector<double> CastVector<double>(const std::vector<float>&);
template std::vector<double> CastVector<double>(const std::vector<double>&);

} // namespace ELL_API
