////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     LoadModel.cpp (common)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "LoadModel.h"

#include <model/include/InputNode.h>
#include <model/include/Map.h>
#include <model/include/Model.h>
#include <model/include/OutputNode.h>
#include <model/include/SliceNode.h>
#include <model/include/SpliceNode.h>

#include <nodes/include/AccumulatorNode.h>
#include <nodes/include/ActivationFunctions.h>
#include <nodes/include/BinaryOperationNode.h>
#include <nodes/include/BinaryPredicateNode.h>
#include <nodes/include/BroadcastFunctionNode.h>
#include <nodes/include/BroadcastOperationNodes.h>
#include <nodes/include/BufferNode.h>
#include <nodes/include/ClockNode.h>
#include <nodes/include/ConcatenationNode.h>
#include <nodes/include/DCTNode.h>
#include <nodes/include/DTWDistanceNode.h>
#include <nodes/include/DelayNode.h>
#include <nodes/include/DiagonalConvolutionNode.h>
#include <nodes/include/DotProductNode.h>
#include <nodes/include/ExtremalValueNode.h>
#include <nodes/include/FFTNode.h>
#include <nodes/include/FastGRNNNode.h>
#include <nodes/include/FilterBankNode.h>
#include <nodes/include/ForestPredictorNode.h>
#include <nodes/include/GRUNode.h>
#include <nodes/include/HammingWindowNode.h>
#include <nodes/include/HannWindowNode.h>
#include <nodes/include/IIRFilterNode.h>
#include <nodes/include/L2NormSquaredNode.h>
#include <nodes/include/LSTMNode.h>
#include <nodes/include/LinearPredictorNode.h>
#include <nodes/include/MatrixMatrixMultiplyCodeNode.h>
#include <nodes/include/MatrixMatrixMultiplyNode.h>
#include <nodes/include/MatrixVectorMultiplyNode.h>
#include <nodes/include/MatrixVectorProductNode.h>
#include <nodes/include/MovingAverageNode.h>
#include <nodes/include/MovingVarianceNode.h>
#include <nodes/include/MultiplexerNode.h>
#include <nodes/include/NeuralNetworkPredictorNode.h>
#include <nodes/include/ProtoNNPredictorNode.h>
#include <nodes/include/RNNNode.h>
#include <nodes/include/ReceptiveFieldMatrixNode.h>
#include <nodes/include/ReinterpretLayoutNode.h>
#include <nodes/include/ReorderDataCodeNode.h>
#include <nodes/include/ReorderDataNode.h>
#include <nodes/include/ScalingNode.h>
#include <nodes/include/SimpleConvolutionNode.h>
#include <nodes/include/SinkNode.h>
#include <nodes/include/SourceNode.h>
#include <nodes/include/SpatialConvolutionNode.h>
#include <nodes/include/UnaryOperationNode.h>
#include <nodes/include/UnrolledConvolutionNode.h>
#include <nodes/include/VoiceActivityDetectorNode.h>
#include <nodes/include/WinogradConvolutionNode.h>

#include <predictors/include/ForestPredictor.h>
#include <predictors/include/LinearPredictor.h>
#include <predictors/include/NeuralNetworkPredictor.h>
#include <predictors/include/SingleElementThresholdPredictor.h>

#include <predictors/neural/include/HardSigmoidActivation.h>
#include <predictors/neural/include/HardTanhActivation.h>
#include <predictors/neural/include/LeakyReLUActivation.h>
#include <predictors/neural/include/ParametricReLUActivation.h>
#include <predictors/neural/include/ReLUActivation.h>
#include <predictors/neural/include/SigmoidActivation.h>
#include <predictors/neural/include/TanhActivation.h>

#include <utilities/include/Archiver.h>
#include <utilities/include/Files.h>
#include <utilities/include/JsonArchiver.h>

#include <cstdint>

using namespace std::string_literals;
using namespace ell::predictors::neural;
using namespace ell::utilities;

namespace ell
{
namespace common
{
    template <typename ElementType>
    void RegisterRealNodeTypes(SerializationContext& context)
    {
        context.GetTypeFactory().AddType<model::Node, model::InputNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, model::OutputNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, model::SpliceNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, model::SliceNode<ElementType>>();

        context.GetTypeFactory().AddType<model::Node, nodes::AccumulatorNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ArgMaxNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ArgMinNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BinaryOperationNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BroadcastUnaryFunctionNode<ElementType, nodes::HardSigmoidActivationFunction<ElementType>>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BroadcastUnaryFunctionNode<ElementType, nodes::HardTanhActivationFunction<ElementType>>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BroadcastUnaryFunctionNode<ElementType, nodes::LeakyReLUActivationFunction<ElementType>>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BroadcastUnaryFunctionNode<ElementType, nodes::ReLUActivationFunction<ElementType>>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BroadcastUnaryFunctionNode<ElementType, nodes::SigmoidActivationFunction<ElementType>>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BroadcastLinearFunctionNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BroadcastUnaryOperationNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BroadcastBinaryOperationNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BroadcastTernaryOperationNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BufferNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ConcatenationNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ConstantNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::DelayNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::DiagonalConvolutionNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::DiagonalConvolutionComputeNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::DotProductNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::DTWDistanceNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::FastGRNNNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::FFTNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::GRUNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::HammingWindowNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::HannWindowNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::L2NormSquaredNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::IIRFilterNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::LinearPredictorNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::LinearFilterBankNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::LSTMNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MelFilterBankNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MatrixVectorProductNode<ElementType, math::MatrixLayout::rowMajor>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MatrixVectorProductNode<ElementType, math::MatrixLayout::columnMajor>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MatrixMatrixMultiplyNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MatrixMatrixMultiplyCodeNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MatrixVectorMultiplyNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MovingAverageNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MovingVarianceNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::NeuralNetworkPredictorNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ReceptiveFieldMatrixNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ReorderDataCodeNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ReorderDataNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ReinterpretLayoutNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::RNNNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ScalingNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::SimpleConvolutionComputeNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::SimpleConvolutionNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::SinkNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::SourceNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::SpatialConvolutionNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::SumNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<bool, ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<int, ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<int64_t, ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<float, ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<double, ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::UnaryOperationNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::UnrolledConvolutionNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::WinogradConvolutionNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::WinogradConvolutionComputeNode<ElementType>>();

        // NN layer nodes
        context.GetTypeFactory().AddType<model::Node, nodes::ActivationLayerNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BatchNormalizationLayerNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BiasLayerNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BinaryConvolutionalLayerNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ConvolutionalLayerNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::FullyConnectedLayerNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ParametricReLUActivationLayerNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::PoolingLayerNode<ElementType, MeanPoolingFunction>>();
        context.GetTypeFactory().AddType<model::Node, nodes::PoolingLayerNode<ElementType, MaxPoolingFunction>>();
        context.GetTypeFactory().AddType<model::Node, nodes::RegionDetectionLayerNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ScalingLayerNode<ElementType>>();
        context.GetTypeFactory().AddType<model::Node, nodes::SoftmaxLayerNode<ElementType>>();

        // Activations
        context.GetTypeFactory().AddType<predictors::neural::ActivationImpl<ElementType>, predictors::neural::HardSigmoidActivation<ElementType>>();
        context.GetTypeFactory().AddType<predictors::neural::ActivationImpl<ElementType>, predictors::neural::HardTanhActivation<ElementType>>();
        context.GetTypeFactory().AddType<predictors::neural::ActivationImpl<ElementType>, predictors::neural::LeakyReLUActivation<ElementType>>();
        context.GetTypeFactory().AddType<predictors::neural::ActivationImpl<ElementType>, predictors::neural::ParametricReLUActivation<ElementType>>();
        context.GetTypeFactory().AddType<predictors::neural::ActivationImpl<ElementType>, predictors::neural::ReLUActivation<ElementType>>();
        context.GetTypeFactory().AddType<predictors::neural::ActivationImpl<ElementType>, predictors::neural::SigmoidActivation<ElementType>>();
        context.GetTypeFactory().AddType<predictors::neural::ActivationImpl<ElementType>, predictors::neural::TanhActivation<ElementType>>();
    }

    void RegisterNodeTypes(SerializationContext& context)
    {
        RegisterRealNodeTypes<float>(context);
        RegisterRealNodeTypes<double>(context);

        // Add type erased nodes.
        context.GetTypeFactory().AddType<model::Node, nodes::VoiceActivityDetectorNode>();

        // additional non-real types, only for nodes that support that.
        context.GetTypeFactory().AddType<model::Node, model::InputNode<bool>>();
        context.GetTypeFactory().AddType<model::Node, model::InputNode<int>>();
        context.GetTypeFactory().AddType<model::Node, model::InputNode<int64_t>>();

        context.GetTypeFactory().AddType<model::Node, model::OutputNode<bool>>();
        context.GetTypeFactory().AddType<model::Node, model::OutputNode<int>>();
        context.GetTypeFactory().AddType<model::Node, model::OutputNode<int64_t>>();

        context.GetTypeFactory().AddType<model::Node, model::SpliceNode<bool>>();
        context.GetTypeFactory().AddType<model::Node, model::SpliceNode<int>>();
        context.GetTypeFactory().AddType<model::Node, model::SpliceNode<int64_t>>();

        context.GetTypeFactory().AddType<model::Node, model::SliceNode<bool>>();
        context.GetTypeFactory().AddType<model::Node, model::SliceNode<int>>();
        context.GetTypeFactory().AddType<model::Node, model::SliceNode<int64_t>>();

        context.GetTypeFactory().AddType<model::Node, nodes::AccumulatorNode<int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::AccumulatorNode<int64_t>>();

        context.GetTypeFactory().AddType<model::Node, nodes::ArgMaxNode<int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ArgMaxNode<int64_t>>();

        context.GetTypeFactory().AddType<model::Node, nodes::ArgMinNode<int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ArgMinNode<int64_t>>();

        context.GetTypeFactory().AddType<model::Node, nodes::BinaryPredicateNode<int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BinaryPredicateNode<double>>();

        context.GetTypeFactory().AddType<model::Node, nodes::BroadcastLinearFunctionNode<int>>();

        // context.GetTypeFactory().AddType<model::Node, nodes::BufferNode<bool>>(); // bugbug: Vector of 'bool' doesn't work.
        context.GetTypeFactory().AddType<model::Node, nodes::BufferNode<int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::BufferNode<int64_t>>();

        context.GetTypeFactory().AddType<model::Node, nodes::ClockNode>();
        context.GetTypeFactory().AddType<model::Node, nodes::SinkNode<int>>();

        context.GetTypeFactory().AddType<model::Node, nodes::ConcatenationNode<bool>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ConcatenationNode<int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ConcatenationNode<int64_t>>();

        context.GetTypeFactory().AddType<model::Node, nodes::ConstantNode<bool>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ConstantNode<int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ConstantNode<int64_t>>();

        context.GetTypeFactory().AddType<model::Node, nodes::DCTNode<float>>();
        context.GetTypeFactory().AddType<model::Node, nodes::DCTNode<double>>();

        context.GetTypeFactory().AddType<model::Node, nodes::DelayNode<bool>>();
        context.GetTypeFactory().AddType<model::Node, nodes::DelayNode<int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::DelayNode<int64_t>>();

        context.GetTypeFactory().AddType<model::Node, nodes::DemultiplexerNode<bool, bool>>();

        context.GetTypeFactory().AddType<model::Node, nodes::MultiplexerNode<bool, bool>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MultiplexerNode<int, bool>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MultiplexerNode<int64_t, bool>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MultiplexerNode<float, bool>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MultiplexerNode<double, bool>>();

        context.GetTypeFactory().AddType<model::Node, nodes::MultiplexerNode<bool, int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MultiplexerNode<int, int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MultiplexerNode<int64_t, int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MultiplexerNode<float, int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::MultiplexerNode<double, int>>();

        context.GetTypeFactory().AddType<model::Node, nodes::ProtoNNPredictorNode>();

        context.GetTypeFactory().AddType<model::Node, nodes::ReinterpretLayoutNode<int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::ReinterpretLayoutNode<bool>>();

        context.GetTypeFactory().AddType<model::Node, nodes::SimpleForestPredictorNode>();

        context.GetTypeFactory().AddType<model::Node, nodes::SingleElementThresholdNode>();

        context.GetTypeFactory().AddType<model::Node, nodes::SumNode<int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::SumNode<int64_t>>();

        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<bool, bool>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<bool, int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<bool, int64_t>>();

        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<int, bool>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<int, int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<int, int64_t>>();

        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<int64_t, bool>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<int64_t, int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<int64_t, int64_t>>();

        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<float, bool>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<float, int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<float, int64_t>>();

        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<double, bool>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<double, int>>();
        context.GetTypeFactory().AddType<model::Node, nodes::TypeCastNode<double, int64_t>>();

        context.GetTypeFactory().AddType<model::Node, nodes::UnaryOperationNode<bool>>();
        context.GetTypeFactory().AddType<model::Node, nodes::UnaryOperationNode<int>>();
    }

    void RegisterMapTypes(SerializationContext& context)
    {
        context.GetTypeFactory().AddType<model::Map, model::Map>();
    }

    template <typename UnarchiverType>
    model::Model LoadArchivedModel(std::istream& stream)
    {
        SerializationContext context;
        RegisterNodeTypes(context);
        UnarchiverType unarchiver(stream, context);
        model::Model model;
        unarchiver.Unarchive(model);
        return model;
    }

    template <typename ArchiverType, typename ObjectType>
    void SaveArchivedObject(const ObjectType& obj, std::ostream& stream)
    {
        ArchiverType archiver(stream);
        archiver.Archive(obj);
    }

    model::Model LoadModel(const std::string& filename)
    {
        if (!IsFileReadable(filename))
        {
            throw SystemException(SystemExceptionErrors::fileNotFound);
        }

        auto filestream = OpenIfstream(filename);
        return LoadArchivedModel<JsonUnarchiver>(filestream);
    }

    void SaveModel(const model::Model& model, const std::string& filename)
    {
        if (!IsFileWritable(filename))
        {
            throw SystemException(SystemExceptionErrors::fileNotWritable);
        }
        auto filestream = OpenOfstream(filename);
        SaveModel(model, filestream);
    }

    void SaveModel(const model::Model& model, std::ostream& outStream)
    {
        SaveArchivedObject<JsonArchiver>(model, outStream);
    }

    //
    // Map
    //
    model::Map LoadMap(const MapLoadArguments& mapLoadArguments)
    {
        if (mapLoadArguments.HasMapFilename())
        {
            return common::LoadMap(mapLoadArguments.inputMapFilename);
        }
        else if (mapLoadArguments.HasModelFilename())
        {
            auto model = common::LoadModel(mapLoadArguments.inputModelFilename);
            model::InputNodeBase* inputNode = nullptr;
            const model::OutputPortBase* output;
            if (mapLoadArguments.modelInputsString != "")
            {
                inputNode = mapLoadArguments.GetInput(model);
                if (inputNode == nullptr)
                {
                    throw InputException(InputExceptionErrors::invalidArgument, "Can't find input node");
                }
            }
            else // look for first input node
            {
                auto inputNodes = model.GetNodesByType<model::InputNodeBase>();
                if (inputNodes.size() == 0)
                {
                    throw InputException(InputExceptionErrors::invalidArgument, "Can't find input node");
                }
                inputNode = inputNodes[0];
            }

            if (mapLoadArguments.modelOutputsString != "")
            {
                output = mapLoadArguments.GetOutput(model);
            }
            else // look for first output node
            {
                auto outputNodes = model.GetNodesByType<model::OutputNodeBase>();
                if (outputNodes.size() == 0)
                {
                    throw InputException(InputExceptionErrors::invalidArgument, "Can't find output node");
                }
                auto outputNode = outputNodes[0];
                auto outputPorts = outputNode->GetOutputPorts();
                if (outputPorts.size() == 0)
                {
                    throw InputException(InputExceptionErrors::invalidArgument, "Can't find output port");
                }

                auto outputPort = outputPorts[0]; // ptr to port base
                output = outputPort;
            }

            return { model, { { "input", inputNode } }, { { "output", *output } } };
        }
        else // No model / map file specified -- return an identity map
        {
            model::Model model;
            auto inputNode = model.AddNode<model::InputNode<double>>(mapLoadArguments.defaultInputSize);
            return { model, { { "input", inputNode } }, { { "output", inputNode->output } } };
        }
    }

    model::Map LoadMap(const std::string& filename)
    {
        if (filename == "")
        {
            return model::Map{};
        }

        if (!IsFileReadable(filename))
        {
            throw SystemException(SystemExceptionErrors::fileNotFound, "File not found '" + filename + "'");
        }

        auto filestream = OpenIfstream(filename);

        try
        {
            return LoadArchivedMap<JsonUnarchiver>(filestream);
        }
        catch (const std::exception& ex)
        {
            throw InputException(InputExceptionErrors::badData, "Error: reading ELL map: " + filename + ": " + ex.what());
        }
    }

    void SaveMap(const model::Map& map, const std::string& filename)
    {
        if (!IsFileWritable(filename))
        {
            throw SystemException(SystemExceptionErrors::fileNotWritable);
        }
        auto filestream = OpenOfstream(filename);
        SaveMap(map, filestream);
    }

    void SaveMap(const model::Map& map, std::ostream& outStream)
    {
        SaveArchivedObject<JsonArchiver>(map, outStream);
    }

    CustomTypeFactoryFunction _func;

    void RegisterCustomTypeFactory(CustomTypeFactoryFunction func)
    {
        _func = func;
    }

    void AddCustomTypes(utilities::SerializationContext& context)
    {
        if (_func)
        {
            _func(context);
        }
    }

} // namespace common
} // namespace ell
