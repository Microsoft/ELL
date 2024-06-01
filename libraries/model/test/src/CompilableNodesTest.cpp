////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CompilableNodesTest.cpp (compile_test)
//  Authors:  Umesh Madan, Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "../clang/DotProductIR.h"

#include <common/include/LoadModel.h> // for RegisterNodeTypes

#include <model_testing/include/ModelTestUtilities.h>

#include <math/include/MathConstants.h>
#include <math/include/MatrixOperations.h>
#include <math/include/Tensor.h>

#include <emitters/include/EmitterException.h>
#include <emitters/include/EmitterTypes.h>
#include <emitters/include/IREmitter.h>

#include <model/include/CompiledMap.h>
#include <model/include/IRCompiledMap.h>
#include <model/include/IRMapCompiler.h>
#include <model/include/InputNode.h>
#include <model/include/Map.h>
#include <model/include/Model.h>
#include <model/include/OutputNode.h>
#include <model/include/PortMemoryLayout.h>
#include <model/include/SpliceNode.h>

#include <nodes/include/AccumulatorNode.h>
#include <nodes/include/ActivationLayerNode.h>
#include <nodes/include/BatchNormalizationLayerNode.h>
#include <nodes/include/BiasLayerNode.h>
#include <nodes/include/BinaryOperationNode.h>
#include <nodes/include/BinaryPredicateNode.h>
#include <nodes/include/BroadcastOperationNodes.h>
#include <nodes/include/BufferNode.h>
#include <nodes/include/ClockNode.h>
#include <nodes/include/ConcatenationNode.h>
#include <nodes/include/ConstantNode.h>
#include <nodes/include/DTWDistanceNode.h>
#include <nodes/include/DelayNode.h>
#include <nodes/include/DotProductNode.h>
#include <nodes/include/ExtremalValueNode.h>
#include <nodes/include/FFTNode.h>
#include <nodes/include/FullyConnectedLayerNode.h>
#include <nodes/include/IRNode.h>
#include <nodes/include/L2NormSquaredNode.h>
#include <nodes/include/MatrixMatrixMultiplyCodeNode.h>
#include <nodes/include/MatrixMatrixMultiplyNode.h>
#include <nodes/include/MatrixVectorMultiplyNode.h>
#include <nodes/include/MatrixVectorProductNode.h>
#include <nodes/include/MultiplexerNode.h>
#include <nodes/include/NeuralNetworkPredictorNode.h>
#include <nodes/include/NodeOperations.h>
#include <nodes/include/PoolingLayerNode.h>
#include <nodes/include/ReceptiveFieldMatrixNode.h>
#include <nodes/include/RegionDetectionLayerNode.h>
#include <nodes/include/ReinterpretLayoutNode.h>
#include <nodes/include/ReorderDataCodeNode.h>
#include <nodes/include/ReorderDataNode.h>
#include <nodes/include/SinkNode.h>
#include <nodes/include/SoftmaxLayerNode.h>
#include <nodes/include/SourceNode.h>
#include <nodes/include/SpatialConvolutionNode.h>
#include <nodes/include/SumNode.h>
#include <nodes/include/TypeCastNode.h>
#include <nodes/include/UnaryOperationNode.h>

#include <predictors/include/NeuralNetworkPredictor.h>

#include <predictors/neural/include/ActivationLayer.h>
#include <predictors/neural/include/BatchNormalizationLayer.h>
#include <predictors/neural/include/BiasLayer.h>
#include <predictors/neural/include/BinaryConvolutionalLayer.h>
#include <predictors/neural/include/ConvolutionalLayer.h>
#include <predictors/neural/include/FullyConnectedLayer.h>
#include <predictors/neural/include/InputLayer.h>
#include <predictors/neural/include/LeakyReLUActivation.h>
#include <predictors/neural/include/MaxPoolingFunction.h>
#include <predictors/neural/include/MeanPoolingFunction.h>
#include <predictors/neural/include/ParametricReLUActivation.h>
#include <predictors/neural/include/PoolingLayer.h>
#include <predictors/neural/include/ReLUActivation.h>
#include <predictors/neural/include/ScalingLayer.h>
#include <predictors/neural/include/SigmoidActivation.h>
#include <predictors/neural/include/SoftMaxActivation.h>
#include <predictors/neural/include/SoftmaxLayer.h>

#include <testing/include/testing.h>

#include <utilities/include/Logger.h>
#include <utilities/include/RandomEngines.h>
#include <utilities/include/TypeName.h>

#include <algorithm>
#include <iostream>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

// set to 1 to print models
#define PRINT_MODELS 0

using namespace ell;
using namespace ell::predictors;
using namespace ell::predictors::neural;
using namespace ell::logging;
using namespace ell::nodes;

void TestCompileIsEqual()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(2);
    auto predicateNode = model.AddNode<BinaryPredicateNode<double>>(model::PortElements<double>{ inputNode->output, 0 }, model::PortElements<double>{ inputNode->output, 1 }, BinaryPredicateType::equal);
    auto outputNode = model.AddNode<model::OutputNode<bool>>(predicateNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });

    std::string name = "TestCompileIsEqual";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<std::vector<double>> signal = { { 1, 1 }, { 2, 2 }, { 1, 3 }, { 1, 4 }, { 5, 5 }, { 1, 4 }, { 3, 3 }, { 2, 2 }, { 1, 0 } };
        std::vector<std::vector<bool>> expected = { { true }, { true }, { false }, { false }, { true }, { false }, { true }, { true }, { false } };
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestCompilableScalarOutputNode()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(1);
    auto outputNode = model.AddNode<model::OutputNode<double>>(ell::model::PortElements<double>{ inputNode->output });
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });

    std::string name = "TestCompileIsEqual";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<std::vector<double>> signal = { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 4 }, { 3 }, { 2 }, { 1 } };
        VerifyCompiledOutputAndResult(map, compiledMap, signal, signal, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestCompilableVectorOutputNode()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto outputNode = model.AddNode<model::OutputNode<double>>(inputNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });

    std::string name = "VectorOutputNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
        VerifyCompiledOutputAndResult(map, compiledMap, signal, signal, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestCompilableAccumulatorNode()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto accumNode = model.AddNode<AccumulatorNode<double>>(inputNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", accumNode->output } });

    std::string name = "AccumulatorNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
        std::vector<std::vector<double>> expected = { { 1, 2, 3 }, { 5, 7, 9 }, { 12, 15, 18 }, { 15, 19, 23 }, { 17, 22, 25 }, { 18, 27, 28 }, { 19, 29, 31 }, { 23, 34, 37 }, { 30, 42, 46 }, { 37, 46, 48 }, { 42, 48, 49 } };
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestCompilableConcatenationNode()
{
    model::Model model;

    auto inputNode = model.AddNode<model::InputNode<double>>(5);
    auto constantNode = model.AddNode<ConstantNode<double>>(std::vector<double>{ 6, 7, 8 });
    auto concatenationInputs = model::PortElements<double>({ inputNode->output, constantNode->output });
    auto outputNode = model.AddNode<ConcatenationNode<double>>(concatenationInputs, model::MemoryShape{ 1, 1, 8 });
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });

    std::string name = "ConcatenationNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<std::vector<double>> signal = { { 1, 2, 3, 4, 5 } };
        std::vector<std::vector<double>> expected = { { 1, 2, 3, 4, 5, 6, 7, 8 } };
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestCompilableDotProductNode()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto constantNode = model.AddNode<ConstantNode<double>>(std::vector<double>{ 1.0, 2.0, 3.0 });
    auto dotNode = model.AddNode<DotProductNode<double>>(inputNode->output, constantNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", dotNode->output } });

    std::string name = "DotProductNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
        std::vector<std::vector<double>> expected = { { 14 }, { 32 }, { 50 }, { 26 }, { 14 }, { 20 }, { 14 }, { 32 }, { 50 }, { 21 }, { 12 } };
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestCompilableDelayNode()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto delayNode = model.AddNode<DelayNode<double>>(inputNode->output, 8);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", delayNode->output } });

    std::string name = "DelayNode";
    TestWithSerialization(map, "DelayNode", [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
        std::vector<std::vector<double>> expected = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestCompilableDTWDistanceNode()
{
    model::Model model;
    std::vector<std::vector<double>> prototype = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto dtwNode = model.AddNode<DTWDistanceNode<double>>(inputNode->output, prototype);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", dtwNode->output } });

    std::string name = "DTWDistanceNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
        std::vector<std::vector<double>> expected = { { 2.7 }, { 1.35 }, { 0 }, { 1.8 }, { 2.55 }, { 2.25 }, { 2.7 }, { 1.35 }, { 0 }, { 1.65 }, { 2.4 } };
        VerifyCompiledOutput(map, compiledMap, signal, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

class LabeledPrototype
{
public:
    LabeledPrototype() = default;
    LabeledPrototype(const LabeledPrototype&) = default;
    LabeledPrototype(int label, const std::vector<std::vector<double>>& prototype) :
        _label(label),
        _prototype(prototype){};
    int Label() const { return _label; }
    size_t Dimension() const { return _prototype[0].size(); }
    std::vector<std::vector<double>> Prototype() const { return _prototype; }

private:
    int _label = 0;
    std::vector<std::vector<double>> _prototype;
};

model::Map GenerateMulticlassDTWClassifier(const std::vector<LabeledPrototype>& prototypes)
{
    auto dim = prototypes[0].Dimension();

    ell::model::Model model;
    auto inputNode = model.AddNode<ell::model::InputNode<double>>(dim);

    std::vector<double> labels = { 0.0 };
    auto threshNode = model.AddNode<ConstantNode<double>>(5.0);
    ell::model::PortElements<double> dtwOutputs(threshNode->output);

    std::vector<std::pair<int, ell::model::Node*>> dtwNodes;
    for (const auto& prototype : prototypes)
    {
        auto label = prototype.Label();
        labels.push_back(static_cast<double>(label));
        auto dtwNode = model.AddNode<DTWDistanceNode<double>>(inputNode->output, prototype.Prototype());
        dtwNodes.emplace_back(label, dtwNode);
        dtwOutputs.Append(dtwNode->output);
    }
    auto labelsNode = model.AddNode<ConstantNode<double>>(labels);
    auto argMinNode = model.AddNode<ArgMinNode<double>>(dtwOutputs); // val, argVal
    auto selectNode = model.AddNode<MultiplexerNode<double, int>>(labelsNode->output, argMinNode->argVal);
    auto combinedNode = model.AddNode<ell::model::SpliceNode<double>>(std::vector<const ell::model::OutputPortBase*>{ &selectNode->output, &argMinNode->val });

    model::Map result(model, { { "input", inputNode } }, { { "output", combinedNode->output } });
    return result;
}

void TestCompilableMulticlassDTW()
{
    model::Model model;
    std::vector<std::vector<double>> prototype1 = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
    std::vector<std::vector<double>> prototype2 = { { 9, 8, 7 }, { 6, 5, 4 }, { 3, 2, 1 } };
    std::vector<LabeledPrototype> prototypes = { { 3, prototype1 }, { 21, prototype2 } };

    auto map = GenerateMulticlassDTWClassifier(prototypes);

    std::string name = "MulticlassDTW";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
        std::vector<std::vector<double>> expected = { { 21, 0.6 }, { 3, 1.35 }, { 3, 0 }, { 21, 0.9 }, { 21, 0.45 }, { 21, 1.05 }, { 21, 0.6 }, { 3, 1.35 }, { 3, 0 }, { 21, 1.05 }, { 21, 0.3 } };
        // bug 1943: this model is not serializing properly so iteration 1 and 2 will fail here.
        if (iteration == 0)
        {
            VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
        }
    });
}

void TestCompilableScalarSumNode()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(1);
    auto sumNode = model.AddNode<SumNode<double>>(inputNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", sumNode->output } });

    std::string name = "SumNode_Scalar";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<std::vector<double>> signal = { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 4 }, { 3 }, { 2 }, { 1 } };
        VerifyCompiledOutputAndResult(map, compiledMap, signal, signal, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestCompilableSumNode()
{
    using ElementType = int64_t;
    std::vector<std::vector<ElementType>> signal = { { 1, 2, 3, 4, 5, 6 }, { 7, 8, 9, 3, 4, 5 }, { 2, 3, 2, 1, 5, 3 }, { 1, 2, 3, 4, 5, 6 }, { 7, 8, 9, 7, 4, 2 }, { 5, 2, 1, 2, 5, 9 } };
    std::vector<std::vector<ElementType>> expected = { { 21 }, { 36 }, { 16 }, { 21 }, { 37 }, { 24 } };
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(signal[0].size());
    auto sumNode = model.AddNode<SumNode<ElementType>>(inputNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", sumNode->output } });

    std::string name = "SumNode_Vector";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::MapCompilerOptions settings;
        settings.compilerSettings.allowVectorInstructions = true;
        model::ModelOptimizerOptions optimizerOptions;
        model::IRMapCompiler compiler(settings, optimizerOptions);
        auto compiledMap = compiler.Compile(map);

        // compare output
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

std::vector<std::vector<double>> GetExpectedUnaryOperationOutput(std::vector<std::vector<double>> signal, UnaryOperationType op)
{
    SigmoidActivationFunction<double> sigmoid;
    HardSigmoidActivationFunction<double> hardSigmoid;
    HardTanhActivationFunction<double> hardTanh;

    std::vector<std::vector<double>> result;
    for (auto v : signal)
    {
        std::vector<double> r;
        if (op == UnaryOperationType::softmax)
        {
            size_t len = v.size();
            r.resize(len);
            double max = *std::max_element(v.begin(), v.end());
            double sum = 0;
            for (size_t i = 0; i < len; i++)
            {
                auto e = std::exp(v[i] - max);
                r[i] = e;
                sum += e;
            }
            for (size_t i = 0; i < len; i++)
            {
                r[i] /= sum;
            }
        }
        else
        {
            for (double d : v)
            {
                switch (op)
                {
                case UnaryOperationType::abs:
                    d = std::abs(d);
                    break;
                case UnaryOperationType::exp:
                    d = std::exp(d);
                    break;
                case UnaryOperationType::hardSigmoid:
                    d = hardSigmoid.Compute(d);
                    break;
                case UnaryOperationType::hardTanh:
                    d = hardTanh.Compute(d);
                    break;
                case UnaryOperationType::log:
                    d = std::log(d);
                    break;
                case UnaryOperationType::log10:
                    d = std::log10(d);
                    break;
                case UnaryOperationType::logicalNot:
                    d = (d == 0) ? 1 : 0;
                    break;
                case UnaryOperationType::sign:
                    d = (d > 0) ? 1 : -1;
                    break;
                case UnaryOperationType::sin:
                    d = std::sin(d);
                    break;
                case UnaryOperationType::sigmoid:
                    d = sigmoid.Compute(d);
                    break;
                case UnaryOperationType::square:
                    d *= d;
                    break;
                case UnaryOperationType::cos:
                    d = std::cos(d);
                    break;
                case UnaryOperationType::sqrt:
                    d = std::sqrt(d);
                    break;
                case UnaryOperationType::tanh:
                    d = std::tanh(d);
                    break;
                default:
                    break;
                }
                r.push_back(d);
            }
        }
        result.push_back(r);
    }
    return result;
}

std::vector<std::vector<int>> GetExpectedUnaryOperationOutput(std::vector<std::vector<int>> signal, UnaryOperationType op)
{
    std::vector<std::vector<int>> result;
    for (auto v : signal)
    {
        std::vector<int> r;
        if (op == UnaryOperationType::square)
        {
            for (int d : v)
            {
                r.push_back(d * d);
            }
        }
        result.push_back(r);
    }
    return result;
}

std::vector<std::vector<bool>> GetExpectedUnaryOperationOutput(std::vector<std::vector<bool>> signal, UnaryOperationType op)
{
    std::vector<std::vector<bool>> result;
    for (auto v : signal)
    {
        std::vector<bool> r;
        if (op == UnaryOperationType::logicalNot)
        {
            for (bool d : v)
            {
                r.push_back(!d);
            }
        }
        result.push_back(r);
    }
    return result;
}

template <typename ElementType>
class Buffer
{
public:
    Buffer(size_t size) :
        _buffer(size)
    {
    }
    std::vector<ElementType>& Compute(std::vector<ElementType> input)
    {
        auto inputSize = input.size();
        if (inputSize > _buffer.size())
        {
            inputSize = _buffer.size();
        }
        auto offset = _buffer.size() - inputSize;
        if (offset > 0)
        {
            // shift the buffer left by the input size to make room for new input
            std::copy_n(_buffer.begin() + inputSize, offset, _buffer.begin());
        }

        // Copy input to right hand side of the buffer
        std::copy_n(input.begin(), inputSize, _buffer.begin() + offset);
        return _buffer;
    }

private:
    std::vector<ElementType> _buffer;
};

template <typename ElementType>
void TestBufferNode()
{
    model::Model model;
    int inputSize = 10;
    int bufferSize = 33;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputSize);
    auto& testOutput = ell::nodes::AddBufferNode<ElementType>(inputNode->output, bufferSize);
    auto outputNode = model.AddNode<model::OutputNode<ElementType>>(model::PortElements<ElementType>{ testOutput }, model::MemoryShape{ bufferSize });
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });

    std::string name = "BufferNode_" + utilities::TypeName<ElementType>::GetName();
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);

        Buffer<ElementType> buffer(bufferSize);
        std::vector<std::vector<ElementType>> signal;
        std::vector<std::vector<ElementType>> expected;
        for (int i = 0; i < 10; i++)
        {
            std::vector<ElementType> input;
            for (int j = 0; j < inputSize; j++)
            {
                input.push_back((i * 10) + j);
            }
            signal.push_back(input);
            auto result = buffer.Compute(input);
            expected.push_back(result);
        }

        // compare output
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

template void TestBufferNode<float>();
template void TestBufferNode<double>();
template void TestBufferNode<int>();
template void TestBufferNode<int64_t>();

void TestCompilableUnaryOperationNode()
{
    using namespace std::string_literals;

#define MAP_OP(op) { #op##s, UnaryOperationType::op }

    std::map<std::string, UnaryOperationType> floatOps{
        MAP_OP(abs),
        MAP_OP(exp),
        MAP_OP(hardSigmoid),
        MAP_OP(hardTanh),
        MAP_OP(log),
        MAP_OP(sign),
        MAP_OP(sin),
        MAP_OP(sigmoid),
        MAP_OP(softmax),
        MAP_OP(square),
        MAP_OP(cos),
        MAP_OP(sqrt),
        MAP_OP(tanh),
    };

    std::map<std::string, UnaryOperationType> intOps{
        MAP_OP(square),
    };

    std::map<std::string, UnaryOperationType> boolOps{
        // MAP_OP(logicalNot), // Boolean operations are still broken
    };
#undef MAP_OP

    // floating-point operations
    {
        model::Model model;
        auto inputNode = model.AddNode<model::InputNode<double>>(3);
        for (auto& op : floatOps)
        {
            auto& opName = op.first;
            auto opValue = op.second;
            auto testNode = model.AddNode<UnaryOperationNode<double>>(inputNode->output, opValue);
            auto map = model::Map(model, { { "input", inputNode } }, { { "output", testNode->output } });

            std::string name = "UnaryOperationNode_" + opName;
            TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
                model::IRMapCompiler compiler;
                auto compiledMap = compiler.Compile(map);

                // compare output
                std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
                std::vector<std::vector<double>> expected = GetExpectedUnaryOperationOutput(signal, opValue);
                VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
            });
        }
    }

    // int operations
    {
        model::Model model;
        auto inputNode = model.AddNode<model::InputNode<int>>(3);
        for (auto& op : intOps)
        {
            auto& opName = op.first;
            auto opValue = op.second;
            auto testNode = model.AddNode<UnaryOperationNode<int>>(inputNode->output, opValue);
            auto map = model::Map(model, { { "input", inputNode } }, { { "output", testNode->output } });

            std::string name = "UnaryOperationNode_" + opName;
            TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
                model::IRMapCompiler compiler;
                auto compiledMap = compiler.Compile(map);

                // compare output
                std::vector<std::vector<int>> signal = { { 0, 1, 2 }, { 0, 1, 3 }, { 0, 3, 1 }, { 2, 3, 3 }, { 3, -3, 0 }, { 3, 1, 0 }, { 0, 0, 0 }, { 3, 3, 3 } };
                std::vector<std::vector<int>> expected = GetExpectedUnaryOperationOutput(signal, opValue);
                VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
            });
        }
    }

    // boolean operations
    {
        model::Model model;
        auto inputNode = model.AddNode<model::InputNode<bool>>(3);
        for (auto& op : boolOps)
        {
            auto& opName = op.first;
            auto opValue = op.second;
            auto testNode = model.AddNode<UnaryOperationNode<bool>>(inputNode->output, opValue);
            auto map = model::Map(model, { { "input", inputNode } }, { { "output", testNode->output } });

            std::string name = "UnaryOperationNode_" + opName;
            TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
                model::IRMapCompiler compiler;
                auto compiledMap = compiler.Compile(map);

                // compare output
                std::vector<std::vector<bool>> signal = { { false, false, false }, { false, false, true }, { false, true, false }, { false, true, true }, { true, false, false }, { true, false, true }, { true, true, false }, { true, true, true } };
                std::vector<std::vector<bool>> expected = GetExpectedUnaryOperationOutput(signal, opValue);
                VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
            });
        }
    }
}

void TestL2NormSquaredNodeCompiled()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto testNode = model.AddNode<L2NormSquaredNode<double>>(inputNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", testNode->output } });
    std::string name = "L2NormSquaredNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::MapCompilerOptions settings;
        settings.compilerSettings.optimize = true;
        model::IRMapCompiler compiler(settings, {});
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
        std::vector<std::vector<double>> expected = {};
        VerifyCompiledOutput(map, compiledMap, signal, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

std::vector<std::vector<double>> GetExpectedMatrixVectorProduct(math::ConstRowMatrixReference<double> m, std::vector<std::vector<double>> signal)
{
    std::vector<std::vector<double>> result;
    for (auto v : signal)
    {
        math::ColumnVector<double> cv(v);
        math::ColumnVector<double> r;
        r.Resize(m.NumRows());
        ell::math::MultiplyScaleAddUpdate<ell::math::ImplementationType::native, double, ell::math::MatrixLayout::rowMajor>(1.0, m, cv, 1.0, r);
        result.push_back(r.ToArray());
    }
    return result;
}

void TestMatrixVectorProductNodeCompile()
{
    math::RowMatrix<double> m{
        { 1.2, 1.1, 0.8 },
        { 0.6, 0.9, 1.3 },
        { 0.3, 1.0, 0.4 },
        { -.4, 0.2, -.7 }
    };
    m.Transform([](double d) { return -2.0 * d; });

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto testNode = model.AddNode<MatrixVectorProductNode<double, math::MatrixLayout::rowMajor>>(inputNode->output, m);
    auto outputNode = model.AddNode<model::OutputNode<double>>(testNode->output, model::MemoryShape{ 1, 4, 1 });
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });

    std::string name = "MatrixVectorProductNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::MapCompilerOptions settings;
        settings.compilerSettings.optimize = false;
        model::ModelOptimizerOptions optimizerOptions;
        model::IRMapCompiler compiler(settings, optimizerOptions);
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
        std::vector<std::vector<double>> expected = GetExpectedMatrixVectorProduct(m, signal);
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

std::vector<std::vector<double>> GetExpectedBinaryOperationResult(std::vector<std::vector<double>> signal, std::vector<double> input, BinaryOperationType op)
{
    std::vector<std::vector<double>> result;
    for (auto v : signal)
    {
        std::vector<double> r;
        for (size_t i = 0, length = v.size(); i < length; i++)
        {
            double a = v[i];
            double b = input[i];
            double d = 0;
            switch (op)
            {
            case BinaryOperationType::add:
                d = a + b;
                break;
            case BinaryOperationType::subtract:
                d = a - b;
                break;
            case BinaryOperationType::multiply:
                d = a * b;
                break;
            case BinaryOperationType::divide:
                d = a / b;
                break;
            case BinaryOperationType::modulo:
                d = fmod(a, b);
                break;
            case BinaryOperationType::maximum:
                d = std::max(a, b);
                break;
            case BinaryOperationType::minimum:
                d = std::min(a, b);
                break;
            default:
                break;
            }
            r.push_back(d);
        }
        result.push_back(r);
    }
    return result;
}

void TestCompilableBinaryOperationNode()
{
    std::string opnames[] = {
        "none",
        "add",
        "subtract",
        "multiply",
        "divide",
        "logicalAnd",
        "logicalOr",
        "logicalXor",
        "modulo",
        "maximum",
        "minimum"
    };

    for (auto op : std::vector<BinaryOperationType>{
             BinaryOperationType::add,
             BinaryOperationType::subtract,
             BinaryOperationType::multiply,
             BinaryOperationType::divide,
             BinaryOperationType::modulo,
             BinaryOperationType::maximum,
             BinaryOperationType::minimum })
    {
        model::Model model;
        auto inputNode = model.AddNode<model::InputNode<double>>(3);
        auto input = std::vector<double>{ 1.0, 2.0, 3.0 };
        auto constantNode = model.AddNode<ConstantNode<double>>(input);
        auto testNode = model.AddNode<BinaryOperationNode<double>>(inputNode->output, constantNode->output, op);
        auto map = model::Map(model, { { "input", inputNode } }, { { "output", testNode->output } });

        std::string name = "BinaryOperationNode_" + opnames[static_cast<int>(op)];
        TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
            model::IRMapCompiler compiler;
            auto compiledMap = compiler.Compile(map);

            // compare output
            std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
            std::vector<std::vector<double>> expected = GetExpectedBinaryOperationResult(signal, input, op);
            VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
        });
    }
}

void TestCompilableBinaryOperationNode2()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 2;
    int numChannels = 2;
    int padding = 1;

    model::PortMemoryLayout input1Shape(model::MemoryShape{ numRows, numColumns, numChannels }, model::MemoryShape{ padding, padding, 0 });
    model::PortMemoryLayout input2Shape(model::MemoryShape{ numRows, numColumns, numChannels });
    model::PortMemoryLayout outputShape(model::MemoryShape{ numRows, numColumns, numChannels });

    auto inputNode = model.AddNode<model::InputNode<double>>(input1Shape.GetMemorySize());
    auto constantNode = model.AddNode<ConstantNode<double>>(std::vector<double>{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
    auto testNode = model.AddNode<BinaryOperationNode<double>>(inputNode->output, input1Shape, constantNode->output, input2Shape, outputShape, BinaryOperationType::add, 0.0);

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", testNode->output } });
    model::IRMapCompiler compiler;
    auto compiledMap = compiler.Compile(map);

    // compare output
    // clang-format off
    std::vector<std::vector<double>> signal = {{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                                 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0,
                                                 0.0, 0.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0,
                                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 }};
    // clang-format on
    std::vector<std::vector<double>> expected = { { 2, 4, 6, 8, 10, 12, 14, 16 } };
    VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, "BinaryOperationNode");
}

std::vector<std::vector<bool>> GetExpectedBinaryPredicateOutput(std::vector<std::vector<double>> signal, std::vector<double> input, BinaryPredicateType op)
{
    std::vector<std::vector<bool>> result;
    for (auto v : signal)
    {
        std::vector<bool> r;
        for (size_t i = 0, length = v.size(); i < length; i++)
        {
            double a = v[i];
            double b = input[i];
            bool d = 0;
            switch (op)
            {
            case BinaryPredicateType::equal:
                d = (a == b);
                break;
            case BinaryPredicateType::less:
                d = (a < b);
                break;
            case BinaryPredicateType::greater:
                d = (a > b);
                break;
            case BinaryPredicateType::notEqual:
                d = (a != b);
                break;
            case BinaryPredicateType::lessOrEqual:
                d = (a <= b);
                break;
            case BinaryPredicateType::greaterOrEqual:
                d = (a >= b);
                break;
            default:
                break;
            }
            r.push_back(d);
        }
        result.push_back(r);
    }
    return result;
}

std::string BinaryPredicateTypeNames[] = {
    "none",
    "equal",
    "less",
    "greater",
    "notEqual",
    "lessOrEqual",
    "greaterOrEqual"
};

// Problem: memory corruption for BinaryPredicateNode (probably because of bool foolishness)
void TestCompilableScalarBinaryPredicateNode()
{
    for (auto op : std::vector<BinaryPredicateType>())
    {
        model::Model model;
        auto inputNode = model.AddNode<model::InputNode<double>>(1);
        auto input = std::vector<double>{ 2 };
        auto constantNode = model.AddNode<ConstantNode<double>>(input);
        auto testNode = model.AddNode<BinaryPredicateNode<double>>(inputNode->output, constantNode->output, BinaryPredicateType::equal);
        auto map = model::Map(model, { { "input", inputNode } }, { { "output", testNode->output } });

        std::string name = "BinaryPredicateNode_Scalar" + BinaryPredicateTypeNames[static_cast<int>(op)];
        TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
            model::IRMapCompiler compiler;
            auto compiledMap = compiler.Compile(map);

            // compare output
            std::vector<std::vector<double>> signal = { { 1 }, { 4 }, { 7 }, { 2 }, { 4 }, { 1 }, { 11 }, { 24 }, { 92 }, { 1 } };
            std::vector<std::vector<bool>> expected = GetExpectedBinaryPredicateOutput(signal, input, op);
            VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
        });
    }
}

// Problem: memory corruption for BinaryPredicateNode (probably because of bool foolishness)
void TestCompilableBinaryPredicateNode()
{
    for (auto op : std::vector<BinaryPredicateType>())
    {
        model::Model model;
        auto inputNode = model.AddNode<model::InputNode<double>>(3);
        auto input = std::vector<double>{ 1.0, 2.0, 3.0 };
        auto constantNode = model.AddNode<ConstantNode<double>>(input);
        auto testNode = model.AddNode<BinaryPredicateNode<double>>(inputNode->output, constantNode->output, BinaryPredicateType::equal);
        auto map = model::Map(model, { { "input", inputNode } }, { { "output", testNode->output } });

        std::string name = "BinaryPredicateNode_Vector" + BinaryPredicateTypeNames[static_cast<int>(op)];
        TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
            model::IRMapCompiler compiler;
            auto compiledMap = compiler.Compile(map);

            // compare output
            std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };

            std::vector<std::vector<bool>> expected = GetExpectedBinaryPredicateOutput(signal, input, op);
            VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
        });
    }
}

void TestCompilableMultiplexerNode()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<int>>(1);
    auto constantNode = model.AddNode<ConstantNode<double>>(std::vector<double>{ 1.0, 2.0, 3.0 });
    auto testNode = model.AddNode<MultiplexerNode<double, int>>(constantNode->output, inputNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", testNode->output } });
    model::IRMapCompiler compiler;
    auto compiledMap = compiler.Compile(map);

    // compare output
    std::vector<std::vector<int>> signal = { { 0 }, { 1 }, { 0 }, { 1 }, { 1 }, { 0 }, { 0 }, { 1 }, { 1 }, { 0 } };
    VerifyCompiledOutput(map, compiledMap, signal, "MultiplexerNode");
}

void TestCompilableTypeCastNode(size_t dimension)
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<int>>(dimension);
    auto testNode = model.AddNode<TypeCastNode<int, double>>(inputNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", testNode->output } });

    std::string name = "TypeCastNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);

        // compare output
        const int numEntries = 10;
        std::vector<std::vector<int>> signal;
        std::generate_n(std::back_inserter(signal), numEntries, [dimension] { return GetRandomVector<int>(dimension, 0, 100); });
        std::vector<std::vector<double>> expected;
        for (auto v : signal)
        {
            std::vector<double> r(v.size());
            std::transform(v.begin(), v.end(), r.begin(), [](auto d) { return static_cast<double>(d); });
            expected.push_back(r);
        }
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestReinterpretLayoutNode()
{
    using ElementType = float;
    const int size = 10;
    std::vector<ElementType> constants(size);
    std::iota(constants.begin(), constants.end(), 0);
    model::Model model;

    // create two inputs that are deliberately different shapes (but same # elements).
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(model::MemoryShape{ 1, 1, size });
    const auto& constantOutput = Constant(model, constants, model::MemoryShape{ size, 1, 1 });

    // now re-interpret the contantNode so its shape matches the input node.
    const auto& reinterpret = ReinterpretLayout(constantOutput, model::MemoryShape{ 1, 1, size });

    // And do a binary operation on the input (binary operation would complain if the shapes don't match).
    auto addition = model.AddNode<BinaryOperationNode<ElementType>>(inputNode->output, reinterpret, BinaryOperationType::add);

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", addition->output } });
    std::string name = "TestReinterpretLayoutNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);
        std::vector<std::vector<ElementType>> signal{ std::vector<ElementType>(size) };
        std::vector<std::vector<ElementType>> expected{ constants };
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestReinterpretLayoutNodeWithPadding()
{
    using ElementType = float;
    const int rows = 3;
    const int cols = 4;
    const int size = rows * cols;
    std::vector<ElementType> constants(size);
    std::iota(constants.begin(), constants.end(), 0); // values 0-(r*c)-1
    model::Model model;

    // create two inputs that are deliberately different shapes
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(model::MemoryShape{ rows - 2, cols - 2 });
    auto constantNode = model.AddNode<ConstantNode<ElementType>>(constants); // implicit layout is a 1D vector of size rows*cols (== 12)

    // reinterpret linear vector as a 4x3 block of memory with 1 element of "padding" around the edge
    auto reinterpret = model.AddNode<ReinterpretLayoutNode<ElementType>>(constantNode->output, model::PortMemoryLayout(model::MemoryShape{ rows - 2, cols - 2 }, model::MemoryShape{ rows, cols }, model::MemoryShape{ 1, 1 }));

    // And do a binary operation on the input (binary operation would complain if the shapes don't match).
    auto addition = model.AddNode<BinaryOperationNode<ElementType>>(inputNode->output, reinterpret->output, BinaryOperationType::add);

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", addition->output } });
    std::string name = "TestReinterpretLayoutNodeWithPadding";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);
        std::vector<ElementType> inputVec((rows - 2) * (cols - 2), static_cast<ElementType>(10));
        std::vector<std::vector<ElementType>> signal{ inputVec };
        auto expectedVec = inputVec;
        for (int i = 0; i < static_cast<int>(inputVec.size()); ++i)
        {
            // compute index into "constants" array
            int r = i / (cols - 2);
            int c = i % (cols - 2);
            int index = ((r + 1) * cols) + (c + 1);

            // add input to appropriate location
            expectedVec[i] += constants[index];
        }
        std::vector<std::vector<ElementType>> expected{ expectedVec };
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestReorderDataNode1()
{
    using ElementType = float;
    int numRows = 3;
    int numColumns = 3;
    int numChannels = 2;
    model::Model model;
    model::PortMemoryLayout inputLayout(model::MemoryShape{ numRows, numColumns, numChannels }); // Default order: 0, 1, 2 == rows, columns, channels
    auto outputLayout = inputLayout.ReorderedCopy({ 2, 0, 1 });

    //        [  (1,2)   (3,4)    (5,6) ]
    // Input: [  (7,8)  (9,10), (11,12))]
    //        [(13,14) (15,16)  (17,18) ]
    //
    // = 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
    //
    //         [  1  3  5 ]   [  2  4  6 ]
    // Output: [  7  9 11 ]   [  8 10 12 ]
    //         [ 13 15 17 ]   [ 14 16 18 ]
    //
    // = 1 3 5 7 9 11 13 15 17 2 4 6 8 10 12 14 16 18

    size_t inputSize = inputLayout.GetMemorySize();
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputSize);
    const auto& testOutput = ReorderData(inputNode->output, inputLayout, outputLayout);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", testOutput } });
    model::IRMapCompiler compiler;
    auto compiledMap = compiler.Compile(map);

    std::vector<ElementType> input(inputSize);
    FillVector(input, 1.0f);
    Log() << "Input:" << EOL << input << EOL;

    // compare output
    std::vector<std::vector<ElementType>> signal = { input };
    VerifyCompiledOutput(map, compiledMap, signal, "ReorderDataNode");
}

void TestReorderDataNode2()
{
    using ElementType = float;
    int numRows = 3;
    int numColumns = 3;
    int numChannels = 2;
    int padding = 1;
    model::Model model;
    model::PortMemoryLayout inputLayout(model::MemoryShape{ numRows, numColumns, numChannels }, model::MemoryShape{ padding, padding, 0 }); // Default order: 0, 1, 2 == rows, columns, channels
    model::PortMemoryLayout outputLayout(model::MemoryShape{ numRows, numColumns, numChannels }); // Transform to order (channels, rows, cols) and remove padding

    //        [    (1,2)   (3,4)   (5,6)   (7,8)   9,10) ]
    // Input: [  (11,12) (13,14) (15,16) (17,18) (19,20) ]
    //        [  (21,22) (23,24) (25,26) (27,28) (29,30) ]
    //        [  (31,32) (33,34) (35,36) (37,38) (39,40) ]
    //        [  (41,42) (43,44) (45,46) (47,48) (49,50) ]
    //
    // = 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 ... 50
    //
    //         [ 13 15 17 ]   [ 14 16 18 ]
    // Output: [ 23 25 27 ]   [ 24 26 28 ]
    //         [ 33 35 37 ]   [ 34 36 38 ]
    //
    // = 13 15 17 23 25 27 33 35 37 14 16 18 24 26 28 34 36 38

    size_t inputSize = inputLayout.GetMemorySize();
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputSize);
    const auto& testOutput = ReorderData(inputNode->output, inputLayout, outputLayout, std::vector<int>{ 2, 0, 1 });
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", testOutput } });
    model::IRMapCompiler compiler;
    auto compiledMap = compiler.Compile(map);

    std::vector<ElementType> input(inputSize);
    FillVector(input, 1.0f);
    std::cout << "Input:\n"
              << input << std::endl;

    // compare output
    std::vector<std::vector<ElementType>> signal = { input };
    VerifyCompiledOutput(map, compiledMap, signal, "ReorderDataNode");
}

void TestReorderDataNode3()
{
    using ElementType = float;
    int numRows = 4;
    int numColumns = 5;
    int numChannels = 2;
    int padding = 1;
    model::Model model;
    model::PortMemoryLayout inputLayout(model::MemoryShape{ numRows, numColumns, numChannels }, model::MemoryShape{ padding, padding, 0 }); // Default order: 0, 1, 2 == rows, columns, channels
    model::PortMemoryLayout outputLayout(model::MemoryShape{ numRows, numColumns, numChannels }, model::DimensionOrder{ 2, 0, 1 }); // Transform to order (channels, rows, cols) and remove padding

    //        [    (1,2)   (3,4)   (5,6)   (7,8)   9,10) ]
    // Input: [  (11,12) (13,14) (15,16) (17,18) (19,20) ]
    //        [  (21,22) (23,24) (25,26) (27,28) (29,30) ]
    //        [  (31,32) (33,34) (35,36) (37,38) (39,40) ]
    //
    // = 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 ... 40
    //
    //         [ 13 15 17 ]   [ 14 16 18 ]
    // Output: [ 23 25 27 ]   [ 24 26 28 ]
    //
    // = 13 15 17 23 25 27 14 16 18 24 26 28

    size_t inputSize = inputLayout.GetMemorySize();
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputSize);
    const auto& testOutput = ReorderData(inputNode->output, inputLayout, outputLayout);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", testOutput } });
    model::IRMapCompiler compiler;
    auto compiledMap = compiler.Compile(map);

    std::vector<ElementType> input(inputSize);
    FillVector(input, 1.0f);
    std::cout << "Input:\n"
              << input << std::endl;

    // compare output
    std::vector<std::vector<ElementType>> signal = { input };
    VerifyCompiledOutput(map, compiledMap, signal, "ReorderDataNode");
}

void TestReorderDataCodeNode1()
{
    using ElementType = float;
    int numRows = 2;
    int numColumns = 3;
    int numChannels = 16;
    model::Model model;
    model::PortMemoryLayout inputLayout(model::MemoryShape{ numRows, numColumns, numChannels }); // Default order: 0, 1, 2 == rows, columns, channels
    auto outputLayout = inputLayout.ReorderedCopy({ 2, 0, 1 });

    size_t inputSize = inputLayout.GetMemorySize();
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputSize);
    const auto& testOutput = ReorderDataWithCodeNode(inputNode->output, inputLayout, outputLayout);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", testOutput } });

    // First, create the input tensor and expected output
    math::ChannelColumnRowTensor<float> input(numRows, numColumns, numChannels);
    math::ColumnRowChannelTensor<float> expectedOutput(numRows, numColumns, numChannels);

    // Next, verify that the compiled output is correct
    FillTensor<float>(input, 1, 1);
    FillTensor<float>(expectedOutput, 1, 1);

    Log() << "Input:" << EOL << input.ToArray() << EOL;

    std::string name = "TestReorderDataCodeNode1";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<std::vector<ElementType>> signal = { input.ToArray() };
        std::vector<std::vector<ElementType>> expected = { expectedOutput.ToArray() };
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestReorderDataCodeNode2()
{
    using ElementType = float;
    int numRows = 3;
    int numColumns = 3;
    int numChannels = 16;
    int padding = 1;
    model::Model model;
    model::PortMemoryLayout inputLayout(model::MemoryShape{ numRows, numColumns, numChannels }, model::MemoryShape{ padding, padding, 0 }); // Default order: 0, 1, 2 == rows, columns, channels
    auto outputLayout = inputLayout.ReorderedCopy({ 2, 0, 1 });

    size_t inputSize = inputLayout.GetMemorySize();
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputSize);
    const auto& testOutput = ReorderDataWithCodeNode(inputNode->output, inputLayout, outputLayout, std::vector<int>{ 2, 0, 1 });
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", testOutput } });
    model::IRMapCompiler compiler;
    auto compiledMap = compiler.Compile(map);

    std::vector<ElementType> input(inputSize);
    FillVector(input, 1.0f);
    Log() << "Input:" << EOL << input << EOL;

    // compare output
    std::vector<std::vector<ElementType>> signal = { input };
    VerifyCompiledOutput(map, compiledMap, signal, "ReorderDataCodeNode2");
}

void TestReorderDataCodeNode3()
{
    using ElementType = float;
    int numRows = 3;
    int numColumns = 4;
    int numChannels = 2;
    int padding = 1;
    model::Model model;
    model::PortMemoryLayout inputLayout(model::MemoryShape{ numRows, numColumns, numChannels }, model::MemoryShape{ padding, padding, 0 }); // Default order: 0, 1, 2 == rows, columns, channels
    auto outputLayout = inputLayout.ReorderedCopy({ 2, 0, 1 });

    size_t inputSize = inputLayout.GetMemorySize();
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputSize);
    const auto& testOutput = ReorderDataWithCodeNode(inputNode->output, inputLayout, outputLayout);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", testOutput } });
    model::IRMapCompiler compiler;
    auto compiledMap = compiler.Compile(map);

    std::vector<ElementType> input(inputSize);
    FillVector(input, 1.0f);
    Log() << "Input:" << EOL << input << EOL;

    // compare output
    std::vector<std::vector<ElementType>> signal = { input };
    VerifyCompiledOutput(map, compiledMap, signal, "ReorderDataCodeNode3");
}

void TestReorderDataCodeNode4()
{
    using ElementType = float;
    int numRows = 2;
    int numColumns = 5;
    model::Model model;
    model::PortMemoryLayout inputLayout(model::MemoryShape{ numRows, numColumns });
    auto outputLayout = inputLayout.ReorderedCopy({ 1, 0 });

    size_t inputSize = inputLayout.GetMemorySize();
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputSize);
    const auto& testOutput = ReorderDataWithCodeNode(inputNode->output, inputLayout, outputLayout);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", testOutput } });

    // First, create the input tensor and expected output
    math::RowMatrix<float> input(numRows, numColumns);
    math::ColumnMatrix<float> expectedOutput(numRows, numColumns);

    // Next, verify that the compiled output is correct
    FillMatrix<float>(input, 1, 1);
    FillMatrix<float>(expectedOutput, 1, 1);

    std::vector<std::vector<ElementType>> signal = { input.ToArray() };
    std::vector<std::vector<ElementType>> expected = { expectedOutput.ToArray() };

    Log() << "Input:" << EOL << input.ToArray() << EOL;

    std::string name = "TestReorderDataCodeNode4";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler;
        auto compiledMap = compiler.Compile(map);

        // compare output
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestReceptiveFieldMatrixNode(size_t numChannels, bool useNewReshape)
{
    const std::array<int, 3> rcdOrder = std::array<int, 3>{ 0, 1, 2 };
    const std::array<int, 3> drcOrder = std::array<int, 3>{ 2, 0, 1 };
    std::array<int, 3> dataOrder = useNewReshape ? drcOrder : rcdOrder;

    using ElementType = float;
    int numRows = 3;
    int numColumns = 3;

    model::PortMemoryLayout inputMemoryLayout({ numRows, numColumns, (int)numChannels });
    size_t filterWidth = 3;
    size_t stride = 1;
    size_t paddingSize = 1;
    size_t outputHeight = numRows;
    size_t outputWidth = numColumns;

    //
    // 1 input channel:
    //
    //        [ 1 2 3 ]
    // Input: [ 4 5 6 ]
    //        [ 7 8 9 ]
    //
    // = 1 2 3 4 5 6 7 8 9
    //
    //         [ 0 0 0  0 1 2  0 4 5 ]
    //         [ 0 0 0  1 2 3  4 5 6 ]
    //         [ 0 0 0  2 3 0  5 6 0 ]
    //         [ 0 1 2  0 4 5  0 7 8 ]
    // Output: [ 1 2 3  4 5 6  7 8 9 ]
    //         [ 2 3 0  5 6 0  8 9 0 ]
    //         [ 0 4 5  0 7 8  0 0 0 ]
    //         [ 4 5 6  7 8 9  0 0 0 ]
    //         [ 5 6 0  8 9 0  0 0 0 ]

    //
    // 2 input channels:
    //
    //        [ 1 2 3 ]  [ 10 11 12 ]
    // Input: [ 4 5 6 ]  [ 13 14 15 ]
    //        [ 7 8 9 ]  [ 16 17 18 ]
    //
    // = 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
    //
    //         [ 0 0 0  0 1 2  0 4 5 ]
    //         [ 0 0 0  0 10 11  0 13 14 ]
    //         [ 0 0 0  1 2 3  4 5 6 ]
    //         [ 0 0 0  10 11 12  13 14 15 ]
    //         [ 0 0 0  2 3 0  5 6 0 ]
    //         [ 0 0 0  11 12 0  14 15 0 ]
    //
    //         [ 0 1 2  0 4 5  0 7 8 ]
    //         [ 0 10 11  0 13 14  0 16 17 ]
    // Output: [ 1 2 3  4 5 6  7 8 9 ]
    //         [ 10 11 12  13 14 15  16 17 18 ]
    //         [ 2 3 0  5 6 0  8 9 0 ]
    //         [ 11 12 0  14 15 0  17 18 0]
    //
    //         [ 0 4 5  0 7 8  0 0 0 ]
    //         [ 0 13 14  0 16 17  0 0 0 ]
    //         [ 4 5 6  7 8 9  0 0 0 ]
    //         [ 13 14 15  16 17 18  0 0 0 ]
    //         [ 5 6 0  8 9 0  0 0 0 ]
    //         [ 14 15 0  17 18 0  0 0 0 ]
    //

    size_t inputSize = inputMemoryLayout.GetExtent().NumElements();
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputSize);
    auto testNode = model.AddNode<ReceptiveFieldMatrixNode<ElementType>>(inputNode->output,
                                                                         inputMemoryLayout,
                                                                         filterWidth,
                                                                         stride,
                                                                         paddingSize,
                                                                         dataOrder,
                                                                         outputWidth,
                                                                         outputHeight);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", testNode->output } });
    model::IRMapCompiler compiler;
    auto compiledMap = compiler.Compile(map);

    std::vector<ElementType> input(inputSize, 1.0);
    FillVector(input, 1.0f);
    compiledMap.SetInputValue(0, input);
    auto compiledResult = compiledMap.ComputeOutput<ElementType>(0);
    std::cout << "Compiled result size: " << compiledResult.size() << std::endl;
    std::cout << "  " << compiledResult << std::endl;

    if (numChannels == 1)
    {
        // Test vs. known output
        std::vector<ElementType> desiredOutput = { 0, 0, 0, 0, 1, 2, 0, 4, 5, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 2, 3, 0, 5, 6, 0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 0, 5, 6, 0, 8, 9, 0, 0, 4, 5, 0, 7, 8, 0, 0, 0, 4, 5, 6, 7, 8, 9, 0, 0, 0, 5, 6, 0, 8, 9, 0, 0, 0, 0 };
        testing::ProcessTest("Testing compiled ReceptiveFieldMatrixNode", testing::IsEqual(compiledResult, desiredOutput));
    }
    else if (numChannels == 2)
    {
        // Test vs. known output
        std::vector<ElementType> desiredOutput = { 0, 0, 0, 0, 1, 2, 0, 4, 5, 0, 0, 0, 0, 10, 11, 0, 13, 14, 0, 0, 0, 1, 2, 3, 4, 5, 6, 0, 0, 0, 10, 11, 12, 13, 14, 15, 0, 0, 0, 2, 3, 0, 5, 6, 0, 0, 0, 0, 11, 12, 0, 14, 15, 0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14, 0, 16, 17, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 2, 3, 0, 5, 6, 0, 8, 9, 0, 11, 12, 0, 14, 15, 0, 17, 18, 0, 0, 4, 5, 0, 7, 8, 0, 0, 0, 0, 13, 14, 0, 16, 17, 0, 0, 0, 4, 5, 6, 7, 8, 9, 0, 0, 0, 13, 14, 15, 16, 17, 18, 0, 0, 0, 5, 6, 0, 8, 9, 0, 0, 0, 0, 14, 15, 0, 17, 18, 0, 0, 0, 0 };
        testing::ProcessTest("Testing compiled ReceptiveFieldMatrixNode", testing::IsEqual(compiledResult, desiredOutput));
    }

    // compare compiled version with computed version (currently disabled because Compute() isn't implemented)
    // std::vector<std::vector<ElementType>> signal = { input };
    // VerifyCompiledOutput(map, compiledMap, signal, "ReceptiveFieldMatrixNode");
}

//
// Now test nodes that compile themselves as a function
//
void TestCompilableAccumulatorNodeFunction()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto accumNode1 = model.AddNode<AccumulatorNode<double>>(inputNode->output);
    auto constNode = model.AddNode<ConstantNode<double>>(std::vector<double>{ 1, 2, 3 });
    auto accumNode2 = model.AddNode<AccumulatorNode<double>>(accumNode1->output);
    auto accumNode3 = model.AddNode<AccumulatorNode<double>>(constNode->output);
    auto dotNode2 = model.AddNode<DotProductNode<double>>(accumNode2->output, accumNode3->output);
    auto accumNode4 = model.AddNode<AccumulatorNode<double>>(dotNode2->output);
    auto outputNode = model.AddNode<model::OutputNode<double>>(model::PortElements<double>{ accumNode4->output, dotNode2->output });

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });
    model::MapCompilerOptions settings;
    settings.compilerSettings.unrollLoops = true;
    settings.compilerSettings.optimize = true;
    model::IRMapCompiler compiler(settings, {});
    auto compiledMap = compiler.Compile(map);

    // compare output
    std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
    VerifyCompiledOutput(map, compiledMap, signal, "AccumulatorNode as function");
}

//
// Now test nodes that compile with callback(s)
//
struct TestCompilableSourceNodeContext
{
    size_t callbackCount;
    size_t inputSize;
};

void TestCompilableSourceNode()
{
    TestCompilableSourceNodeContext context{ 0, 5 };
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<TimeTickType>>(2);
    auto testNode = model.AddNode<SourceNode<double>>(
        inputNode->output,
        context.inputSize,
        "CompiledSourceNode_InputCallback");

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", testNode->output } });

    std::vector<std::vector<double>> signal = { { 5, 10 }, { 100, 200 }, { 456, 789 } };
    std::vector<std::vector<double>> expected = { { 42, 42, 42, 42, 42 }, { 42, 42, 42, 42, 42 }, { 42, 42, 42, 42, 42 } };

    std::string name = "SourceNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::MapCompilerOptions settings;
        settings.moduleName = "TestSourceNode";
        settings.compilerSettings.optimize = true;
        model::ModelOptimizerOptions optimizerOptions;

        // the Compute callback function cannot be serialized, so we set it here.
        const SourceNode<double>* constSourceNode = static_cast<const SourceNode<double>*>(map.GetSourceNodes()[0]);
        SourceNode<double>* sourceNode = const_cast<SourceNode<double>*>(constSourceNode);
        sourceNode->SetSourceFunction(
            [&context](auto& input) {
                input.assign(context.inputSize, 42.0);
                context.callbackCount++;
                return true;
            });

        model::IRMapCompiler compiler(settings, optimizerOptions);
        auto compiledMap = compiler.Compile(map);
        bool exception = false;
        try
        {
            compiledMap.SetContext(&context);
        }
        catch (const ell::utilities::InputException&)
        {
            exception = true;
        }
        testing::ProcessTest("SetContext throws an exception when SetSourceFunction is used", exception);

        // compare output
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });

    // Verify that jitted source callbacks are actually called, we have 3 inputs and 3 iterations, so 9 times in total.
    size_t expectedCount = 3 * 3 * 2;
    testing::ProcessTest("Testing callback values", testing::IsEqual(context.callbackCount, expectedCount));
}

struct CallbackContext
{
    size_t inputSize;
    bool called;
    std::vector<double> outputValues;
};

// C callback (called by emitted code)
extern "C" {
void TestSinkNode_CompiledSinkNode_OutputCallback(void* context, double* output, int size)
{
    CallbackContext* cc = static_cast<CallbackContext*>(context);
    if (!cc)
    {
        throw utilities::InputException(utilities::InputExceptionErrors::nullReference);
    }

    cc->called = true;
    Log() << "Sink Output Callback (size=" << size << ") " << *output << EOL;
    testing::ProcessTest("Callback size is correct", size == static_cast<int>(cc->inputSize));
    cc->outputValues.assign(output, output + size); // assign reallocates as needed
}
TESTING_FORCE_DEFINE_SYMBOL(TestSinkNode_CompiledSinkNode_OutputCallback, void, void*, double*, int);
}

void TestCompilableSinkNode(size_t inputSize, bool triggerValue)
{
    std::string sinkFunctionName = "CompiledSinkNode_OutputCallback";

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(inputSize);
    auto condition = model.AddNode<ConstantNode<bool>>(triggerValue);
    auto testNode = model.AddNode<SinkNode<double>>(inputNode->output, condition->output, sinkFunctionName);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", testNode->output } });

    std::string name = "SourceNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        CallbackContext context;
        context.inputSize = inputSize;
        context.called = false;

        model::MapCompilerOptions settings;
        settings.moduleName = "TestSinkNode";
        settings.compilerSettings.optimize = true;
        model::ModelOptimizerOptions optimizerOptions;
        model::IRMapCompiler compiler(settings, optimizerOptions);

        auto compiledMap = compiler.Compile(map);
        compiledMap.SetContext(&context);

        // compare output
        std::vector<std::vector<double>> signal = { {} };
        for (size_t i = 0; i < inputSize; ++i)
        {
            signal[0].push_back(i * 10);
        }
        VerifyCompiledOutput(map, compiledMap, signal, "SinkNode");

        if (triggerValue)
        {
            // Verify that sink callbacks are actually called
            testing::ProcessTest("Testing callback values", context.outputValues.size() == signal[0].size() && testing::IsEqual(context.outputValues, signal[0]));
            for (auto x : context.outputValues)
                Log() << x << "  ";
            Log() << EOL;
        }
        else
        {
            // Verify that sink callbacks are never called
            testing::ProcessTest("Testing callback was never called", testing::IsFalse(context.called));
        }
    });
}

void TestCompilableSinkNode()
{
    TestCompilableSinkNode(1, true); // fails
    TestCompilableSinkNode(1, false);
    TestCompilableSinkNode(100, true); // fails
    TestCompilableSinkNode(100, false);
}

void TestFloatNode()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<float>>(3);
    auto accumNode1 = model.AddNode<AccumulatorNode<float>>(inputNode->output);

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", accumNode1->output } });
    model::MapCompilerOptions settings;
    settings.compilerSettings.unrollLoops = true;
    settings.compilerSettings.optimize = true;

    std::string name = "AccumulatorNode_Float";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler(settings, {});
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<std::vector<float>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
        std::vector<std::vector<float>> expected = { { 1, 2, 3 }, { 5, 7, 9 }, { 12, 15, 18 }, { 15, 19, 23 }, { 17, 22, 25 }, { 18, 27, 28 }, { 19, 29, 31 }, { 23, 34, 37 }, { 30, 42, 46 }, { 37, 46, 48 }, { 42, 48, 49 } };
        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestMultipleOutputNodes()
{
    model::Model model;
    model::MemoryShape shape{ 224, 224, 3 };
    auto inputNode = model.AddNode<model::InputNode<double>>(shape);
    auto outputNode = model.AddNode<model::OutputNode<double>>(inputNode->output, shape);
    auto outputNode2 = model.AddNode<model::OutputNode<double>>(inputNode->output);

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output }, { "output2", outputNode2->output } });
    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;
    model::IRMapCompiler compiler;
    auto compiledMap = compiler.Compile(map);

#if PRINT_MODELS
    PrintIR(compiledMap);
#endif
}

void TestShapeFunctionGeneration()
{
    auto npos = std::string::npos;

    model::Model model;
    model::MemoryShape shape{ 224, 224, 3 };
    auto inputNode = model.AddNode<model::InputNode<double>>(shape);
    auto outputNode = model.AddNode<model::OutputNode<double>>(inputNode->output, shape);

    // this is blocked by IRMapCompiler.cpp line 42 which throws, so uncomment this when we decide to fix that.
    //    auto outputNode2 = model.AddNode<model::OutputNode<double>>(inputNode->output);

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } }); // , { "output2", outputNode2->output } });
    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;
    model::IRMapCompiler compiler;
    auto compiledMap = compiler.Compile(map);

#if PRINT_MODELS
    PrintIR(compiledMap);
#endif

    std::ostringstream buffer;
    compiledMap.WriteCode(buffer, emitters::ModuleOutputFormat::ir);

    std::string result = buffer.str();
    std::cout << result << std::endl;
    // some minimal strings for testing, full verbose string comparison might be too fragile to future code gen changes.
    auto inputFuncPos = result.find("define dso_local void @ELL_GetInputShape(i32 %index, %TensorShape* %shape");
    auto inputFuncPos2 = result.find("define dso_local void @ELL_GetInputShape(i32 %index, %TensorShape* nocapture %shape");
    auto outputFuncPos = result.find("define dso_local void @ELL_GetOutputShape(i32 %index, %TensorShape* %shape");
    auto outputFuncPos2 = result.find("define dso_local void @ELL_GetOutputShape(i32 %index, %TensorShape* nocapture %shape");
    auto storePos = result.find("store i32 224, i32* %rows, align 4");

    auto hasInputFunc = inputFuncPos != npos || inputFuncPos2 != npos;
    auto hasOutputFunc = outputFuncPos != npos || outputFuncPos2 != npos;
    auto hasStoreInstruction = storePos != npos;

    testing::ProcessTest("Testing GetOutputShape generation",
                         hasInputFunc && hasOutputFunc && hasStoreInstruction);
}

void TestMatrixVectorMultiplyNode(int m, int n, bool useBlas)
{
    using ValueType = float;
    std::vector<ValueType> vectorVals(n);
    FillVector(vectorVals);

    model::Model model;
    auto inputMatrixNode = model.AddNode<model::InputNode<ValueType>>(m * n);
    auto inputVectorNode = model.AddNode<ConstantNode<ValueType>>(vectorVals);

    auto matVecMultNode = model.AddNode<MatrixVectorMultiplyNode<ValueType>>(inputMatrixNode->output, m, n, n, inputVectorNode->output);

    auto map = model::Map(model, { { "inputMatrix", inputMatrixNode } }, { { "output", matVecMultNode->output } });

    std::string name = "MatrixVectorMultiplyNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::MapCompilerOptions settings;
        settings.compilerSettings.useBlas = useBlas;
        model::ModelOptimizerOptions optimizerOptions;
        model::IRMapCompiler compiler(settings, optimizerOptions);
        auto compiledMap = compiler.Compile(map);

        // compare output
        std::vector<ValueType> matrixVals(m * n);
        FillVector(matrixVals);
        std::vector<std::vector<ValueType>> signal = { matrixVals };
        VerifyCompiledOutput(map, compiledMap, signal, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestMatrixMatrixMultiplyNode(int m, int n, int k, bool useBlas)
{
    using ValueType = float;
    std::vector<ValueType> matrixBVals(k * n);
    FillVector(matrixBVals);

    model::Model model;
    auto inputMatrixNode = model.AddNode<model::InputNode<ValueType>>(m * k);
    auto matrixBNode = model.AddNode<ConstantNode<ValueType>>(matrixBVals);

    int lda = k;
    int ldb = n;
    int ldc = n;

    auto matMatMultNode = model.AddNode<MatrixMatrixMultiplyNode<ValueType>>(inputMatrixNode->output, m, n, k, lda, matrixBNode->output, ldb, ldc);

    auto map = model::Map(model, { { "inputMatrix", inputMatrixNode } }, { { "output", matMatMultNode->output } });

    std::string name = "MatrixMatrixMultiplyNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        // compare output
        std::vector<ValueType> matrixAVals(m * k);
        FillVector(matrixAVals);
        std::vector<std::vector<ValueType>> signal = { matrixAVals };

        model::MapCompilerOptions settings;
        settings.compilerSettings.useBlas = useBlas;
        model::ModelOptimizerOptions optimizerOptions;
        model::IRMapCompiler compiler(settings, optimizerOptions);
        auto compiledMap = compiler.Compile(map);

        VerifyCompiledOutput(map, compiledMap, signal, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

void TestOrderedMatrixMatrixMultiplyNode(int m, int n, int k, bool transposeA, bool transposeB, bool transposeC, bool useBlas)
{
    using ValueType = float;

    auto orderA = transposeA ? model::DimensionOrder{ 1, 0 } : model::DimensionOrder{ 0, 1 };
    auto orderB = transposeB ? model::DimensionOrder{ 1, 0 } : model::DimensionOrder{ 0, 1 };
    auto orderC = transposeC ? model::DimensionOrder{ 1, 0 } : model::DimensionOrder{ 0, 1 };
    auto outputLayout = model::PortMemoryLayout(model::MemoryShape{ m, n }).ReorderedCopy(orderC);

    model::Model model;
    auto inputMatrixNode = model.AddNode<model::InputNode<ValueType>>(model::MemoryShape{ m, k });
    const auto& reorderedInputMatrix = ReorderData(inputMatrixNode->output, orderA);

    std::vector<ValueType> matrixBVals(k * n);
    FillVector(matrixBVals);
    const auto& matrixB = Constant(model, matrixBVals, model::MemoryShape{ k, n });
    const auto& reorderedMatrixB = ReorderData(matrixB, orderB);

    const auto& matMatMultResult = MatrixMatrixMultiply(reorderedInputMatrix, reorderedMatrixB, outputLayout);

    auto map = model::Map(model, { { "inputMatrix", inputMatrixNode } }, { { "output", matMatMultResult } });

    // compare output
    std::vector<ValueType> matrixAVals(m * k);
    FillVector(matrixAVals);
    std::vector<std::vector<ValueType>> signal = { matrixAVals };

    std::string name = "MatrixMatrixMultiplyNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::MapCompilerOptions settings;
        settings.compilerSettings.useBlas = useBlas;
        model::ModelOptimizerOptions optimizerOptions;
        model::IRMapCompiler compiler(settings, optimizerOptions);
        auto compiledMap = compiler.Compile(map);

        std::stringstream id;
        id << std::boolalpha << "OrderedMatrixMatrixMultiplyNode(m = " << m << ", n = " << n << ", k = " << k << ", transposeA = "
           << transposeA << ", transposeB = " << transposeB << ", transposeC = " << transposeC << ", useBlas = " << useBlas << ") iteration " << iteration;
        VerifyCompiledOutput(map, compiledMap, signal, id.str());
    });
}

void TestMatrixMatrixMultiplyCodeNode(int m, int n, int k, int panelM, int panelN, int panelK, int kernelM, int kernelN, int kernelK, MatrixMatrixMultiplyImplementation gemmImpl)
{
    using ValueType = float;
    std::vector<ValueType> matrixBVals(k * n);
    FillRandomVector(matrixBVals);

    model::Model model;
    auto inputMatrixNode = model.AddNode<model::InputNode<ValueType>>(m * k);
    auto matrixBNode = model.AddNode<ConstantNode<ValueType>>(matrixBVals);

    int lda = k;
    int ldb = n;
    int ldc = n;

    auto matMatMultNode = model.AddNode<MatrixMatrixMultiplyCodeNode<ValueType>>(inputMatrixNode->output, m, n, k, lda, matrixBNode->output, ldb, ldc, panelM, panelN, panelK, kernelM, kernelN, kernelK, gemmImpl);

    auto map = model::Map(model, { { "inputMatrix", inputMatrixNode } }, { { "output", matMatMultNode->output } });

    std::string name = "MatrixMatrixMultiplyCodeNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        // compare output
        std::vector<ValueType> matrixAVals(m * k);
        FillRandomVector(matrixAVals);
        std::vector<std::vector<ValueType>> signal = { matrixAVals };

        model::MapCompilerOptions settings;
        model::ModelOptimizerOptions optimizerOptions;
        model::IRMapCompiler compiler(settings, optimizerOptions);
        auto compiledMap = compiler.Compile(map);

        std::vector<ValueType> expectedResult(m * n);
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                for (int kVal = 0; kVal < k; kVal++)
                {
                    expectedResult[i * n + j] += matrixAVals[i * k + kVal] * matrixBVals[kVal * n + j];
                }
            }
        }

        std::vector<std::vector<ValueType>> expected{ expectedResult };
        std::stringstream id;
        id << std::boolalpha << "MatrixMatrixMultiplyCodeNode(impl = " << static_cast<int>(gemmImpl)
           << ", m = " << m << ", n = " << n << ", k = " << k
           << ", panelM = " << panelM << ", panelN =" << panelN << ", panelK = " << panelK
           << ", kernelM = " << kernelM << ", kernelN = " << kernelN << ", kernelK" << kernelK << ") iteration " << iteration;

        VerifyCompiledOutputAndResult(map, compiledMap, signal, expected, id.str());
    });
}

// C callback (called by emitted code)
static int lagNotificationCallbackCount = 0;
extern "C" {
void Test_ClockNode_LagNotificationCallback(void* context, double lag)
{
    testing::EnableLoggingHelper helper;
    Log() << "ClockNode Lag Notification Callback " << lag << EOL;
    lagNotificationCallbackCount++;
}
}
TESTING_FORCE_DEFINE_SYMBOL(Test_ClockNode_LagNotificationCallback, void, void*, double);

void TestCompilableClockNode()
{
    using GetTicksUntilNextInterval = TimeTickType(TimeTickType);
    using GetLagThreshold = TimeTickType();
    using GetStepInterval = TimeTickType();

    constexpr TimeTickType lagThreshold = 125;
    constexpr TimeTickType interval = 50;
    constexpr TimeTickType start = 1511889201834.5767; // timestamp from python: time.time() * 1000

    model::Model model;

    auto inputNode = model.AddNode<model::InputNode<TimeTickType>>(1);
    auto clockNote = model.AddNode<ClockNode>(inputNode->output, interval, lagThreshold, "ClockNode_LagNotificationCallback");
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", clockNote->output } });

    model::MapCompilerOptions settings;
    settings.moduleName = "Test";
    settings.compilerSettings.optimize = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    auto& jitter = compiledMap.GetJitter();

    auto getStepIntervalFunction = reinterpret_cast<GetStepInterval*>(jitter.ResolveFunctionAddress("Test_GetStepInterval"));
    testing::ProcessTest("Testing compiled GetStepInterval", testing::IsEqual(getStepIntervalFunction(), interval));

    auto getLagThresholdFunction = reinterpret_cast<GetLagThreshold*>(jitter.ResolveFunctionAddress("Test_GetLagThreshold"));
    testing::ProcessTest("Testing compiled GetLagThreshold", testing::IsEqual(getLagThresholdFunction(), lagThreshold));

    auto getTicksFunction = reinterpret_cast<GetTicksUntilNextInterval*>(jitter.ResolveFunctionAddress("Test_GetTicksUntilNextInterval"));

    // compare output
    std::vector<std::vector<TimeTickType>> signal = {
        { start },
        { start + interval * 1 + lagThreshold / 2 }, // within threshold
        { start + interval * 2 }, // on time
        { start + interval * 3 + lagThreshold }, // late (expect notification)
        { start + interval * 4 + lagThreshold * 20 }, // really late (expect notification)
        { start + interval * 5 } // on time
    };

    std::vector<TimeTickType> getTicksResults;
    std::vector<TimeTickType> expectedGetTicksResults = {
        interval,
        interval - lagThreshold / 2,
        interval,
        interval - lagThreshold,
        interval - lagThreshold * 20,
        interval
    };

    lagNotificationCallbackCount = 0;
    for (const auto& input : signal)
    {
        // interleave calls to the map and getTicks, so that we can test updates to the last interval state
        VerifyCompiledOutput(map, compiledMap, std::vector<std::vector<TimeTickType>>{ input }, "ClockNode");
        getTicksResults.push_back(getTicksFunction(input[0]));
    }
    testing::ProcessTest("Testing compiled GetTicksUntilNextInterval", testing::IsEqual(getTicksResults, expectedGetTicksResults));
    testing::ProcessTest("Testing lag notification count", testing::IsEqual(lagNotificationCallbackCount, 2));
}

void TestCompilableFFTNode()
{
    using ValueType = float;
    const int N = 8;
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ValueType>>(N);
    auto fftNode = model.AddNode<FFTNode<ValueType>>(inputNode->output, N);

    std::vector<ValueType> input1(N, 1.0); // DC
    std::vector<ValueType> input2(N, 0); // impulse
    input2[0] = 1.0;
    std::vector<ValueType> input3(N, 0);
    for (int index = 0; index < N; ++index)
    {
        input3[index] = std::sin(2 * math::Constants<ValueType>::pi * index / N);
    }
    std::vector<std::vector<ValueType>> signal = { input1, input2, input3 };

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", fftNode->output } });

    std::string name = "FFTNode";
    TestWithSerialization(map, name, [&](model::Map& map, int iteration) {
        model::MapCompilerOptions settings;
        model::ModelOptimizerOptions optimizerOptions;
        model::IRMapCompiler compiler(settings, optimizerOptions);
        auto compiledMap = compiler.Compile(map);
        // compiledMap.WriteCode("FFTNode.ll", emitters::ModuleOutputFormat::ir);

        // compare output
        VerifyCompiledOutput(map, compiledMap, signal, utilities::FormatString("%s iteration %d", name.c_str(), iteration));
    });
}

class BinaryFunctionIRNode : public IRNode
{
public:
    /// @name Input and Output Ports
    /// @{
    static constexpr const char* input1PortName = "input1";
    static constexpr const char* input2PortName = "input2";
    static constexpr const char* outputPortName = "output";
    const model::InputPort<double>& input1 = _input1;
    const model::InputPort<double>& input2 = _input2;
    const model::OutputPort<double>& output = _output;
    /// @}

    BinaryFunctionIRNode(const model::OutputPort<double>& in1, const model::OutputPort<double>& in2, const std::string& functionName, const std::string& irCode, const emitters::NamedVariableTypeList& otherArgs) :
        IRNode({ &_input1, &_input2 }, { &_output }, functionName, irCode, otherArgs),
        _input1(this, in1, input1PortName),
        _input2(this, in2, input2PortName),
        _output(this, outputPortName, 1)
    {
    }

protected:
    std::vector<emitters::LLVMValue> GetNodeFunctionStateArguments(model::IRMapCompiler& compiler, emitters::IRFunctionEmitter& currentFunction) const override
    {
        size_t inputSize = _input1.Size();
        assert(inputSize == _input2.Size());
        return { currentFunction.Literal(static_cast<int>(inputSize)) };
    }

private:
    void Copy(model::ModelTransformer& transformer) const override
    {
        const auto& newInput1 = transformer.GetCorrespondingInputs(_input1);
        const auto& newInput2 = transformer.GetCorrespondingInputs(_input2);
        auto newNode = transformer.AddNode<BinaryFunctionIRNode>(newInput1, newInput2, GetFunctionName(), GetIRCode(), GetExtraArgs());
        transformer.MapNodeOutput(output, newNode->output);
    }

    // Inputs
    model::InputPort<double> _input1;
    model::InputPort<double> _input2;

    // Output
    model::OutputPort<double> _output;
};

void TestIRNode()
{
    int dimension = 3;
    std::vector<double> constValue(dimension);
    for (int index = 0; index < dimension; ++index)
    {
        constValue[index] = index + 0.5;
    }

    model::Model dotNodeModel;
    auto inputNode1 = dotNodeModel.AddNode<model::InputNode<double>>(dimension);
    auto constantNode1 = dotNodeModel.AddNode<ConstantNode<double>>(constValue);
    auto dotNode = dotNodeModel.AddNode<DotProductNode<double>>(inputNode1->output, constantNode1->output);
    auto dotNodeMap = model::Map(dotNodeModel, { { "input", inputNode1 } }, { { "output", dotNode->output } });

    model::Model irNodeModel;
    auto inputNode2 = irNodeModel.AddNode<model::InputNode<double>>(dimension);
    auto constantNode2 = irNodeModel.AddNode<ConstantNode<double>>(constValue);
    std::vector<model::PortElementsBase> inputs{ inputNode2->output, constantNode2->output };
    emitters::NamedVariableTypeList extraArgs{ { "count", emitters::VariableType::Int32 } };

    // Get the precompiled IR for dot product (defined in the DotProductIR.h file)
    auto dotProductIR = GetDotProductIR();
    auto dotProductFunctionName = GetDotProductFunctionName();
    auto irNode = irNodeModel.AddNode<BinaryFunctionIRNode>(inputNode2->output, constantNode2->output, dotProductFunctionName, dotProductIR, extraArgs);
    auto irNodeMap = model::Map(irNodeModel, { { "input", inputNode2 } }, { { "output", *irNode->GetOutputPort(0) } });

    model::IRMapCompiler compiler;
    auto compiledMap = compiler.Compile(irNodeMap);

#if PRINT_MODELS
    PrintIR(compiledMap);
#endif

    // compare output
    std::vector<std::vector<double>> signal;
    for (int index1 = 0; index1 < 8; ++index1)
    {
        std::vector<double> entry;
        for (int index2 = 0; index2 < dimension; ++index2)
        {
            entry.push_back(index2);
        }
        signal.push_back(entry);
    }

    VerifyCompiledOutput(dotNodeMap, compiledMap, signal, "DotProductNode");
}

//
// Neural network layer nodes
//

// Helper function
template <typename ElementType>
void VerifyLayerMap(ell::model::Map& map, const ell::model::Node* computeNode, const typename Layer<ElementType>::TensorType& inputWithPadding, const typename Layer<ElementType>::ConstTensorReferenceType& output, const std::string& additionalMessage = "")
{
    std::vector<std::vector<ElementType>> signal = { inputWithPadding.ToArray() };
    std::vector<std::vector<ElementType>> expectedOutput = { output.ToArray() };
    VerifyMapOutput(map, signal, expectedOutput, computeNode->GetRuntimeTypeName(), additionalMessage);

    model::MapCompilerOptions settings;
    settings.compilerSettings.useBlas = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    // compare output
    VerifyCompiledOutput(map, compiledMap, signal, computeNode->GetRuntimeTypeName(), additionalMessage);
}

template <typename ElementType>
void VerifyArchiveAndUnarchivingMap(const ell::model::Map& map, const ell::model::Node* computeNode, const typename Layer<ElementType>::TensorType& inputWithPadding, const typename Layer<ElementType>::ConstTensorReferenceType& output, const std::string& additionalMessage = "")
{
    // Test archiving / unarchiving produces same result as map before
    // archiving.
    utilities::SerializationContext context;
    common::RegisterNodeTypes(context);
    std::stringstream strstream;
    utilities::JsonArchiver archiver(strstream);

    archiver << map;
    utilities::JsonUnarchiver unarchiver(strstream, context);
    model::Map unarchivedMap;
    unarchiver >> unarchivedMap;

    VerifyLayerMap<ElementType>(unarchivedMap, computeNode, inputWithPadding, output, additionalMessage + " ArchiveAndUnarchivingMap");
}

void TestNeuralNetworkPredictorNode1()
{
    // Create a simple neural net model with the following layers:
    // input -> bias
    using ElementType = double;
    using InputParameters = typename predictors::neural::InputLayer<ElementType>::InputParameters;
    using LayerParameters = typename predictors::neural::Layer<ElementType>::LayerParameters;
    using VectorType = typename predictors::neural::Layer<ElementType>::VectorType;
    using DataVectorType = typename predictors::NeuralNetworkPredictor<ElementType>::DataVectorType;

    // Build a net
    typename predictors::NeuralNetworkPredictor<ElementType>::InputLayerReference inputLayer;
    typename predictors::NeuralNetworkPredictor<ElementType>::Layers layers;

    InputParameters inputParams{ { 1, 1, 3 }, { predictors::neural::PaddingScheme::zeros, 0 }, { 1, 1, 3 }, { predictors::neural::PaddingScheme::zeros, 0 }, 1 };
    inputLayer = std::make_unique<predictors::neural::InputLayer<ElementType>>(inputParams);

    LayerParameters layerParameters{ inputLayer->GetOutput(), predictors::neural::NoPadding(), { 1, 1, 3 }, predictors::neural::NoPadding() };
    VectorType bias1({ -0.43837756f, -0.90868396f, -0.0323102f });
    layers.push_back(std::unique_ptr<predictors::neural::Layer<ElementType>>(new predictors::neural::BiasLayer<ElementType>(layerParameters, bias1)));
    predictors::NeuralNetworkPredictor<ElementType> neuralNetwork(std::move(inputLayer), std::move(layers));

    std::vector<ElementType> input = { 0, 1, 2 };
    auto output = neuralNetwork.Predict(DataVectorType(input));

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(GetShapeSize(neuralNetwork.GetInputShape()));
    auto predictorNode = model.AddNode<NeuralNetworkPredictorNode<double>>(inputNode->output, neuralNetwork);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", predictorNode->output } });
    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

#if PRINT_MODELS
    PrintIR(compiledMap);
#endif

    // compare output
    std::vector<std::vector<double>> signal = { input };
    VerifyCompiledOutput(map, compiledMap, signal, predictorNode->GetRuntimeTypeName() + "_1");

    // Test that archiving / unarchiving produces same results
    utilities::SerializationContext context;
    common::RegisterNodeTypes(context);
    std::stringstream strstream;
    utilities::JsonArchiver archiver(strstream);

    archiver << map;
    utilities::JsonUnarchiver unarchiver(strstream, context);
    model::Map unarchivedMap;
    unarchiver >> unarchivedMap;

    VerifyCompiledOutput(unarchivedMap, compiledMap, signal, predictorNode->GetRuntimeTypeName() + "_1");
}

void TestBroadcastLinearFunctionNode()
{
    using ElementType = double;

    // Create model
    model::Model model;

    int rows = 5;
    int cols = 7;

    math::RowMatrix<double> m(rows, cols);
    m.Fill(1);

    auto inputNode = model.AddNode<model::InputNode<double>>(model::MemoryShape{ rows, cols });

    std::vector<ElementType> scaleValues = { 1, 2, 3, 4, 5 };
    auto scaleValuesNode = model.AddNode<ConstantNode<ElementType>>(scaleValues, model::MemoryShape{ rows });
    auto biasValuesNode = model.AddNode<ConstantNode<ElementType>>(); // nothing

    const size_t secondaryInputDimension = 0; // broadcast the scale vector across our input "rows".
    auto computeNode = model.AddNode<BroadcastLinearFunctionNode<ElementType>>(inputNode->output,
                                                                               inputNode->output.GetMemoryLayout(),
                                                                               scaleValuesNode->output,
                                                                               biasValuesNode->output,
                                                                               secondaryInputDimension,
                                                                               inputNode->output.GetMemoryLayout());

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });

    model::MapCompilerOptions settings;
    settings.compilerSettings.parallelize = false;
    settings.compilerSettings.optimize = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    // compare output
    std::vector<std::vector<ElementType>> signal;
    ElementType* start = m.GetDataPointer();
    ElementType* end = start + m.Size();
    signal.push_back(std::vector<ElementType>(start, end));

    // to compute expected output we need to copy the scale values into each column of the expected matrix.
    math::ColumnVector<ElementType> scaleVector(scaleValues);
    math::RowMatrix<ElementType> expected(rows, cols);
    for (int i = 0; i < cols; i++)
    {
        expected.GetColumn(i).CopyFrom(scaleVector);
    }

    bool ok = VerifyCompiledOutputAndResult<ElementType, ElementType>(map, compiledMap, signal, { expected.ToArray() }, "TestBroadcastLinearFunctionNode");
    if (!ok)
    {
        map.SetInputValue(0, signal[0]);
        std::vector<ElementType> result = map.ComputeOutput<ElementType>(0);

        math::RowMatrix<ElementType> actual(rows, cols, result);
        std::stringstream stream1;
        math::Print(expected, stream1);

        std::stringstream stream2;
        math::Print(actual, stream2);
    }
}

void TestNeuralNetworkPredictorNode2()
{
    // Create a simple neural net model with the following layers:
    // input -> fully-connected -> bias -> activation -> fully-connected -> bias
    using ElementType = double;
    using InputParameters = typename InputLayer<ElementType>::InputParameters;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using VectorType = typename Layer<ElementType>::VectorType;
    using MatrixType = typename Layer<ElementType>::MatrixType;
    using DataVectorType = typename NeuralNetworkPredictor<ElementType>::DataVectorType;

    // Build a net
    typename NeuralNetworkPredictor<ElementType>::InputLayerReference inputLayer;
    typename NeuralNetworkPredictor<ElementType>::Layers layers;

    InputParameters inputParams = { { 1, 1, 2 }, { PaddingScheme::zeros, 0 }, { 1, 1, 2 }, { PaddingScheme::zeros, 0 }, 1 };
    inputLayer = std::make_unique<InputLayer<ElementType>>(inputParams);

    LayerParameters layerParameters{ inputLayer->GetOutput(), NoPadding(), { 1, 1, 3 }, NoPadding() };
    MatrixType weights1(3, 2);
    weights1(0, 0) = -0.97461396f;
    weights1(0, 1) = 1.40845299f;
    weights1(1, 0) = -0.14135513f;
    weights1(1, 1) = -0.54136097f;
    weights1(2, 0) = 0.99313086f;
    weights1(2, 1) = -0.99083692f;
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new FullyConnectedLayer<ElementType>(layerParameters, weights1)));

    layerParameters = { layers[0]->GetOutput(), NoPadding(), { 1, 1, 3 }, NoPadding() };
    VectorType bias1({ -0.43837756f, -0.90868396f, -0.0323102f });
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new BiasLayer<ElementType>(layerParameters, bias1)));

    layerParameters = { layers[1]->GetOutput(), NoPadding(), { 1, 1, 3 }, NoPadding() };
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ActivationLayer<ElementType>(layerParameters, new ReLUActivation<ElementType>())));

    layerParameters = { layers[2]->GetOutput(), NoPadding(), { 1, 1, 1 }, NoPadding() };
    MatrixType weights2(1, 3);
    weights2(0, 0) = 1.03084767f;
    weights2(0, 1) = -0.10772263f;
    weights2(0, 2) = 1.04077697f;
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new FullyConnectedLayer<ElementType>(layerParameters, weights2)));

    layerParameters = { layers[3]->GetOutput(), NoPadding(), { 1, 1, 1 }, NoPadding() };
    VectorType bias2({ 1.40129846e-20f });
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new BiasLayer<ElementType>(layerParameters, bias2)));

    NeuralNetworkPredictor<ElementType> neuralNetwork(std::move(inputLayer), std::move(layers));
    std::vector<ElementType> input = { 0, 1 };
    auto output = neuralNetwork.Predict(DataVectorType(input));

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(GetShapeSize(neuralNetwork.GetInputShape()));
    auto predictorNode = model.AddNode<NeuralNetworkPredictorNode<double>>(inputNode->output, neuralNetwork);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", predictorNode->output } });

    model::MapCompilerOptions settings;
    settings.compilerSettings.parallelize = false;
    settings.compilerSettings.optimize = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

#if PRINT_MODELS
    PrintIR(compiledMap);
#endif

    // compare output
    std::vector<std::vector<double>> signal = { input };
    VerifyCompiledOutput(map, compiledMap, signal, predictorNode->GetRuntimeTypeName() + "_2");
}

void TestNeuralNetworkPredictorNode3()
{
    // Create a simple neural net model with the following layers:
    // input -> convolutional
    using ElementType = double;
    using InputParameters = typename InputLayer<ElementType>::InputParameters;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;

    // Build a net
    typename NeuralNetworkPredictor<ElementType>::InputLayerReference inputLayer;
    typename NeuralNetworkPredictor<ElementType>::Layers layers;

    size_t imageSize = 3;
    size_t numChannels = 1;
    size_t k = 3;
    size_t numFilters = 1;
    size_t padding = 1;
    size_t stride = 1;

    // convolve 3x3 image of all 1s with a 3x3 filter of all 1s, with 0-padding on image
    // result should be:
    //
    //   4 6 4
    //   6 9 6
    //   4 6 4

    InputParameters inputParams = { { imageSize, imageSize, numChannels }, NoPadding(), { imageSize + 2 * padding, imageSize + 2 * padding, numChannels }, { PaddingScheme::zeros, padding }, 1.0 };
    inputLayer = std::make_unique<InputLayer<ElementType>>(inputParams);

    LayerParameters layerParameters{ inputLayer->GetOutput(), { PaddingScheme::zeros, padding }, { imageSize, imageSize, numFilters }, NoPadding() };
    auto convolutionMethod = ConvolutionMethod::unrolled;
    ConvolutionalParameters convolutionalParams{ k, stride, convolutionMethod, numFilters };
    TensorType convWeights(numFilters * k, k, numChannels);
    FillTensor(convWeights);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ConvolutionalLayer<ElementType>(layerParameters, convolutionalParams, convWeights)));

    NeuralNetworkPredictor<ElementType> neuralNetwork(std::move(inputLayer), std::move(layers));

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(GetShapeSize(neuralNetwork.GetInputShape()));
    auto predictorNode = model.AddNode<NeuralNetworkPredictorNode<double>>(inputNode->output, neuralNetwork);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", predictorNode->output } });

    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    // compare output
    const auto inputSize = imageSize * imageSize * numChannels;
    std::vector<ElementType> input(inputSize, 1.0);
    FillVector(input, 1.0);
    std::vector<std::vector<double>> signal = { input };

    compiledMap.SetInputValue(0, input);
    auto compiledResult = compiledMap.ComputeOutput<ElementType>(0);
    std::cout << "Compiled result size: " << compiledResult.size() << std::endl;
    std::cout << "  " << compiledResult << std::endl;

    VerifyCompiledOutput(map, compiledMap, signal, predictorNode->GetRuntimeTypeName() + "_3");
}

void TestNeuralNetworkPredictorNode4()
{
    // Create a simple neural net model with the following layers:
    // input -> convolutional (no padding)

    using ElementType = double;
    using InputParameters = typename InputLayer<ElementType>::InputParameters;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;

    // Build a net
    typename NeuralNetworkPredictor<ElementType>::InputLayerReference inputLayer;
    typename NeuralNetworkPredictor<ElementType>::Layers layers;

    size_t imageSize = 5;
    size_t numChannels = 1;
    size_t k = 3;
    size_t numFilters = 1;
    size_t padding = 0;
    size_t stride = 1;

    // convolve 5x5 image of all 1s with a 3x3 filter of all 1s, with no padding on image
    // result should be:
    //
    //   9 9 9
    //   9 9 9
    //   9 9 9

    InputParameters inputParams = { { imageSize, imageSize, numChannels }, NoPadding(), { imageSize + 2 * padding, imageSize + 2 * padding, numChannels }, { PaddingScheme::zeros, padding }, 1.0 };
    inputLayer = std::make_unique<InputLayer<ElementType>>(inputParams);

    LayerParameters layerParameters{ inputLayer->GetOutput(), { PaddingScheme::zeros, padding }, { imageSize - 2 * (k / 2), imageSize - 2 * (k / 2), numFilters }, NoPadding() };
    auto convolutionMethod = ConvolutionMethod::unrolled;
    ConvolutionalParameters convolutionalParams{ k, stride, convolutionMethod, numFilters }; //
    TensorType convWeights(numFilters * k, k, numChannels);
    // FillTensor(convWeights);
    convWeights.Fill(1);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ConvolutionalLayer<ElementType>(layerParameters, convolutionalParams, convWeights)));

    NeuralNetworkPredictor<ElementType> neuralNetwork(std::move(inputLayer), std::move(layers));

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(GetShapeSize(neuralNetwork.GetInputShape()));
    auto predictorNode = model.AddNode<NeuralNetworkPredictorNode<double>>(inputNode->output, neuralNetwork);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", predictorNode->output } });

    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    // compare output
    const auto inputSize = imageSize * imageSize * numChannels;
    std::vector<ElementType> input(inputSize, 1.0);
    // FillVector(input, 1.0);
    std::vector<std::vector<double>> signal = { input };

    map.SetInputValue(0, input);
    auto computedResult = map.ComputeOutput<ElementType>(0);
    std::cout << "Computed result size: " << computedResult.size() << std::endl;
    std::cout << "  " << computedResult << std::endl;

    compiledMap.SetInputValue(0, input);
    auto compiledResult = compiledMap.ComputeOutput<ElementType>(0);
    std::cout << "Compiled result size: " << compiledResult.size() << std::endl;
    std::cout << "  " << compiledResult << std::endl;

    VerifyCompiledOutput(map, compiledMap, signal, predictorNode->GetRuntimeTypeName() + "_4");
}

void TestNeuralNetworkPredictorNode5()
{
    // Create a simple neural net model with the following layers:
    // input -> convolutional -> max_pool -> convolutional -> mean-pooling -> softmax
    using ElementType = float;
    using InputParameters = typename InputLayer<ElementType>::InputParameters;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;

    // Build a net
    typename NeuralNetworkPredictor<ElementType>::InputLayerReference inputLayer;
    typename NeuralNetworkPredictor<ElementType>::Layers layers;

    size_t k = 3;
    size_t w = 8;
    size_t d = 2;
    size_t f1 = 3;
    size_t f2 = 6;

    // Input Layer
    InputParameters inputParams = { { w, w, d }, NoPadding(), { w + 2, w + 2, d }, ZeroPadding(1), 1 };
    inputLayer = std::make_unique<InputLayer<ElementType>>(inputParams);

    // ConvolutionalLayer
    LayerParameters layerParameters{ inputLayer->GetOutput(), ZeroPadding(1), { w, w, f1 }, NoPadding() };
    auto convolutionMethod = ConvolutionMethod::unrolled;
    ConvolutionalParameters convolutionalParams{ k, 1, convolutionMethod, 1 };
    TensorType convWeights1(f1 * k, k, d);
    FillTensor(convWeights1, -10.0f, 0.0625f);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ConvolutionalLayer<ElementType>(layerParameters, convolutionalParams, convWeights1)));

    // Max PoolingLayer
    layerParameters = { layers.back()->GetOutput(), ZeroPadding(1), { w / 2 + 2, w / 2 + 2, f1 }, ZeroPadding(1) };
    PoolingParameters poolingParameters{ 2, 2 }; // window size, stride
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new PoolingLayer<ElementType, MaxPoolingFunction>(layerParameters, poolingParameters)));

    // ConvolutionalLayer
    layerParameters = { layers.back()->GetOutput(), ZeroPadding(1), { w / 2, w / 2, f2 }, NoPadding() };
    ConvolutionalParameters convolutionalParams2{ k, 1, convolutionMethod, 1 };
    TensorType convWeights2(f2 * k, k, f1);
    FillTensor(convWeights2, -2.0f, 0.0625f);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ConvolutionalLayer<ElementType>(layerParameters, convolutionalParams2, convWeights2)));

    // Mean PoolingLayer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), { (w / 2) / 2, (w / 2) / 2, f2 }, NoPadding() };
    PoolingParameters poolingParameters2{ 2, 2 };
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new PoolingLayer<ElementType, MeanPoolingFunction>(layerParameters, poolingParameters2)));

    // Create the predictor
    NeuralNetworkPredictor<ElementType> neuralNetwork(std::move(inputLayer), std::move(layers));

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(GetShapeSize(neuralNetwork.GetInputShape()));
    auto predictorNode = model.AddNode<NeuralNetworkPredictorNode<ElementType>>(inputNode->output, neuralNetwork);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", predictorNode->output } });

    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;
    settings.compilerSettings.useBlas = true; // !!! if BLAS is off, this fails
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    // Create an input vector
    std::vector<ElementType> input(w * w * d);
    FillVector(input);

    map.SetInputValue(0, input);
    auto computedResult = map.ComputeOutput<ElementType>(0);
    std::cout << "Computed result size: " << computedResult.size() << std::endl;
    std::cout << "  " << computedResult << std::endl;

    compiledMap.SetInputValue(0, input);
    auto compiledResult = compiledMap.ComputeOutput<ElementType>(0);
    std::cout << "Compiled result size: " << compiledResult.size() << std::endl;
    std::cout << "  " << compiledResult << std::endl;

    // compare output
    std::vector<std::vector<ElementType>> signal = { input };
    VerifyCompiledOutput(map, compiledMap, signal, predictorNode->GetRuntimeTypeName() + "_5");
}

void TestNeuralNetworkPredictorNode6()
{
    // Create a simple neural net model with the following layers:
    // input -> convolutional -> bias -> activation -> batch-norm -> scaling -> max-pooling -> mean-pooling
    using ElementType = double;
    using InputParameters = typename InputLayer<ElementType>::InputParameters;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;
    using VectorType = typename Layer<ElementType>::VectorType;

    // Build a net
    typename NeuralNetworkPredictor<ElementType>::InputLayerReference inputLayer;
    typename NeuralNetworkPredictor<ElementType>::Layers layers;

    // Input Layer
    InputParameters inputParams = { { 3, 3, 3 }, { PaddingScheme::zeros, 0 }, { 5, 5, 3 }, { PaddingScheme::zeros, 1 }, 1 };
    inputLayer = std::make_unique<InputLayer<ElementType>>(inputParams);

    // ConvolutionalLayer
    LayerParameters layerParameters{ inputLayer->GetOutput(), { PaddingScheme::zeros, 1 }, { 3, 3, 8 }, NoPadding() };
    auto convolutionMethod = ConvolutionMethod::unrolled;
    ConvolutionalParameters convolutionalParams{ 3, 1, convolutionMethod, 1 };
    TensorType convWeights1(8 * 3, 3, 3);
    FillTensor(convWeights1, -10.0);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ConvolutionalLayer<ElementType>(layerParameters, convolutionalParams, convWeights1)));

    // BiasLayer
    layerParameters = { layers[0]->GetOutput(), NoPadding(), { 3, 3, 8 }, NoPadding() };
    VectorType bias1(layerParameters.outputShape.NumChannels());
    FillVector(bias1);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new BiasLayer<ElementType>(layerParameters, bias1)));

    // ActivationLayer
    layerParameters = { layers[1]->GetOutput(), NoPadding(), { 3, 3, 8 }, NoPadding() };
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ActivationLayer<ElementType>(layerParameters, new ReLUActivation<ElementType>())));

    // BatchNormalizationLayer
    layerParameters = { layers[2]->GetOutput(), NoPadding(), { 3, 3, 8 }, NoPadding() };
    VectorType mean(layerParameters.outputShape.NumChannels());
    VectorType variance(layerParameters.outputShape.NumChannels());
    FillVector(mean);
    FillVector(variance);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new BatchNormalizationLayer<ElementType>(layerParameters, mean, variance, 1.0e-6f, EpsilonSummand::SqrtVariance)));

    // ScalingLayer
    layerParameters = { layers[3]->GetOutput(), NoPadding(), { 5, 5, 8 }, { PaddingScheme::zeros, 1 } };
    VectorType scales(layerParameters.outputShape.NumChannels());
    FillVector(scales, -3.0);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ScalingLayer<ElementType>(layerParameters, scales)));

    // Max PoolingLayer
    layerParameters = { layers[4]->GetOutput(), { PaddingScheme::zeros, 1 }, { 2, 2, 8 }, NoPadding() };
    PoolingParameters poolingParameters{ 2, 1 };
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new PoolingLayer<ElementType, MaxPoolingFunction>(layerParameters, poolingParameters)));

    // Mean PoolingLayer
    layerParameters = { layers[5]->GetOutput(), NoPadding(), { 1, 1, 8 }, NoPadding() };
    PoolingParameters poolingParameters2{ 2, 2 };
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new PoolingLayer<ElementType, MeanPoolingFunction>(layerParameters, poolingParameters2)));

    // Create the predictor
    NeuralNetworkPredictor<ElementType> neuralNetwork(std::move(inputLayer), std::move(layers));
    std::vector<ElementType> input(3 * 3 * 3);
    FillVector(input);

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(GetShapeSize(neuralNetwork.GetInputShape()));
    auto predictorNode = model.AddNode<NeuralNetworkPredictorNode<double>>(inputNode->output, neuralNetwork);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", predictorNode->output } });

    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    // compare output
    std::vector<std::vector<double>> signal = { input };
    VerifyCompiledOutput(map, compiledMap, signal, predictorNode->GetRuntimeTypeName() + "_6");
}

// tinyYolo prefix test
void TestNeuralNetworkPredictorNode7()
{
    using ElementType = double;
    using InputParameters = typename InputLayer<ElementType>::InputParameters;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;
    using Shape = typename Layer<ElementType>::Shape;
    using VectorType = typename Layer<ElementType>::VectorType;

    // Build a net (a prefix of darknet's tinyYolo)

    typename NeuralNetworkPredictor<ElementType>::InputLayerReference inputLayer;
    typename NeuralNetworkPredictor<ElementType>::Layers layers;

    auto convolutionMethod = ConvolutionMethod::unrolled;
    const Shape inputSize = { 224, 224, 3 };
    const Shape paddedInputSize = { 226, 226, 3 };

    // Input layer
    InputParameters inputParams{ inputSize, NoPadding(), paddedInputSize, ZeroPadding(1), 1.0 };
    inputLayer = std::make_unique<InputLayer<ElementType>>(inputParams);

    // layer_0 = ConvolutionalLayer<float>(shape=[224,224,16])
    // ConvolutionalLayer
    LayerParameters layerParameters{ inputLayer->GetOutput(), ZeroPadding(1), { 224, 224, 16 }, NoPadding() };
    ConvolutionalParameters convolutionalParams{ 3, 1, convolutionMethod, 1 };
    TensorType convWeights(16 * 3, 3, 3);
    FillRandomTensor(convWeights);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ConvolutionalLayer<ElementType>(layerParameters, convolutionalParams, convWeights)));

    // layer_1 = BatchNormalizationLayer<float>(shape=[224,224,16])
    // BatchNormalizationLayer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), { 224, 224, 16 }, NoPadding() };
    VectorType mean(layerParameters.outputShape.NumChannels());
    VectorType variance(layerParameters.outputShape.NumChannels());
    FillRandomVector(mean);
    FillRandomVector(variance, 0.125, 1.0);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new BatchNormalizationLayer<ElementType>(layerParameters, mean, variance, 1.0e-6f, EpsilonSummand::SqrtVariance)));

    // layer_2 = ScalingLayer<float>(shape=[224,224,16])
    // ScalingLayer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), { 224, 224, 16 }, NoPadding() };
    VectorType scales(layerParameters.outputShape.NumChannels());
    FillRandomVector(scales);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ScalingLayer<ElementType>(layerParameters, scales)));

    // layer_3 = BiasLayer<float>(shape=[224,224,16])
    // BiasLayer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), { 224, 224, 16 }, NoPadding() };
    VectorType bias(layerParameters.outputShape.NumChannels());
    FillRandomVector(bias);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new BiasLayer<ElementType>(layerParameters, scales)));

    // layer_4 = ActivationLayer<float,LeakyReLUActivation>(shape=[224,224,16])
    // ActivationLayer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), { 224, 224, 16 }, NoPadding() };
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ActivationLayer<ElementType>(layerParameters, new LeakyReLUActivation<ElementType>(0))));

    // layer_5 = PoolingLayer<float,MaxPoolingFunction>(shape=[114,114,16])
    // Max PoolingLayer
    layerParameters = { layers.back()->GetOutput(), ZeroPadding(1), { 114, 114, 16 }, ZeroPadding(1) };
    PoolingParameters poolingParameters{ 2, 2 };
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new PoolingLayer<ElementType, MaxPoolingFunction>(layerParameters, poolingParameters)));

    // layer_6 = ConvolutionalLayer<float>(shape=[112,112,32])
    // ConvolutionalLayer
    layerParameters = { layers.back()->GetOutput(), ZeroPadding(1), { 112, 112, 32 }, NoPadding() };
    convolutionalParams = { 3, 1, convolutionMethod, 1 };
    convWeights = { 32 * 3, 3, 16 };
    FillRandomTensor(convWeights);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ConvolutionalLayer<ElementType>(layerParameters, convolutionalParams, convWeights)));

    // layer_7 = BatchNormalizationLayer<float>(shape=[112,112,32])
    // BatchNormalizationLayer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), { 112, 112, 32 }, NoPadding() };
    mean = VectorType(layerParameters.outputShape.NumChannels());
    variance = VectorType(layerParameters.outputShape.NumChannels());
    FillRandomVector(mean);
    FillRandomVector(variance, 0.125, 1.0);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new BatchNormalizationLayer<ElementType>(layerParameters, mean, variance, 1.0e-4f, EpsilonSummand::SqrtVariance)));

    // layer_8 = ScalingLayer<float>(shape=[112,112,32])
    // ScalingLayer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), { 112, 112, 32 }, NoPadding() };
    scales = VectorType(layerParameters.outputShape.NumChannels());
    FillRandomVector(scales);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ScalingLayer<ElementType>(layerParameters, scales)));

    // layer_9 = BiasLayer<float>(shape=[112,112,32])
    // BiasLayer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), { 112, 112, 32 }, NoPadding() };
    bias = VectorType(layerParameters.outputShape.NumChannels());
    FillRandomVector(bias);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new BiasLayer<ElementType>(layerParameters, scales)));

    // layer_10 = ActivationLayer<float,LeakyReLUActivation>(shape=[112,112,32])
    // ActivationLayer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), { 112, 112, 32 }, NoPadding() };
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ActivationLayer<ElementType>(layerParameters, new LeakyReLUActivation<ElementType>(0))));

    // layer_11 = PoolingLayer<float,MaxPoolingFunction>(shape=[56,56,32])
    // Max PoolingLayer
    layerParameters = { layers.back()->GetOutput(), ZeroPadding(1), { 58, 58, 32 }, ZeroPadding(1) };
    poolingParameters = { 2, 2 };
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new PoolingLayer<ElementType, MaxPoolingFunction>(layerParameters, poolingParameters)));

    // layer_12 = ConvolutionalLayer<float>(shape=[56,56,16])
    // ConvolutionalLayer
    layerParameters = { layers.back()->GetOutput(), ZeroPadding(1), { 56, 56, 16 }, NoPadding() };
    convolutionalParams = { 3, 1, convolutionMethod, 1 };
    convWeights = { 16 * 3, 3, 32 };
    FillRandomTensor(convWeights);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ConvolutionalLayer<ElementType>(layerParameters, convolutionalParams, convWeights)));

    // layer_13 = BatchNormalizationLayer<float>(shape=[56,56,16])
    // BatchNormalizationLayer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), { 56, 56, 16 }, NoPadding() };
    mean = VectorType(layerParameters.outputShape.NumChannels());
    variance = VectorType(layerParameters.outputShape.NumChannels());
    FillRandomVector(mean);
    FillRandomVector(variance, 0.125, 1.0);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new BatchNormalizationLayer<ElementType>(layerParameters, mean, variance, 1.0e-6f, EpsilonSummand::SqrtVariance)));

    // layer_14 = ScalingLayer<float>(shape=[56,56,16])
    // ScalingLayer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), { 56, 56, 16 }, NoPadding() };
    scales = VectorType(layerParameters.outputShape.NumChannels());
    FillRandomVector(scales);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ScalingLayer<ElementType>(layerParameters, scales)));

    // layer_15 = BiasLayer<float>(shape=[56,56,16])
    // BiasLayer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), { 56, 56, 16 }, NoPadding() };
    bias = VectorType(layerParameters.outputShape.NumChannels());
    FillRandomVector(bias);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new BiasLayer<ElementType>(layerParameters, scales)));

    // layer_16 = ActivationLayer<float,LeakyReLUActivation>(shape=[58,58,16])
    // ActivationLayer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), { 58, 58, 16 }, ZeroPadding(1) };
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ActivationLayer<ElementType>(layerParameters, new LeakyReLUActivation<ElementType>(0))));

    // layer_17 = ConvolutionalLayer<float>(shape=[56,56,128])
    layerParameters = { layers.back()->GetOutput(), ZeroPadding(1), { 56, 56, 128 }, NoPadding() };
    convolutionalParams = { 3, 1, convolutionMethod, 1 };
    convWeights = { 128 * 3, 3, 16 };
    FillRandomTensor(convWeights);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ConvolutionalLayer<ElementType>(layerParameters, convolutionalParams, convWeights)));

    // Create the predictor
    NeuralNetworkPredictor<ElementType> neuralNetwork(std::move(inputLayer), std::move(layers));
    std::vector<ElementType> input(inputSize.Size());
    FillRandomVector(input);

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(GetShapeSize(neuralNetwork.GetInputShape()));
    auto predictorNode = model.AddNode<NeuralNetworkPredictorNode<double>>(inputNode->output, neuralNetwork);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", predictorNode->output } });

    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    // compare output
    double epsilon = 0.0001;
    std::vector<std::vector<double>> signal = { input };
    VerifyCompiledOutput(map, compiledMap, signal, predictorNode->GetRuntimeTypeName(), "_7", epsilon);
}

void TestInputLayerNode(size_t outputPadding)
{
    using ElementType = double;
    using InputParameters = typename predictors::neural::InputLayer<ElementType>::InputParameters;
    using LayerParameters = typename predictors::neural::Layer<ElementType>::LayerParameters;
    using DataVectorType = typename predictors::NeuralNetworkPredictor<ElementType>::DataVectorType;

    // Build a net
    typename predictors::NeuralNetworkPredictor<ElementType>::InputLayerReference inputLayer;
    typename predictors::NeuralNetworkPredictor<ElementType>::Layers layers;

    // Input layer
    InputParameters inputParams{ { 1, 1, 3 }, predictors::neural::NoPadding(), { 2 * outputPadding + 1, 2 * outputPadding + 1, 3 }, predictors::neural::ZeroPadding(outputPadding), 1.0 };
    inputLayer = std::make_unique<predictors::neural::InputLayer<ElementType>>(inputParams);

    // Pooling layer
    const size_t poolingSize = 3;
    const size_t poolingStride = 1;

    LayerParameters layerParameters{ inputLayer->GetOutput(), predictors::neural::ZeroPadding(outputPadding), { 1, 1, 3 }, predictors::neural::NoPadding() };
    predictors::neural::PoolingParameters poolingParameters{ poolingSize, poolingStride };
    layers.push_back(std::unique_ptr<predictors::neural::Layer<ElementType>>(new predictors::neural::PoolingLayer<ElementType, predictors::neural::MaxPoolingFunction>(layerParameters, poolingParameters)));
    predictors::NeuralNetworkPredictor<ElementType> neuralNetwork(std::move(inputLayer), std::move(layers));

    std::vector<ElementType> input = { 0, 1, 2 };
    auto output = neuralNetwork.Predict(DataVectorType(input));

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(GetShapeSize(neuralNetwork.GetInputShape()));
    auto predictorNode = model.AddNode<NeuralNetworkPredictorNode<ElementType>>(inputNode->output, neuralNetwork);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", predictorNode->output } });

    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    // compare output
    std::vector<std::vector<ElementType>> signal = { input };
    VerifyCompiledOutput(map, compiledMap, signal, "InputLayer");
}

template <typename ElementType>
void TestActivationLayerNode(size_t inputPaddingSize, size_t outputPaddingSize, Activation<ElementType> activation)
{
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;
    using TensorReferenceType = typename Layer<ElementType>::TensorReferenceType;
    using Shape = typename Layer<ElementType>::Shape;

    // Build a model
    TensorType inputWithPadding(2 + 2 * inputPaddingSize, 2 + 2 * inputPaddingSize, 2);
    TensorReferenceType input = inputWithPadding.GetSubTensor(inputPaddingSize, inputPaddingSize, 0, 2, 2, 2);
    input(0, 0, 0) = 1.0;
    input(0, 1, 0) = -2.0;
    input(1, 0, 1) = 3.0;
    input(1, 1, 1) = -4.0;
    Shape outputShape = { 2 + 2 * outputPaddingSize, 2 + 2 * outputPaddingSize, 2 };
    LayerParameters layerParameters{ inputWithPadding, ZeroPadding(inputPaddingSize), outputShape, ZeroPadding(outputPaddingSize) };

    ActivationLayer<ElementType> layer(layerParameters, activation);
    layer.Compute();
    auto output = layer.GetOutput();

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputWithPadding.Size());
    auto computeNode = model.AddNode<ActivationLayerNode<ElementType>>(inputNode->output, layer);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });
    VerifyLayerMap<ElementType>(map, computeNode, inputWithPadding, output);

    // Test archiving / unarchiving produces same result
    VerifyArchiveAndUnarchivingMap<ElementType>(map, computeNode, inputWithPadding, output);
}

void TestHardSigmoidActivationLayerNode(size_t inputPaddingSize, size_t outputPaddingSize)
{
    TestActivationLayerNode<double>(inputPaddingSize, outputPaddingSize, new HardSigmoidActivation<double>());
}

void TestHardTanhActivationLayerNode(size_t inputPaddingSize, size_t outputPaddingSize)
{
    TestActivationLayerNode<double>(inputPaddingSize, outputPaddingSize, new HardTanhActivation<double>());
}

void TestReLUActivationLayerNode(size_t inputPaddingSize, size_t outputPaddingSize)
{
    TestActivationLayerNode<double>(inputPaddingSize, outputPaddingSize, new ReLUActivation<double>());
}

void TestLeakyReLUActivationLayerNode(size_t inputPaddingSize, size_t outputPaddingSize)
{
    TestActivationLayerNode<double>(inputPaddingSize, outputPaddingSize, new LeakyReLUActivation<double>());
}

void TestSigmoidActivationLayerNode(size_t inputPaddingSize, size_t outputPaddingSize)
{
    TestActivationLayerNode<double>(inputPaddingSize, outputPaddingSize, new SigmoidActivation<double>());
}

void TestParametricReLUActivationLayerNode(size_t inputPaddingSize, size_t outputPaddingSize)
{
    using ElementType = double;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;
    using TensorReferenceType = typename Layer<ElementType>::TensorReferenceType;
    using Shape = typename Layer<ElementType>::Shape;

    // Build a model
    TensorType inputWithPadding(2 + 2 * inputPaddingSize, 2 + 2 * inputPaddingSize, 2);
    TensorReferenceType input = inputWithPadding.GetSubTensor(inputPaddingSize, inputPaddingSize, 0, 2, 2, 2);
    input(0, 0, 0) = 1.0;
    input(0, 1, 0) = -2.0;
    input(1, 0, 1) = 3.0;
    input(1, 1, 1) = -4.0;
    Shape outputShape = { 2 + 2 * outputPaddingSize, 2 + 2 * outputPaddingSize, 2 };
    LayerParameters layerParameters{ inputWithPadding, ZeroPadding(inputPaddingSize), outputShape, ZeroPadding(outputPaddingSize) };

    TensorType alphaWithPadding(2 + 2 * inputPaddingSize, 2 + 2 * inputPaddingSize, 2);
    TensorReferenceType alpha = alphaWithPadding.GetSubTensor(inputPaddingSize, inputPaddingSize, 0, 2, 2, 2);
    alpha(0, 0, 0) = 0.1;
    alpha(0, 1, 0) = 0.2;
    alpha(1, 0, 1) = 0.3;
    alpha(1, 1, 1) = 0.4;

    ActivationLayer<ElementType> layer(layerParameters, new ParametricReLUActivation<ElementType>(alphaWithPadding));
    layer.Compute();
    auto output = layer.GetOutput();

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputWithPadding.Size());
    auto computeNode = model.AddNode<ParametricReLUActivationLayerNode<ElementType>>(inputNode->output, layer);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });
    VerifyLayerMap<ElementType>(map, computeNode, inputWithPadding, output);

    // Test archiving / unarchiving produces same result
    VerifyArchiveAndUnarchivingMap<ElementType>(map, computeNode, inputWithPadding, output);
}

void TestBatchNormalizationLayerNode(size_t inputPaddingSize, size_t outputPaddingSize)
{
    using ElementType = double;
    using LayerType = predictors::neural::BatchNormalizationLayer<double>;
    using LayerParameters = typename LayerType::LayerParameters;
    using TensorType = typename LayerType::TensorType;
    using TensorReferenceType = typename LayerType::TensorReferenceType;
    using Shape = typename LayerType::Shape;
    using VectorType = typename LayerType::VectorType;

    // Build a model
    TensorType inputWithPadding(2 + 2 * inputPaddingSize, 2 + 2 * inputPaddingSize, 2);
    TensorReferenceType input = inputWithPadding.GetSubTensor(inputPaddingSize, inputPaddingSize, 0, 2, 2, 2);
    input(0, 0, 0) = 11;
    input(0, 1, 0) = 7;
    input(1, 0, 1) = 30;
    input(1, 1, 1) = 50;

    auto inputPadding = inputPaddingSize == 0 ? predictors::neural::NoPadding() : predictors::neural::ZeroPadding(inputPaddingSize);
    auto outputPadding = outputPaddingSize == 0 ? predictors::neural::NoPadding() : predictors::neural::ZeroPadding(outputPaddingSize);
    Shape outputShape = { 2 + 2 * outputPaddingSize, 2 + 2 * outputPaddingSize, 2 };
    LayerParameters layerParameters{ inputWithPadding, inputPadding, outputShape, outputPadding };
    VectorType mean({ 5, 10 });
    VectorType variance({ 4.0, 16.0 });

    predictors::neural::BatchNormalizationLayer<double> layer(layerParameters, mean, variance, 1.0e-6f, predictors::neural::EpsilonSummand::SqrtVariance);
    layer.Compute();
    auto output = layer.GetOutput();

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(inputWithPadding.Size());
    auto computeNode = model.AddNode<BatchNormalizationLayerNode<double>>(inputNode->output, layer);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });

    VerifyLayerMap<ElementType>(map, computeNode, inputWithPadding, output);

    // Test archiving / unarchiving produces same result
    VerifyArchiveAndUnarchivingMap<ElementType>(map, computeNode, inputWithPadding, output);
}

void TestBiasLayerNode(size_t inputPaddingSize, size_t outputPaddingSize)
{
    using ElementType = double;
    using LayerType = predictors::neural::BiasLayer<double>;
    using LayerParameters = typename LayerType::LayerParameters;
    using TensorType = typename LayerType::TensorType;
    using TensorReferenceType = typename LayerType::TensorReferenceType;
    using Shape = typename LayerType::Shape;
    using VectorType = typename LayerType::VectorType;

    // Set up bias layer
    TensorType inputWithPadding(2 + 2 * inputPaddingSize, 2 + 2 * inputPaddingSize, 2);
    TensorReferenceType input = inputWithPadding.GetSubTensor(inputPaddingSize, inputPaddingSize, 0, 2, 2, 2);

    input(0, 0, 0) = 1;
    input(0, 1, 0) = 2;
    input(1, 0, 1) = 3;
    input(1, 1, 1) = 4;

    auto inputPadding = inputPaddingSize == 0 ? predictors::neural::NoPadding() : predictors::neural::ZeroPadding(inputPaddingSize);
    auto outputPadding = outputPaddingSize == 0 ? predictors::neural::NoPadding() : predictors::neural::ZeroPadding(outputPaddingSize);
    Shape outputShape = { 2 + 2 * outputPaddingSize, 2 + 2 * outputPaddingSize, 2 };
    LayerParameters parameters{ inputWithPadding, inputPadding, outputShape, outputPadding };
    VectorType bias({ 10, 100 });

    predictors::neural::BiasLayer<double> layer(parameters, bias);
    layer.Compute();
    auto output = layer.GetOutput();

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(inputWithPadding.Size());
    auto computeNode = model.AddNode<BiasLayerNode<double>>(inputNode->output, layer);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });

    VerifyLayerMap<ElementType>(map, computeNode, inputWithPadding, output);

    // Test archiving / unarchiving produces same result
    VerifyArchiveAndUnarchivingMap<ElementType>(map, computeNode, inputWithPadding, output);
}

void TestBinaryConvolutionalLayerNode(size_t imageRows, size_t imageColumns, size_t numChannels, size_t numFilters, size_t inputPaddingSize, size_t outputPaddingSize, PaddingScheme paddingScheme, bool scaleByFilterMeans)
{
    using ElementType = float;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;
    using TensorReferenceType = typename Layer<ElementType>::TensorReferenceType;
    using Shape = typename Layer<ElementType>::Shape;

    const size_t k = 3;
    const size_t stride = 1;

    // Verify BinaryConvolutionalLayer with bitwise method
    TensorType inputWithPadding(imageRows + 2 * inputPaddingSize, imageColumns + 2 * inputPaddingSize, numChannels);
    TensorReferenceType input = inputWithPadding.GetSubTensor(inputPaddingSize, inputPaddingSize, 0, imageRows, imageColumns, numChannels);
    input.Fill(0);
    int inputSize = input.Size();
    FillTensor(input, -2 * static_cast<ElementType>(inputSize) / 3);

    Shape outputShape = { imageRows + 2 * outputPaddingSize, imageColumns + 2 * outputPaddingSize, numFilters };

    LayerParameters parameters{ inputWithPadding, { paddingScheme, inputPaddingSize }, outputShape, { paddingScheme, outputPaddingSize } };
    BinaryConvolutionalParameters convolutionalParams{ k, stride, BinaryConvolutionMethod::bitwise, scaleByFilterMeans ? BinaryWeightsScale::mean : BinaryWeightsScale::none };
    TensorType weights(convolutionalParams.receptiveField * outputShape.NumChannels(), convolutionalParams.receptiveField, input.NumChannels());
    int weightsSize = weights.Size();
    FillTensor(weights, -static_cast<ElementType>(weightsSize) / 2);

    BinaryConvolutionalLayer<ElementType> layer(parameters, convolutionalParams, weights);
    layer.Compute();
    auto output = layer.GetOutput();
    UNUSED(output);

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputWithPadding.Size());
    auto computeNode = model.AddNode<BinaryConvolutionalLayerNode<ElementType>>(inputNode->output, layer);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });

    // Compile it
    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;
    settings.compilerSettings.useBlas = true; // !!! if BLAS is off, this fails
    settings.compilerSettings.allowVectorInstructions = false;
    settings.compilerSettings.vectorWidth = 2;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    auto signal = std::vector<std::vector<ElementType>>{ inputWithPadding.ToArray() };
    VerifyCompiledOutput<ElementType>(map, compiledMap, { signal }, computeNode->GetRuntimeTypeName());

    // Test archiving / unarchiving produces same result
    VerifyArchiveAndUnarchivingMap<ElementType>(map, computeNode, inputWithPadding, output);
}

void TestConvolutionalLayerNode(ConvolutionMethod convolutionMethod, size_t inputPaddingSize, size_t outputPaddingSize)
{
    // Abbreviations:
    //
    // r == # input rows
    // c == # input columns
    // ch == # input channels
    // fw == filter width
    // nf == # filters
    // pi == input padding amount
    // po == output padding amount

    // Data dimensions:
    //
    // Input: r x c x ch, with padding -> r+2pi x c+2pi x ch
    //     == 1 x 2 x 2, with padding == 1 -> 3 x 4 x 2
    // Weights: nf x fw x fw x ch
    //       == 2 x 3 x 3 x 2, (2 3x3 filters, with 2 input channels each)
    // Output: r x c x nf, with padding -> 1+2po x 2+2po x 2
    //      == 1 x 2 x 2, with padding == 0 -> 1 x 2 x 2

    using ElementType = double;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;
    using TensorReferenceType = typename Layer<ElementType>::TensorReferenceType;
    using Shape = typename Layer<ElementType>::Shape;

    assert(inputPaddingSize == 1);
    TensorType inputWithPadding(1 + 2 * inputPaddingSize, 2 + 2 * inputPaddingSize, 2);
    TensorReferenceType input = inputWithPadding.GetSubTensor({ inputPaddingSize, inputPaddingSize, 0 }, { 1, 2, 2 });
    inputWithPadding.Fill(0);
    input(0, 0, 0) = 2;
    input(0, 1, 0) = 1;
    input(0, 0, 1) = 3;
    input(0, 1, 1) = 2;
    // Input channel 0: [2, 3], input channel 1: [1, 2]

    Shape outputShape = { 1 + 2 * outputPaddingSize, 2 + 2 * outputPaddingSize, 2 };

    LayerParameters parameters{ inputWithPadding, ZeroPadding(inputPaddingSize), outputShape, ZeroPadding(outputPaddingSize) };
    ConvolutionalParameters convolutionalParams{ 3, 1, convolutionMethod, 2 }; // 2 == batch size

    // Filter weights in `weightsVector` are in numFilters x numChannels x filterSize x filterSize order
    // clang-format off
    std::vector<ElementType> weightsVector {
        1, 3, 2,   3, 1, 1,   2, 3, 1,   // Filter 1, channel 1
        2, 4, 1,   3, 1, 2,   1, 4, 2,   // Filter 1, channel 2

        1, 2, 1,   2, 3, 2,   1, 2, 1,   // Filter 2, channel 1
        0, 3, 2,   3, 1, 2,   1, 0, 2 }; // Filter 2, channel 2
    // clang-format on

    // Viewed as planar filters (ch x fw x fw):
    //
    //       1 3 2   2 4 1
    // f0 =  3 1 1   3 1 2
    //       2 3 1   1 4 2
    //
    //       1 2 1   0 3 2
    // f1 =  2 3 2   3 1 2
    //       1 2 1   1 0 2

    // Filter weights in `weights` tensor are in numFilters x filterSize x filterSize x numChannels order
    TensorType weights(convolutionalParams.receptiveField * outputShape.NumChannels(), convolutionalParams.receptiveField, input.NumChannels());

    size_t vectorIndex = 0;
    for (size_t f = 0; f < outputShape.NumChannels(); ++f)
    {
        for (size_t k = 0; k < input.NumChannels(); ++k)
        {
            for (size_t i = 0; i < convolutionalParams.receptiveField; ++i)
            {
                for (size_t j = 0; j < convolutionalParams.receptiveField; ++j)
                {
                    weights(f * convolutionalParams.receptiveField + i, j, k) = weightsVector[vectorIndex++];
                }
            }
        }
    }

    //
    // Verify ConvolutionalLayerNode
    //
    ConvolutionalLayer<ElementType> layer(parameters, convolutionalParams, weights);
    layer.Compute();
    auto output = layer.GetOutput();
    ElementType eps = 1e-6;
    ElementType v1 = 10;
    ElementType v2 = 15;
    ElementType v3 = 18;
    ElementType v4 = 18;
    testing::ProcessTest("Testing ConvolutionalLayer, values",
                         testing::IsEqual(output(0, 0, 0), v1, eps) &&
                             testing::IsEqual(output(0, 0, 1), v2, eps) &&
                             testing::IsEqual(output(0, 1, 0), v3, eps) &&
                             testing::IsEqual(output(0, 1, 1), v4, eps));

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(inputWithPadding.Size());
    auto computeNode = model.AddNode<ConvolutionalLayerNode<double>>(inputNode->output, layer);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });

    const auto info = "(TestConvolutionalLayerNode1, method = " + std::to_string(int(convolutionMethod)) + ")";

    VerifyLayerMap<ElementType>(map, computeNode, inputWithPadding, output, info);

    // Test archiving / unarchiving produces same result
    VerifyArchiveAndUnarchivingMap<ElementType>(map, computeNode, inputWithPadding, output, info);
}

void TestConvolutionalLayerNode2(ConvolutionMethod convolutionMethod, size_t inputPaddingSize, size_t outputPaddingSize)
{
    using ElementType = double;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;
    using TensorReferenceType = typename Layer<ElementType>::TensorReferenceType;
    using Shape = typename Layer<ElementType>::Shape;

    const size_t numRows = 56;
    const size_t numCols = 56;
    const size_t numChannels = 16;
    const size_t numFilters = 128;

    auto rng = utilities::GetRandomEngine("123");
    auto rand = [&rng]() { return (double)rng() / (double)(rng.max() - rng.min()); };

    assert(inputPaddingSize == 1);
    TensorType inputWithPadding(numRows + 2 * inputPaddingSize, numCols + 2 * inputPaddingSize, numChannels);
    inputWithPadding.Fill(0);
    TensorReferenceType input = inputWithPadding.GetSubTensor(inputPaddingSize, inputPaddingSize, 0, numRows, numCols, numChannels);
    for (size_t rowIndex = 0; rowIndex < numRows; ++rowIndex)
    {
        for (size_t colIndex = 0; colIndex < numCols; ++colIndex)
        {
            for (size_t channelIndex = 0; channelIndex < numChannels; ++channelIndex)
            {
                // input(rowIndex, colIndex, channelIndex) = 1.25 * rowIndex + 0.75 * colIndex + channelIndex;
                input(rowIndex, colIndex, channelIndex) = rand() - 0.5;
            }
        }
    }
    Shape outputShape = { numRows + 2 * outputPaddingSize, numCols + 2 * outputPaddingSize, numFilters };

    LayerParameters parameters{ inputWithPadding, ZeroPadding(inputPaddingSize), outputShape, ZeroPadding(outputPaddingSize) };
    auto actualConvolutionMethod = convolutionMethod == ConvolutionMethod::diagonal ? convolutionMethod : ConvolutionMethod::unrolled;
    ConvolutionalParameters convolutionalParams{ 3, 1, actualConvolutionMethod, 2 }; // 2 == batch size
    TensorType weights(convolutionalParams.receptiveField * numFilters, convolutionalParams.receptiveField, input.NumChannels());
    weights.Fill(1.0);
    for (size_t rowIndex = 0; rowIndex < convolutionalParams.receptiveField * numFilters; ++rowIndex)
    {
        for (size_t colIndex = 0; colIndex < convolutionalParams.receptiveField; ++colIndex)
        {
            for (size_t channelIndex = 0; channelIndex < numChannels; ++channelIndex)
            {
                // weights(rowIndex, colIndex, channelIndex) = 1.5 * rowIndex + 3.3 * colIndex + 0.15 * channelIndex;
                weights(rowIndex, colIndex, channelIndex) = rand() - 0.5;
            }
        }
    }

    //
    // Verify ConvolutionalLayerNode
    //
    ConvolutionalLayer<ElementType> layer(parameters, convolutionalParams, weights);
    layer.Compute();
    auto output = layer.GetOutput();

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(inputWithPadding.Size());
    auto computeNode = model.AddNode<ConvolutionalLayerNode<double>>(inputNode->output, layer);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });

    const auto info = "(TestConvolutionalLayerNode2, method = " + std::to_string(int(convolutionMethod)) + ")";

    VerifyLayerMap<ElementType>(map, computeNode, inputWithPadding, output, info);

    // Test archiving / unarchiving produces same result
    VerifyArchiveAndUnarchivingMap<ElementType>(map, computeNode, inputWithPadding, output, info);
}

// Test separable convolutions
void TestConvolutionalLayerNode3(ConvolutionMethod convolutionMethod, size_t inputPaddingSize, size_t outputPaddingSize)
{
    using ElementType = double;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;
    using TensorReferenceType = typename Layer<ElementType>::TensorReferenceType;
    using Shape = typename Layer<ElementType>::Shape;

    const size_t numRows = 2;
    const size_t numCols = 2;
    const size_t numChannels = 2;
    const size_t numFilters = 2;

    auto rng = utilities::GetRandomEngine("123");
    auto rand = [&rng]() { return (double)rng() / (double)(rng.max() - rng.min()); };

    assert(inputPaddingSize == 1);
    TensorType inputWithPadding(numRows + 2 * inputPaddingSize, numCols + 2 * inputPaddingSize, numChannels);
    inputWithPadding.Fill(0);
    TensorReferenceType input = inputWithPadding.GetSubTensor(inputPaddingSize, inputPaddingSize, 0, numRows, numCols, numChannels);
    for (size_t rowIndex = 0; rowIndex < numRows; ++rowIndex)
    {
        for (size_t colIndex = 0; colIndex < numCols; ++colIndex)
        {
            for (size_t channelIndex = 0; channelIndex < numChannels; ++channelIndex)
            {
                input(rowIndex, colIndex, channelIndex) = rand() - 0.5;
            }
        }
    }
    Shape outputShape = { numRows + 2 * outputPaddingSize, numCols + 2 * outputPaddingSize, numFilters };

    LayerParameters parameters{ inputWithPadding, ZeroPadding(inputPaddingSize), outputShape, ZeroPadding(outputPaddingSize) };
    auto actualConvolutionMethod = convolutionMethod;
    ConvolutionalParameters convolutionalParams{ 3, 1, actualConvolutionMethod, 2 }; // 2 == batch size
    TensorType weights(convolutionalParams.receptiveField * numFilters, convolutionalParams.receptiveField, numChannels);
    weights.Fill(1.0);
    for (size_t rowIndex = 0; rowIndex < convolutionalParams.receptiveField * numFilters; ++rowIndex)
    {
        for (size_t colIndex = 0; colIndex < convolutionalParams.receptiveField; ++colIndex)
        {
            for (size_t channelIndex = 0; channelIndex < 1; ++channelIndex)
            {
                weights(rowIndex, colIndex, channelIndex) = rand() - 0.5;
            }
        }
    }

    //
    // Verify ConvolutionalLayerNode
    //
    ConvolutionalLayer<ElementType> layer(parameters, convolutionalParams, weights);
    layer.Compute();
    auto output = layer.GetOutput();

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(inputWithPadding.Size());
    auto computeNode = model.AddNode<ConvolutionalLayerNode<double>>(inputNode->output, layer);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });

    const auto info = "(TestConvolutionalLayerNode3 - depthwise separable, method = " + std::to_string(int(convolutionMethod)) + ")";

    VerifyLayerMap<ElementType>(map, computeNode, inputWithPadding, output, info);

    // Test archiving / unarchiving produces same result
    VerifyArchiveAndUnarchivingMap<ElementType>(map, computeNode, inputWithPadding, output, info);
}

void TestFullyConnectedLayerNode(size_t inputPaddingSize, size_t outputPaddingSize)
{
    using ElementType = double;
    using LayerType = predictors::neural::FullyConnectedLayer<double>;
    using LayerParameters = typename LayerType::LayerParameters;
    using TensorType = typename LayerType::TensorType;
    using TensorReferenceType = typename LayerType::TensorReferenceType;
    using MatrixType = typename LayerType::MatrixType;
    using Shape = typename LayerType::Shape;

    // Set up layer
    TensorType inputWithPadding(2 + 2 * inputPaddingSize, 2 + 2 * inputPaddingSize, 2);
    TensorReferenceType input = inputWithPadding.GetSubTensor(inputPaddingSize, inputPaddingSize, 0, 2, 2, 2);
    input(0, 0, 0) = 1;
    input(0, 1, 0) = 2;
    input(1, 0, 1) = 3;
    input(1, 1, 1) = 4;

    auto inputPadding = inputPaddingSize == 0 ? predictors::neural::NoPadding() : predictors::neural::ZeroPadding(inputPaddingSize);
    auto outputPadding = outputPaddingSize == 0 ? predictors::neural::NoPadding() : predictors::neural::ZeroPadding(outputPaddingSize);
    Shape outputShape = { 4 + 2 * outputPaddingSize, 1 + 2 * outputPaddingSize, 1 };
    LayerParameters parameters{ input, inputPadding, outputShape, outputPadding };
    MatrixType weights(4, 8);
    for (int index = 0; index < 8; index++)
        weights(1, index) = static_cast<double>(index);
    for (int index = 0; index < 8; index++)
        weights(2, index) = static_cast<double>(7 - index);
    for (int index = 0; index < 8; index++)
        weights(3, index) = 1.0;

    LayerType layer(parameters, weights);
    layer.Compute();
    auto output = layer.GetOutput();

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(inputWithPadding.Size());
    auto computeNode = model.AddNode<FullyConnectedLayerNode<double>>(inputNode->output, layer);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });

    VerifyLayerMap<ElementType>(map, computeNode, inputWithPadding, output);

    // Test archiving / unarchiving produces same result
    VerifyArchiveAndUnarchivingMap<ElementType>(map, computeNode, inputWithPadding, output);
}

template <typename ElementType, template <typename> class PoolingFunction>
void TestPoolingLayerNode(size_t inRows, size_t inCols, size_t numChannels, size_t outRows, size_t outCols, size_t poolingSize, size_t poolingStride, size_t inputPaddingSize, size_t outputPaddingSize, double epsilon)
{
    std::string tname = typeid(ElementType).name();
    Log() << "TestPoolingLayerNode< " << tname << ">([" << inRows << "," << inCols << "],[" << outRows << "," << outCols << "], pool=" << poolingSize << ", stride=" << poolingStride << ", inpad=" << inputPaddingSize << ", outpad=" << outputPaddingSize << ", e=" << epsilon << ")" << EOL;

    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;
    using TensorReferenceType = typename Layer<ElementType>::TensorReferenceType;
    using Shape = typename Layer<ElementType>::Shape;

    // Build a model
    TensorType inputWithPadding(inRows + 2 * inputPaddingSize, inCols + 2 * inputPaddingSize, numChannels);
    TensorReferenceType input = inputWithPadding.GetSubTensor(inputPaddingSize, inputPaddingSize, 0, inRows, inCols, numChannels);
    FillTensor(input);

    Shape outputShape = { outRows + 2 * outputPaddingSize, outCols + 2 * outputPaddingSize, numChannels };
    LayerParameters layerParameters{ inputWithPadding, ZeroPadding(inputPaddingSize), outputShape, ZeroPadding(outputPaddingSize) };
    PoolingParameters poolingParameters{ poolingSize, poolingStride };
    PoolingLayer<ElementType, PoolingFunction> layer(layerParameters, poolingParameters);
    layer.Compute();
    auto output = layer.GetOutput();

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputWithPadding.Size());
    auto computeNode = model.AddNode<PoolingLayerNode<ElementType, PoolingFunction>>(inputNode->output, layer);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });

    VerifyLayerMap<ElementType>(map, computeNode, inputWithPadding, output);

    // Test archiving / unarchiving produces same result
    utilities::SerializationContext context;
    common::RegisterNodeTypes(context);
    std::stringstream strstream;
    utilities::JsonArchiver archiver(strstream);

    archiver << map;
    utilities::JsonUnarchiver unarchiver(strstream, context);
    model::Map unarchivedMap;
    unarchiver >> unarchivedMap;

    std::vector<std::vector<ElementType>> signal = { inputWithPadding.ToArray() };
    std::vector<std::vector<ElementType>> expectedOutput = { output.ToArray() };
    VerifyMapOutput(unarchivedMap, signal, expectedOutput, "Unarchived model with MaxPoolingLayerNode");
}

void TestMaxPoolingLayerNode(size_t inRows, size_t inCols, size_t numChannels, size_t outRows, size_t outCols, size_t poolingSize, size_t poolingStride, size_t inputPaddingSize, size_t outputPaddingSize)
{
    TestPoolingLayerNode<double, predictors::neural::MaxPoolingFunction>(inRows, inCols, numChannels, outRows, outCols, poolingSize, poolingStride, inputPaddingSize, outputPaddingSize, 1e-10);
    TestPoolingLayerNode<float, predictors::neural::MaxPoolingFunction>(inRows, inCols, numChannels, outRows, outCols, poolingSize, poolingStride, inputPaddingSize, outputPaddingSize, 1e-5);
}

void TestMeanPoolingLayerNode(size_t inRows, size_t inCols, size_t numChannels, size_t outRows, size_t outCols, size_t poolingSize, size_t poolingStride, size_t inputPaddingSize, size_t outputPaddingSize)
{
    TestPoolingLayerNode<double, MeanPoolingFunction>(inRows, inCols, numChannels, outRows, outCols, poolingSize, poolingStride, inputPaddingSize, outputPaddingSize, 1e-10);
    TestPoolingLayerNode<float, MeanPoolingFunction>(inRows, inCols, numChannels, outRows, outCols, poolingSize, poolingStride, inputPaddingSize, outputPaddingSize, 1e-5);
}

void TestScalingLayerNode(size_t inputPaddingSize, size_t outputPaddingSize)
{
    using ElementType = double;
    using LayerType = predictors::neural::ScalingLayer<ElementType>;
    using LayerParameters = typename LayerType::LayerParameters;
    using TensorType = typename LayerType::TensorType;
    using TensorReferenceType = typename LayerType::TensorReferenceType;
    using Shape = typename LayerType::Shape;
    using VectorType = typename LayerType::VectorType;

    // Set up scaling layer
    TensorType inputWithPadding(2 + 2 * inputPaddingSize, 2 + 2 * inputPaddingSize, 2);
    TensorReferenceType input = inputWithPadding.GetSubTensor(inputPaddingSize, inputPaddingSize, 0, 2, 2, 2);
    input(0, 0, 0) = 1;
    input(0, 1, 0) = 2;
    input(1, 0, 1) = 3;
    input(1, 1, 1) = 4;
    Shape outputShape = { 2 + 2 * outputPaddingSize, 2 + 2 * outputPaddingSize, 2 };
    LayerParameters layerParameters{ inputWithPadding, ZeroPadding(inputPaddingSize), outputShape, ZeroPadding(outputPaddingSize) };
    VectorType scale({ 5, 10 });

    LayerType layer(layerParameters, scale);
    layer.Compute();
    auto output = layer.GetOutput();

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputWithPadding.Size());
    auto computeNode = model.AddNode<ScalingLayerNode<ElementType>>(inputNode->output, layer);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });

    VerifyLayerMap<ElementType>(map, computeNode, inputWithPadding, output);

    // Test archiving / unarchiving produces same result
    VerifyArchiveAndUnarchivingMap<ElementType>(map, computeNode, inputWithPadding, output);
}

void TestSoftmaxLayerNode(size_t inputPaddingSize, size_t outputPaddingSize)
{
    using ElementType = double;
    using LayerType = predictors::neural::SoftmaxLayer<ElementType>;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;
    using TensorReferenceType = typename Layer<ElementType>::TensorReferenceType;
    using Shape = typename Layer<ElementType>::Shape;

    // Build a model
    TensorType inputWithPadding(2 + 2 * inputPaddingSize, 2 + 2 * inputPaddingSize, 2);
    TensorReferenceType input = inputWithPadding.GetSubTensor(inputPaddingSize, inputPaddingSize, 0, 2, 2, 2);
    input(0, 0, 0) = 1.0;
    input(0, 1, 0) = -2.0;
    input(1, 0, 1) = 3.0;
    input(1, 1, 1) = -4.0;
    Shape outputShape = { 2 + 2 * outputPaddingSize, 2 + 2 * outputPaddingSize, 2 };
    LayerParameters layerParameters{ inputWithPadding, ZeroPadding(inputPaddingSize), outputShape, ZeroPadding(outputPaddingSize) };
    LayerType layer(layerParameters);
    layer.Compute();
    auto output = layer.GetOutput();

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputWithPadding.Size());
    auto computeNode = model.AddNode<SoftmaxLayerNode<ElementType>>(inputNode->output, layer);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });

    VerifyLayerMap<ElementType>(map, computeNode, inputWithPadding, output);

    // Test archiving / unarchiving produces same result
    VerifyArchiveAndUnarchivingMap<ElementType>(map, computeNode, inputWithPadding, output);
}

void TestFusedLinearLayerNodes(size_t rows, size_t columns, size_t channels)
{
    // Create a simple neural net model with the following layers:
    // input -> scaling -> bias
    using ElementType = double;
    using InputParameters = typename InputLayer<ElementType>::InputParameters;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using VectorType = typename Layer<ElementType>::VectorType;

    // Build a net
    typename NeuralNetworkPredictor<ElementType>::InputLayerReference inputLayer;
    typename NeuralNetworkPredictor<ElementType>::Layers layers;
    Layer<ElementType>::Shape dataShape = { rows, columns, channels };
    auto dataSize = rows * columns * channels;
    UNUSED(dataSize);

    // Input layer
    InputParameters inputParams = { dataShape, NoPadding(), dataShape, NoPadding(), 1 };
    inputLayer = std::make_unique<InputLayer<ElementType>>(inputParams);

    LayerParameters layerParameters{ inputLayer->GetOutput(), NoPadding(), dataShape, NoPadding() };

    // Set up initial bias layer
    layerParameters = { inputLayer->GetOutput(), NoPadding(), dataShape, NoPadding() };
    VectorType bias1(channels);
    FillRandomVector(bias1);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new BiasLayer<ElementType>(layerParameters, bias1)));

    // Set up scaling layer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), dataShape, NoPadding() };
    VectorType scale1(channels);
    FillRandomVector(scale1);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new ScalingLayer<ElementType>(layerParameters, scale1)));

    // Set up bias layer
    layerParameters = { layers.back()->GetOutput(), NoPadding(), dataShape, NoPadding() };
    VectorType bias2(channels);
    FillRandomVector(bias2);
    layers.push_back(std::unique_ptr<Layer<ElementType>>(new BiasLayer<ElementType>(layerParameters, bias2)));

    NeuralNetworkPredictor<ElementType> neuralNetwork(std::move(inputLayer), std::move(layers));
    std::vector<ElementType> input(rows * columns * channels);
    FillRandomVector(input);

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(GetShapeSize(neuralNetwork.GetInputShape()));
    auto predictorNode = model.AddNode<NeuralNetworkPredictorNode<double>>(inputNode->output, neuralNetwork);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", predictorNode->output } });

    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;

    model::ModelOptimizerOptions optimizerOptions;
    optimizerOptions.SetEntry("fuseLinearFunctionNodes", true);

    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    // compare output
    std::vector<std::vector<double>> signal = { input };
    VerifyCompiledOutput(map, compiledMap, signal, "Fused linear layers");
}

template <typename TensorType>
static auto PostProcessRegionLayerOutput(TensorType tensor, RegionDetectionParameters detectionParams)
{
    SigmoidActivation<typename TensorType::TensorElementType> sigmoid;
    SoftMaxActivation<typename TensorType::TensorElementType> softmax;

    for (int i = 0; i < detectionParams.width; ++i)
    {
        for (int j = 0; j < detectionParams.height; ++j)
        {
            auto slice = tensor.template GetSlice<math::Dimension::channel>(i, j);

            for (int c = 0; c < detectionParams.numBoxesPerCell; ++c)
            {
                int offset = c * (detectionParams.numAnchors + 1 + detectionParams.numClasses);
                slice[offset + 0] = sigmoid(slice[offset + 0]);
                slice[offset + 1] = sigmoid(slice[offset + 1]);
                slice[offset + 2] = std::exp(slice[offset + 2]);
                slice[offset + 3] = std::exp(slice[offset + 3]);

                // If softmax wasn't applied by the layer, do so in post-processing
                if (!detectionParams.applySoftmax)
                {
                    auto classProbabilities = slice.GetSubVector(offset + 5, detectionParams.numClasses);
                    softmax(classProbabilities);
                }
            }
        }
    }

    return tensor;
}

void TestRegionDetectionNode()
{
    using ElementType = double;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;
    using Shape = typename Layer<ElementType>::Shape;

    // Input created by running
    /*
    git clone https://github.com/pjreddie/darknet
    git checkout 80d9bec20f0a44ab07616215c6eadb2d633492fe
    wget https://pjreddie.com/media/files/yolov2-tiny-voc.weights
    make DEBUG=1
    gdb --args ./darknet detector test cfg/voc.data cfg/tiny-yolo-voc.cfg yolov2-tiny-voc.weights data/dog.jpg
    */
    // setting a breakpoint in src/region_layer.c:162, and dumping the contents of l.output to a file,
    // using gdb:
    /*
    dump binary memory data.bin l.output (l.output + l.outputs * l.batch)
    */
    // loading the contents of said file in numpy, followed by reordering and reshaping into a 13x13x125 tensor
    /*
    import numpy as np
    import itertools
    raw_data = np.fromfile('./data.bin', dtype=np.float32)
    data = np.zeros((13, 13, 125), dtype=np.float32)
    for i, j in itertools.product(range(13), range(13)):
        data[i, j, ...] = raw_data[(i * 13 + j) * 125 : (i * 13 + j + 1) * 125]
    data = data.reshape(13, 13, 125)
    */
    // Creating the brace-formatted output for C++ requires the following snippet:
    /*
    s = '{'
    for i in range(13):
        s += '\n{'
        for j in range(13):
            s += '\n{' + ', '.join(np.char.mod('%f', data[i, j, ...])) + '},'
        s += '\n},'
    s += '\n}'
    with open('data.inc', 'w') as f: f.write(s)
    */
    // clang-format off
    TensorType input =
    {
        #include "TestRegionDetectionNode_input.inc"
    };
    // clang-format on
    testing::ProcessTest("Verifying input dimensions", testing::IsEqual(input.GetShape(), math::TensorShape{ 13, 13, 125 }));

    // Expected output created by running the following operation for every 1D slice in aforementioned
    // input in the channel dimension in numpy:
    /*
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    for c in range(5):
      boxOffset = c * 25
      expected[boxOffset + 0] = sigmoid(input[boxOffset + 0])
      expected[boxOffset + 1] = sigmoid(input[boxOffset + 1])
      expected[boxOffset + 2] = math.exp(input[boxOffset + 2])
      expected[boxOffset + 3] = math.exp(input[boxOffset + 3])
      expected[boxOffset + 4] = sigmoid(input[boxOffset + 4])
      expected[boxOffset + 5 : boxOffset + 5 + 20] = softmax(input[boxOffset + 5 : boxOffset + 5 + 20])
    */
    // clang-format off
    TensorType expectedOutput =
    {
        #include "TestRegionDetectionNode_expectedOutput.inc"
    };
    // clang-format on
    testing::ProcessTest("Verifying expected output dimensions", testing::IsEqual(expectedOutput.GetShape(), math::TensorShape{ 13, 13, 125 }));

    Shape outputShape = { 13, 13, 125 };
    LayerParameters layerParams{ input, NoPadding(), outputShape, NoPadding() };

    for (bool applySoftmax : { false, true })
    {
        RegionDetectionParameters detectionParams{ 13, 13, 5, 20, 4, applySoftmax };

        RegionDetectionLayer<ElementType> detectionLayer(layerParams, detectionParams);
        detectionLayer.Compute();
        auto output = detectionLayer.GetOutput();

        // Expected output has been pre-processed already, so do the same with the output
        auto processedOutput = PostProcessRegionLayerOutput(RegionDetectionLayer<ElementType>::TensorType(output), detectionParams);
        testing::ProcessTest("Layer output == expectedOutput", testing::IsEqual(processedOutput.ToArray(), expectedOutput.ToArray(), 1e-5));

        // Create model
        model::Model model;
        auto inputNode = model.AddNode<model::InputNode<ElementType>>(input.Size());
        auto computeNode = model.AddNode<RegionDetectionLayerNode<ElementType>>(inputNode->output, detectionLayer);
        auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });

        // Make a copy to ensure remaining tests aren't affected
        auto mapCopy = map;
        mapCopy.SetInputValue(0, input.ToArray());
        auto mapOutput = mapCopy.ComputeOutput<ElementType>(0);
        testing::ProcessTest("Map output == expectedOutput", testing::IsEqual(mapOutput, output.ToArray(), 1e-5));

        // Compile model
        model::MapCompilerOptions settings;
        settings.compilerSettings.useBlas = true;
        model::ModelOptimizerOptions optimizerOptions;
        model::IRMapCompiler compiler(settings, optimizerOptions);
        auto compiledMap = compiler.Compile(map);

        // compare computed vs. compiled output
        std::vector<std::vector<ElementType>> signal = { input.ToArray() };
        VerifyCompiledOutput(map, compiledMap, signal, computeNode->GetRuntimeTypeName());
    }
}

void TestBroadcasUnaryOperationNodeCompile()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 3;
    int numChannels = 4;

    model::PortMemoryLayout inputLayout({ numRows, numColumns, numChannels });
    model::PortMemoryLayout outputLayout({ numRows, numColumns, numChannels });

    // clang-format off
    std::vector<double> inputVals{ 1, -2, -1, 2,    -1, -2, 1, 2,    1, -2, 1, 2,        3, -4, 3, -4,    3, -4, 3, 4,    3, -4, 3, 4 };
    std::vector<double>  expected{ 1,  2,  1, 2,     1,  2, 1, 2,    1,  2, 1, 2,        3,  4, 3,  4,    3,  4, 3, 4,    3,  4, 3, 4 };
    // clang-format on

    auto inputNode = model.AddNode<model::InputNode<double>>(inputLayout);
    auto op = BroadcastUnaryOperationNode<double>::OperationType::abs;
    auto outputNode = model.AddNode<BroadcastUnaryOperationNode<double>>(inputNode->output, op);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });

    // Compile model
    model::MapCompilerOptions settings;
    settings.compilerSettings.useBlas = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    auto computed = compiledMap.Compute<double>(inputVals);
    testing::ProcessTest("TestBroadcastUnaryOperationNodeCompile", testing::IsEqual(computed, expected));
}

void TestBroadcasBinaryOperationNodeCompileAdd()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 3;
    int numChannels = 4;

    model::PortMemoryLayout input1Layout({ numRows, numColumns, numChannels });
    model::PortMemoryLayout input2Layout({ 1, numColumns, 1 });
    model::PortMemoryLayout input3Layout({ 1, numColumns, 1 });
    model::PortMemoryLayout outputLayout({ numRows, numColumns, numChannels });

    // clang-format off
    std::vector<double> input1Vals{ 1, 2, 1, 2,    1, 2, 1, 2,    1, 2, 1, 2,        3, 4, 3, 4,    3, 4, 3, 4,    3, 4, 3, 4 };
    std::vector<double> input2Vals{ 2, 4, 6 };
    // broadcasts to:             { 2, 2, 2, 2,    4, 4, 4, 4,    6, 6, 6, 6,        2, 2, 2, 2,    4, 4, 4, 4,    6, 6, 6, 6 }
    std::vector<double>   expected{ 3, 4, 3, 4,    5, 6, 5, 6,    7, 8, 7, 8,        5, 6, 5, 6,    7, 8, 7, 8,    9, 10, 9, 10 };
    // clang-format on

    auto input1Node = model.AddNode<model::InputNode<double>>(input1Layout);
    auto input2Node = model.AddNode<ConstantNode<double>>(input2Vals, input2Layout);
    auto op = BroadcastBinaryOperationNode<double>::OperationType::add;
    auto outputNode = model.AddNode<BroadcastBinaryOperationNode<double>>(input1Node->output,
                                                                          input2Node->output,
                                                                          op);
    auto map = model::Map(model, { { "input", input1Node } }, { { "output", outputNode->output } });

    // Compile model
    model::MapCompilerOptions settings;
    settings.compilerSettings.useBlas = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    auto computed = compiledMap.Compute<double>(input1Vals);
    testing::ProcessTest("TestBroadcastBinaryOperationNodeCompileAdd", testing::IsEqual(computed, expected));
}

void TestBroadcasBinaryOperationNodeCompileSubtract()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 3;
    int numChannels = 4;

    model::PortMemoryLayout input1Layout({ numRows, numColumns, numChannels });
    model::PortMemoryLayout input2Layout({ 1, numColumns, 1 });
    model::PortMemoryLayout input3Layout({ 1, numColumns, 1 });
    model::PortMemoryLayout outputLayout({ numRows, numColumns, numChannels });

    // clang-format off
    std::vector<double> input1Vals{ 1, 2, 1, 2,      1, 2, 1, 2,      1, 2, 1, 2,            3, 4, 3, 4,    3, 4, 3, 4,    3, 4, 3, 4 };
    std::vector<double> input2Vals{ 2, 4, 6 };
    // broadcasts to:             { 2, 2, 2, 2,      4, 4, 4, 4,      6, 6, 6, 6,            2, 2, 2, 2,    4, 4, 4, 4,    6, 6, 6, 6 }
    std::vector<double>   expected{ -1, 0, -1, 0,    -3, -2, -3, -2,    -5, -4, -5, -4,      1, 2, 1, 2,   -1, 0, -1, 0,  -3, -2, -3, -2 };
    // clang-format on

    auto input1Node = model.AddNode<model::InputNode<double>>(input1Layout);
    auto input2Node = model.AddNode<ConstantNode<double>>(input2Vals, input2Layout);
    auto op = BroadcastBinaryOperationNode<double>::OperationType::subtract;
    auto outputNode = model.AddNode<BroadcastBinaryOperationNode<double>>(input1Node->output,
                                                                          input2Node->output,
                                                                          op);
    auto map = model::Map(model, { { "input", input1Node } }, { { "output", outputNode->output } });

    // Compile model
    model::MapCompilerOptions settings;
    settings.compilerSettings.useBlas = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    auto computed = compiledMap.Compute<double>(input1Vals);
    testing::ProcessTest("TestBroadcastBinaryOperationNodeCompileSubtract", testing::IsEqual(computed, expected));
}

void TestBroadcasBinaryOperationNodeCompileWithOrdering()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 3;

    model::PortMemoryLayout input1Layout({ numRows, numColumns });
    model::PortMemoryLayout input2Layout = input1Layout.ReorderedCopy({ 1, 0 });
    model::PortMemoryLayout outputLayout({ numRows, numColumns });

    // clang-format off
    std::vector<double> input1Vals{ 1, 2, 3,
                                    4, 5, 6 };
    // std::vector<double> input1Vals{ 10, 20, 30,
    //                                 40, 50, 60 };
    // transposed to:
    std::vector<double> input2Vals{ 10, 40,
                                    20, 50,
                                    30, 60 };
    std::vector<double>   expected{ 11, 22, 33,
                                    44, 55, 66 };
    // clang-format on

    auto input1Node = model.AddNode<model::InputNode<double>>(input1Layout);
    auto input2Node = model.AddNode<ConstantNode<double>>(input2Vals, input2Layout);
    auto op = BroadcastBinaryOperationNode<double>::OperationType::add;
    auto outputNode = model.AddNode<BroadcastBinaryOperationNode<double>>(input1Node->output,
                                                                          input2Node->output,
                                                                          op);
    auto map = model::Map(model, { { "input", input1Node } }, { { "output", outputNode->output } });

    // Compile model
    model::MapCompilerOptions settings;
    settings.compilerSettings.useBlas = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    auto computed = compiledMap.Compute<double>(input1Vals);
    testing::ProcessTest("TestBroadcastBinaryOperationNodeCompileWithOrdering", testing::IsEqual(computed, expected));
}

void TestSpatialConvolutionNode(size_t inputPaddingSize, size_t outputPaddingSize)
{
    // Abbreviations:
    //
    // r == # input rows
    // c == # input columns
    // ch == # input channels
    // fw == filter width
    // nf == # filters
    // pi == input padding amount
    // po == output padding amount

    // Data dimensions:
    //
    // Input: r x c x ch, with padding -> r+2pi x c+2pi x ch
    //     == 1 x 2 x 2, with padding == 1 -> 3 x 4 x 2
    // Weights: nf x fw x fw x ch
    //       == 2 x 3 x 3 x 2, (2 3x3 filters, with 2 input channels each)
    // Output: r x c x nf, with padding -> 1+2po x 2+2po x 2
    //      == 1 x 2 x 2, with padding == 0 -> 1 x 2 x 2

    using ElementType = double;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;
    using TensorReferenceType = typename Layer<ElementType>::TensorReferenceType;
    using Shape = typename Layer<ElementType>::Shape;

    assert(inputPaddingSize == 1);
    TensorType inputWithPadding(2 + 2 * inputPaddingSize, 2 + 2 * inputPaddingSize, 2);
    TensorReferenceType input = inputWithPadding.GetSubTensor({ inputPaddingSize, inputPaddingSize, 0 }, { 2, 2, 2 });
    inputWithPadding.Fill(0);
    input(0, 0, 0) = 2;
    input(0, 1, 0) = 1;
    input(0, 0, 1) = 3;
    input(0, 1, 1) = 2;
    // Input channel 0: [2, 3], input channel 1: [1, 2]

    Shape outputShape = { 2, 2, 2 };

    LayerParameters parameters{ inputWithPadding, ZeroPadding(inputPaddingSize), outputShape, ZeroPadding(outputPaddingSize) };
    ConvolutionalParameters convolutionalParams{ 3, 1, ConvolutionMethod::automatic, 2 };

    // Filter weights in `weightsVector` are in numFilters x numChannels x filterSize x filterSize order
    // clang-format off
    std::vector<ElementType> weightsVector {
        1, 3, 2,   3, 1, 1,   2, 3, 1,   // Filter 1, channel 1
        1, 2, 1,   2, 3, 2,   1, 2, 1}; // Filter 2, channel 2
    // clang-format on

    // Viewed as planar filters (ch x fw x fw):
    //
    //       1 3 2
    // f0 =  3 1 1
    //       2 3 1
    //
    //       1 2 1
    // f1 =  2 3 2
    //       1 2 1

    // Filter weights in `weights` tensor are in numFilters x filterSize x filterSize x 1 order
    TensorType weights(convolutionalParams.receptiveField * outputShape.NumChannels(), convolutionalParams.receptiveField, 1);

    size_t vectorIndex = 0;
    for (size_t f = 0; f < outputShape.NumChannels(); ++f)
    {
        for (size_t k = 0; k < 1; ++k)
        {
            for (size_t i = 0; i < convolutionalParams.receptiveField; ++i)
            {
                for (size_t j = 0; j < convolutionalParams.receptiveField; ++j)
                {
                    weights(f * convolutionalParams.receptiveField + i, j, k) = weightsVector[vectorIndex++];
                }
            }
        }
    }

    //
    // Verify ConvolutionalLayerNode
    //
    ConvolutionalLayer<ElementType> layer(parameters, convolutionalParams, weights);
    layer.Compute();
    auto output = layer.GetOutput();

    // Create model
    model::Model model;
    auto inputMemoryLayout = utilities::MemoryLayout(
        utilities::MemoryShape{ 2, 2, 2 },
        utilities::MemoryShape{ static_cast<int>(inputPaddingSize), static_cast<int>(inputPaddingSize), 0 });
    // BUGBUG: This fails when the order is not canonical order.
    auto inputNode = model.AddNode<model::InputNode<double>>(inputMemoryLayout.ReorderedCopy({ 2, 0, 1 }));
    auto outputMemoryLayout = utilities::MemoryLayout(utilities::MemoryShape{ 2, 2, 2 });
    auto computeNode = model.AddNode<SpatialConvolutionNode<double>>(inputNode->output, layer, outputMemoryLayout);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });

    const auto info = "TestSpatialConvolutionalLayer";

    VerifyLayerMap<ElementType>(map, computeNode, inputWithPadding, output, info);

    // Test archiving / unarchiving produces same result
    VerifyArchiveAndUnarchivingMap<ElementType>(map, computeNode, inputWithPadding, output, info);
}
