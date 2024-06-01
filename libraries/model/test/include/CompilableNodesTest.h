////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CompilableNodesTest.h (compile_test)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <model_testing/include/ModelTestUtilities.h>

#include <emitters/include/IREmitter.h>

#include <model/include/CompiledMap.h>
#include <model/include/IRCompiledMap.h>
#include <model/include/IRMapCompiler.h>
#include <model/include/Map.h>

#include <nodes/include/ConstantNode.h>
#include <nodes/include/DotProductNode.h>
#include <nodes/include/MatrixMatrixMultiplyCodeNode.h>

#include <predictors/neural/include/ConvolutionalLayer.h>
#include <predictors/neural/include/Layer.h>

#include <cstring>

using namespace ell;

void TestCompileIsEqual();
void TestCompilableScalarOutputNode();
void TestCompilableVectorOutputNode();
void TestCompilableAccumulatorNode();
void TestCompilableDotProductNode();
void TestCompilableDelayNode();
void TestCompilableDTWDistanceNode();
void TestCompilableMulticlassDTW();
void TestCompilableScalarSumNode();
void TestCompilableSumNode();
void TestCompilableUnaryOperationNode();
void TestL2NormSquaredNodeCompiled();
void TestMatrixVectorProductNodeCompile();
void TestCompilableBinaryOperationNode();
void TestCompilableBinaryOperationNode2();
void TestCompilableScalarBinaryPredicateNode();
void TestCompilableBinaryPredicateNode();
void TestCompilableMultiplexerNode();
void TestCompilableTypeCastNode(size_t dimension);
void TestReinterpretLayoutNode();
void TestReorderDataNode1();
void TestReorderDataNode2();
void TestReorderDataNode3();
void TestReorderDataCodeNode1();
void TestReorderDataCodeNode2();
void TestReorderDataCodeNode3();
void TestReorderDataCodeNode4();
void TestReceptiveFieldMatrixNode(size_t numChannels, bool useNewReshape);
void TestCompilableAccumulatorNodeFunction();
void TestCompilableSourceNode();
void TestCompilableSinkNode();
template <typename ElementType>
void TestCompilableDotProductNode2(int dimension);
void TestFloatNode();
void TestMultipleOutputNodes();
void TestShapeFunctionGeneration();
void TestCompilableClockNode();
void TestCompilableFFTNode();
template <typename ElementType>
void TestBufferNode();

//
// mathy nodes
//
void TestMatrixVectorMultiplyNode(int m, int n, bool useBlas);
void TestMatrixMatrixMultiplyNode(int m, int n, int k, bool useBlas);
void TestOrderedMatrixMatrixMultiplyNode(int m, int n, int k, bool transposeA, bool transposeB, bool transposeC, bool useBlas);
void TestMatrixMatrixMultiplyCodeNode(int m, int n, int k, int panelM, int panelN, int panelK, int kernelM, int kernelN, int kernelK, nodes::MatrixMatrixMultiplyImplementation gemmImpl);

void TestBroadcasUnaryOperationNodeCompile();
void TestBroadcasBinaryOperationNodeCompileAdd();
void TestBroadcasBinaryOperationNodeCompileSubtract();
void TestBroadcasBinaryOperationNodeCompileWithOrdering();

//
// NN layer nodes
//
void TestNeuralNetworkPredictorNode1();
void TestNeuralNetworkPredictorNode2();
void TestNeuralNetworkPredictorNode3();
void TestNeuralNetworkPredictorNode4();
void TestNeuralNetworkPredictorNode5();
void TestNeuralNetworkPredictorNode6();

using ConvolutionMethod = ell::predictors::neural::ConvolutionMethod;

void TestInputLayerNode(size_t outputPadding = 0);
void TestHardSigmoidActivationLayerNode(size_t inputPaddingSize = 0, size_t outputPaddingSize = 0);
void TestHardTanhActivationLayerNode(size_t inputPaddingSize = 0, size_t outputPaddingSize = 0);
void TestReLUActivationLayerNode(size_t inputPadding = 0, size_t outputPadding = 0);
void TestLeakyReLUActivationLayerNode(size_t inputPadding = 0, size_t outputPadding = 0);
void TestParametricReLUActivationLayerNode(size_t inputPadding = 0, size_t outputPadding = 0);
void TestSigmoidActivationLayerNode(size_t inputPadding = 0, size_t outputPadding = 0);
void TestBatchNormalizationLayerNode(size_t inputPadding = 0, size_t outputPadding = 0);
void TestBiasLayerNode(size_t inputPadding = 0, size_t outputPadding = 0);
void TestBinaryConvolutionalLayerNode(size_t imageRows, size_t imageColumns, size_t numChannels, size_t numFilters, size_t inputPadding = 1, size_t outputPadding = 0, ell::predictors::neural::PaddingScheme = ell::predictors::neural::PaddingScheme::zeros, bool scaleByFilterMeans = true);
void TestBroadcastLinearFunctionNode();
void TestConvolutionalLayerNode(ConvolutionMethod convolutionMethod, size_t inputPadding = 1, size_t outputPadding = 0);
void TestConvolutionalLayerNode2(ConvolutionMethod convolutionMethod, size_t inputPadding = 1, size_t outputPadding = 0);
void TestConvolutionalLayerNode3(ConvolutionMethod convolutionMethod, size_t inputPadding = 1, size_t outputPadding = 0);
void TestFullyConnectedLayerNode(size_t inputPadding = 0, size_t outputPadding = 0);
void TestMaxPoolingLayerNode(size_t inRows, size_t inCols, size_t numChannels, size_t outRows, size_t outCols, size_t poolingSize, size_t poolingStride, size_t inputPadding = 0, size_t outputPadding = 0);
void TestMeanPoolingLayerNode(size_t inRows, size_t inCols, size_t numChannels, size_t outRows, size_t outCols, size_t poolingSize, size_t poolingStride, size_t inputPadding = 0, size_t outputPadding = 0);
void TestScalingLayerNode(size_t inputPadding = 0, size_t outputPadding = 0);
void TestSoftmaxLayerNode(size_t inputPadding = 0, size_t outputPadding = 0);
void TestSpatialConvolutionNode(size_t inputPadding = 1, size_t outputPadding = 0);
void TestFusedLinearLayerNodes(size_t rows, size_t columns, size_t channels);
void TestRegionDetectionNode();
void TestIRNode();

#pragma region implementation

template <typename ElementType>
void TestCompilableDotProductNode2(int dimension)
{
    model::Model model;
    std::vector<ElementType> constValue(dimension);
    for (int index = 0; index < dimension; ++index)
    {
        constValue[index] = index + 0.5;
    }
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(dimension);
    auto constantNode = model.AddNode<nodes::ConstantNode<ElementType>>(constValue);
    auto dotNode = model.AddNode<nodes::DotProductNode<ElementType>>(inputNode->output, constantNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", dotNode->output } });
    model::IRMapCompiler compiler;
    auto compiledMap = compiler.Compile(map);
    PrintIR(compiledMap);

    // compare output
    std::vector<std::vector<ElementType>> signal;
    for (int index1 = 0; index1 < 8; ++index1)
    {
        std::vector<ElementType> x;
        for (int index2 = 0; index2 < dimension; ++index2)
        {
            x.push_back(index2);
        }
        signal.push_back(x);
    }

    VerifyCompiledOutput(map, compiledMap, signal, "DotProductNode");
}

#pragma endregion implementation
