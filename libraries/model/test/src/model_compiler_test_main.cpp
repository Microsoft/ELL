////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     model_compiler_test_main.cpp (model_compiler_test)
//  Authors:  Umesh Madan, Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CompilableCodeNode_test.h"
#include "CompilableNodesTest.h"
#include "CompilerTest.h"
#include "ModelHeaderOutputTest.h"
#include "PerformanceCountersTest.h"
#include <nodes/include/MatrixMatrixMultiplyCodeNode.h>

#include <model_testing/include/ModelTestUtilities.h>

#include <testing/include/testing.h>

#include <iostream>

using namespace ell;
using namespace ell::emitters;
using namespace ell::predictors::neural;

void TestMatrixMatrixMultiplyCodeNodeImplementations()
{
    const int fallbackPanelM = 1;
    const int fallbackPanelN = 1;
    const int fallbackPanelK = 1;
    const int fallbackKernelM = 1;
    const int fallbackKernelN = 1;
    const int fallbackKernelK = 1;
    // Naive for-loop implementation
    TestMatrixMatrixMultiplyCodeNode(1, 1, 1, fallbackPanelM, fallbackPanelN, fallbackPanelK, fallbackKernelM, fallbackKernelN, fallbackKernelK, nodes::MatrixMatrixMultiplyImplementation::SimpleForLoops);
    TestMatrixMatrixMultiplyCodeNode(4, 4, 4, fallbackPanelM, fallbackPanelN, fallbackPanelK, fallbackKernelM, fallbackKernelN, fallbackKernelK, nodes::MatrixMatrixMultiplyImplementation::SimpleForLoops);
    TestMatrixMatrixMultiplyCodeNode(4, 8, 8, fallbackPanelM, fallbackPanelN, fallbackPanelK, fallbackKernelM, fallbackKernelN, fallbackKernelK, nodes::MatrixMatrixMultiplyImplementation::SimpleForLoops);
    TestMatrixMatrixMultiplyCodeNode(4, 4, 8, fallbackPanelM, fallbackPanelN, fallbackPanelK, fallbackKernelM, fallbackKernelN, fallbackKernelK, nodes::MatrixMatrixMultiplyImplementation::SimpleForLoops);
}

void TestIRCompiler()
{
    // TestIRNode(); // Failing on Windows

    TestBufferNode<float>();
    TestBufferNode<double>();
    TestBufferNode<int>();
    TestBufferNode<int64_t>();

    TestReinterpretLayoutNode();
    TestBroadcastLinearFunctionNode();

    TestNodeMetadata();
    TestMultiOutputMap();
    TestCombineOutputMap();

    // Header file generation
    TestModelHeaderOutput();

    TestFloatNode();
    // TestMultipleOutputNodes(); // Multiple output nodes aren't currently supported
    TestShapeFunctionGeneration();
    TestCompilableDotProductNode2<float>(3);
    TestCompilableDotProductNode2<double>(3);
    TestCompilableDotProductNode2<float>(4);
    TestCompilableDotProductNode2<double>(4);

    TestCompileIsEqual();
    TestSimpleMap(false);
    TestSimpleMap(true);
    TestCompiledMapMove();
    TestCompiledMapClone();
    TestCompiledMapParallelClone();

    TestBinaryScalar();
    TestBinaryVector(true);
    TestBinaryVector(false);
    TestBinaryVector(true, true);
    TestDotProduct();
    TestSum(false);
    TestSum(true);
    TestAccumulator(false);
    TestAccumulator(true);
    TestDelay();
    TestSqrt();
    TestBinaryPredicate(false);
    TestSlidingAverage();
    TestDotProductOutput();
    TestLinearPredictor<double>();
    TestLinearPredictor<float>();
    // TestMultiplexer(); // FAILS -- crash
    // TestForest(); // FAILS -- crash
#ifdef USE_BLAS
    TestMatrixVectorMultiplyNode(10, 5, true);
#endif
    TestMatrixVectorMultiplyNode(10, 5, false);

#ifdef USE_BLAS
    TestMatrixMatrixMultiplyNode(4, 5, 6, true);
#endif
    TestMatrixMatrixMultiplyNode(4, 5, 6, false);

#ifdef USE_BLAS
    // Using BLAS
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, false, false, false, true);
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, true, false, false, true);
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, false, true, false, true);
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, true, true, false, true);
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, false, false, true, true);
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, true, false, true, true);
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, false, true, true, true);
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, true, true, true, true);
#endif

    // Not using BLAS
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, false, false, false, false);
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, true, false, false, false);
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, false, true, false, false);
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, true, true, false, false);
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, false, false, true, false);
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, true, false, true, false);
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, false, true, true, false);
    TestOrderedMatrixMatrixMultiplyNode(4, 5, 6, true, true, true, false);

    // TestMatrixMatrixMultiplyNode(15, 25600, 27, false); // Fails due to numerical  issues

    TestMatrixMatrixMultiplyCodeNodeImplementations();

    TestCompilableScalarOutputNode();
    TestCompilableVectorOutputNode();
    TestCompilableAccumulatorNode();
    TestCompilableDotProductNode();
    TestCompilableDelayNode();
    TestCompilableDTWDistanceNode();
    TestCompilableMulticlassDTW();
    TestCompilableScalarSumNode();
    TestCompilableSumNode();
    TestCompilableUnaryOperationNode();
    TestCompilableBinaryOperationNode();
    TestCompilableBinaryOperationNode2();
    TestCompilableScalarBinaryPredicateNode();
    TestCompilableBinaryPredicateNode();
    TestCompilableMultiplexerNode();
    TestCompilableTypeCastNode(1);
    TestCompilableTypeCastNode(2);
    TestCompilableTypeCastNode(10);
    TestReorderDataNode1();
    TestReorderDataNode2();
    TestReorderDataNode3();
    TestReorderDataCodeNode1();
    TestReorderDataCodeNode2();
    TestReorderDataCodeNode3();
    TestReorderDataCodeNode4();
    TestReceptiveFieldMatrixNode(1, true); // new version
    TestReceptiveFieldMatrixNode(1, false); // old (slow) version
    TestReceptiveFieldMatrixNode(2, true); // new version
    // TestReceptiveFieldMatrixNode(2, false); // old (slow) version -- Fails
    TestCompilableAccumulatorNodeFunction();
    TestCompilableSourceNode();
    TestCompilableSinkNode();
    TestCompilableClockNode();
    TestCompilableFFTNode();

    TestPerformanceCounters();
    TestCompilableDotProductNode2<float>(3); // uses IR
    TestCompilableDotProductNode2<double>(3); // uses IR
    TestCompilableDotProductNode2<float>(4); // uses IR
    TestCompilableDotProductNode2<double>(4); // uses IR

    //
    // Neural net nodes
    //
    TestNeuralNetworkPredictorNode1();
    TestNeuralNetworkPredictorNode2();
    TestNeuralNetworkPredictorNode3();
    // TestNeuralNetworkPredictorNode4(); // Currently fails
    // TestNeuralNetworkPredictorNode5(); // Currently fails (but just barely...)
    // TestNeuralNetworkPredictorNode6();

    TestFusedLinearLayerNodes(4, 6, 8);

    // TestInputLayerNode(0);
    TestInputLayerNode(1);

    TestHardSigmoidActivationLayerNode();
    TestHardSigmoidActivationLayerNode(0, 1);
    TestHardSigmoidActivationLayerNode(0, 2);

    TestHardTanhActivationLayerNode();
    TestHardTanhActivationLayerNode(0, 1);
    TestHardTanhActivationLayerNode(0, 2);

    TestReLUActivationLayerNode();
    TestReLUActivationLayerNode(0, 1);
    TestReLUActivationLayerNode(0, 2);
    // TestReLUActivationLayerNode(1, 0); // Input padding not supported (yet)

    TestLeakyReLUActivationLayerNode();
    TestLeakyReLUActivationLayerNode(0, 1);
    TestLeakyReLUActivationLayerNode(0, 2);

    TestParametricReLUActivationLayerNode();
    TestParametricReLUActivationLayerNode(0, 1);
    TestParametricReLUActivationLayerNode(0, 2);

    TestSigmoidActivationLayerNode();
    TestSigmoidActivationLayerNode(0, 1);
    TestSigmoidActivationLayerNode(0, 2);

    TestBatchNormalizationLayerNode();
    TestBatchNormalizationLayerNode(0, 1);
    TestBatchNormalizationLayerNode(0, 2);
    // TestBatchNormalizationLayerNode(1, 0); // Input padding not supported (yet)

    TestBiasLayerNode();
    TestBiasLayerNode(0, 1);
    TestBiasLayerNode(0, 2);
    // TestBiasLayerNode(1, 0); // Input padding not supported (yet)

    TestMaxPoolingLayerNode(5, 5, 16, 5, 5, 4, 1, 1, 0); // params: inW, inH, inChannels, outW, outH, poolingWindowSize, stride, inputPadding, outputPadding

    TestMaxPoolingLayerNode(8, 8, 16, 6, 6, 3, 1, 0, 0); // params: inW, inH, inChannels, outW, outH, poolingWindowSize, stride, inputPadding, outputPadding
    TestMaxPoolingLayerNode(8, 8, 16, 6, 6, 3, 1, 0, 1);
    TestMaxPoolingLayerNode(8, 8, 16, 6, 6, 3, 1, 0, 2);

    TestMaxPoolingLayerNode(10, 10, 16, 5, 5, 3, 2, 1, 0);
    TestMaxPoolingLayerNode(10, 10, 16, 9, 9, 2, 1, 0, 0);

    // test weird case we are seeing in some cntk models
    TestMaxPoolingLayerNode(7, 7, 16, 4, 4, 2, 2, 0, 0);

    TestMeanPoolingLayerNode(8, 8, 16, 6, 6, 3, 1, 0, 0);
    TestMeanPoolingLayerNode(8, 8, 16, 6, 6, 3, 1, 0, 1);
    TestMeanPoolingLayerNode(8, 8, 16, 6, 6, 3, 1, 0, 2);
    // TestMeanPoolingLayerNode(8, 8, 16, 6, 6, 3, 1, 1, 0);

    // TestMeanPoolingLayerNode(8, 8, 16, 2, 1, 2, 1, 0, 0);

    TestScalingLayerNode();
    TestScalingLayerNode(0, 1);
    TestScalingLayerNode(0, 2);
    // TestScalingLayerNode(1, 0); // Input padding not supported (yet)

    TestSoftmaxLayerNode();
    TestSoftmaxLayerNode(0, 1);
    TestSoftmaxLayerNode(0, 2);
    // TestSoftmaxLayerNode(1, 0); // Input padding not supported (yet)

    TestBinaryConvolutionalLayerNode(32, 32, 3, 4);
    TestBinaryConvolutionalLayerNode(32, 32, 3, 4, 1, 0, PaddingScheme::zeros, true);
    TestBinaryConvolutionalLayerNode(32, 32, 3, 4, 1, 0, PaddingScheme::minusOnes, false);
    TestBinaryConvolutionalLayerNode(32, 32, 3, 4, 1, 0, PaddingScheme::minusOnes, true);

    // TestConvolutionalLayerNode(ConvolutionMethod::unrolled);
    TestConvolutionalLayerNode(ConvolutionMethod::unrolled, 1, 0);

    TestConvolutionalLayerNode2(ConvolutionMethod::unrolled, 1, 0);
    TestConvolutionalLayerNode3(ConvolutionMethod::unrolled, 1, 0);
    // TestConvolutionalLayerNode(ConvolutionMethod::unrolled, 2, 0);
    // TestConvolutionalLayerNode(ConvolutionMethod::unrolled, 1, 1); // Convolutional layer output padding not supported

    TestConvolutionalLayerNode(ConvolutionMethod::diagonal); // Input padding must be set correctly (to floor(filterWidth/2))

    // Input padding must be set correctly (to floor(filterWidth/2))
    TestConvolutionalLayerNode(ConvolutionMethod::simple, 1, 0);
    TestConvolutionalLayerNode2(ConvolutionMethod::simple, 1, 0);
    TestConvolutionalLayerNode3(ConvolutionMethod::simple, 1, 0);

    TestConvolutionalLayerNode(ConvolutionMethod::winograd, 1, 0);
    TestConvolutionalLayerNode2(ConvolutionMethod::winograd, 1, 0);
    TestConvolutionalLayerNode3(ConvolutionMethod::winograd, 1, 0);

    //BUGBUG: This test currently fails for Compute but passes for Compile.
    //TestSpatialConvolutionNode(1, 0);

    TestFullyConnectedLayerNode();
    // TestFullyConnectedLayerNode(0, 1); // Fully-connected layer nodes can't have padding (yet)
    // TestFullyConnectedLayerNode(0, 2); // Fully-connected layer nodes can't have padding (yet)
    // TestFullyConnectedLayerNode(1, 1); // Fully-connected layer nodes can't have padding (yet)

    TestProtoNNPredictorMap();
    TestMultiSourceSinkMap();

    TestRegionDetectionNode();

    TestMatrixVectorProductNodeCompile();

    TestBroadcasUnaryOperationNodeCompile();
    TestBroadcasBinaryOperationNodeCompileAdd();
    TestBroadcasBinaryOperationNodeCompileSubtract();
    TestBroadcasBinaryOperationNodeCompileAdd();
    TestBroadcasBinaryOperationNodeCompileWithOrdering();
}

int main(int argc, char* argv[])
{
    try
    {
        TestIRCompiler();

        CompilableCodeNode_test1();
    }
    catch (const std::exception& exception)
    {
        std::cerr << "ERROR, got exception. Message: " << exception.what() << std::endl;
        return 1;
    }

    if (testing::DidTestFail())
    {
        return 1;
    }

    return 0;
}
