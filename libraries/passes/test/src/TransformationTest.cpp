////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     TransformationTest.cpp (passes/passes_test)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TransformationTest.h"

#include <passes/include/FuseLinearOperationsTransformation.h>
#include <passes/include/OptimizeReorderDataNodesTransformation.h>
#include <passes/include/SetConvolutionMethodTransformation.h>

#include <model/include/InputNode.h>
#include <model/include/TransformContext.h>
#include <model/include/Transformation.h>

#include <nodes/include/BroadcastFunctionNode.h>
#include <nodes/include/ConstantNode.h>
#include <nodes/include/ConvolutionalLayerNode.h>
#include <nodes/include/MatrixMatrixMultiplyNode.h>
#include <nodes/include/ReorderDataCodeNode.h>

#include <predictors/neural/include/ConvolutionalLayer.h>

#include <testing/include/testing.h>

#include <utilities/include/JsonArchiver.h>

#include <iostream>

#define PRINT_MODELS 0

using namespace ell;
using namespace ell::emitters;
using namespace ell::model;
using namespace ell::testing;
using namespace ell::passes;
using namespace ell::utilities;

namespace
{
[[maybe_unused]] void PrintModel(const model::Model& model)
{
    std::cout << "------ Model start ------" << std::endl;
    model.Print(std::cout);
    std::cout << "------ Model end ------" << std::endl;
}

bool HasNodeWithTypeName(const model::Model& model, std::string typeName)
{
    auto iter = model.GetNodeIterator();
    while (iter.IsValid())
    {
        auto node = iter.Get();
        if (node->GetRuntimeTypeName() == typeName)
        {
            return true;
        }
        iter.Next();
    }
    return false;
}

template <typename ValueType>
auto Increment(ValueType start, ValueType inc = static_cast<ValueType>(1))
{
    return [start, inc]() mutable { auto t = start; start += inc; return t; };
}

template <typename ValueType>
model::Map GenerateLinearOpsTestModel(const model::PortMemoryLayout& inputLayout, const model::PortMemoryLayout& outputLayout, std::vector<std::pair<bool, bool>> functionInfos)
{
    auto numRows = inputLayout.GetActiveSize(0);
    auto numColumns = inputLayout.GetActiveSize(1);
    auto numChannels = inputLayout.GetActiveSize(2);

    // Create a model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ValueType>>(numRows * numColumns * numChannels);
    model::MemoryShape scaleShape({ 1, 1, numChannels });
    model::MemoryShape biasShape({ 1, 1, numChannels });

    const model::OutputPort<ValueType>* prevOutput = &inputNode->output;
    int scaleStart = static_cast<ValueType>(1);
    int biasStart = static_cast<ValueType>(2);
    for (auto info : functionInfos)
    {
        nodes::ConstantNode<ValueType>* scaleNode = nullptr;
        nodes::ConstantNode<ValueType>* biasNode = nullptr;
        if (info.first) // has scale
        {
            std::vector<ValueType> scaleValues(numChannels);
            std::generate(scaleValues.begin(), scaleValues.end(), Increment(static_cast<ValueType>(scaleStart)));
            scaleNode = model.AddNode<nodes::ConstantNode<ValueType>>(scaleValues, scaleShape);
        }
        else
        {
            scaleNode = model.AddNode<nodes::ConstantNode<ValueType>>();
        }

        if (info.second) // has bias
        {
            std::vector<ValueType> biasValues(numChannels);
            std::generate(biasValues.begin(), biasValues.end(), Increment(static_cast<ValueType>(biasStart)));
            biasNode = model.AddNode<nodes::ConstantNode<ValueType>>(biasValues, biasShape);
        }
        else
        {
            biasNode = model.AddNode<nodes::ConstantNode<ValueType>>();
        }
        auto functionNode = model.AddNode<nodes::BroadcastLinearFunctionNode<ValueType>>(*prevOutput, inputLayout, scaleNode->output, biasNode->output, 2, outputLayout);
        prevOutput = &functionNode->output;
        ++scaleStart;
        ++biasStart;
    }

    // Make a map from it
    model::Map map(model, { { "input", inputNode } }, { { "output", *prevOutput } });
    return map;
}
} // namespace

//
// Transformation class tests
//

void TestTransformations()
{
    TestFuseLinearOperationsTransformation();
    TestSetConvolutionMethodTransformation();
    TestOptimizeReorderDataNodesTransformation();
}

void TestFuseLinearOperationsTransformation(std::vector<std::pair<bool, bool>> functionInfos)
{
    using ValueType = float;

    int numRows = 1;
    int numColumns = 1;
    int numChannels = 1;
    model::PortMemoryLayout inputLayout({ numRows, numColumns, numChannels });
    model::PortMemoryLayout outputLayout({ numRows, numColumns, numChannels });

    auto map = GenerateLinearOpsTestModel<ValueType>(inputLayout, outputLayout, functionInfos);
    auto oldSize = map.GetModel().Size();

    // Generate test data
    std::vector<ValueType> testInput(numRows * numColumns * numChannels);
    std::generate(testInput.begin(), testInput.end(), Increment<ValueType>(0.0f));

    // Evaluate it pre-optimization
    map.SetInputValue("input", testInput);
    auto referenceOutput = map.ComputeOutput<ValueType>("output");

#if PRINT_MODELS
    PrintModel(map.GetModel());
#endif

    model::MapCompilerOptions settings;
    model::ModelOptimizerOptions optimizerOptions;
    optimizerOptions["fuseLinearFunctionNodes"] = true;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    model::TransformContext context(&compiler);
    FuseLinearOperationsTransformation fuseOps;
    map.GetModel().GetMetadata().SetEntry("compileOptions", optimizerOptions.AsPropertyBag());
    map.Transform(fuseOps, context);
    map.Refine();
    map.Prune();

#if PRINT_MODELS
    PrintModel(map.GetModel());
#endif

    auto newSize = map.GetModel().Size();
    auto numLinearNodes = functionInfos.size();
    testing::ProcessTest("Testing linear ops count", oldSize == (3 * numLinearNodes) + 1 && newSize == 4);

    // Evaluate model post-optimization
    map.SetInputValue("input", testInput);
    auto optimizedOutput = map.ComputeOutput<ValueType>("output");
    testing::ProcessTest("Testing result", testing::IsEqual(referenceOutput, optimizedOutput));

    //
    // Now test compiled codepath
    //

    // Compile the model
    // model::ModelOptimizerOptions optimizerOptions;
    // optimizerOptions["fuseLinearFunctionNodes"] = true;
    // model::IRMapCompiler compiler(settings, optimizerOptions);
    // auto compiledMap = compiler.Compile(map);
    // auto newCompiledSize = compiledMap.GetModel().Size();
    // testing::ProcessTest("Testing compiled linear ops count", newCompiledSize <= newSize);

    // Evaluate the compiled model
    // compiledMap.SetInputValue("input", testInput);
    // auto compiledOutput = compiledMap.ComputeOutput<ValueType>("output");
    // testing::ProcessTest("Testing compiled result", testing::IsEqual(referenceOutput, compiledOutput));
}

void TestFuseLinearOperationsTransformation()
{
    std::pair<bool, bool> linear = { true, true };
    std::pair<bool, bool> scale = { true, false };
    std::pair<bool, bool> bias = { false, true };

    TestFuseLinearOperationsTransformation({ linear, linear });
    TestFuseLinearOperationsTransformation({ linear, scale });
    TestFuseLinearOperationsTransformation({ linear, bias });

    TestFuseLinearOperationsTransformation({ scale, linear });
    TestFuseLinearOperationsTransformation({ scale, scale });
    TestFuseLinearOperationsTransformation({ scale, bias });

    TestFuseLinearOperationsTransformation({ bias, linear });
    TestFuseLinearOperationsTransformation({ bias, scale });
    TestFuseLinearOperationsTransformation({ bias, bias });

    TestFuseLinearOperationsTransformation({ bias, scale, linear });
    TestFuseLinearOperationsTransformation({ scale, bias, linear });
    TestFuseLinearOperationsTransformation({ linear, bias, scale });
    TestFuseLinearOperationsTransformation({ linear, scale, bias });
    TestFuseLinearOperationsTransformation({ bias, linear, scale });
    TestFuseLinearOperationsTransformation({ scale, linear, bias });

    TestFuseLinearOperationsTransformation({ linear, scale, scale });
    TestFuseLinearOperationsTransformation({ linear, bias, bias });
}

void TestSetConvolutionMethodTransformation(model::PreferredConvolutionMethod convolutionMethod, std::string expectedNodeTypeName)
{
    using namespace predictors::neural;

    using ElementType = float;
    using LayerParameters = typename Layer<ElementType>::LayerParameters;
    using TensorType = typename Layer<ElementType>::TensorType;
    using TensorReferenceType = typename Layer<ElementType>::TensorReferenceType;
    using Shape = typename Layer<ElementType>::Shape;

    const size_t inputPaddingSize = 1;
    const size_t outputPaddingSize = 0;
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
    ConvolutionalParameters convolutionalParams{ 3, 1, ConvolutionMethod::automatic, 2 };

    TensorType weights(convolutionalParams.receptiveField * outputShape.NumChannels(), convolutionalParams.receptiveField, input.NumChannels());
    ConvolutionalLayer<ElementType> layer(parameters, convolutionalParams, weights);

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputWithPadding.Size());
    auto computeNode = model.AddNode<nodes::ConvolutionalLayerNode<ElementType>>(inputNode->output, layer);

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", computeNode->output } });
#if PRINT_MODELS
    PrintModel(map.GetModel());
#endif

    model::MapCompilerOptions settings;
    model::ModelOptimizerOptions optimizerOptions;
    optimizerOptions["preferredConvolutionMethod"] = convolutionMethod;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    model::TransformContext context(&compiler);
    passes::SetConvolutionMethodTransformation setConvMethod;
    map.Transform(setConvMethod, context);
    map.Prune();

#if PRINT_MODELS
    PrintModel(map.GetModel());
#endif

    testing::ProcessTest("Testing SetConvolutionMethodTransformation for " + expectedNodeTypeName, HasNodeWithTypeName(map.GetModel(), expectedNodeTypeName));
}

void TestSetConvolutionMethodTransformation()
{
    TestSetConvolutionMethodTransformation(model::PreferredConvolutionMethod::diagonal, "DiagonalConvolutionNode<float>");
    TestSetConvolutionMethodTransformation(model::PreferredConvolutionMethod::simple, "SimpleConvolutionNode<float>");
    TestSetConvolutionMethodTransformation(model::PreferredConvolutionMethod::winograd, "WinogradConvolutionNode<float>");
    TestSetConvolutionMethodTransformation(model::PreferredConvolutionMethod::unrolled, "UnrolledConvolutionNode<float>");
}

void TestOptimizeReorderDataNodesTransformation1()
{
    using ValueType = float;
    constexpr int m = 4, n = 5, k = 6;
    bool transposeA = false, transposeB = false, transposeC = false;

    auto orderA = transposeA ? model::DimensionOrder{ 1, 0 } : model::DimensionOrder{ 0, 1 };
    auto orderB = transposeB ? model::DimensionOrder{ 1, 0 } : model::DimensionOrder{ 0, 1 };
    auto orderC = transposeC ? model::DimensionOrder{ 1, 0 } : model::DimensionOrder{ 0, 1 };
    auto outputLayout = model::PortMemoryLayout(model::MemoryShape{ m, n }).ReorderedCopy(orderC);

    model::Model model;
    auto inputMatrixNode = model.AddNode<model::InputNode<ValueType>>(model::MemoryShape{ m, k });
    auto reorderedInputMatrixNode = model.AddNode<nodes::ReorderDataCodeNode<ValueType>>(inputMatrixNode->output, orderA);

    std::vector<ValueType> matrixBVals(k * n);
    auto matrixBNode = model.AddNode<nodes::ConstantNode<ValueType>>(matrixBVals, model::MemoryShape{ k, n });
    auto reorderedMatrixBNode = model.AddNode<nodes::ReorderDataCodeNode<ValueType>>(matrixBNode->output, orderB);

    auto matMatMultNode = model.AddNode<nodes::MatrixMatrixMultiplyNode<ValueType>>(reorderedInputMatrixNode->output, reorderedMatrixBNode->output, outputLayout);

    auto map = model::Map(model, { { "inputMatrix", inputMatrixNode } }, { { "output", matMatMultNode->output } });

    auto oldSize = map.GetModel().Size();
#if PRINT_MODELS
    PrintModel(map.GetModel());
#endif

    // Transform model
    passes::OptimizeReorderDataNodesTransformation optReorders;
    map.Transform(optReorders);
    map.Refine();
    auto newSize = map.GetModel().Size();

#if PRINT_MODELS
    PrintModel(map.GetModel());
#endif

    testing::ProcessTest("Testing OptimizeReorderDataNodesTransform1", oldSize == 5 && newSize == 3);
}

void TestOptimizeReorderDataNodesTransformation2()
{
    using ValueType = float;
    constexpr int m = 4, n = 5, k = 6;
    bool transposeA = false, transposeB = true, transposeC = false;

    auto orderA = transposeA ? model::DimensionOrder{ 1, 0 } : model::DimensionOrder{ 0, 1 };
    auto orderB = transposeB ? model::DimensionOrder{ 1, 0 } : model::DimensionOrder{ 0, 1 };
    auto orderC = transposeC ? model::DimensionOrder{ 1, 0 } : model::DimensionOrder{ 0, 1 };
    auto outputLayout = model::PortMemoryLayout(model::MemoryShape{ m, n }).ReorderedCopy(orderC);

    model::Model model;
    auto inputMatrixNode = model.AddNode<model::InputNode<ValueType>>(model::MemoryShape{ m, k });
    auto reorderedInputMatrixNode = model.AddNode<nodes::ReorderDataCodeNode<ValueType>>(inputMatrixNode->output, orderA);

    std::vector<ValueType> matrixBVals(k * n);
    auto matrixBNode = model.AddNode<nodes::ConstantNode<ValueType>>(matrixBVals, model::MemoryShape{ k, n });
    auto reorderedMatrixBNode = model.AddNode<nodes::ReorderDataCodeNode<ValueType>>(matrixBNode->output, orderB);

    auto matMatMultNode = model.AddNode<nodes::MatrixMatrixMultiplyNode<ValueType>>(reorderedInputMatrixNode->output, reorderedMatrixBNode->output, outputLayout);

    auto map = model::Map(model, { { "inputMatrix", inputMatrixNode } }, { { "output", matMatMultNode->output } });
    auto oldSize = map.GetModel().Size();
#if PRINT_MODELS
    PrintModel(map.GetModel());
#endif

    // Transform model
    passes::OptimizeReorderDataNodesTransformation t;
    map.Transform(t);
    map.Refine();
    auto newSize = map.GetModel().Size();

#if PRINT_MODELS
    PrintModel(map.GetModel());
#endif

    testing::ProcessTest("Testing OptimizeReorderDataNodesTransform2", oldSize == 5 && newSize == 4);
}

void TestOptimizeReorderDataNodesTransformation3()
{
    using ValueType = float;
    constexpr int m = 4, n = 5, k = 6;
    bool transposeA = true, transposeB = true, transposeC = true;

    auto orderA = transposeA ? model::DimensionOrder{ 1, 0 } : model::DimensionOrder{ 0, 1 };
    auto orderB = transposeB ? model::DimensionOrder{ 1, 0 } : model::DimensionOrder{ 0, 1 };
    auto orderC = transposeC ? model::DimensionOrder{ 1, 0 } : model::DimensionOrder{ 0, 1 };
    auto outputLayout = model::PortMemoryLayout(model::MemoryShape{ m, n }).ReorderedCopy(orderC);

    model::Model model;
    auto inputMatrixNode = model.AddNode<model::InputNode<ValueType>>(model::MemoryShape{ m, k });
    auto reorderedInputMatrixNode = model.AddNode<nodes::ReorderDataCodeNode<ValueType>>(inputMatrixNode->output, orderA);

    std::vector<ValueType> matrixBVals(k * n);
    auto matrixBNode = model.AddNode<nodes::ConstantNode<ValueType>>(matrixBVals, model::MemoryShape{ k, n });
    auto reorderedMatrixBNode = model.AddNode<nodes::ReorderDataCodeNode<ValueType>>(matrixBNode->output, orderB);

    auto matMatMultNode = model.AddNode<nodes::MatrixMatrixMultiplyNode<ValueType>>(reorderedInputMatrixNode->output, reorderedMatrixBNode->output, outputLayout);

    auto map = model::Map(model, { { "inputMatrix", inputMatrixNode } }, { { "output", matMatMultNode->output } });

    //
    // test pass
    //

    auto oldSize = map.GetModel().Size();
#if PRINT_MODELS
    PrintModel(map.GetModel());
#endif

    // Transform model
    passes::OptimizeReorderDataNodesTransformation optReorders;
    map.Transform(optReorders);
    map.Refine();
    auto newSize = map.GetModel().Size();

#if PRINT_MODELS
    PrintModel(map.GetModel());
#endif

    testing::ProcessTest("Testing OptimizeReorderDataNodesTransform3", oldSize == 5 && newSize == 5);
}

void TestOptimizeReorderDataNodesTransformation4()
{
    using ValueType = float;
    constexpr int m = 4, n = 5, k = 6;

    auto rowMajor = model::DimensionOrder{ 0, 1 };
    auto colMajor = model::DimensionOrder{ 1, 0 };
    auto outputLayout = model::PortMemoryLayout(model::MemoryShape{ m, n }).ReorderedCopy(colMajor);

    model::Model model;
    auto rowMajorLayout = model::PortMemoryLayout(model::MemoryShape{ m, k }).ReorderedCopy(rowMajor);
    auto colMajorLayout = model::PortMemoryLayout(model::MemoryShape{ m, k }).ReorderedCopy(colMajor);
    auto inputMatrixNode = model.AddNode<model::InputNode<ValueType>>(rowMajorLayout.GetActiveSize());
    auto reorderedInputMatrixNode1 = model.AddNode<nodes::ReorderDataCodeNode<ValueType>>(inputMatrixNode->output, rowMajorLayout, colMajorLayout);
    auto reorderedInputMatrixNode2 = model.AddNode<nodes::ReorderDataCodeNode<ValueType>>(reorderedInputMatrixNode1->output, colMajorLayout, rowMajorLayout);
    auto reorderedInputMatrixNode3 = model.AddNode<nodes::ReorderDataCodeNode<ValueType>>(reorderedInputMatrixNode2->output, rowMajorLayout, rowMajorLayout);

    std::vector<ValueType> matrixBVals(k * n);
    rowMajorLayout = model::PortMemoryLayout(model::MemoryShape{ k, n }).ReorderedCopy(rowMajor);
    colMajorLayout = model::PortMemoryLayout(model::MemoryShape{ k, n }).ReorderedCopy(colMajor);
    auto matrixBNode = model.AddNode<nodes::ConstantNode<ValueType>>(matrixBVals, rowMajorLayout);
    auto reorderedMatrixBNode1 = model.AddNode<nodes::ReorderDataCodeNode<ValueType>>(matrixBNode->output, rowMajorLayout, rowMajorLayout);
    auto reorderedMatrixBNode2 = model.AddNode<nodes::ReorderDataCodeNode<ValueType>>(reorderedMatrixBNode1->output, rowMajorLayout, colMajorLayout);
    auto reorderedMatrixBNode3 = model.AddNode<nodes::ReorderDataCodeNode<ValueType>>(reorderedMatrixBNode2->output, colMajorLayout, colMajorLayout);

    auto matMatMultNode = model.AddNode<nodes::MatrixMatrixMultiplyNode<ValueType>>(reorderedInputMatrixNode3->output, reorderedMatrixBNode3->output, outputLayout);

    auto map = model::Map(model, { { "inputMatrix", inputMatrixNode } }, { { "output", matMatMultNode->output } });
    auto oldSize = map.GetModel().Size();
#if PRINT_MODELS
    PrintModel(map.GetModel());
#endif

    // Transform model
    passes::OptimizeReorderDataNodesTransformation optReorders;
    map.Transform(optReorders);
    map.Refine();
    auto newSize = map.GetModel().Size();

#if PRINT_MODELS
    PrintModel(map.GetModel());
#endif

    testing::ProcessTest("Testing OptimizeReorderDataNodesTransform4", oldSize == 9 && newSize == 4);
}

void TestOptimizeReorderDataNodesTransformation()
{
    TestOptimizeReorderDataNodesTransformation1();
    TestOptimizeReorderDataNodesTransformation2();
    TestOptimizeReorderDataNodesTransformation3();
    TestOptimizeReorderDataNodesTransformation4();
}
