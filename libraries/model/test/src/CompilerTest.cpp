////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CompilerTest.cpp (compile_test)
//  Authors:  Umesh Madan, Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CompilerTest.h"

#include <model_testing/include/ModelTestUtilities.h>

#include <model/include/CompilableNode.h>
#include <model/include/IRCompiledMap.h>
#include <model/include/IRMapCompiler.h>
#include <model/include/InputNode.h>
#include <model/include/Map.h>
#include <model/include/Model.h>

#include <nodes/include/AccumulatorNode.h>
#include <nodes/include/ClockNode.h>
#include <nodes/include/ConstantNode.h>
#include <nodes/include/DelayNode.h>
#include <nodes/include/DotProductNode.h>
#include <nodes/include/ForestPredictorNode.h>
#include <nodes/include/L2NormSquaredNode.h>
#include <nodes/include/LinearPredictorNode.h>
#include <nodes/include/MatrixVectorProductNode.h>
#include <nodes/include/ProtoNNPredictorNode.h>
#include <nodes/include/SinkNode.h>
#include <nodes/include/SourceNode.h>
#include <nodes/include/SquaredEuclideanDistanceNode.h>
#include <nodes/include/SumNode.h>

#include <emitters/include/EmitterException.h>
#include <emitters/include/EmitterTypes.h>
#include <emitters/include/IREmitter.h>
#include <emitters/include/IRFunctionEmitter.h>
#include <emitters/include/IRModuleEmitter.h>
#include <emitters/include/ScalarVariable.h>
#include <emitters/include/VectorVariable.h>

#include <predictors/include/LinearPredictor.h>
#include <predictors/include/ProtoNNPredictor.h>

#include <utilities/include/Logger.h>
#include <utilities/include/RandomEngines.h>

#include <testing/include/testing.h>

#include <algorithm>
#include <chrono>
#include <future>
#include <iostream>
#include <memory>
#include <ostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace ell;
using namespace ell::logging;

std::string outputBasePath = "";

void SetOutputPathBase(std::string path)
{
    outputBasePath = path;
}

std::string OutputPath(std::string relPath)
{
    return outputBasePath + relPath;
}

//
// Helper functions for constructing example models/maps
//

model::Map MakeForestMap()
{
    // define some abbreviations
    using SplitAction = predictors::SimpleForestPredictor::SplitAction;
    using SplitRule = predictors::SingleElementThresholdPredictor;
    using EdgePredictorVector = std::vector<predictors::ConstantPredictor>;

    // build a forest
    predictors::SimpleForestPredictor forest;
    auto root = forest.Split(SplitAction{ forest.GetNewRootId(), SplitRule{ 0, 0.3 }, EdgePredictorVector{ -1.0, 1.0 } });
    auto child1 = forest.Split(SplitAction{ forest.GetChildId(root, 0), SplitRule{ 1, 0.6 }, EdgePredictorVector{ -2.0, 2.0 } });
    forest.Split(SplitAction{ forest.GetChildId(child1, 0), SplitRule{ 1, 0.4 }, EdgePredictorVector{ -2.1, 2.1 } });
    forest.Split(SplitAction{ forest.GetChildId(child1, 1), SplitRule{ 1, 0.7 }, EdgePredictorVector{ -2.2, 2.2 } });
    forest.Split(SplitAction{ forest.GetChildId(root, 1), SplitRule{ 2, 0.9 }, EdgePredictorVector{ -4.0, 4.0 } });

    auto root2 = forest.Split(SplitAction{ forest.GetNewRootId(), SplitRule{ 0, 0.2 }, EdgePredictorVector{ -3.0, 3.0 } });
    forest.Split(SplitAction{ forest.GetChildId(root2, 0), SplitRule{ 1, 0.21 }, EdgePredictorVector{ -3.1, 3.1 } });
    forest.Split(SplitAction{ forest.GetChildId(root2, 1), SplitRule{ 1, 0.22 }, EdgePredictorVector{ -3.2, 3.2 } });

    // build the model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto forestNode = model.AddNode<nodes::SimpleForestPredictorNode>(inputNode->output, forest);

    return { model, { { "input", inputNode } }, { { "output", forestNode->output } } };
}

//
// Tests
//
void TestNodeMetadata()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(10);
    inputNode->GetMetadata().SetEntry("window_size", std::string("80"));
    auto outputNode = model.AddNode<model::OutputNode<double>>(inputNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });
    model::MapCompilerOptions settings;
    settings.moduleName = "Model";
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    auto compiledFunction = compiledMap.GetJitter().GetFunction<char*(const char*)>("Model_GetMetadata");
    char* result = compiledFunction("window_size");

    testing::ProcessTest("Test compiled node metadata", testing::IsEqual(std::string(result), std::string("80")));
}

void TestSimpleMap(bool optimize)
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    // auto bufferNode = model.AddNode<model::OutputNode<double>>(inputNode->output);
    auto accumNode = model.AddNode<nodes::AccumulatorNode<double>>(inputNode->output);
    auto accumNode2 = model.AddNode<nodes::AccumulatorNode<double>>(accumNode->output);
    // auto outputNode = model.AddNode<model::OutputNode<double>>(accumNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", accumNode2->output } });
    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = optimize;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);
    testing::ProcessTest("Testing IsValid of original map", testing::IsEqual(compiledMap.IsValid(), true));

    std::vector<double> input({ 4, 5, 6 });
    std::vector<double> output({ 0, 0, 0 });
    std::vector<double> expected({ 12, 15, 18 });

    // AccumulatorNode just keeps accumulating, so we have to make sure we make the same number of calls
    // to Compute, and compiled.Compute.
    auto actual = map.Compute<double>(input);
    auto compiledActual = compiledMap.Compute<double>(input);

    actual = map.Compute<double>(input);
    compiledMap.ComputeMultiple(std::vector<void*>({ (void*)input.data() }), std::vector<void*>({ (void*)output.data() }));
    testing::ProcessTest("Test ComputeDispatch", testing::IsEqual(output, expected));

    // compare output
    std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
    VerifyCompiledOutput(map, compiledMap, signal, " map");
}

void TestSqEuclideanDistanceMap()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    math::RowMatrix<double> m{
        { 1.2, 1.1, 0.8 },
        { 0.6, 0.9, 1.3 },
        { 0.3, 1.0, 0.4 },
        { -.4, 0.2, -.7 }
    };
    auto sqEuclidDistNode = model.AddNode<nodes::SquaredEuclideanDistanceNode<double, math::MatrixLayout::rowMajor>>(inputNode->output, m);
    auto map = model::Map{ model, { { "input", inputNode } }, { { "output", sqEuclidDistNode->output } } };

    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    testing::ProcessTest("Testing IsValid of original map", testing::IsEqual(compiledMap.IsValid(), true));

    // compare output
    std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
    VerifyCompiledOutput(map, compiledMap, signal, " map");
}

void TestProtoNNPredictorMap()
{
    // the values of dim, gamma, and matrices come from the result of running protoNNTrainer with the following command line
    // protoNNTrainer -v --inputDataFilename Train-28x28_sparse.txt -dd 784 -sw 0.29 -sb 0.8 -sz 0.8 -pd 15 -l 10 -mp 5 --outputModelFilename mnist-94.model --evaluationFrequency 1 -plf L4 -ds 0.003921568627451

    size_t dim = 784, projectedDim = 15, numPrototypes = 50, numLabels = 10;
    double gamma = 0.0733256;
    predictors::ProtoNNPredictor protonnPredictor(dim, projectedDim, numPrototypes, numLabels, gamma);

    // projectedDim * dim
    auto W = protonnPredictor.GetProjectionMatrix() = {
#include "TestProtoNNPredictorMap_Projection.inc"
    };

    // projectedDim * numPrototypes
    auto B = protonnPredictor.GetPrototypes() = {
#include "TestProtoNNPredictorMap_Prototypes.inc"
    };

    // numLabels * numPrototypes
    auto Z = protonnPredictor.GetLabelEmbeddings() = {
#include "TestProtoNNPredictorMap_LabelEmbeddings.inc"
    };

    // MNIST training data features
    std::vector<std::vector<double>> features = {
#include "TestProtoNNPredictorMap_features.inc"
    };

    std::vector<std::vector<int>> labels{ { 0, 0, 0, 0, 1, 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 1, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 } };

    testing::IsEqual(protonnPredictor.GetProjectedDimension(), projectedDim);
    testing::IsEqual(protonnPredictor.GetNumPrototypes(), numPrototypes);
    testing::IsEqual(protonnPredictor.GetNumLabels(), numLabels);

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(dim);
    auto protonnPredictorNode = model.AddNode<nodes::ProtoNNPredictorNode>(inputNode->output, protonnPredictor);
    auto outputNode = model.AddNode<model::OutputNode<double>>(protonnPredictorNode->output);
    auto map = model::Map{ model, { { "input", inputNode } }, { { "output", outputNode->output } } };

    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = false;
    settings.compilerSettings.includeDiagnosticInfo = true;
    settings.compilerSettings.inlineOperators = false;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    testing::ProcessTest("Testing IsValid of original map", testing::IsEqual(compiledMap.IsValid(), true));

    for (unsigned i = 0; i < features.size(); ++i)
    {
        auto& input = features[i];
        std::transform(input.begin(), input.end(), input.begin(), [](double d) { return d / 255; });

        const auto& label = labels[i];

        IsEqual(input.size(), dim);

        inputNode->SetInput(input);
        auto computeOutput = model.ComputeOutput(outputNode->output);
        testing::ProcessTest("ProtoNN: one hot indices are incorrect for computed and actual label",
                             IsEqual(std::max_element(label.begin(), label.end()) - label.begin(),
                                     std::max_element(computeOutput.begin(), computeOutput.end()) - computeOutput.begin()));

        map.SetInputValue(0, input);
        auto refinedOutput = map.ComputeOutput<double>(0);
        testing::ProcessTest("ProtoNN: computed and refined output vectors don't match", IsEqual(computeOutput, refinedOutput, 1e-5));

        compiledMap.SetInputValue(0, input);
        auto compiledOutput = compiledMap.ComputeOutput<double>(0);
        testing::ProcessTest("ProtoNN: refined and compiled output vectors don't match", IsEqual(refinedOutput, compiledOutput, 1e-5));
    }
}

void TestCombineOutputMap()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto accumNode = model.AddNode<nodes::AccumulatorNode<double>>(inputNode->output);
    // combine inputNode and accumNode as a bigger vector of size 6 and output that.
    auto outputNode = model.AddNode<model::OutputNode<double>>(model::PortElements<double>{ inputNode->output, accumNode->output });

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });
    model::MapCompilerOptions settings;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    testing::ProcessTest("Testing TestCombineOutputMap IsValid", testing::IsEqual(compiledMap.IsValid(), true));

    std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
    std::vector<double> expected = { 5, 2, 1, 42, 48, 49 };
    std::vector<double> result = VerifyCompiledOutput<double, double>(map, compiledMap, signal, "TestCombineOutputMap");

    // compare final output
    double epsilon = 1e-5;
    bool ok = IsEqual(result, expected, epsilon);
    if (IsVerbose() || !ok)
    {
        std::cout << "result versus expected: " << std::endl;
        std::cout << "  result:   " << result << std::endl;
        std::cout << "  expected: " << expected << std::endl;
        std::cout << "  Largest difference: " << LargestDifference(result, expected) << std::endl;
    }

    testing::ProcessTest("TestCombineOutputMap matches expected result", ok);
}

void TestMultiOutputMap()
{
    // Map with 2 OutputNodes
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto sumNode = model.AddNode<nodes::SumNode<double>>(inputNode->output);
    auto dotNode = model.AddNode<nodes::DotProductNode<double>>(inputNode->output, inputNode->output);
    auto outputNode = model.AddNode<model::OutputNode<double>>(model::PortElements<double>{ sumNode->output });
    auto outputNode2 = model.AddNode<model::OutputNode<double>>(model::PortElements<double>{ dotNode->output });

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output }, { "output2", outputNode2->output } });

    // can't compile multiple maps with outputs yet...
    //model::IRMapCompiler compiler;
    //auto compiledMap = compiler.Compile(map);
    //PrintIR(compiledMap);

    //// compare output
    //std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
    //auto result = VerifyCompiledOutput<double, double>(map, compiledMap, signal, "TestMultiOutputMap");

    testing::ProcessTest("Testing TestMultiOutputMap clone and prune", map.GetModel().Size() == 5);
}

void TestCompiledMapMove()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto accumNode = model.AddNode<nodes::AccumulatorNode<double>>(inputNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", accumNode->output } });
    model::IRMapCompiler compiler1;
    auto compiledMap1 = compiler1.Compile(map);

    PrintIR(compiledMap1);
    testing::ProcessTest("Testing IsValid of original map", testing::IsEqual(compiledMap1.IsValid(), true));

    // compare output
    std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
    VerifyCompiledOutput(map, compiledMap1, signal, " original compiled map");

    auto compiledMap2 = std::move(compiledMap1);
    testing::ProcessTest("Testing IsValid of moved-from map", testing::IsEqual(compiledMap1.IsValid(), false));
    testing::ProcessTest("Testing IsValid of moved-to map", testing::IsEqual(compiledMap2.IsValid(), true));

    // compare output
    VerifyCompiledOutput(map, compiledMap2, signal, " moved compiled map");
}

void TestCompiledMapClone()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto accumNode = model.AddNode<nodes::AccumulatorNode<double>>(inputNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", accumNode->output } });
    model::IRMapCompiler compiler1;
    auto compiledMap1 = compiler1.Compile(map);

    PrintIR(compiledMap1);
    testing::ProcessTest("Testing IsValid of original map", testing::IsEqual(compiledMap1.IsValid(), true));

    // compare output
    std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };
    VerifyCompiledOutput(map, compiledMap1, signal, " original compiled map");

    auto compiledMap2 = compiledMap1.Clone();
    auto map2 = model::Map(model, { { "input", inputNode } }, { { "output", accumNode->output } });
    testing::ProcessTest("Testing IsValid of cloned-from map", testing::IsEqual(compiledMap1.IsValid(), true));
    testing::ProcessTest("Testing IsValid of cloned-to map", testing::IsEqual(compiledMap2.IsValid(), true));

    // compare output
    VerifyCompiledOutput(map2, compiledMap2, signal, " cloned compiled map");
}

void TestCompiledMapParallelClone()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto accumNode = model.AddNode<nodes::AccumulatorNode<double>>(inputNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", accumNode->output } });
    std::vector<std::vector<double>> signal = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 3, 4, 5 }, { 2, 3, 2 }, { 1, 5, 3 }, { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 7, 4, 2 }, { 5, 2, 1 } };

    // get original map output as gold standard
    std::vector<std::vector<double>> expected;
    for (const auto& input : signal)
    {
        map.SetInputValue(0, input);
        expected.push_back(map.ComputeOutput<double>(0));
    }

    model::IRMapCompiler compiler;
    std::vector<std::pair<int, std::unique_ptr<model::IRCompiledMap>>> compiledMaps;
    compiledMaps.push_back({ 0, std::make_unique<model::IRCompiledMap>(compiler.Compile(map)) });
    const int numParallelComputations = 100;
    for (int i = 1; i < numParallelComputations; ++i)
    {
        compiledMaps.push_back({ i, std::make_unique<model::IRCompiledMap>(compiledMaps.front().second->Clone()) });
    }

    std::vector<std::future<bool>> futures;
    for (const auto& mapRef : compiledMaps)
    {
        futures.push_back(std::async(
            std::launch::async,
            [&](int index, model::IRCompiledMap* map) {
                auto engine = utilities::GetRandomEngine("123");
                std::uniform_int_distribution<int> dist(0, 500);
                std::this_thread::sleep_for(std::chrono::milliseconds(dist(engine)));
                VerifyMapOutput(*map, signal, expected, "Parallel map test");
                return true;
            },
            mapRef.first,
            mapRef.second.get()));
    }

    // wait on futures
    for (auto& fut : futures)
    {
        [[maybe_unused]] auto x = fut.get();
    }
}

typedef void (*MapPredictFunction)(void* context, double*, double*);

void TestBinaryVector(bool expanded, bool runJit)
{
    std::vector<double> data = { 5, 10, 15, 20 };
    std::vector<double> data2 = { 4, 4, 4, 4 };
    const int c_InputSize = data.size();
    const std::string modelFunctionName = "TestBinaryVector";
    model::Model model;

    auto inputNode1 = model.AddNode<model::InputNode<double>>(c_InputSize);
    const auto& c1 = nodes::Constant(model, data);
    const auto& c2 = nodes::Constant(model, data2);

    const auto& sum = nodes::Add(c1, inputNode1->output);
    const auto& product = nodes::Multiply(sum, c2);

    model::MapCompilerOptions settings;
    settings.compilerSettings.unrollLoops = expanded;
    settings.mapFunctionName = modelFunctionName;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);

    model::Map map{ model, { { "input", inputNode1 } }, { { "output", product } } };
    model::IRCompiledMap compiledMap = compiler.Compile(map);

    std::vector<double> testInput = { 1, 1, 1, 1 };
    std::vector<double> testOutput(testInput.size());
    PrintIR(compiledMap);
    PrintDiagnostics(compiledMap.GetModule().GetDiagnosticHandler());
    if (runJit)
    {
        auto& jitter = compiledMap.GetJitter();
        auto predict = reinterpret_cast<MapPredictFunction>(jitter.ResolveFunctionAddress(modelFunctionName));
        predict(nullptr, &testInput[0], testOutput.data());
    }
    else
    {
        auto mainFunction = compiledMap.GetModule().BeginMainDebugFunction();
        emitters::IRFunctionCallArguments args(mainFunction);
        auto& emitter = compiledMap.GetModule().GetIREmitter();
        args.Append(emitter.NullPointer(emitter.GetIRBuilder().getInt8Ty()->getPointerTo()));
        args.Append(compiledMap.GetModule().ConstantArray("c_data", testInput));
        auto* pResult = args.AppendOutput(emitters::VariableType::Double, testInput.size());
        mainFunction.Call(modelFunctionName, args);
        mainFunction.PrintForEach("%f,", pResult, testInput.size());
        mainFunction.Return();
        compiledMap.GetModule().EndFunction();
        compiledMap.GetModule().WriteToFile(OutputPath(expanded ? "BinaryVector_E.asm" : "BinaryVector.asm"));
    }
}

void TestBinaryScalar()
{
    std::vector<double> data = { 5 };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(1);
    const auto& c1 = nodes::Constant(model, data);
    const auto& sum = nodes::Add(c1, inputNode->output);

    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    model::Map map{ model, { { "input", inputNode } }, { { "output", sum } } };
    model::IRCompiledMap compiledMap = compiler.Compile(map);
    PrintIR(compiledMap);
}

void TestDotProduct(model::MapCompilerOptions& settings)
{
    std::vector<double> data = { 5, 10, 15, 20 };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(4);
    const auto& c1 = nodes::Constant(model, data);
    const auto& dotProduct = nodes::DotProduct(c1, inputNode->output);

    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    model::Map map{ model, { { "input", inputNode } }, { { "output", dotProduct } } };
    model::IRCompiledMap compiledMap = compiler.Compile(map);
    PrintIR(compiledMap);
}

void TestDotProduct()
{
    model::MapCompilerOptions settings;

    settings.compilerSettings.unrollLoops = false;
    settings.compilerSettings.inlineOperators = true;
    TestDotProduct(settings);

    settings.compilerSettings.unrollLoops = true;
    settings.compilerSettings.inlineOperators = true;
    TestDotProduct(settings);

    settings.compilerSettings.unrollLoops = false;
    settings.compilerSettings.inlineOperators = false;
    TestDotProduct(settings);
}

void TestSimpleSum(bool expanded, bool optimized)
{
    std::vector<double> data = { 5, 10, 15, 20 };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(4);
    const auto& sum = nodes::Sum(inputNode->output);

    model::MapCompilerOptions settings;
    settings.compilerSettings.unrollLoops = expanded;
    settings.compilerSettings.optimize = optimized;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);

    model::Map map{ model, { { "input", inputNode } }, { { "output", sum } } };
    model::IRCompiledMap compiledMap = compiler.Compile(map);
    PrintIR(compiledMap);
    PrintDiagnostics(compiledMap.GetModule().GetDiagnosticHandler());
}

void TestSum(bool expanded, bool optimized)
{
    std::vector<double> data = { 5, 10, 15, 20 };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(4);
    const auto& c1 = nodes::Constant(model, data);
    const auto& product = nodes::Multiply(c1, inputNode->output);
    const auto& sum = nodes::Sum(product);

    model::MapCompilerOptions settings;
    settings.compilerSettings.unrollLoops = expanded;
    settings.compilerSettings.optimize = optimized;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    model::Map map{ model, { { "input", inputNode } }, { { "output", sum } } };
    model::IRCompiledMap compiledMap = compiler.Compile(map);
    PrintIR(compiledMap);

    // Print out any diagnostic messages
    PrintDiagnostics(compiledMap.GetModule().GetDiagnosticHandler());
}

void TestAccumulator(bool expanded)
{
    std::vector<double> data = { 5, 10, 15, 20 };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(4);
    const auto& c1 = nodes::Constant(model, data);
    const auto& product = nodes::Multiply(c1, inputNode->output);
    const auto& accumulate = nodes::Accumulate(product);

    model::MapCompilerOptions settings;
    settings.compilerSettings.unrollLoops = expanded;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    model::Map map{ model, { { "input", inputNode } }, { { "output", accumulate } } };
    model::IRCompiledMap compiledMap = compiler.Compile(map);
    PrintIR(compiledMap);
}

void TestDelay()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(4);
    const auto& delay = nodes::Delay(inputNode->output, 3);

    model::IRMapCompiler compiler;
    model::Map map{ model, { { "input", inputNode } }, { { "output", delay } } };
    model::IRCompiledMap compiledMap = compiler.Compile(map);
    PrintIR(compiledMap);
}

void TestSqrt()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(1);
    const auto& sqrt = nodes::Sqrt(inputNode->output);

    model::IRMapCompiler compiler;
    model::Map map{ model, { { "input", inputNode } }, { { "output", sqrt } } };
    model::IRCompiledMap compiledMap = compiler.Compile(map);
    PrintIR(compiledMap);
}

void TestBinaryPredicate(bool expanded)
{
    std::vector<double> data = { 5 };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(data.size());
    const auto& c1 = nodes::Constant(model, data);
    const auto& eq = nodes::Equal(inputNode->output, c1);

    model::IRMapCompiler compiler;
    model::Map map{ model, { { "input", inputNode } }, { { "output", eq } } };
    model::IRCompiledMap compiledMap = compiler.Compile(map);
    PrintIR(compiledMap);
}

void TestMultiplexer()
{
    std::vector<double> data = { 5, 10 };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(data.size());
    const auto& c1 = nodes::Constant(model, true);
    const auto& selectedOutput = nodes::Multiplexer(inputNode->output, c1);

    model::IRMapCompiler compiler;
    model::Map map{ model, { { "input", inputNode } }, { { "output", selectedOutput } } };
    model::IRCompiledMap compiledMap = compiler.Compile(map);
    PrintIR(compiledMap);
}

void TestSlidingAverage()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(4);
    const auto& dim = nodes::Constant(model, (double)4.0);
    const auto& delay = nodes::Delay(inputNode->output, 2);
    const auto& sum = nodes::Sum(delay);
    const auto& avg = nodes::Divide(sum, dim);

    model::MapCompilerOptions settings;
    settings.mapFunctionName = "TestSlidingAverage";
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);

    model::Map map{ model, { { "input", inputNode } }, { { "output", avg } } };
    model::IRCompiledMap compiledMap = compiler.Compile(map);

    auto& module = compiledMap.GetModule();
    module.DeclarePrintf();
    auto mainFunction = module.BeginMainFunction();
    std::vector<double> data = { 5, 10, 15, 20 };
    auto& emitter = compiledMap.GetModule().GetIREmitter();
    emitters::LLVMValue pContext = emitter.NullPointer(emitter.GetIRBuilder().getInt8Ty()->getPointerTo());
    emitters::LLVMValue pData = module.ConstantArray("c_data", data);
    emitters::LLVMValue pResult = mainFunction.Variable(emitters::VariableType::Double, 1);
    mainFunction.Call("TestSlidingAverage", { pContext, mainFunction.PointerOffset(pData, 0), mainFunction.PointerOffset(pResult, 0) });
    mainFunction.PrintForEach("%f\n", pResult, 1);
    mainFunction.Call("TestSlidingAverage", { pContext, mainFunction.PointerOffset(pData, 0), mainFunction.PointerOffset(pResult, 0) });
    mainFunction.PrintForEach("%f\n", pResult, 1);
    mainFunction.Call("TestSlidingAverage", { pContext, mainFunction.PointerOffset(pData, 0), mainFunction.PointerOffset(pResult, 0) });
    mainFunction.PrintForEach("%f\n", pResult, 1);
    mainFunction.Return();
    module.EndFunction();

    PrintIR(module);
}

void TestDotProductOutput()
{
    model::MapCompilerOptions settings;
    settings.compilerSettings.inlineOperators = false;
    settings.mapFunctionName = "TestDotProduct";
    std::vector<double> data = { 5, 10, 15, 20 };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(4);
    const auto& c1 = nodes::Constant(model, data);
    const auto& dotProduct = nodes::DotProduct(c1, inputNode->output);

    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    model::Map map{ model, { { "input", inputNode } }, { { "output", dotProduct } } };
    model::IRCompiledMap compiledMap = compiler.Compile(map);

    auto mainFunction = compiledMap.GetModule().BeginMainDebugFunction();
    emitters::IRFunctionCallArguments args(mainFunction);
    auto& emitter = compiledMap.GetModule().GetIREmitter();
    args.Append(emitter.NullPointer(emitter.GetIRBuilder().getInt8Ty()->getPointerTo()));
    args.Append(compiledMap.GetModule().ConstantArray("c_data", data));
    auto pResult = args.AppendOutput(emitters::VariableType::Double, 1);
    mainFunction.Call("TestDotProduct", args);
    mainFunction.PrintForEach("%f\n", pResult, 1);
    mainFunction.Return();
    compiledMap.GetModule().EndFunction();

    PrintIR(compiledMap);
}

void TestForest()
{
    auto map = MakeForestMap();

    std::vector<double> data = { 0.2, 0.5, 0.0 };

    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;
    settings.compilerSettings.includeDiagnosticInfo = false;
    settings.mapFunctionName = "TestForest";
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    auto& module = compiledMap.GetModule();
    module.DeclarePrintf();

    auto mainFunction = module.BeginMainFunction();
    emitters::LLVMValue pData = module.ConstantArray("c_data", data);
    emitters::LLVMValue pResult = nullptr;
    auto args = module.GetFunction("TestForest")->args();
    emitters::IRValueList callArgs;
    callArgs.push_back(mainFunction.PointerOffset(pData, 0));
    size_t i = 0;
    for (auto& arg : args)
    {
        (void)arg; // stifle compiler warning
        if (i > 0)
        {
            emitters::LLVMValue pArg = nullptr;
            if (pResult == nullptr)
            {
                pArg = mainFunction.Variable(emitters::VariableType::Double, 1);
                pResult = pArg;
            }
            else
            {
                pArg = mainFunction.Variable(emitters::VariableType::Int32, 1);
            }
            callArgs.push_back(mainFunction.PointerOffset(pArg, 0));
        }
        ++i;
    }

    //mainFunction.Call("TestForest", { mainFunction.PointerOffset(pData, 0), mainFunction.PointerOffset(pResult, 0) });
    mainFunction.Print("Calling TestForest\n");
    mainFunction.Call("TestForest", callArgs);
    mainFunction.Print("Done Calling TestForest\n");

    mainFunction.PrintForEach("%f\n", pResult, 1);
    mainFunction.Return();
    mainFunction.Verify();

    PrintIR(compiledMap);
    compiledMap.GetModule().WriteToFile(OutputPath("forest_map.ll"));
}

extern "C" {
// Callbacks used by compiled map
bool TestMulti_DataCallback1(void* context, double* input, int size)
{
    Log() << "Data callback 1" << EOL;
    const std::vector<double> input1{ 1, 3, 5, 7, 9, 11, 13 };
    testing::ProcessTest("Sink callback size is correct", size == static_cast<int>(input1.size()));
    std::copy(input1.begin(), input1.end(), input);
    return true;
}
bool TestMulti_DataCallback2(void* context, double* input, int size)
{
    Log() << "Data callback 2" << EOL;
    const std::vector<double> input2{ 42 };
    testing::ProcessTest("Sink callback size is correct", size == static_cast<int>(input2.size()));
    std::copy(input2.begin(), input2.end(), input);
    return true;
}

void TestMulti_ResultsCallback_Scalar(void* context, double result, int size)
{
    Log() << "Results callback (scalar): " << result << EOL;
}

void TestMulti_ResultsCallback_Vector(void* context, double* result, int size)
{
    Log() << "Results callback (vector): " << result[0] << EOL;
}

void TestMulti_LagNotificationCallback(void* context, double lag)
{
    Log() << "Lag callback:" << lag << EOL;
}
}

// Ensure that LLVM jit can find these symbols
TESTING_FORCE_DEFINE_SYMBOL(TestMulti_DataCallback1, bool, void*, double*, int);
TESTING_FORCE_DEFINE_SYMBOL(TestMulti_DataCallback2, bool, void*, double*, int);
TESTING_FORCE_DEFINE_SYMBOL(TestMulti_ResultsCallback_Scalar, void, void*, double, int);
TESTING_FORCE_DEFINE_SYMBOL(TestMulti_ResultsCallback_Vector, void, void*, double*, int);
TESTING_FORCE_DEFINE_SYMBOL(TestMulti_LagNotificationCallback, void, void*, double);

void TestMultiSourceSinkMap(bool expanded, bool optimized)
{
    // Create the map
    constexpr nodes::TimeTickType lagThreshold = 200;
    constexpr nodes::TimeTickType interval = 40;
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<nodes::TimeTickType>>(1 /*currentTime*/);
    auto clockNode = model.AddNode<nodes::ClockNode>(inputNode->output, interval, lagThreshold, "LagNotificationCallback");
    auto sourceNode1 = model.AddNode<nodes::SourceNode<double>>(clockNode->output, 7, "DataCallback1", [](auto& v) { return TestMulti_DataCallback1(nullptr, &v[0], 7); });
    auto sourceNode2 = model.AddNode<nodes::SourceNode<double>>(clockNode->output, 1, "DataCallback2", [](auto& v) { return TestMulti_DataCallback2(nullptr, &v[0], 1); });
    auto sumNode = model.AddNode<nodes::SumNode<double>>(sourceNode1->output);
    auto minusNode = model.AddNode<nodes::BinaryOperationNode<double>>(sumNode->output,
                                                                       sourceNode2->output,
                                                                       nodes::BinaryOperationType::subtract);
    auto conditionNode = model.AddNode<nodes::ConstantNode<bool>>(true);
    auto sinkNode1 = model.AddNode<nodes::SinkNode<double>>(sumNode->output, conditionNode->output, "ResultsCallback_Scalar");
    auto sinkNode2 = model.AddNode<nodes::SinkNode<double>>(model::PortElements<double>{ minusNode->output, sumNode->output },
                                                            conditionNode->output,
                                                            "ResultsCallback_Vector");

    // compiled maps require a single output, so we concatenate the ports for the sink nodes
    auto outputNode = model.AddNode<model::OutputNode<double>>(model::PortElements<double>{ sinkNode1->output, sinkNode2->output });
    auto map = model::Map(model, { { "time", inputNode } }, { { "output", outputNode->output } });

    // Compile the map
    model::MapCompilerOptions settings;
    settings.moduleName = "TestMulti";
    settings.compilerSettings.optimize = optimized;
    settings.compilerSettings.unrollLoops = expanded;

    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    bool found = false;
    for (std::string name : compiler.GetModule().GetCallbackFunctionNames())
    {
        if (name == "TestMulti_LagNotificationCallback")
        {
            found = true;
        }
    }
    testing::ProcessTest("Testing GetCallbackFunctionNames", found);

    // Compare output
    std::vector<std::vector<nodes::TimeTickType>> signal = {
        { 0 },
        { interval * 1 + lagThreshold / 2 }, // within threshold
        { interval * 2 }, // on time
        { interval * 3 + lagThreshold }, // late
        { interval * 4 + lagThreshold * 20 }, // really late
        { interval * 5 } // on time
    };

    VerifyCompiledOutput(map, compiledMap, signal, " multi-sink and source map");
}

void TestMultiSourceSinkMap()
{
    TestMultiSourceSinkMap(true, true);
    TestMultiSourceSinkMap(true, false);
    TestMultiSourceSinkMap(false, true);
    TestMultiSourceSinkMap(false, false);
}
