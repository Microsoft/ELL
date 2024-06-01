////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     BasicMathNodesTests.cpp (nodes_test)
//  Authors:  Chuck Jacobs, Byron Changuion, Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "BasicMathNodesTests.h"
#include "NodesTestData.h"
#include "NodesTestUtilities.h"

#include <nodes/include/BinaryOperationNode.h>
#include <nodes/include/BroadcastFunctionNode.h>
#include <nodes/include/BroadcastOperationNodes.h>
#include <nodes/include/ConstantNode.h>
#include <nodes/include/UnaryOperationNode.h>

#include <math/include/MathConstants.h>
#include <math/include/Matrix.h>
#include <math/include/MatrixOperations.h>
#include <math/include/Tensor.h>
#include <math/include/TensorOperations.h>

#include <model/include/InputNode.h>
#include <model/include/Model.h>
#include <model/include/Node.h>
#include <model/include/OutputNode.h>

#include <model_testing/include/ModelTestUtilities.h>

#include <testing/include/testing.h>

#include <utilities/include/RandomEngines.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

using namespace ell;
using namespace nodes;
using namespace value;
using namespace std::string_literals;

//
// Helpers
//
namespace
{
template <typename ValueType>
std::ostream& operator<<(std::ostream& os, const std::vector<ValueType>& vec)
{
    os << "[";
    for (auto x : vec)
    {
        os << x << " ";
    }
    os << "]";
    return os;
}
} // namespace

void TestBasicMathNodes()
{
    TestBinaryOperationNodeCompute();
    TestBinaryOperationNodeCompute2();
    TestUnaryOperationNodeCompute();
    TestLogicalUnaryOperationNodeCompute();

    TestBroadcastUnaryOperationNodeCompute();

    TestBroadcastBinaryOperationNodeComputeFull();
    TestBroadcastBinaryOperationNodeComputeAdd();
    TestBroadcastBinaryOperationNodeComputeSubtract();
    TestBroadcastBinaryOperationNodeComputeWithOrdering();
    TestBroadcastBinaryOperationNodeComputeWithLayout();
    TestBroadcastBinaryOperationNodeComputeWithBadLayout();
    TestBroadcastBinaryOperationNodeComputeDifferentBroadcastDimensions();
    TestBroadcastTernaryOperationNodeComputeFMA();

    TestBroadcastLinearFunctionNodeCompute();
}

void TestUnaryOperationNodeCompute(UnaryOperationType op, double (*expectedTransform)(double))
{
    ComputeContext context("TestUnaryOperationNodeCompute<double>");
    ContextGuard<> guard(context);

    std::vector<std::vector<double>> data = { { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 } };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(data[0].size());
    auto outputNode = model.AddNode<nodes::UnaryOperationNode<double>>(inputNode->output, op);

    for (size_t index = 0; index < data.size(); ++index)
    {
        auto inputValue = data[index];

        inputNode->SetInput(inputValue);
        auto outputVec = model.ComputeOutput(outputNode->output);

        for (size_t d = 0; d < inputValue.size(); ++d)
        {
            auto expectedOutput = expectedTransform(inputValue[d]);
            testing::ProcessTest("Testing UnaryOperationNode<double> compute for " + ToString(op),
                                 testing::IsEqual(outputVec[d], expectedOutput));
        }
    }
}

void TestUnaryOperationNodeCompute()
{
    TestUnaryOperationNodeCompute(UnaryOperationType::abs, std::abs);
    TestUnaryOperationNodeCompute(UnaryOperationType::cos, std::cos);
    TestUnaryOperationNodeCompute(UnaryOperationType::exp, std::exp);
    TestUnaryOperationNodeCompute(UnaryOperationType::log, std::log);
    TestUnaryOperationNodeCompute(UnaryOperationType::log10, std::log10);
    TestUnaryOperationNodeCompute(UnaryOperationType::max, [](double d) { return 10.0; }); // see data above
    TestUnaryOperationNodeCompute(UnaryOperationType::min, [](double d) { return 0.0; }); // see data above
    TestUnaryOperationNodeCompute(UnaryOperationType::sin, std::sin);
    TestUnaryOperationNodeCompute(UnaryOperationType::sqrt, std::sqrt);
    TestUnaryOperationNodeCompute(UnaryOperationType::square, [](double d) { return d * d; });
    TestUnaryOperationNodeCompute(UnaryOperationType::tanh, std::tanh);
}

void TestLogicalUnaryOperationNodeCompute(UnaryOperationType op, bool (*expectedTransform)(bool))
{
    ComputeContext context("TestUnaryOperationNodeCompute<bool>");
    ContextGuard<> guard(context);

    std::vector<std::vector<bool>> data = { { false }, { true } };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<bool>>(data[0].size());
    auto outputNode = model.AddNode<nodes::UnaryOperationNode<bool>>(inputNode->output, op);

    for (size_t index = 0; index < data.size(); ++index)
    {
        auto inputValue = data[index];

        inputNode->SetInput(inputValue);
        auto outputVec = model.ComputeOutput(outputNode->output);

        for (size_t d = 0; d < inputValue.size(); ++d)
        {
            auto expectedOutput = expectedTransform(inputValue[d]);
            testing::ProcessTest("Testing UnaryOperationNode<bool> compute for " + ToString(op),
                                 testing::IsEqual(outputVec[d], expectedOutput));
        }
    }
}

void TestLogicalUnaryOperationNodeCompute()
{
    TestLogicalUnaryOperationNodeCompute(UnaryOperationType::logicalNot, [](bool b) { return !b; });
}

void TestBinaryOperationNodeCompute()
{
    std::vector<std::vector<double>> data = { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }, { 10 } };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(data[0].size());
    auto outputNode = model.AddNode<nodes::BinaryOperationNode<double>>(inputNode->output, inputNode->output, BinaryOperationType::add);

    for (size_t index = 0; index < data.size(); ++index)
    {
        auto inputValue = data[index];

        inputNode->SetInput(inputValue);
        std::vector<double> outputVec = model.ComputeOutput(outputNode->output);

        for (size_t d = 0; d < inputValue.size(); ++d)
        {
            auto expectedOutput = 2 * inputValue[d];
            testing::ProcessTest("Testing BinaryOperationNode compute", testing::IsEqual(outputVec[d], expectedOutput));
        }
    }
}

void TestBinaryOperationNodeCompute2()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 2;
    int numChannels = 2;
    int padding = 1;

    model::PortMemoryLayout input1Shape(model::MemoryShape{ numRows, numColumns, numChannels }, model::MemoryShape{ padding, padding, 0 });
    model::PortMemoryLayout input2Shape(model::MemoryShape{ numRows, numColumns, numChannels });
    model::PortMemoryLayout outputShape(model::MemoryShape{ numRows, numColumns, numChannels });

    auto input1Node = model.AddNode<model::InputNode<double>>(input1Shape.GetMemorySize());
    auto constantNode = model.AddNode<nodes::ConstantNode<double>>(std::vector<double>{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 });
    auto outputNode = model.AddNode<nodes::BinaryOperationNode<double>>(input1Node->output, input1Shape, constantNode->output, input2Shape, outputShape, BinaryOperationType::add, 0);
    auto map = model::Map(model, { { "input", input1Node } }, { { "output", outputNode->output } });

    std::vector<double> input{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    std::vector<double> expected{ 2, 4, 6, 8, 10, 12, 14, 16 };

    auto result = map.Compute<double>(input);
    testing::ProcessTest("TestBinaryOperationNodeCompute2", testing::IsEqual(result, expected));
}

void TestBroadcastLinearFunctionNodeCompute()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 3;
    int numChannels = 4;

    model::PortMemoryLayout input1Layout({ numRows, numColumns, numChannels });
    model::PortMemoryLayout input2Layout({ numColumns });
    model::PortMemoryLayout input3Layout({ numColumns });
    model::PortMemoryLayout outputLayout({ numRows, numColumns, numChannels });

    // clang-format off
    std::vector<double> input1Vals{ 1, 2, 1, 2,    1, 2, 1, 2,    1, 2, 1, 2,
                                    3, 4, 3, 4,    3, 4, 3, 4,    3, 4, 3, 4 };
    std::vector<double> input2Vals{ 2, 4, 6 };
    // broadcasts to              { 2, 2, 2, 2,    4, 4, 4, 4,    6, 6, 6, 6,
    //                              2, 2, 2, 2,    4, 4, 4, 4,    6, 6, 6, 6 }
    std::vector<double> input3Vals{ 1, 2, 3 };
    // broadcasts to              { 1, 1, 1, 1,    2, 2, 2, 2     3, 3, 3, 3,
    //                              1, 1, 1, 1,    2, 2, 2, 2,    3, 3, 3, 3 }
    std::vector<double>   expected{ 3, 5, 3, 5,    6, 10, 6, 10,     9, 15, 9, 15,
                                    7, 9, 7, 9,    14, 18, 14, 18,   21, 27, 21, 27};
    // clang-format on
    auto input1Node = model.AddNode<model::InputNode<double>>(input1Layout);
    auto input2Node = model.AddNode<nodes::ConstantNode<double>>(input2Vals, input2Layout);
    auto input3Node = model.AddNode<nodes::ConstantNode<double>>(input3Vals, input3Layout);

    // y = x*a + b (in1*in2 + in3)
    auto outputNode = model.AddNode<nodes::BroadcastLinearFunctionNode<double>>(input1Node->output,
                                                                                input1Layout,
                                                                                input2Node->output,
                                                                                input3Node->output,
                                                                                1,
                                                                                outputLayout);
    auto map = model::Map(model, { { "input", input1Node } }, { { "output", outputNode->output } });

    auto result = map.Compute<double>(input1Vals);
    testing::ProcessTest("TestBroadcastLinearFunctionNodeCompute", testing::IsEqual(result, expected));
}

void TestBroadcastUnaryOperationNodeCompute()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 3;
    int numChannels = 4;

    model::PortMemoryLayout inputLayout({ numRows, numColumns, numChannels });

    // clang-format off
    std::vector<double> inputVals{ 1, -2, -1, 2,    -1, -2, 1, 2,    1, -2, 1, 2,        3, -4, 3, -4,    3, -4, 3, 4,    3, -4, 3, 4 };
    std::vector<double>  expected{ 1,  2,  1, 2,     1,  2, 1, 2,    1,  2, 1, 2,        3,  4, 3,  4,    3,  4, 3, 4,    3,  4, 3, 4 };
    // clang-format on

    auto inputNode = model.AddNode<model::InputNode<double>>(inputLayout);
    auto op = UnaryOperationType::abs;
    auto outputNode = model.AddNode<nodes::BroadcastUnaryOperationNode<double>>(inputNode->output, op);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });
    TestWithSerialization(map, "BroadcastUnaryOpNodeAbs", [&](model::Map& map, int iter) {
        auto computed = map.Compute<double>(inputVals);
        testing::ProcessTest("TestBroadcastUnaryOperationNodeComputeAbs_"s + std::to_string(iter), testing::IsEqual(computed, expected));
    });
}

void TestBroadcastBinaryOperationNodeComputeFull()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 3;
    int numChannels = 4;

    model::PortMemoryLayout input1Layout({ numRows, numColumns, numChannels });
    model::PortMemoryLayout input2Layout({ numRows, numColumns, numChannels });

    // clang-format off
    std::vector<double> input1Vals{ 1, 2, 1, 2,    1, 2, 1, 2,    1, 2, 1, 2,        3, 4, 3, 4,    3, 4, 3, 4,    3, 4, 3, 4 };
    std::vector<double> input2Vals{ 2, 2, 2, 2,    4, 4, 4, 4,    6, 6, 6, 6,        2, 2, 2, 2,    4, 4, 4, 4,    6, 6, 6, 6 };
    std::vector<double>   expected{ 3, 4, 3, 4,    5, 6, 5, 6,    7, 8, 7, 8,        5, 6, 5, 6,    7, 8, 7, 8,    9, 10, 9, 10 };
    // clang-format on

    auto input1Node = model.AddNode<model::InputNode<double>>(input1Layout);
    auto input2Node = model.AddNode<nodes::ConstantNode<double>>(input2Vals, input2Layout);
    auto op = BinaryOperationType::add;
    auto outputNode = model.AddNode<nodes::BroadcastBinaryOperationNode<double>>(input1Node->output,
                                                                                 input2Node->output,
                                                                                 op);
    auto map = model::Map(model, { { "input", input1Node } }, { { "output", outputNode->output } });
    TestWithSerialization(map, "BroadcastBinaryOpNodeFull", [&](model::Map& map, int iter) {
        auto computed = map.Compute<double>(input1Vals);
        testing::ProcessTest("TestBroadcastBinaryOperationNodeComputeFull_"s + std::to_string(iter), testing::IsEqual(computed, expected));
    });
}

void TestBroadcastBinaryOperationNodeComputeAdd()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 3;
    int numChannels = 4;

    model::PortMemoryLayout input1Layout({ numRows, numColumns, numChannels });
    model::PortMemoryLayout input2Layout({ 1, numColumns, 1 });

    // clang-format off
    std::vector<double> input1Vals{ 1, 2, 1, 2,    1, 2, 1, 2,    1, 2, 1, 2,        3, 4, 3, 4,    3, 4, 3, 4,    3, 4, 3, 4 };
    std::vector<double> input2Vals{ 2, 4, 6 };
    // broadcasts to:             { 2, 2, 2, 2,    4, 4, 4, 4,    6, 6, 6, 6,        2, 2, 2, 2,    4, 4, 4, 4,    6, 6, 6, 6 }
    std::vector<double>   expected{ 3, 4, 3, 4,    5, 6, 5, 6,    7, 8, 7, 8,        5, 6, 5, 6,    7, 8, 7, 8,    9, 10, 9, 10 };
    // clang-format on

    auto input1Node = model.AddNode<model::InputNode<double>>(input1Layout);
    auto input2Node = model.AddNode<nodes::ConstantNode<double>>(input2Vals, input2Layout);
    auto op = BinaryOperationType::add;
    auto outputNode = model.AddNode<nodes::BroadcastBinaryOperationNode<double>>(input1Node->output,
                                                                                 input2Node->output,
                                                                                 op);
    auto map = model::Map(model, { { "input", input1Node } }, { { "output", outputNode->output } });
    TestWithSerialization(map, "BroadcastBinaryOpNodeAdd", [&](model::Map& map, int iter) {
        auto computed = map.Compute<double>(input1Vals);
        testing::ProcessTest("TestBroadcastBinaryOperationNodeComputeAdd_"s + std::to_string(iter), testing::IsEqual(computed, expected));
    });
}

void TestBroadcastBinaryOperationNodeComputeSubtract()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 3;
    int numChannels = 4;

    model::PortMemoryLayout input1Layout({ numRows, numColumns, numChannels });
    model::PortMemoryLayout input2Layout({ 1, numColumns, 1 });

    // clang-format off
    std::vector<double> input1Vals{ 1, 2, 1, 2,      1, 2, 1, 2,      1, 2, 1, 2,            3, 4, 3, 4,    3, 4, 3, 4,    3, 4, 3, 4 };
    std::vector<double> input2Vals{ 2, 4, 6 };
    // broadcasts to:             { 2, 2, 2, 2,      4, 4, 4, 4,      6, 6, 6, 6,            2, 2, 2, 2,    4, 4, 4, 4,    6, 6, 6, 6 }
    std::vector<double>   expected{ -1, 0, -1, 0,    -3, -2, -3, -2,    -5, -4, -5, -4,      1, 2, 1, 2,   -1, 0, -1, 0,  -3, -2, -3, -2 };
    // clang-format on

    auto input1Node = model.AddNode<model::InputNode<double>>(input1Layout);
    auto input2Node = model.AddNode<nodes::ConstantNode<double>>(input2Vals, input2Layout);
    auto op = BinaryOperationType::subtract;
    auto outputNode = model.AddNode<nodes::BroadcastBinaryOperationNode<double>>(input1Node->output,
                                                                                 input2Node->output,
                                                                                 op);
    auto map = model::Map(model, { { "input", input1Node } }, { { "output", outputNode->output } });

    TestWithSerialization(map, "BroadcastBinaryOpNodeSubtract", [&](model::Map& map, int iter) {
        auto computed = map.Compute<double>(input1Vals);
        testing::ProcessTest("TestBroadcastBinaryOperationNodeComputeSubtract_"s + std::to_string(iter), testing::IsEqual(computed, expected));
    });
}

void TestBroadcastBinaryOperationNodeComputeWithOrdering()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 3;

    model::PortMemoryLayout input1Layout({ numRows, numColumns });
    model::PortMemoryLayout input2Layout = input1Layout.ReorderedCopy({ 1, 0 });

    // clang-format off
    std::vector<double> input1Vals{ 1, 2, 3,
                                    4, 5, 6 };
    // std::vector<double> input2Vals{ 10, 20, 30,
    //                                 40, 50, 60 };
    // in column-major order:
    std::vector<double> input2Vals{ 10, 40,
                                    20, 50,
                                    30, 60 };
    std::vector<double>   expected{ 11, 22, 33,
                                    44, 55, 66 };
    // clang-format on

    auto input1Node = model.AddNode<model::InputNode<double>>(input1Layout);
    auto input2Node = model.AddNode<nodes::ConstantNode<double>>(input2Vals, input2Layout);
    auto op = BinaryOperationType::add;
    auto outputNode = model.AddNode<nodes::BroadcastBinaryOperationNode<double>>(input1Node->output,
                                                                                 input2Node->output,
                                                                                 op);
    auto map = model::Map(model, { { "input", input1Node } }, { { "output", outputNode->output } });

    TestWithSerialization(map, "BroadcastBinaryOpNodeWithOrdering", [&](model::Map& map, int iter) {
        auto computed = map.Compute<double>(input1Vals);
        testing::ProcessTest("TestBroadcastBinaryOperationNodeComputeWithOrdering_"s + std::to_string(iter), testing::IsEqual(computed, expected));
    });
}

void TestBroadcastBinaryOperationNodeComputeWithLayout()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 3;

    model::PortMemoryLayout input1Layout({ numRows, numColumns });
    model::PortMemoryLayout input2Layout({ numRows, numColumns });
    model::PortMemoryLayout outputLayout = input1Layout.ReorderedCopy({ 1, 0 });

    // clang-format off
    std::vector<double> input1Vals{ 1, 2, 3,
                                    4, 5, 6 };
    std::vector<double> input2Vals{ 10, 20, 30,
                                    40, 50, 60 };
    // std::vector<double>   expected{ 11, 22, 33,
    //                                 44, 55, 66 };
    // in column-major order:
    std::vector<double>   expected{ 11, 44,
                                    22, 55,
                                    33, 66 };
    // clang-format on

    auto input1Node = model.AddNode<model::InputNode<double>>(input1Layout);
    auto input2Node = model.AddNode<nodes::ConstantNode<double>>(input2Vals, input2Layout);
    auto op = BinaryOperationType::add;
    auto outputNode = model.AddNode<nodes::BroadcastBinaryOperationNode<double>>(input1Node->output,
                                                                                 input2Node->output,
                                                                                 outputLayout,
                                                                                 op);
    auto map = model::Map(model, { { "input", input1Node } }, { { "output", outputNode->output } });

    TestWithSerialization(map, "BroadcastBinaryOpNodeWithLayout", [&](model::Map& map, int iter) {
        auto computed = map.Compute<double>(input1Vals);
        testing::ProcessTest("TestBroadcastBinaryOperationNodeComputeWithLayout_"s + std::to_string(iter), testing::IsEqual(computed, expected));
    });
}

void TestBroadcastBinaryOperationNodeComputeWithBadLayout()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 3;

    model::PortMemoryLayout input1Layout({ numRows, numColumns });
    model::PortMemoryLayout input2Layout({ numRows, numColumns });
    model::PortMemoryLayout outputLayout({ numColumns, numColumns });

    // clang-format off
    std::vector<double> input1Vals{ 1, 2, 3,
                                    4, 5, 6 };
    std::vector<double> input2Vals{ 10, 20, 30,
                                    40, 50, 60 };
    // clang-format on

    auto input1Node = model.AddNode<model::InputNode<double>>(input1Layout);
    auto input2Node = model.AddNode<nodes::ConstantNode<double>>(input2Vals, input2Layout);
    auto op = BinaryOperationType::add;

    bool threw = false;
    try
    {
        [[maybe_unused]] auto outputNode = model.AddNode<nodes::BroadcastBinaryOperationNode<double>>(input1Node->output,
                                                                                                      input2Node->output,
                                                                                                      outputLayout,
                                                                                                      op);
    }
    catch (const utilities::InputException&)
    {
        threw = true;
    }

    testing::ProcessTest("TestBroadcastBinaryOperationNodeComputeWithBadLayout", threw);
}

void TestBroadcastBinaryOperationNodeComputeDifferentBroadcastDimensions()
{
    model::Model model;
    int numRows = 2;
    int numColumns = 3;

    model::PortMemoryLayout input1Layout({ numRows, 1 });
    model::PortMemoryLayout input2Layout({ 1, numColumns });
    model::PortMemoryLayout outputLayout({ numRows, numColumns });

    // clang-format off
    std::vector<double> input1Vals{ 1,
                                    2 };
    // broadcasts to:             { 1, 1, 1,
    //                              2, 2, 2 }
    std::vector<double> input2Vals{ 2, 4, 6 };
    // broadcasts to:             { 2, 4, 6,
    //                              2, 4, 6 }
    std::vector<double>   expected{ 3, 5, 7,
                                    4, 6, 8 };
    // clang-format on

    auto input1Node = model.AddNode<model::InputNode<double>>(input1Layout);
    auto input2Node = model.AddNode<nodes::ConstantNode<double>>(input2Vals, input2Layout);
    auto op = BinaryOperationType::add;
    auto outputNode = model.AddNode<nodes::BroadcastBinaryOperationNode<double>>(input1Node->output,
                                                                                 input2Node->output,
                                                                                 op);

    testing::ProcessTest("Testing BroadcastBinaryOperationNode output layout", outputLayout == outputNode->output.GetMemoryLayout());
    auto map = model::Map(model, { { "input", input1Node } }, { { "output", outputNode->output } });
    TestWithSerialization(map, "BroadcastBinaryOpNodeDifferentBroadcastDimensions", [&](model::Map& map, int iter) {
        auto computed = map.Compute<double>(input1Vals);
        testing::ProcessTest("TestBroadcastBinaryOperationNodeComputeDifferentBroadcastDimensions_"s + std::to_string(iter), testing::IsEqual(computed, expected));
    });
}

void TestBroadcastTernaryOperationNodeComputeFMA()
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
    std::vector<double> input1Vals{ 1, 2, 1, 2,    1, 2, 1, 2,    1, 2, 1, 2,
                                    3, 4, 3, 4,    3, 4, 3, 4,    3, 4, 3, 4 };
    std::vector<double> input2Vals{ 2, 4, 6 };
    // broadcasts to              { 2, 2, 2, 2,    4, 4, 4, 4,    6, 6, 6, 6,
    //                              2, 2, 2, 2,    4, 4, 4, 4,    6, 6, 6, 6 }
    std::vector<double> input3Vals{ 1, 2, 3 };
    // broadcasts to              { 1, 1, 1, 1,    2, 2, 2, 2     3, 3, 3, 3,
    //                              1, 1, 1, 1,    2, 2, 2, 2,    3, 3, 3, 3 }
    std::vector<double>   expected{ 3, 5, 3, 5,    6, 10, 6, 10,     9, 15, 9, 15,
                                    7, 9, 7, 9,    14, 18, 14, 18,   21, 27, 21, 27};
    // clang-format on
    auto input1Node = model.AddNode<model::InputNode<double>>(input1Layout);
    auto input2Node = model.AddNode<nodes::ConstantNode<double>>(input2Vals, input2Layout);
    auto input3Node = model.AddNode<nodes::ConstantNode<double>>(input3Vals, input3Layout);

    // y = x*a + b (in1*in2 + in3)
    auto op = TernaryOperationType::fma;
    auto outputNode = model.AddNode<nodes::BroadcastTernaryOperationNode<double>>(input1Node->output,
                                                                                  input2Node->output,
                                                                                  input3Node->output,
                                                                                  op);
    auto map = model::Map(model, { { "input", input1Node } }, { { "output", outputNode->output } });
    TestWithSerialization(map, "BroadcasTernaryOperationNodeComputeFMA", [&](model::Map& map, int iter) {
        auto computed = map.Compute<double>(input1Vals);
        testing::ProcessTest("TestBroadcastTernaryOperationNodeComputeFMA_"s + std::to_string(iter), testing::IsEqual(computed, expected));
    });
}
