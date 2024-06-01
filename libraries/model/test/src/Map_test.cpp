////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Map_test.cpp (model_test)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Map_test.h"

#include <common/include/LoadModel.h> // for RegisterNodeTypes

#include <model_testing/include/ModelTestUtilities.h>

#include <data/include/DenseDataVector.h>

#include <model/include/InputNode.h>
#include <model/include/Map.h>
#include <model/include/Model.h>
#include <model/include/OutputNode.h>

#include <nodes/include/ClockNode.h>
#include <nodes/include/ExtremalValueNode.h>
#include <nodes/include/MovingAverageNode.h>
#include <nodes/include/SinkNode.h>
#include <nodes/include/SourceNode.h>

#include <utilities/include/JsonArchiver.h>

#include <testing/include/testing.h>

#include <cassert>
#include <iostream>
#include <sstream>
#include <thread>
#include <tuple>

using namespace ell;

//
// Test code for dynamic map class
//

// Returns a model with 1 3-dimensional double input and 1 2-dimensional double output
void TestMapCreate()
{
    auto model = GetSimpleModel();

    auto inputNodes = model.GetNodesByType<model::InputNode<double>>();
    auto outputNodes = model.GetNodesByType<model::OutputNode<double>>();
    assert(outputNodes.size() == 1);
    auto map = model::Map(model, { { "doubleInput", inputNodes[0] } }, { { "doubleOutput", outputNodes[0]->output } });
}

void TestMapCompute()
{
    auto model = GetSimpleModel();
    auto inputNodes = model.GetNodesByType<model::InputNode<double>>();
    auto outputNodes = model.GetNodesByType<model::OutputNode<double>>();
    assert(outputNodes.size() == 1);
    auto map = model::Map(model, { { "doubleInput", inputNodes[0] } }, { { "doubleOutput", outputNodes[0]->output } });

    assert(inputNodes.size() == 1);

    auto input = std::vector<std::vector<double>>{ { 1.0, 2.0, 3.0 },
                                                   { 4.0, 5.0, 6.0 },
                                                   { 7.0, 8.0, 9.0 },
                                                   { 10.0, 11.0, 12.0 } };
    std::vector<double> resultValues;
    for (const auto& inVec : input)
    {
        map.SetInputValue("doubleInput", inVec);
        resultValues = map.ComputeOutput<double>("doubleOutput");
    }

    testing::ProcessTest("Testing map compute 1", testing::IsEqual(resultValues[0], 8.5) && testing::IsEqual(resultValues[1], 10.5));
}

void TestMapComputeDataVector()
{
    auto model = GetSimpleModel();
    auto inputNodes = model.GetNodesByType<model::InputNode<double>>();
    auto outputNodes = model.GetNodesByType<model::OutputNode<double>>();
    assert(outputNodes.size() == 1);
    auto map = model::Map(model, { { "doubleInput", inputNodes[0] } }, { { "doubleOutput", outputNodes[0]->output } });

    assert(inputNodes.size() == 1);

    auto signal = std::vector<std::vector<double>>{ { 1.0, 2.0, 3.0 },
                                                    { 4.0, 5.0, 6.0 },
                                                    { 7.0, 8.0, 9.0 },
                                                    { 10.0, 11.0, 12.0 } };
    std::vector<double> resultValues;
    for (const auto& sample : signal)
    {
        data::DoubleDataVector sampleVec(sample);
        map.SetInputValue("doubleInput", sampleVec);
        resultValues = map.ComputeOutput<double>("doubleOutput");
    }

    testing::ProcessTest("Testing map compute 2", testing::IsEqual(resultValues[0], 8.5) && testing::IsEqual(resultValues[1], 10.5));
}

void TestMapRefine()
{
    auto model = GetSimpleModel();
    auto inputNodes = model.GetNodesByType<model::InputNode<double>>();
    auto outputNodes = model.GetNodesByType<model::OutputNode<double>>();
    assert(outputNodes.size() == 1);

    auto map1 = model::Map(model, { { "doubleInput", inputNodes[0] } }, { { "doubleOutput", outputNodes[0]->output } });
    auto map2 = model::Map(model, { { "doubleInput", inputNodes[0] } }, { { "doubleOutput", outputNodes[0]->output } });

    map2.Refine();

    auto input = std::vector<std::vector<double>>{ { 1.0, 2.0, 3.0 },
                                                   { 4.0, 5.0, 6.0 },
                                                   { 7.0, 8.0, 9.0 },
                                                   { 10.0, 11.0, 12.0 } };
    std::vector<double> resultValues1;
    std::vector<double> resultValues2;
    for (const auto& inVec : input)
    {
        map1.SetInputValue("doubleInput", inVec);
        map2.SetInputValue("doubleInput", inVec);

        resultValues1 = map1.ComputeOutput<double>("doubleOutput");
        resultValues2 = map2.ComputeOutput<double>("doubleOutput");
    }

    // make sure they're the same
    testing::ProcessTest("Testing refined map compute", testing::IsEqual(resultValues1, resultValues2));
}

void TestMapSerialization(const model::Map& map)
{
    std::stringstream outStream;
    utilities::JsonArchiver archiver(outStream);
    archiver << map;

    std::cout << "Archived map" << std::endl;
    std::cout << outStream.str() << std::endl;

    // Now read it back in
    utilities::SerializationContext context;
    common::RegisterNodeTypes(context);
    common::RegisterMapTypes(context);
    std::stringstream inStream(outStream.str());
    utilities::JsonUnarchiver unarchiver(inStream, context);
    model::Map map2;
    unarchiver >> map2;
}

void TestMapSerialization()
{
    auto model = GetSimpleModel();
    auto inputNodes = model.GetNodesByType<model::InputNode<double>>();
    auto outputNodes = model.GetNodesByType<model::OutputNode<double>>();
    auto map = model::Map(model, { { "doubleInput", inputNodes[0] } }, { { "doubleOutput", outputNodes[0]->output } });

    TestMapSerialization(map);
}

void TestMapClockNode()
{
    constexpr nodes::TimeTickType lagThreshold = 75;
    constexpr nodes::TimeTickType interval = 20;
    std::vector<nodes::TimeTickType> lagValues;
    std::vector<std::vector<double>> outputValues;
    std::vector<std::vector<double>> inputValues;

    model::Model model;
    auto in = model.AddNode<model::InputNode<nodes::TimeTickType>>(1);
    auto clock = model.AddNode<nodes::ClockNode>(in->output, interval, lagThreshold, "LagNotificationCallback", [&lagValues](auto timeLag) {
        std::cout << "LagNotificationCallback: " << timeLag << "\n";
        lagValues.push_back(timeLag);
    });
    auto source = model.AddNode<nodes::SourceNode<double>>(clock->output, 3, "SourceCallback", [&inputValues](auto& input) {
        std::cout << "SourceCallback\n";
        input.assign(3, 42.0);
        inputValues.push_back(input);
        return true;
    });
    auto condition = model.AddNode<nodes::ConstantNode<bool>>(true);
    auto sink = model.AddNode<nodes::SinkNode<double>>(source->output,
                                                       condition->output,
                                                       "SinkCallback",
                                                       [&outputValues](const auto& values) {
                                                           std::cout << "SinkCallback\n";
                                                           outputValues.push_back(values);
                                                       });

    auto map = model::Map(model, { { "clockInput", in } }, { { "sinkOutput", sink->output } });
    TestMapSerialization(map);

    std::vector<std::vector<nodes::TimeTickType>> clockValues = {
        { 0 },
        { interval * 1 + lagThreshold / 2 }, // within threshold
        { interval * 2 }, // on time
        { interval * 3 + lagThreshold }, // late
        { interval * 4 + lagThreshold * 20 }, // really late
        { interval * 5 } // on time
    };

    for (const auto& input : clockValues)
    {
        map.SetInputValue("clockInput", input);
        map.ComputeOutput<double>("sinkOutput");
    }

    testing::ProcessTest("Testing source and sink callbacks", testing::IsEqual(inputValues, outputValues));

    std::vector<nodes::TimeTickType> expectedLagValues = { lagThreshold, lagThreshold * 20 };
    testing::ProcessTest("Testing lag callbacks", testing::IsEqual(lagValues, expectedLagValues));
}
