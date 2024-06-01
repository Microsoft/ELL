////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ObjectiveTest.cpp (model/optimizer_test)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ObjectiveTest.h"
#include "ExampleObjectives.h"
#include "OptimizerTestUtil.h"

#include <testing/include/testing.h>

using namespace ell::model::optimizer;
using namespace ell::testing;

//
// Objective class tests
//
void TestObjectives()
{
    TestSimpleObjective();
}

void TestSimpleObjective()
{
    SimpleObjective objective;
    Cost cost;
    cost["runtime"] = HeuristicCostValue(10);
    ProcessTest("Testing SimpleObjective", objective.Evaluate(cost) == 10);
}
