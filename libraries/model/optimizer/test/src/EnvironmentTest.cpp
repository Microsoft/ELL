////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     EnvironmentTest.cpp (model/optimizer_test)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "EnvironmentTest.h"
#include "Environment.h"

#include <emitters/include/CompilerOptions.h>
#include <emitters/include/IRModuleEmitter.h>
#include <emitters/include/TargetDevice.h>

#include <testing/include/testing.h>

using namespace ell;
using namespace ell::model::optimizer;
using namespace ell::testing;

//
// Environment class tests
//
void TestEnvironment()
{
    // Creating an instance of `IRModuleEmitter` will initialize LLVM so we can retrieve the host device
    emitters::CompilerOptions options;
    emitters::IRModuleEmitter module("testModule", options);

    Environment emptyEnv;
    ProcessTest("Testing default environment", !emptyEnv.HasTargetDevice());

    Environment hostEnv(emitters::GetTargetDevice("host"));
    ProcessTest("Testing host environment", hostEnv.HasTargetDevice());
}
