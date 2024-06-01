////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     main.cpp (nodes_test)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "AppendNodesTests.h"
#include "BasicMathNodesTests.h"
#include "NodesTests.h"

#include <testing/include/testing.h>

#include <utilities/include/Exception.h>
#include <utilities/include/Unused.h>

#include <iostream>

using namespace ell;
using namespace ell::testing;

/// Runs all tests
///
int main(int argc, char** argv)
{
    UNUSED(argc);
    UNUSED(argv);
    try
    {
        FailOnException(TestNodes);
        FailOnException(TestAppendNodes);
        FailOnException(TestBasicMathNodes);
    }
    catch (const utilities::Exception& exception)
    {
        std::cerr << "ERROR, got ELL exception. Message: " << exception.GetMessage() << std::endl;
        throw;
    }

    int rc = GetExitCode();
    if (rc != 0)
    {
        std::cerr << "ERROR: a test failed" << std::endl;
    }
    return rc;
}
