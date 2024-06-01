////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     DSPNodesTest_main.cpp (nodes_test)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DSPCodeNodesTests.h"
#include "DSPNodesTests.h"

#include <testing/include/testing.h>

#include <utilities/include/Exception.h>
#include <utilities/include/Files.h>
#include <utilities/include/Unused.h>

#include <iostream>
#include <string>

using namespace ell;

/// Runs all tests
///
int main(int argc, char** argv)
{
    UNUSED(argc);
    std::string path = utilities::GetDirectoryPath(argv[0]);
    std::cerr << "main argv[0]=" << path << "\n";
    try
    {
        TestDSPCodeNodes(path);
        TestDSPNodes(path);
    }
    catch (const utilities::Exception& exception)
    {
        std::cerr << "ERROR, got ELL exception. Message: " << exception.GetMessage() << std::endl;
        throw;
    }

    if (testing::DidTestFail())
    {
        return 1;
    }

    return 0;
}
