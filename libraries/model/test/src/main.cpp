////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     main.cpp (model_test)
//  Authors:  Umesh Madan, Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Model tests
//

#include "Map_test.h"
#include "Metadata_test.h"
#include "ModelBuilder_test.h"
#include "ModelOptimizerOptions_test.h"
#include "ModelTransformerTest.h"
#include "Model_test.h"
#include "PortElements_test.h"
#include "Submodel_test.h"

#include <testing/include/testing.h>

#include <utilities/include/Exception.h>

#include <iostream>

using namespace ell;

int main()
{
    try
    {
        // Model tests
        TestStaticModel();
        TestNodeIterator();
        TestReverseNodeIterator();
        TestModelSerialization();
        TestInputRouting();

        TestDeepCopyModel();
        TestShallowCopyModel();
        TestRefineSplitOutputs();
        TestChangeInputForNode();

        // PortElements tests
        TestSlice();
        TestAppend();
        TestParsePortElements();
        TestConvertPortElements();

        // Map tests
        TestMapCreate();
        TestMapCompute();
        TestMapComputeDataVector();
        TestMapRefine();
        TestMapSerialization();
        TestMapClockNode();

        TestCustomRefine();

        // Metadata tests
        TestModelMetadata();

        // ModelBuilder tests

        // Test passthrough to Model::AddNode
        TestTemplateAddNode();

        // Test ModelBuilder::GetAddNodeArgs(string)
        TestInputNodeGetArgumentTypes();
        TestConstantNodeGetArgumentTypes();
        TestOutputNodeGetArgumentTypes();
        TestBinaryOpNodeGetArgumentTypes();

        // Test ModelBuilder::AddNode(string, vector<Variant>) with exact types
        TestVariantAddConstantNodes();
        TestVariantAddInputNodes();
        TestVariantAddOutputNode();
        TestVariantAddBinaryOpNode();

        // Test ModelBuilder::AddNode(string, vector<Variant>) with convertable types
        TestVariantAddInputNodesConvertableArgs();

        // Test ModelBuilder::AddNode(string, vector<string>)
        TestVariantAddInputNodesParsedArgs();

        // ModelTransformer tests
        TestCopySubmodel();
        TestCopySubmodelOnto();
        TestTransformSubmodelOnto();
        TestTransformSubmodelInPlace();

        // Submodel tests
        TestSubmodels();

        // ModelOptimizerOptions tests
        TestModelOptimizerOptions();
    }
    catch (const utilities::Exception& exception)
    {
        std::cerr << "ERROR, got ELL exception. Message: " << exception.GetMessage() << std::endl;
        return 1;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "ERROR, got unhandled exception. Message: " << exception.what() << std::endl;
        return 1;
    }

    if (testing::DidTestFail())
    {
        return 1;
    }

    return 0;
}
