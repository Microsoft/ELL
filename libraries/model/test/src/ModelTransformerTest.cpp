////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ModelTransformerTest.cpp (model_test)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ModelTransformerTest.h"

#include <model_testing/include/ModelTestUtilities.h>

#include <model/include/InputNode.h>
#include <model/include/Model.h>
#include <model/include/OutputNode.h>

#include <nodes/include/BinaryOperationNode.h>
#include <nodes/include/UnaryOperationNode.h>

#include <testing/include/testing.h>

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ell;
using namespace ell::model;
using namespace ell::nodes;
using namespace ell::testing;

// prototypes of sub-tests
void TestCopySubmodel_Full(const Model& model);
void TestCopySubmodel_Prefix(const Model& model);
void TestCopySubmodel_DoublePrefix(const Model& model);

void TestCopySubmodelOnto_InPlace();
void TestCopySubmodelOnto_OutOfPlace();
void TestCopySubmodelOnto_PrefixInPlace();
void TestCopySubmodelOnto_PrefixOutOfPlace();
template <bool useDestModel>
void TestCopySubmodelOnto_MidsectionInPlace();
template <bool useDestModel>
void TestCopySubmodelOnto_MidsectionOutOfPlace();

void TestTransformSubmodelOnto_CopyInPlace();
void TestTransformSubmodelOnto_CopyOutOfPlace();
void TestTransformSubmodelOnto_CopyPrefixInPlace();
void TestTransformSubmodelOnto_CopyPrefixOutOfPlace();
void TestTransformSubmodelOnto_ModifyInPlace();
void TestTransformSubmodelOnto_ModifyOutOfPlace();

void TestTransformSubmodelInPlace_Copy();
void TestTransformSubmodelInPlace_CopyPrefix();
void TestTransformSubmodelInPlace_Modify();

namespace
{
const std::vector<const OutputPortBase*> noOutput = {};
} // namespace

// Transform functions
void CopyNode(const Node& node, ModelTransformer& transformer)
{
    transformer.CopyNode(node);
}

class ModifyFirstDebugNode
{
public:
    void operator()(const Node& node, ModelTransformer& transformer)
    {
        auto debugNode = dynamic_cast<const DebugNode<double, int>*>(&node);
        if (debugNode == nullptr || _didModify)
        {
            transformer.CopyNode(node);
        }
        else
        {
            const auto& newInputs = transformer.GetCorrespondingInputs(debugNode->input);
            auto newNode = transformer.AddNode<DebugNode<double, int>>(newInputs, 101);
            transformer.MapNodeOutput(debugNode->output, newNode->output);
            _didModify = true;
        }
    }

private:
    bool _didModify = false;
};

// Tests
void TestCopySubmodel()
{
    // Tests the function:
    //
    // Model ModelTransformer::CopySubmodel(const Submodel& submodel, const TransformContext& context);

    auto model = GetLinearDebugNodeModel(8); // in -> n1 -> n2 -> ... -> n8:out
    FailOnException(TestCopySubmodel_Full, model);
    FailOnException(TestCopySubmodel_Prefix, model);
    FailOnException(TestCopySubmodel_DoublePrefix, model);
}

void TestCopySubmodelOnto()
{
    // Tests the functions:
    //
    // Model ModelTransformer::CopySubmodelOnto(const Submodel& submodel, Model& destModel, const std::vector<const OutputPortBase*>& onto, const TransformContext& context);
    // Model ModelTransformer::CopySubmodelOnto(const Submodel& submodel, const std::vector<const OutputPortBase*>& onto, const TransformContext& context);

    FailOnException(TestCopySubmodelOnto_InPlace);
    FailOnException(TestCopySubmodelOnto_OutOfPlace);
    FailOnException(TestCopySubmodelOnto_PrefixInPlace);
    FailOnException(TestCopySubmodelOnto_PrefixOutOfPlace);
    FailOnException(TestCopySubmodelOnto_MidsectionInPlace<true>);
    FailOnException(TestCopySubmodelOnto_MidsectionInPlace<false>);
    FailOnException(TestCopySubmodelOnto_MidsectionOutOfPlace<true>);
    FailOnException(TestCopySubmodelOnto_MidsectionOutOfPlace<false>);
}

void TestTransformSubmodelOnto()
{
    // Tests the function:
    //
    // Submodel ModelTransformer::TransformSubmodelOnto(const Submodel& submodel, Model& destModel, const std::vector<const OutputPortBase*>& onto,
    //                                                  const TransformContext& context, const NodeTransformFunction& transformFunction);

    FailOnException(TestTransformSubmodelOnto_CopyInPlace);
    FailOnException(TestTransformSubmodelOnto_CopyOutOfPlace);
    FailOnException(TestTransformSubmodelOnto_CopyPrefixInPlace);
    FailOnException(TestTransformSubmodelOnto_CopyPrefixOutOfPlace);
    FailOnException(TestTransformSubmodelOnto_ModifyInPlace);
    FailOnException(TestTransformSubmodelOnto_ModifyOutOfPlace);
}

void TestTransformSubmodelInPlace()
{
    // Tests the function:
    //
    // Submodel ModelTransformer::TransformSubmodelOnto(const Submodel& submodel, const std::vector<const OutputPortBase*>& onto,
    //                                                  const TransformContext& context, const NodeTransformFunction& transformFunction);

    FailOnException(TestTransformSubmodelInPlace_Copy);
    FailOnException(TestTransformSubmodelInPlace_CopyPrefix);
    FailOnException(TestTransformSubmodelInPlace_Modify);
}

// Individual tests
void TestCopySubmodel_Full(const Model& model)
{
    TransformContext context;
    ModelTransformer transformer;

    Submodel submodel(model);
    auto newModel = transformer.CopySubmodel(submodel, context);
    ProcessTest("TestCopySubmodel_Full", newModel.Size() == static_cast<int>(model.Size()));
}

void TestCopySubmodel_Prefix(const Model& model)
{
    TransformContext context;
    ModelTransformer transformer;

    std::vector<const OutputPortBase*> outputs = { &FindDebugNode(model, 3)->output }; // in -> n1 -> n2 -> n3:out
    Submodel submodel(outputs);
    auto newModel = transformer.CopySubmodel(submodel, context);
    ProcessTest("TestCopySubmodel_Prefix", newModel.Size() == 4);
}

void TestCopySubmodel_DoublePrefix(const Model& model)
{
    TransformContext context;
    ModelTransformer transformer;

    std::vector<const OutputPortBase*> outputs = { &FindDebugNode(model, 3)->output, &FindDebugNode(model, 5)->output }; // in -> n1 -> n2 -> n3 -> n4 -> n5:out
    Submodel submodel(outputs);
    auto newModel = transformer.CopySubmodel(submodel, context);
    ProcessTest("TestCopySubmodel_DoublePrefix", newModel.Size() == 6);
}

void TestCopySubmodelOnto_InPlace()
{
    auto srcModel = GetLinearDebugNodeModel(4);
    auto oldSize = srcModel.Size();

    TransformContext context;
    ModelTransformer transformer;

    // Copy the submodel onto itself (should be a no-op)
    Submodel fullSubmodel(srcModel);
    transformer.CopySubmodelOnto(fullSubmodel, srcModel, noOutput, context);
    ProcessTest("TestCopySubmodelOnto_InPlace", srcModel.Size() == oldSize);
}

void TestCopySubmodelOnto_OutOfPlace()
{
    auto srcModel = GetLinearDebugNodeModel(4);
    auto oldSize = srcModel.Size();

    Model destModel;
    TransformContext context;
    ModelTransformer transformer;

    Submodel submodel(srcModel);
    transformer.CopySubmodelOnto(submodel, destModel, noOutput, context);
    ProcessTest("TestCopySubmodelOnto_OutOfPlace", destModel.Size() == oldSize);
}

void TestCopySubmodelOnto_PrefixInPlace()
{
    auto srcModel = GetLinearDebugNodeModel(4);
    auto oldSize = srcModel.Size();

    TransformContext context;
    ModelTransformer transformer;

    auto branchPoint = &FindDebugNode(srcModel, 1)->output;
    Submodel submodel({ branchPoint });
    transformer.CopySubmodelOnto(submodel, srcModel, noOutput, context);
    ProcessTest("TestCopySubmodelOnto_PrefixInPlace", srcModel.Size() == oldSize);
}

void TestCopySubmodelOnto_PrefixOutOfPlace()
{
    auto srcModel = GetLinearDebugNodeModel(4);

    Model destModel;
    TransformContext context;
    ModelTransformer transformer;

    auto branchPoint = &FindDebugNode(srcModel, 1)->output;
    Submodel submodel({ branchPoint });
    transformer.CopySubmodelOnto(submodel, destModel, noOutput, context);
    ProcessTest("TestCopySubmodelOnto_PrefixOutOfPlace", destModel.Size() == 2);
}

template <bool useDestModel>
void TestCopySubmodelOnto_MidsectionInPlace()
{
    auto srcModel = GetLinearDebugNodeModel(4);
    auto oldSize = srcModel.Size();

    TransformContext context;
    ModelTransformer transformer;

    auto midsectionInput = &FindDebugNode(srcModel, 2)->input;
    auto midsectionOutput = &FindDebugNode(srcModel, 3)->output;
    auto onto = &FindDebugNode(srcModel, 1)->output;
    Submodel submodel({ midsectionInput }, { midsectionOutput });
    if (useDestModel)
    {
        transformer.CopySubmodelOnto(submodel, srcModel, { onto }, context);
    }
    else
    {
        transformer.CopySubmodelOnto(submodel, { onto }, context);
    }
    auto newNode2 = FindDebugNode(srcModel, 2);
    ProcessTest("TestCopySubmodelOnto_MidsectionInPlace", (srcModel.Size() == oldSize) && ((&newNode2->input.GetReferencedPort()) == onto));
}

template <bool useDestModel>
void TestCopySubmodelOnto_MidsectionOutOfPlace()
{
    auto srcModel = GetLinearDebugNodeModel(4);
    auto destModel = GetLinearDebugNodeModel(1);
    auto oldSize = destModel.Size();

    TransformContext context;
    ModelTransformer transformer;

    auto midsectionInput = &FindDebugNode(srcModel, 2)->input;
    auto midsectionOutput = &FindDebugNode(srcModel, 3)->output;
    auto onto = &FindDebugNode(destModel, 1)->output;

    Submodel submodel({ midsectionInput }, { midsectionOutput });
    if (useDestModel)
    {
        transformer.CopySubmodelOnto(submodel, destModel, { onto }, context);
    }
    else
    {
        transformer.CopySubmodelOnto(submodel, { onto }, context);
    }
    auto newNode2 = FindDebugNode(destModel, 2);
    ProcessTest("TestCopySubmodelOnto_MidsectionOutOfPlace", (destModel.Size() == oldSize + 2) && ((&newNode2->input.GetReferencedPort()) == onto));
}

void TestTransformSubmodelOnto_CopyInPlace()
{
    auto srcModel = GetLinearDebugNodeModel(4);
    auto oldSize = srcModel.Size();

    TransformContext context;
    ModelTransformer transformer;

    Submodel submodel(srcModel);
    transformer.TransformSubmodelOnto(submodel, srcModel, noOutput, context, CopyNode);
    ProcessTest("TestTransformSubmodelOnto_CopyInPlace", srcModel.Size() == oldSize);
}

void TestTransformSubmodelOnto_CopyOutOfPlace()
{
    auto srcModel = GetLinearDebugNodeModel(4);
    auto oldSize = srcModel.Size();

    Model destModel;
    TransformContext context;
    ModelTransformer transformer;

    Submodel submodel(srcModel);
    transformer.TransformSubmodelOnto(submodel, destModel, noOutput, context, CopyNode);
    ProcessTest("TestTransformSubmodelOnto_CopyOutOfPlace", destModel.Size() == oldSize);
}

void TestTransformSubmodelOnto_CopyPrefixInPlace()
{
    auto srcModel = GetLinearDebugNodeModel(4);
    auto oldSize = srcModel.Size();

    TransformContext context;
    ModelTransformer transformer;

    auto branchPoint = &FindDebugNode(srcModel, 1)->output;
    Submodel submodel({ branchPoint });
    transformer.TransformSubmodelOnto(submodel, srcModel, noOutput, context, CopyNode);
    ProcessTest("TestTransformSubmodelOnto_CopyPrefixInPlace", srcModel.Size() == oldSize);
}

void TestTransformSubmodelOnto_CopyPrefixOutOfPlace()
{
    auto srcModel = GetLinearDebugNodeModel(4);

    Model destModel;
    TransformContext context;
    ModelTransformer transformer;

    auto branchPoint = &FindDebugNode(srcModel, 1)->output;
    Submodel submodel({ branchPoint });
    transformer.TransformSubmodelOnto(submodel, destModel, noOutput, context, CopyNode);
    ProcessTest("TestTransformSubmodelOnto_CopyPrefixOutOfPlace", destModel.Size() == 2);
}

void TestTransformSubmodelOnto_ModifyInPlace()
{
    auto srcModel = GetLinearDebugNodeModel(4);
    auto oldSize = srcModel.Size();

    TransformContext context;
    ModelTransformer transformer;

    Submodel submodel(srcModel);
    // transforms first debug node to have an ID of 101, then copies the rest
    auto oldNode1 = FindDebugNode(srcModel, 1);
    transformer.TransformSubmodelOnto(submodel, srcModel, noOutput, context, ModifyFirstDebugNode());
    auto newNode1 = FindDebugNode(srcModel, 101);
    ProcessTest("TestTransformSubmodelOnto_ModifyInPlace", (srcModel.Size() == 2 * oldSize - 1) && ((&oldNode1->input.GetReferencedPort()) == (&newNode1->input.GetReferencedPort())));
}

void TestTransformSubmodelOnto_ModifyOutOfPlace()
{
    auto srcModel = GetLinearDebugNodeModel(4);
    auto oldSize = srcModel.Size();

    Model destModel;
    TransformContext context;
    ModelTransformer transformer;

    Submodel submodel(srcModel);
    transformer.TransformSubmodelOnto(submodel, destModel, noOutput, context, ModifyFirstDebugNode());
    ProcessTest("TestTransformSubmodelOnto_ModifyOutOfPlace", destModel.Size() == oldSize);
}

void TestTransformSubmodelInPlace_Copy()
{
    auto srcModel = GetLinearDebugNodeModel(4);
    auto oldSize = srcModel.Size();

    TransformContext context;
    ModelTransformer transformer;

    Submodel submodel(srcModel);
    transformer.TransformSubmodelOnto(submodel, srcModel, noOutput, context, CopyNode);
    ProcessTest("TestTransformSubmodelInPlace_Copy", srcModel.Size() == oldSize);
}

void TestTransformSubmodelInPlace_CopyPrefix()
{
    auto srcModel = GetLinearDebugNodeModel(4);
    auto oldSize = srcModel.Size();

    TransformContext context;
    ModelTransformer transformer;

    auto branchPoint = &FindDebugNode(srcModel, 1)->output;
    Submodel submodel({ branchPoint });
    transformer.TransformSubmodelOnto(submodel, srcModel, noOutput, context, CopyNode);
    ProcessTest("TestTransformSubmodelInPlace_CopyPrefix", srcModel.Size() == oldSize);
}

void TestTransformSubmodelInPlace_Modify()
{
    auto srcModel = GetLinearDebugNodeModel(4);
    auto oldSize = srcModel.Size();

    TransformContext context;
    ModelTransformer transformer;

    Submodel submodel(srcModel);
    auto oldNode1 = FindDebugNode(srcModel, 1);
    transformer.TransformSubmodelOnto(submodel, srcModel, noOutput, context, ModifyFirstDebugNode());
    auto newNode1 = FindDebugNode(srcModel, 101);
    ProcessTest("TestTransformSubmodelInPlace_Modify", (srcModel.Size() == 2 * oldSize - 1) && ((&oldNode1->input.GetReferencedPort()) == (&newNode1->input.GetReferencedPort())));
}
