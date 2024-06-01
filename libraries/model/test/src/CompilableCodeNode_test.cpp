////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CompilableCodeNode_test.cpp (model)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CompilableCodeNode_test.h"

#include <common/include/LoadModel.h>

#include <model/include/CompilableCodeNode.h>
#include <model/include/IRMapCompiler.h>
#include <model/include/InputNode.h>
#include <model/include/Map.h>
#include <model/include/MapCompilerOptions.h>
#include <model/include/Model.h>
#include <model/include/ModelTransformer.h>

#include <model_testing/include/ModelTestUtilities.h>

#include <nodes/include/ConstantNode.h>

#include <testing/include/testing.h>

#include <utilities/include/Logger.h>
#include <utilities/include/StringUtil.h>

#include <value/include/FunctionDeclaration.h>
#include <value/include/Scalar.h>
#include <value/include/Value.h>
#include <value/include/Vector.h>

#include <vector>

namespace ell
{
using namespace common;
using namespace model;

namespace detail
{
    using namespace value;

    class DotProductCodeNode : public CompilableCodeNode
    {
    public:
        const InputPortBase& input1 = _input1;
        const InputPortBase& input2 = _input2;
        const OutputPortBase& output = _output;

        DotProductCodeNode() :
            model::CompilableCodeNode("DotProduct", { &_input1, &_input2 }, { &_output }),
            _input1(this, {}, defaultInput1PortName),
            _input2(this, {}, defaultInput2PortName),
            _output(this, defaultOutputPortName, ell::model::Port::PortType::real, utilities::ScalarLayout)
        {
        }

        DotProductCodeNode(const OutputPortBase& input1, const OutputPortBase& input2) :
            model::CompilableCodeNode("DotProduct", { &_input1, &_input2 }, { &_output }),
            _input1(this, input1, defaultInput1PortName),
            _input2(this, input2, defaultInput2PortName),
            _output(this, defaultOutputPortName, _input1.GetType(), utilities::ScalarLayout)
        {
            assert(_input1.GetType() == _input2.GetType());
        }

        void Define(FunctionDeclaration& fn) override
        {
            (void)fn.Define([](Vector v1, Vector v2, Scalar s) {
                s = Dot(v1, v2);
            });
        }

        static std::string GetTypeName() { return "DotProductCodeNode"; }

    protected:
        void WriteToArchive(utilities::Archiver& archiver) const override
        {
            Node::WriteToArchive(archiver);
            archiver[defaultInput1PortName] << _input1;
            archiver[defaultInput2PortName] << _input2;
        }

        void ReadFromArchive(utilities::Unarchiver& archiver) override
        {
            Node::ReadFromArchive(archiver);
            archiver[defaultInput1PortName] >> _input1;
            archiver[defaultInput2PortName] >> _input2;
        }

    private:
        void Copy(model::ModelTransformer& transformer) const override
        {
            const auto& newInput1 = transformer.GetCorrespondingInputs(_input1);
            const auto& newInput2 = transformer.GetCorrespondingInputs(_input2);
            auto newNode = transformer.AddNode<DotProductCodeNode>(newInput1, newInput2);
            transformer.MapNodeOutput(output, newNode->output);
        }

        InputPortBase _input1;
        InputPortBase _input2;
        OutputPortBase _output;
    };

} // namespace detail
using ::ell::detail::DotProductCodeNode;

void CompilableCodeNode_test1()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(4);
    auto constantNode = model.AddNode<nodes::ConstantNode<double>>(std::vector<double>{ 5.0, 5.0, 7.0, 3.0 });
    auto dotNode = model.AddNode<DotProductCodeNode>(inputNode->output, constantNode->output);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", dotNode->output } });

    // make sure we can serialize it.
    RegisterCustomTypeFactory([](utilities::SerializationContext& context) {
        context.GetTypeFactory().AddType<model::Node, DotProductCodeNode>();
    });

    // compare output
    std::vector<std::vector<double>> signal = {
        { 1, 2, 3, 7 },
        { 4, 5, 6, 7 },
        { 7, 8, 9, 7 },
        { 3, 4, 5, 7 },
        { 2, 3, 2, 7 },
        { 1, 5, 3, 7 },
        { 1, 2, 3, 7 },
        { 4, 5, 6, 7 },
        { 7, 8, 9, 7 },
        { 7, 4, 2, 7 },
        { 5, 2, 1, 7 }
    };

    TestWithSerialization(map, "DotProductCodeNode", [&](model::Map& map, int iteration) {
        model::IRMapCompiler compiler({}, {});
        auto compiledMap = compiler.Compile(map);

        VerifyCompiledOutput(map, compiledMap, signal, utilities::FormatString("DotProductCodeNode iteration %d", iteration));
    });

    RegisterCustomTypeFactory(nullptr);
}

} // namespace ell
