////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ScalingNode.cpp(nodes)
//  Authors:  Chris Lovett
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ScalingNode.h"

#include <value/include/EmitterContext.h>

#include <utilities/include/MemoryLayout.h>

namespace ell
{
namespace nodes
{
    using namespace utilities;
    using namespace value;

    template <typename ValueType>
    ScalingNode<ValueType>::ScalingNode(const model::OutputPort<ValueType>& input, ValueType scale) :
        CompilableCodeNode("ScalingNode", { &_input }, { &_output }),
        _input(this, input, defaultInputPortName),
        _output(this, defaultOutputPortName, input.Size()),
        _scale(scale)
    {
    }

    template <typename ValueType>
    ScalingNode<ValueType>::ScalingNode() :
        CompilableCodeNode("ScalingNode", { &_input }, { &_output }),
        _input(this, {}, defaultInputPortName),
        _output(this, defaultOutputPortName, 0),
        _scale(0)
    {
    }

    template <typename ValueType>
    void ScalingNode<ValueType>::Define(FunctionDeclaration& fn)
    {
        fn.Define([this](const Value data, Value result) {
            // flatten the MemoryLayout so we can accept any shaped input data and produce any shape result.
            Vector input = ToVector(data);
            Vector output = ToVector(result);

            For(input, [&input, &output, this](Scalar index) {
                output[index] = _scale * input[index];
            });
        });
    }

    template <typename ValueType>
    void ScalingNode<ValueType>::DefineReset(FunctionDeclaration& fn)
    {
    }

    template <typename ValueType>
    void ScalingNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInputs = transformer.GetCorrespondingInputs(_input);
        auto newNode = transformer.AddNode<ScalingNode<ValueType>>(newInputs, _scale);
        transformer.MapNodeOutput(output, newNode->output);
    }

    template <typename ValueType>
    void ScalingNode<ValueType>::WriteToArchive(utilities::Archiver& archiver) const
    {
        Node::WriteToArchive(archiver);
        archiver[defaultInputPortName] << _input;
        double s = _scale;
        archiver["scale"] << s;
    }

    template <typename ValueType>
    void ScalingNode<ValueType>::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        Node::ReadFromArchive(archiver);
        archiver[defaultInputPortName] >> _input;
        double s;
        archiver["scale"] >> s;
        _scale = static_cast<ValueType>(s);
        _output.SetSize(_input.Size());
    }

    // Explicit specializations
    template class ScalingNode<float>;
    template class ScalingNode<double>;
    template class ScalingNode<int>;
    template class ScalingNode<int64_t>;
    template class ScalingNode<Boolean>;

    template <typename ValueType>
    const model::OutputPort<ValueType>& AddScalingNode(const model::OutputPort<ValueType>& input, ValueType scale)
    {
        model::Model* model = input.GetNode()->GetModel();
        if (model == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input not part of a model");
        }
        auto node = model->AddNode<ScalingNode<ValueType>>(input, scale);
        return node->output;
    }

    template const model::OutputPort<float>& AddScalingNode(const model::OutputPort<float>& input, float scale);
    template const model::OutputPort<double>& AddScalingNode(const model::OutputPort<double>& input, double scale);

} // namespace nodes
} // namespace ell
