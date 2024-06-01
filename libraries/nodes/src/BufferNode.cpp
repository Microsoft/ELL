////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     BufferNode.cpp(nodes)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "BufferNode.h"

#include <value/include/EmitterContext.h>

#include <utilities/include/Boolean.h>
#include <utilities/include/MemoryLayout.h>

namespace ell
{
namespace nodes
{
    using namespace utilities;
    using namespace value;

    template <typename ValueType>
    BufferNode<ValueType>::BufferNode(const model::OutputPort<ValueType>& input, size_t windowSize) :
        CompilableCodeNode("BufferNode", { &_input }, { &_output }),
        _input(this, input, defaultInputPortName),
        _output(this, defaultOutputPortName, windowSize),
        _windowSize(windowSize)
    {
    }

    template <typename ValueType>
    BufferNode<ValueType>::BufferNode() :
        CompilableCodeNode("BufferNode", { &_input }, { &_output }),
        _input(this, {}, defaultInputPortName),
        _output(this, defaultOutputPortName, 0),
        _windowSize(0)
    {
    }

    template <typename ValueType>
    void BufferNode<ValueType>::Define(FunctionDeclaration& fn)
    {
        fn.Define([this](const Value data, Value result) {
            // flatten the MemoryLayout so we can accept any shaped input data and produce any shape result.
            Vector input = ToVector(data);
            Vector output = ToVector(result);
            int inputSize = static_cast<int>(input.Size());
            int windowSize = static_cast<int>(_windowSize);
            _buffer = StaticAllocate("buffer", GetValueType<ValueType>(), MemoryLayout({ _windowSize }));
            if (inputSize > windowSize)
            {
                inputSize = windowSize;
            }
            auto remainder = windowSize - inputSize;
            if (remainder > 0)
            {
                // shift down the tail to the front of the buffer, this is potentially overlapping
                // so we have to use our own For loop here.
                auto shift = _buffer.SubVector(inputSize, remainder);
                For(shift, [this, &shift](Scalar index) {
                    _buffer[index] = shift[index];
                });
            }
            else
            {
                remainder = 0;
            }

            // copy the input data to the end of the buffer
            Vector tail = _buffer.SubVector(remainder, inputSize);

            // tail = input, should work, but it is currently broken.  (bug 2208)
            For(data, [&tail, &input](Scalar index) {
                tail[index] = input[index];
            });

            // copy data to the output
            output = _buffer;
        });
    }

    template <typename ValueType>
    void BufferNode<ValueType>::DefineReset(FunctionDeclaration& fn)
    {
        fn.Define([this] {
            // bugbug: how to emit a "memset" operation here instead?
            Scalar zero(static_cast<ValueType>(0));
            For(_buffer, [this, zero](Scalar index) {
                _buffer[index] = zero;
            });
        });
    }

    template <typename ValueType>
    void BufferNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInputs = transformer.GetCorrespondingInputs(_input);
        auto newNode = transformer.AddNode<BufferNode<ValueType>>(newInputs, _windowSize);
        transformer.MapNodeOutput(output, newNode->output);
    }

    template <typename ValueType>
    void BufferNode<ValueType>::WriteToArchive(utilities::Archiver& archiver) const
    {
        Node::WriteToArchive(archiver);
        archiver[defaultInputPortName] << _input;
        archiver["windowSize"] << _windowSize;
    }

    template <typename ValueType>
    void BufferNode<ValueType>::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        Node::ReadFromArchive(archiver);
        archiver[defaultInputPortName] >> _input;
        archiver["windowSize"] >> _windowSize;
        _output.SetSize(_windowSize);
    }

    // Explicit specializations
    template class BufferNode<float>;
    template class BufferNode<double>;
    template class BufferNode<int>;
    template class BufferNode<int64_t>;
    template class BufferNode<Boolean>;

    template <typename ValueType>
    const model::OutputPort<ValueType>& AddBufferNode(const model::OutputPort<ValueType>& input, size_t windowSize)
    {
        model::Model* model = input.GetNode()->GetModel();
        if (model == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input not part of a model");
        }
        auto node = model->AddNode<BufferNode<ValueType>>(input, windowSize);
        return node->output;
    }

    template const model::OutputPort<float>& AddBufferNode(const model::OutputPort<float>& input, size_t windowSize);
    template const model::OutputPort<double>& AddBufferNode(const model::OutputPort<double>& input, size_t windowSize);
    template const model::OutputPort<int>& AddBufferNode(const model::OutputPort<int>& input, size_t windowSize);
    template const model::OutputPort<int64_t>& AddBufferNode(const model::OutputPort<int64_t>& input, size_t windowSize);
    template const model::OutputPort<Boolean>& AddBufferNode(const model::OutputPort<Boolean>& input, size_t windowSize);

} // namespace nodes
} // namespace ell
