////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     BufferNode.h (nodes)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <model/include/CompilableCodeNode.h>
#include <model/include/IRMapCompiler.h>
#include <model/include/InputPort.h>
#include <model/include/MapCompiler.h>
#include <model/include/ModelTransformer.h>
#include <model/include/OutputPort.h>

#include <utilities/include/IArchivable.h>
#include <utilities/include/TypeName.h>

#include <value/include/Vector.h>

#include <string>
#include <vector>

namespace ell
{
namespace nodes
{
    /// <summary> A node that combines one or more input buffers returning a larger window over that input.
    /// On each new input the buffer is shifted left by the size of that input.  For example, if the input size
    /// is 8 and the windowSize is 16 and the inputs are given in the sequence i1, i2, i3, i4 then the output
    /// of the buffer node will be [0 i1], [i1 i2], [i2, i3], [i3 i4].  So if you think of the input as a
    /// series of values over time (like audio signal) then the BufferNode provides a sliding window over that
    /// input data.
    /// </summary>
    template <typename ValueType>
    class BufferNode : public model::CompilableCodeNode
    {
    public:
        /// @name Input and Output Ports
        /// @{
        const model::InputPort<ValueType>& input = _input;
        const model::OutputPort<ValueType>& output = _output;
        /// @}

        BufferNode();

        /// <summary> Constructor </summary>
        ///
        /// <param name="input"> The input to buffer. </param>
        /// <param name="windowSize"> The size of the output of this node which should not be smaller than the input size. </param>
        BufferNode(const model::OutputPort<ValueType>& input, size_t windowSize);

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return utilities::GetCompositeTypeName<ValueType>("BufferNode"); }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

        /// <summary> Return the window size </summary>
        ///
        /// <returns> The window size </returns>
        size_t GetWindowSize() const { return _windowSize; }

    protected:
        void Define(value::FunctionDeclaration& fn) override;
        void DefineReset(value::FunctionDeclaration& fn) override;

        bool HasState() const override { return true; } // stored state: windowSize
        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;

    private:
        void Copy(model::ModelTransformer& transformer) const override;

        // Inputs
        model::InputPort<ValueType> _input;

        // Output
        model::OutputPort<ValueType> _output;

        // Buffer
        value::Vector _buffer;
        size_t _windowSize;
    };

    /// <summary> Convenience function for adding a buffer node to a model. </summary>
    ///
    /// <param name="input"> The input to the buffer node. </param>
    /// <param name="windowSize"> The size of the buffer. </param>
    ///
    /// <returns> The output of the new node. </returns>
    template <typename ValueType>
    const model::OutputPort<ValueType>& AddBufferNode(const model::OutputPort<ValueType>& input, size_t windowSize);

} // namespace nodes
} // namespace ell
