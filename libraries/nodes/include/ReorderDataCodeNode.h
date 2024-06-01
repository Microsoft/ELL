////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ReorderDataCodeNode.h (nodes)
//  Authors:  Byron Changuion
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <model/include/CompilableCodeNode.h>
#include <model/include/IRMapCompiler.h>
#include <model/include/InputPort.h>
#include <model/include/OutputPort.h>

#include <value/include/FunctionDeclaration.h>
#include <value/include/Scalar.h>
#include <value/include/ScalarOperations.h>
#include <value/include/Tensor.h>

#include <value/include/loopnests/CodeGenerator.h>
#include <value/include/loopnests/Kernel.h>
#include <value/include/loopnests/LoopNest.h>

#include <string>
#include <vector>

namespace ell
{
namespace nodes
{
    /// <summary> A node that takes data from its input and outputs it in a different order.  </summary>
    template <typename ValueType>
    class ReorderDataCodeNode : public model::CompilableCodeNode
    {

    public:
        /// @name Input and Output Ports
        /// @{
        const model::InputPort<ValueType>& input = _input;
        const model::OutputPort<ValueType>& output = _output;
        /// @}

        /// <summary> Default Constructor </summary>
        ReorderDataCodeNode();

        /// <summary> Constructor with no reordering </summary>
        ///
        /// <param name="input"> The input to reorder. </param>
        /// <param name="outputMemoryLayout"> The memory layout of the output. Data will be copied into the "active" area, and the rest will be zeroed out according to the padding value. </param>
        /// <param name="paddingValue"> The value to fill the inactive area with. </param>
        ReorderDataCodeNode(const model::OutputPort<ValueType>& input, const model::PortMemoryLayout& outputMemoryLayout, ValueType paddingValue = 0);

        /// <summary> Constructor with no reordering </summary>
        ///
        /// <param name="input"> The input to reorder. </param>
        /// <param name="inputMemoryLayout"> The memory layout of the input. Only data in the "active" area will be copied. </param>
        /// <param name="outputMemoryLayout"> The memory layout of the output. Data will be copied into the "active" area, and the rest will be zeroed out according to the padding value. </param>
        /// <param name="paddingValue"> The value to fill the inactive area with. </param>
        ReorderDataCodeNode(const model::OutputPort<ValueType>& input, const model::PortMemoryLayout& inputMemoryLayout, const model::PortMemoryLayout& outputMemoryLayout, ValueType paddingValue = 0);

        /// <summary> Constructor with reordering </summary>
        ///
        /// <param name="input"> The input to reorder. </param>
        /// <param name="order"> The permutation vector to apply to the dimensions when copying. Input dimension `i` will get copied to output dimension `order[i]`. If left empty, no reordering is done.
        //    For instance, to reorder the normal interleaved image order into a planar order, the `order` parameter would be
        ///   set to {2, 0, 1} --- reordering {row, column, channel} to {channel, row, column} </param>
        ReorderDataCodeNode(const model::OutputPort<ValueType>& input, const model::DimensionOrder& order);

        /// <summary> Constructor with reordering </summary>
        ///
        /// <param name="input"> The input to reorder. </param>
        /// <param name="outputMemoryLayout"> The memory layout of the output. Data will be copied into the "active" area, and the rest will be zeroed out according to the padding value. </param>
        /// <param name="order"> The permutation vector to apply to the dimensions when copying. Input dimension `i` will get copied to output dimension `order[i]`. If left empty, no reordering is done.
        ///    For instance, to reorder the normal interleaved image order into a planar order, the `order` parameter would be
        ///    set to {2, 0, 1} --- reordering {row, column, channel} to {channel, row, column} </param>
        /// <param name="paddingValue"> The value to fill the inactive area with. </param>
        ReorderDataCodeNode(const model::OutputPort<ValueType>& input, const model::PortMemoryLayout& outputMemoryLayout, const model::DimensionOrder& order, ValueType paddingValue = 0);

        /// <summary> Constructor with reordering </summary>
        ///
        /// <param name="input"> The input to reorder. </param>
        /// <param name="inputMemoryLayout"> The memory layout of the input. Only data in the "active" area is guaranteed to be copied. </param>
        /// <param name="outputMemoryLayout"> The memory layout of the output. Data will be copied into the "active" area, and the rest will be zeroed out according to the padding value. </param>
        /// <param name="order"> The permutation vector to apply to the dimensions when copying. Input dimension `i` will get copied to output dimension `order[i]`. If left empty, no reordering is done.
        ///   For instance, to reorder the normal interleaved image order into a planar order, the `order` parameter would be
        ///   set to {2, 0, 1} --- reordering {row, column, channel} to {channel, row, column} </param>
        /// <param name="paddingValue"> The value to fill the inactive area with. </param>
        ReorderDataCodeNode(const model::OutputPort<ValueType>& input, const model::PortMemoryLayout& inputMemoryLayout, const model::PortMemoryLayout& outputMemoryLayout, const model::DimensionOrder& order, ValueType paddingValue = 0);

        /// <summary> Gets information about the input memory layout </summary>
        const model::PortMemoryLayout& GetInputMemoryLayout() const { return _inputMemoryLayout; }

        /// <summary> Gets information about the input memory layout </summary>
        model::PortMemoryLayout GetOutputMemoryLayout() const { return _outputMemoryLayout; }

        /// <summary> Returns padding value </summary>
        ///
        /// <returns> Padding value </returns>
        ValueType GetPaddingValue() const { return _paddingValue; }

        /// <summary> Returns true if the node can accept input with this memory layout order, else false </summary>
        ///
        /// <param name="order"> The memory layout order for all the input ports </summary>
        /// <returns> If the node can accept the input memory layout order, true, else false </returns>
        bool CanAcceptInputLayout(const utilities::DimensionOrder& order) const override
        {
            return GetInputMemoryLayout().GetLogicalDimensionOrder() == order;
        }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return utilities::GetCompositeTypeName<ValueType>("ReorderDataCodeNode"); }

    protected:
        void Define(ell::value::FunctionDeclaration& fn) override;
        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;
        bool HasState() const override { return true; } // stored state: operation
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

    private:
        void Copy(model::ModelTransformer& transformer) const override;

        void reorder_kernel_optimized_columns(value::Tensor source, value::Tensor dest, value::Scalar i, value::Scalar j, value::Scalar k);
        void reorder_kernel_optimized_channels(value::Tensor source, value::Tensor dest, value::Scalar i, value::Scalar j, value::Scalar k);
        static void reorder_kernel_basic(value::Tensor source, value::Tensor dest, value::Scalar i, value::Scalar j, value::Scalar k);

        // Inputs
        model::InputPort<ValueType> _input;

        // Output
        model::OutputPort<ValueType> _output;

        // Memory Layouts
        model::PortMemoryLayout _inputMemoryLayout;
        model::PortMemoryLayout _outputMemoryLayout;

        ValueType _paddingValue;

        // This is used in the Define function as a workaround for passing in constant Scalar values
        // to the kernel
        int _kernel_size;
    };

    /// <summary> Convenience function for adding a node to a model. </summary>
    ///
    /// <param name="input"> The input to reorder. </param>
    /// <param name="outputMemoryLayout"> The memory layout of the output. Data will be copied into the "active" area, and the rest will be zeroed out according to the padding value. </param>
    /// <param name="paddingValue"> The value to fill the inactive area with. </param>
    ///
    /// <returns> The output of the new node. </returns>
    template <typename ValueType>
    const model::OutputPort<ValueType>& ReorderDataWithCodeNode(const model::OutputPort<ValueType>& input, const model::PortMemoryLayout& outputMemoryLayout, ValueType paddingValue = 0);

    /// <summary> Convenience function for adding a node to a model. </summary>
    ///
    /// <param name="input"> The input to reorder. </param>
    /// <param name="inputMemoryLayout"> The memory layout of the input. Only data in the "active" area will be copied. </param>
    /// <param name="outputMemoryLayout"> The memory layout of the output. Data will be copied into the "active" area, and the rest will be zeroed out according to the padding value. </param>
    /// <param name="paddingValue"> The value to fill the inactive area with. </param>
    ///
    /// <returns> The output of the new node. </returns>
    template <typename ValueType>
    const model::OutputPort<ValueType>& ReorderDataWithCodeNode(const model::OutputPort<ValueType>& input, const model::PortMemoryLayout& inputMemoryLayout, const model::PortMemoryLayout& outputMemoryLayout, ValueType paddingValue = 0);

    /// <summary> Convenience function for adding a node to a model. </summary>
    ///
    /// <param name="input"> The input to reorder. </param>
    /// <param name="outputMemoryLayout"> The memory layout of the output. Data will be copied into the "active" area, and the rest will be zeroed out according to the padding value. </param>
    /// <param name="order"> The permutation vector to apply to the dimensions when copying. Input dimension `i` will get copied to output dimension `order[i]`. If left empty, no reordering is done.
    ///    For instance, to reorder the normal interleaved image order into a planar order, the `order` parameter would be
    ///    set to {2, 0, 1} --- reordering {row, column, channel} to {channel, row, column} </param>
    /// <param name="paddingValue"> The value to fill the inactive area with. </param>
    ///
    /// <returns> The output of the new node. </returns>
    template <typename ValueType>
    const model::OutputPort<ValueType>& ReorderDataWithCodeNode(const model::OutputPort<ValueType>& input, const model::PortMemoryLayout& outputMemoryLayout, const model::DimensionOrder& order, ValueType paddingValue = 0);

    /// <summary> Convenience function for adding a node to a model. </summary>
    ///
    /// <param name="input"> The input to reorder. </param>
    /// <param name="inputMemoryLayout"> The memory layout of the input. Only data in the "active" area will be copied. </param>
    /// <param name="outputMemoryLayout"> The memory layout of the output. Data will be copied into the "active" area, and the rest will be zeroed out according to the padding value. </param>
    /// <param name="order"> The permutation vector to apply to the dimensions when copying. Input dimension `i` will get copied to output dimension `order[i]`. If left empty, no reordering is done.
    ///    For instance, to reorder the normal interleaved image order into a planar order, the `order` parameter would be
    ///    set to {2, 0, 1} --- reordering {row, column, channel} to {channel, row, column} </param>
    /// <param name="paddingValue"> The value to fill the inactive area with. </param>
    ///
    /// <returns> The output of the new node. </returns>
    template <typename ValueType>
    const model::OutputPort<ValueType>& ReorderDataWithCodeNode(const model::OutputPort<ValueType>& input, const model::PortMemoryLayout& inputMemoryLayout, const model::PortMemoryLayout& outputMemoryLayout, const model::DimensionOrder& order, ValueType paddingValue = 0);

    /// <summary> Convenience function for adding a node to a model. </summary>
    ///
    /// <param name="input"> The input to reorder. </param>
    /// <param name="order"> The permutation vector to apply to the dimensions when copying. Input dimension `i` will get copied to output dimension `order[i]`. If left empty, no reordering is done.
    ///    For instance, to reorder the normal interleaved image order into a planar order, the `order` parameter would be
    ///    set to {2, 0, 1} --- reordering {row, column, channel} to {channel, row, column} </param>
    ///
    /// <returns> The output of the new node. </returns>
    template <typename ValueType>
    const model::OutputPort<ValueType>& ReorderDataWithCodeNode(const model::OutputPort<ValueType>& input, const model::DimensionOrder& order);

} // namespace nodes
} // namespace ell

#pragma region implementation

namespace ell
{
namespace nodes
{
    //
    // ReorderDataCodeNode
    //
    template <typename ValueType>
    ReorderDataCodeNode<ValueType>::ReorderDataCodeNode() :
        CompilableCodeNode("ReorderDataCodeNode", { &_input }, { &_output }),
        _input(this, {}, defaultInputPortName),
        _output(this, defaultOutputPortName, 0),
        _inputMemoryLayout(utilities::MemoryShape{}),
        _outputMemoryLayout(_output.GetMemoryLayout()),
        _paddingValue(0),
        _kernel_size(1)
    {}

    //
    // Without reordering ("reshape" / slicing)
    //
    template <typename ValueType>
    ReorderDataCodeNode<ValueType>::ReorderDataCodeNode(const model::OutputPort<ValueType>& input,
                                                        const model::PortMemoryLayout& outputMemoryLayout,
                                                        ValueType paddingValue) :
        CompilableCodeNode("ReorderDataCodeNode", { &_input }, { &_output }),
        _input(this, input, defaultInputPortName),
        _output(this, defaultOutputPortName, outputMemoryLayout),
        _inputMemoryLayout(_input.GetMemoryLayout()),
        _outputMemoryLayout(_output.GetMemoryLayout()),
        _paddingValue(paddingValue),
        _kernel_size(1)
    {
        if (_inputMemoryLayout.NumDimensions() != outputMemoryLayout.NumDimensions())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument,
                                            "Error: input and output layouts must have same dimension");
        }
    }

    template <typename ValueType>
    ReorderDataCodeNode<ValueType>::ReorderDataCodeNode(const model::OutputPort<ValueType>& input,
                                                        const model::PortMemoryLayout& inputMemoryLayout,
                                                        const model::PortMemoryLayout& outputMemoryLayout,
                                                        ValueType paddingValue) :
        CompilableCodeNode("ReorderDataCodeNode", { &_input }, { &_output }),
        _input(this, input, defaultInputPortName),
        _output(this, defaultOutputPortName, outputMemoryLayout),
        _inputMemoryLayout(inputMemoryLayout),
        _outputMemoryLayout(_output.GetMemoryLayout()),
        _paddingValue(paddingValue),
        _kernel_size(1)
    {
        if (inputMemoryLayout.NumDimensions() != outputMemoryLayout.NumDimensions())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument,
                                            "Error: input and output layouts must have same dimension");
        }
    }

    //
    // With reordering ("reshape" / slicing, followed by transpose / dimension reordering)
    //
    template <typename ValueType>
    ReorderDataCodeNode<ValueType>::ReorderDataCodeNode(const model::OutputPort<ValueType>& input,
                                                        const model::DimensionOrder& order) :
        CompilableCodeNode("ReorderDataCodeNode", { &_input }, { &_output }),
        _input(this, input, defaultInputPortName),
        _output(this, defaultOutputPortName, _input.GetMemoryLayout().ReorderedCopy(order)),
        _inputMemoryLayout(_input.GetMemoryLayout()),
        _outputMemoryLayout(_output.GetMemoryLayout()),
        _paddingValue(0),
        _kernel_size(1)
    {
        if (_inputMemoryLayout.NumDimensions() != order.NumDimensions())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument,
                                            "Error: input and output layouts must have same dimension");
        }
    }

    template <typename ValueType>
    ReorderDataCodeNode<ValueType>::ReorderDataCodeNode(const model::OutputPort<ValueType>& input,
                                                        const model::PortMemoryLayout& outputMemoryLayout,
                                                        const model::DimensionOrder& order,
                                                        ValueType paddingValue) :
        CompilableCodeNode("ReorderDataCodeNode", { &_input }, { &_output }),
        _input(this, input, defaultInputPortName),
        _output(this, defaultOutputPortName, outputMemoryLayout.ReorderedCopy(order)),
        _inputMemoryLayout(_input.GetMemoryLayout()),
        _outputMemoryLayout(_output.GetMemoryLayout()),
        _paddingValue(paddingValue),
        _kernel_size(1)
    {
        if (_inputMemoryLayout.NumDimensions() != outputMemoryLayout.NumDimensions())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument,
                                            "Error: input and output layouts must have same dimension");
        }
    }

    template <typename ValueType>
    ReorderDataCodeNode<ValueType>::ReorderDataCodeNode(const model::OutputPort<ValueType>& input,
                                                        const model::PortMemoryLayout& inputMemoryLayout,
                                                        const model::PortMemoryLayout& outputMemoryLayout,
                                                        const model::DimensionOrder& order,
                                                        ValueType paddingValue) :
        CompilableCodeNode("ReorderDataCodeNode", { &_input }, { &_output }),
        _input(this, input, defaultInputPortName),
        _output(this, defaultOutputPortName, outputMemoryLayout.ReorderedCopy(order)),
        _inputMemoryLayout(inputMemoryLayout),
        _outputMemoryLayout(_output.GetMemoryLayout()),
        _paddingValue(paddingValue),
        _kernel_size(1)
    {
        if (inputMemoryLayout.NumDimensions() != outputMemoryLayout.NumDimensions())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument,
                                            "Error: input and output layouts must have same dimension");
        }
    }

    //
    // A reorder kernel that is optimized for when channels are the minor increment
    //
    template <typename ValueType>
    void ReorderDataCodeNode<ValueType>::reorder_kernel_optimized_channels(value::Tensor source, value::Tensor dest, value::Scalar i, value::Scalar j, value::Scalar k)
    {
        value::Vector cache = value::MakeVector<ValueType>(_kernel_size);
        for (int l = 0; l < _kernel_size; ++l)
        {
            cache(l) = source(i, j, k * _kernel_size + l);
        }

        for (int l = 0; l < _kernel_size; ++l)
        {
            dest(i, j, k * _kernel_size + l) = cache(l);
        }
    }

    //
    // A reorder kernel that is optimized for when columns are the minor increment
    //
    template <typename ValueType>
    void ReorderDataCodeNode<ValueType>::reorder_kernel_optimized_columns(value::Tensor source, value::Tensor dest, value::Scalar i, value::Scalar j, value::Scalar k)
    {
        value::Vector cache = value::MakeVector<ValueType>(_kernel_size);
        for (int l = 0; l < _kernel_size; ++l)
        {
            cache(l) = source(i, j * _kernel_size + l, k);
        }

        for (int l = 0; l < _kernel_size; ++l)
        {
            dest(i, j * _kernel_size + l, k) = cache(l);
        }
    }

    //
    // A basic, unoptimized reorder kernel
    //
    template <typename ValueType>
    void ReorderDataCodeNode<ValueType>::reorder_kernel_basic(value::Tensor source, value::Tensor dest, value::Scalar i, value::Scalar j, value::Scalar k)
    {
        dest(i, j, k) = source(i, j, k);
    }

    template <typename ValueType>
    void ReorderDataCodeNode<ValueType>::Define(ell::value::FunctionDeclaration& fn)
    {
        (void)fn.Define([this](const value::Value value_input, value::Value output) {
            namespace loopnests = ell::value::loopnests;

            auto input = value_input;
            // Set the layout to use for the input view.
            input.SetLayout(_inputMemoryLayout);

            // Check if this is a Tensor
            if (input.GetLayout().NumDimensions() == 3)
            {
                auto data = value::Tensor(input);
                auto result = value::Tensor(output);

                _kernel_size = 8;
                if (output.GetLayout().GetLogicalDimensionOrder() == utilities::DimensionOrder({ 0, 1, 2 }))
                {
                    if (result.Channels() % _kernel_size != 0)
                    {
                        _kernel_size = 4;
                    }
                }
                else if (output.GetLayout().GetLogicalDimensionOrder() == utilities::DimensionOrder({ 2, 0, 1 }))
                {
                    if (result.Columns() % _kernel_size != 0)
                    {
                        _kernel_size = 4;
                    }
                }

                // Check the order to see which kernel to use. Additionally, verify that an optimized kernel can run on this input, else fallback
                // to the simple one.
                if (output.GetLayout().GetLogicalDimensionOrder() == utilities::DimensionOrder({ 2, 0, 1 }) && result.Columns() % _kernel_size == 0)
                {
                    // Declare the indexes
                    loopnests::IndexRange i("i", { 0, (int)(data.Rows()) });
                    loopnests::IndexRange j("j", { 0, (int)(data.Columns() / _kernel_size) });
                    loopnests::IndexRange k("k", { 0, (int)(data.Channels()) });

                    auto kernel = loopnests::Kernel("kernel")
                                      .Inputs(input, output)
                                      .Indices(i.GetIndex(), j.GetIndex(), k.GetIndex())
                                      .Define([this](value::Tensor source, value::Tensor dest, value::Scalar i, value::Scalar j, value::Scalar k) {
                                          reorder_kernel_optimized_columns(source, dest, i, j, k);
                                      });

                    loopnests::LoopNest loop(std::vector<loopnests::IndexRange>{ i, j, k });
                    loop.AddKernel(kernel);

                    loopnests::CodeGenerator generator;
                    generator.Run(loop);
                }
                else if (output.GetLayout().GetLogicalDimensionOrder() == utilities::DimensionOrder({ 0, 1, 2 }) && result.Channels() % _kernel_size == 0)
                {
                    // Declare the indexes
                    loopnests::IndexRange i("i", { 0, (int)(data.Rows()) });
                    loopnests::IndexRange j("j", { 0, (int)(data.Columns()) });
                    loopnests::IndexRange k("k", { 0, (int)(data.Channels() / _kernel_size) });

                    auto kernel = loopnests::Kernel("kernel")
                                      .Inputs(input, output)
                                      .Indices(i.GetIndex(), j.GetIndex(), k.GetIndex())
                                      .Define([this](value::Tensor source, value::Tensor dest, value::Scalar i, value::Scalar j, value::Scalar k) {
                                          reorder_kernel_optimized_channels(source, dest, i, j, k);
                                      });

                    loopnests::LoopNest loop(std::vector<loopnests::IndexRange>{ i, j, k });
                    loop.AddKernel(kernel);

                    loopnests::CodeGenerator generator;
                    generator.Run(loop);
                }
                else
                {
                    // Declare the indexes
                    loopnests::IndexRange i("i", { 0, (int)(data.Rows()) });
                    loopnests::IndexRange j("j", { 0, (int)(data.Columns()) });
                    loopnests::IndexRange k("k", { 0, (int)(data.Channels()) });

                    // This is the basic fallback kernel that can do one element at a time
                    auto kernel = loopnests::Kernel("kernel")
                                      .Inputs(input, output)
                                      .Indices(i.GetIndex(), j.GetIndex(), k.GetIndex())
                                      .Define(reorder_kernel_basic);

                    loopnests::LoopNest loop(std::vector<loopnests::IndexRange>{ i, j, k });
                    loop.AddKernel(kernel);

                    loopnests::CodeGenerator generator;
                    generator.Run(loop);
                }
            }
            else if (input.GetLayout().NumDimensions() == 2)
            {
                auto data = value::Matrix(input);
                auto result = value::Matrix(output);

                value::Scalar v = value::Allocate(result.Type(), ell::utilities::ScalarLayout);
                For(data, [&](value::Scalar row, value::Scalar column) {
                    result(row, column) = data(row, column);
                });
            }
            else
            {
                auto data = value::Vector(input);
                auto result = value::Vector(output);

                For(data, [&](value::Scalar index) {
                    result[index] = data[index];
                });
            }
        });
    }

    template <typename ValueType>
    void ReorderDataCodeNode<ValueType>::WriteToArchive(utilities::Archiver& archiver) const
    {
        CompilableNode::WriteToArchive(archiver);
        archiver[defaultInputPortName] << _input;
        archiver["inputLayout"] << _inputMemoryLayout;
        archiver["outputLayout"] << _outputMemoryLayout;
        archiver["paddingValue"] << _paddingValue;
    }

    template <typename ValueType>
    void ReorderDataCodeNode<ValueType>::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        CompilableNode::ReadFromArchive(archiver);
        archiver[defaultInputPortName] >> _input;
        archiver["inputLayout"] >> _inputMemoryLayout;
        archiver["outputLayout"] >> _outputMemoryLayout;
        archiver["paddingValue"] >> _paddingValue;
        _output.SetMemoryLayout(_outputMemoryLayout);
    }

    template <typename ValueType>
    void ReorderDataCodeNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInputs = transformer.GetCorrespondingInputs(_input);
        auto newNode = transformer.AddNode<ReorderDataCodeNode<ValueType>>(newInputs, _inputMemoryLayout, _outputMemoryLayout, _paddingValue);
        transformer.MapNodeOutput(output, newNode->output);
    }

    template <typename ValueType>
    const model::OutputPort<ValueType>& ReorderDataWithCodeNode(const model::OutputPort<ValueType>& input, const model::PortMemoryLayout& outputMemoryLayout, ValueType paddingValue)
    {
        model::Model* model = input.GetNode()->GetModel();
        if (model == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input not part of a model");
        }
        auto node = model->AddNode<ReorderDataCodeNode<ValueType>>(input, outputMemoryLayout, paddingValue);
        return node->output;
    }

    template <typename ValueType>
    const model::OutputPort<ValueType>& ReorderDataWithCodeNode(const model::OutputPort<ValueType>& input, const model::PortMemoryLayout& inputMemoryLayout, const model::PortMemoryLayout& outputMemoryLayout, ValueType paddingValue)
    {
        model::Model* model = input.GetNode()->GetModel();
        if (model == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input not part of a model");
        }
        auto node = model->AddNode<ReorderDataCodeNode<ValueType>>(input, inputMemoryLayout, outputMemoryLayout, paddingValue);
        return node->output;
    }

    template <typename ValueType>
    const model::OutputPort<ValueType>& ReorderDataWithCodeNode(const model::OutputPort<ValueType>& input, const model::PortMemoryLayout& outputMemoryLayout, const model::DimensionOrder& order, ValueType paddingValue)
    {
        model::Model* model = input.GetNode()->GetModel();
        if (model == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input not part of a model");
        }
        auto node = model->AddNode<ReorderDataCodeNode<ValueType>>(input, outputMemoryLayout, order, paddingValue);
        return node->output;
    }

    template <typename ValueType>
    const model::OutputPort<ValueType>& ReorderDataWithCodeNode(const model::OutputPort<ValueType>& input, const model::PortMemoryLayout& inputMemoryLayout, const model::PortMemoryLayout& outputMemoryLayout, const model::DimensionOrder& order, ValueType paddingValue)
    {
        model::Model* model = input.GetNode()->GetModel();
        if (model == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input not part of a model");
        }
        auto node = model->AddNode<ReorderDataCodeNode<ValueType>>(input, inputMemoryLayout, outputMemoryLayout, order, paddingValue);
        return node->output;
    }

    template <typename ValueType>
    const model::OutputPort<ValueType>& ReorderDataWithCodeNode(const model::OutputPort<ValueType>& input, const model::DimensionOrder& order)
    {
        model::Model* model = input.GetNode()->GetModel();
        if (model == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input not part of a model");
        }
        auto node = model->AddNode<ReorderDataCodeNode<ValueType>>(input, order);
        return node->output;
    }

} // namespace nodes
} // namespace ell

#pragma endregion implementation
