////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     BroadcastOperationNodes.h (nodes)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "NodeOperations.h"

#include <model/include/CompilableCodeNode.h>
#include <model/include/Model.h>
#include <model/include/ModelTransformer.h>
#include <model/include/Node.h>
#include <model/include/PortMemoryLayout.h>

#include <utilities/include/Exception.h>
#include <utilities/include/TypeName.h>

#include <value/include/Array.h>
#include <value/include/EmitterContext.h>
#include <value/include/Scalar.h>

#include <value/include/loopnests/CodeGenerator.h>
#include <value/include/loopnests/LoopNest.h>

#include <functional>
#include <numeric>
#include <string>
#include <type_traits>
#include <vector>

namespace ell
{
namespace nodes
{
    using UnaryScalarFunction = value::Scalar (*)(value::Scalar);
    using BinaryScalarFunction = value::Scalar (*)(value::Scalar, value::Scalar);
    using TernaryScalarFunction = value::Scalar (*)(value::Scalar, value::Scalar, value::Scalar);

    /// <summary>
    ///
    /// Broadcast operation nodes perform elementwise operations on multidimensional arrays, using "broadcast" semantics. If the
    /// shape of the operands doesn't match, any dimension with a size of '1' will be "broadcast" to match the other operands.
    /// For instance, if one input is a 2x3x4 array, the other input could be an array of size 1x3x4, 2x1x4, or 2x3x1 (in addition
    /// to matching exactly).
    ///
    /// BroadcastUnaryOperationNodes don't really broadcast anything, but perform unary operations of the form out = f(x), where x is
    /// an element from the primary input.
    ///
    /// </summary>

    // Base class for broadcast nodes
    template <typename ValueType, typename FunctionType>
    class BroadcastOperationNode : public model::CompilableCodeNode
    {
    public:
        /// @name Output Port
        /// @{
        const model::OutputPort<ValueType>& output = _output;
        /// @}

    protected:
        using KernelFunctionType = std::function<void(const std::vector<value::Value>& args)>;

        BroadcastOperationNode(const std::vector<model::InputPortBase*>& inputsPortRefs,
                               const std::vector<const model::OutputPortBase*>& inputs,
                               ValueType padding = 0);

        BroadcastOperationNode(const std::vector<model::InputPortBase*>& inputsPortRefs,
                               const std::vector<const model::OutputPortBase*>& inputs,
                               const model::PortMemoryLayout& outputLayout,
                               ValueType padding = 0);

        int NumDimensions() const { return GetOutputMemoryLayout().NumDimensions(); }
        model::PortMemoryLayout GetOutputMemoryLayout() const;
        model::PortMemoryLayout GetInputMemoryLayout(int index) const;
        const model::OutputPort<ValueType>& GetOutput() const;
        const model::InputPort<ValueType>& GetInput(int index) const;

        void Define(ell::value::FunctionDeclaration& fn) override;
        KernelFunctionType MakeKernel(FunctionType f) const;
        virtual KernelFunctionType GetKernelFunction() const = 0;
        value::Scalar CallKernelFunction(FunctionType f, std::vector<value::Array> inputs, std::vector<std::vector<value::Scalar>> indices) const;

        bool HasState() const override { return true; } // stored state: function and padding value

        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;

    private:
        utilities::ArchiveVersion GetArchiveVersion() const override;
        bool CanReadArchiveVersion(const utilities::ArchiveVersion& version) const override;

        ValueType GetOutputPadding() const { return _paddingValue; }

        model::OutputPort<ValueType> _output;
        ValueType _paddingValue;
    };

    //
    // BroadcastUnaryOperationNode
    //

    template <typename ValueType>
    class BroadcastUnaryOperationNode : public BroadcastOperationNode<ValueType, UnaryScalarFunction>
    {
    public:
        using OperationType = UnaryOperationType;
        using FunctionType = UnaryScalarFunction;

        /// <summary> Default constructor. </summary>
        BroadcastUnaryOperationNode();

        /// <summary> Constructor. </summary>
        ///
        /// <param name="input"> The input of the operation. </param>
        /// <param name="operation"> The type of operation to perform. </param>
        ///
        /// Note: the output will use a contiguous block of memory in canonical order
        BroadcastUnaryOperationNode(const model::OutputPort<ValueType>& input,
                                    OperationType operation,
                                    ValueType padding = 0);

        /// <summary> Constructor. </summary>
        ///
        /// <param name="input"> The input of the operation. </param>
        /// <param name="outputLayout"> The layout to use for the output. </param>
        /// <param name="operation"> The type of operation to perform. </param>
        ///
        /// Note: the output will use a contiguous block of memory in canonical order
        BroadcastUnaryOperationNode(const model::OutputPort<ValueType>& input,
                                    const model::PortMemoryLayout& outputLayout,
                                    OperationType operation,
                                    ValueType padding = 0);

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return utilities::GetCompositeTypeName<ValueType>("BroadcastUnaryOperationNode"); }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

    protected:
        using BroadcastOperationNode<ValueType, FunctionType>::GetOutput;
        using BroadcastOperationNode<ValueType, FunctionType>::MakeKernel;
        using KernelFunctionType = typename BroadcastOperationNode<ValueType, FunctionType>::KernelFunctionType;

        KernelFunctionType GetKernelFunction() const override;

        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;

    private:
        void Copy(model::ModelTransformer& transformer) const override;

        model::InputPort<ValueType> _input;
        OperationType _operation = OperationType::none;
    };

    //
    // BroadcastBinaryOperationNode
    //
    template <typename ValueType>
    class BroadcastBinaryOperationNode : public BroadcastOperationNode<ValueType, BinaryScalarFunction>
    {
    public:
        using OperationType = BinaryOperationType;
        using FunctionType = BinaryScalarFunction;

        /// <summary> Default constructor. </summary>
        BroadcastBinaryOperationNode();

        /// <summary> Constructor. </summary>
        ///
        /// <param name="input1"> The left-hand input of the operation. </param>
        /// <param name="input2"> The right-hand input of the operation. </param>
        /// <param name="operation"> The type of operation to perform. </param>
        ///
        /// Note: the output will use a contiguous block of memory in canonical order
        BroadcastBinaryOperationNode(const model::OutputPort<ValueType>& input1,
                                     const model::OutputPort<ValueType>& input2,
                                     OperationType operation,
                                     ValueType padding = 0);

        /// <summary> Constructor. </summary>
        ///
        /// <param name="input1"> The left-hand input of the operation. </param>
        /// <param name="input2"> The right-hand input of the operation. </param>
        /// <param name="outputLayout"> The layout to use for the output. </param>
        /// <param name="operation"> The type of operation to perform. </param>
        ///
        /// Note: the output will use a contiguous block of memory in canonical order
        BroadcastBinaryOperationNode(const model::OutputPort<ValueType>& input1,
                                     const model::OutputPort<ValueType>& input2,
                                     const model::PortMemoryLayout& outputLayout,
                                     OperationType operation,
                                     ValueType padding = 0);

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return utilities::GetCompositeTypeName<ValueType>("BroadcastBinaryOperationNode"); }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

        /// <summary> Get the type of binary operation this node will perform </summary>
        OperationType GetOperation() const { return _operation; }

    protected:
        using BroadcastOperationNode<ValueType, FunctionType>::GetOutput;
        using BroadcastOperationNode<ValueType, FunctionType>::MakeKernel;
        using KernelFunctionType = typename BroadcastOperationNode<ValueType, FunctionType>::KernelFunctionType;

        KernelFunctionType GetKernelFunction() const override;

        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;

    private:
        void Copy(model::ModelTransformer& transformer) const override;

        model::InputPort<ValueType> _input1;
        model::InputPort<ValueType> _input2;
        OperationType _operation = OperationType::none;
    };

    //
    // BroadcastTernaryOperationNode
    //
    template <typename ValueType>
    class BroadcastTernaryOperationNode : public BroadcastOperationNode<ValueType, TernaryScalarFunction>
    {
    public:
        using OperationType = TernaryOperationType;
        using FunctionType = TernaryScalarFunction;

        /// <summary> Default constructor. </summary>
        BroadcastTernaryOperationNode();

        /// <summary> Constructor. </summary>
        ///
        /// <param name="input1"> The first input of the operation. </param>
        /// <param name="input2"> The second input of the operation. </param>
        /// <param name="input3"> The third input of the operation. </param>
        /// <param name="operation"> The type of operation to perform. </param>
        ///
        /// Note: the output will use a contiguous block of memory in canonical order
        BroadcastTernaryOperationNode(const model::OutputPort<ValueType>& input1,
                                      const model::OutputPort<ValueType>& input2,
                                      const model::OutputPort<ValueType>& input3,
                                      OperationType operation,
                                      ValueType padding = 0);

        /// <summary> Constructor. </summary>
        ///
        /// <param name="input1"> The first input of the operation. </param>
        /// <param name="input2"> The second input of the operation. </param>
        /// <param name="input3"> The third input of the operation. </param>
        /// <param name="outputLayout"> The layout to use for the output. </param>
        /// <param name="operation"> The type of operation to perform. </param>
        ///
        /// Note: the output will use a contiguous block of memory in canonical order
        BroadcastTernaryOperationNode(const model::OutputPort<ValueType>& input1,
                                      const model::OutputPort<ValueType>& input2,
                                      const model::OutputPort<ValueType>& input3,
                                      const model::PortMemoryLayout& outputLayout,
                                      OperationType operation,
                                      ValueType padding = 0);

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return utilities::GetCompositeTypeName<ValueType>("BroadcastTernaryOperationNode"); }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

    protected:
        using BroadcastOperationNode<ValueType, FunctionType>::GetOutput;
        using BroadcastOperationNode<ValueType, FunctionType>::MakeKernel;
        using KernelFunctionType = typename BroadcastOperationNode<ValueType, FunctionType>::KernelFunctionType;

        KernelFunctionType GetKernelFunction() const override;

        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;

    private:
        void Copy(model::ModelTransformer& transformer) const override;

        model::InputPort<ValueType> _input1;
        model::InputPort<ValueType> _input2;
        model::InputPort<ValueType> _input3;
        OperationType _operation = OperationType::none;
    };
} // namespace nodes
} // namespace ell

#pragma region implementation

namespace ell
{
namespace nodes
{
    utilities::MemoryLayout ComputeBroadcastedLayout(const std::vector<utilities::MemoryLayout>& arguments);
    utilities::MemoryLayout ComputeBroadcastedLayout(const std::vector<const model::OutputPortBase*>& arguments);
    void VerifyLayoutsCompatible(const std::vector<utilities::MemoryLayout>& inputs);

    //
    // BroadcastOperationNode
    //
    template <typename ValueType, typename FunctionType>
    BroadcastOperationNode<ValueType, FunctionType>::BroadcastOperationNode(const std::vector<model::InputPortBase*>& inputPortRefs,
                                                                            const std::vector<const model::OutputPortBase*>& inputs,
                                                                            ValueType paddingValue) :
        CompilableCodeNode("BroadcastOperationNode", inputPortRefs, { &_output }),
        _output(this, ell::model::Node::defaultOutputPortName, ComputeBroadcastedLayout(inputs)),
        _paddingValue(paddingValue)
    {
    }

    template <typename ValueType, typename FunctionType>
    BroadcastOperationNode<ValueType, FunctionType>::BroadcastOperationNode(const std::vector<model::InputPortBase*>& inputPortRefs,
                                                                            const std::vector<const model::OutputPortBase*>& inputs,
                                                                            const model::PortMemoryLayout& outputLayout,
                                                                            ValueType paddingValue) :
        CompilableCodeNode("BroadcastOperationNode", inputPortRefs, { &_output }),
        _output(this, ell::model::Node::defaultOutputPortName, outputLayout),
        _paddingValue(paddingValue)
    {
        auto canonicalOutputLayout = ComputeBroadcastedLayout(inputs);
        VerifyLayoutsCompatible({ outputLayout, canonicalOutputLayout });
    }

    template <typename ValueType, typename FunctionType>
    model::PortMemoryLayout BroadcastOperationNode<ValueType, FunctionType>::GetOutputMemoryLayout() const
    {
        return GetOutputPort(0)->GetMemoryLayout();
    }

    template <typename ValueType, typename FunctionType>
    model::PortMemoryLayout BroadcastOperationNode<ValueType, FunctionType>::GetInputMemoryLayout(int inputIndex) const
    {
        return GetInputPort(inputIndex)->GetMemoryLayout();
    }

    template <typename ValueType, typename FunctionType>
    const model::OutputPort<ValueType>& BroadcastOperationNode<ValueType, FunctionType>::GetOutput() const
    {
        return _output;
    }

    template <typename ValueType, typename FunctionType>
    const model::InputPort<ValueType>& BroadcastOperationNode<ValueType, FunctionType>::GetInput(int index) const
    {
        return static_cast<const model::InputPort<ValueType>&>(*GetInputPort(index));
    }

    template <typename ValueType, typename FunctionType>
    void BroadcastOperationNode<ValueType, FunctionType>::Define(ell::value::FunctionDeclaration& fn)
    {
        using namespace value::loopnests;

        (void)fn.Define([this](const std::vector<value::Value>& args) {
            if (static_cast<int>(args.size()) != (this->NumInputPorts() + this->NumOutputPorts()))
            {
                throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState);
            }

            auto outputLayout = this->GetOutputMemoryLayout();
            auto numDim = outputLayout.NumDimensions();

            // Create the indices and ranges for the loop nest
            std::vector<Index> indices;
            std::vector<IndexRange> ranges;
            for (int d = 0; d < numDim; ++d)
            {
                auto name = "i_" + std::to_string(d);
                Index i(name);
                indices.emplace_back(i);
                int size = static_cast<int>(outputLayout.GetLogicalDimensionActiveSize()[d]);
                ranges.push_back({ i, { 0, size } });
            }

            LoopNest loop(ranges);
            auto kernel = value::loopnests::Kernel("kernel")
                              .Inputs(args)
                              .Indices(indices)
                              .Define(GetKernelFunction());
            loop.AddKernel(kernel);

            CodeGenerator generator;
            generator.Run(loop);
        });
    }

    template <typename ValueType, typename FunctionType>
    auto BroadcastOperationNode<ValueType, FunctionType>::MakeKernel(FunctionType f) const -> KernelFunctionType
    {
        // # args = # inputs + # outputs
        // the rest are indices
        return [this, f = std::move(f)](const std::vector<value::Value>& args) {
            if (static_cast<int>(args.size()) != NumDimensions() + NumInputPorts() + 1)
            {
                throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState);
            }

            const int numInputs = this->NumInputPorts();
            std::vector<value::Array> inputs;
            auto it = args.begin();
            for (int i = 0; i < numInputs; ++i)
            {
                inputs.push_back({ *it++ });
            }
            auto output = value::Array(*it++);
            std::vector<std::vector<value::Scalar>> indices(numInputs);
            std::vector<value::Scalar> outputIndices;
            int dimension = 0;
            for (; it != args.end(); ++it)
            {
                for (int i = 0; i < numInputs; ++i)
                {
                    indices[i].push_back(GetInputMemoryLayout(i).GetLogicalDimensionActiveSize(dimension) > 1 ? value::Scalar{ *it } : value::Scalar(0));
                }
                outputIndices.push_back({ *it });
                ++dimension;
            }

            output(outputIndices) = CallKernelFunction(f, inputs, indices);
        };
    }

    template <typename ValueType, typename FunctionType>
    value::Scalar BroadcastOperationNode<ValueType, FunctionType>::CallKernelFunction(FunctionType f, std::vector<value::Array> inputs, std::vector<std::vector<value::Scalar>> indices) const
    {
        // TODO: if FunctionType was a function that took a vector of inputs, then we could dispense with this `if constexpr` block
        if constexpr (std::is_same_v<FunctionType, UnaryScalarFunction>)
        {
            return f(inputs[0](indices[0]));
        }
        else if constexpr (std::is_same_v<FunctionType, BinaryScalarFunction>)
        {
            return f(inputs[0](indices[0]), inputs[1](indices[1]));
        }
        else if constexpr (std::is_same_v<FunctionType, TernaryScalarFunction>)
        {
            return f(inputs[0](indices[0]), inputs[1](indices[1]), inputs[2](indices[2]));
        }
        else
        {
            throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState);
        }
    }

    template <typename ValueType, typename FunctionType>
    utilities::ArchiveVersion BroadcastOperationNode<ValueType, FunctionType>::GetArchiveVersion() const
    {
        constexpr utilities::ArchiveVersion archiveVersion = { utilities::ArchiveVersionNumbers::v5_refined_nodes };

        return archiveVersion;
    }

    template <typename ValueType, typename FunctionType>
    bool BroadcastOperationNode<ValueType, FunctionType>::CanReadArchiveVersion(const utilities::ArchiveVersion& version) const
    {
        constexpr utilities::ArchiveVersion archiveVersion = { utilities::ArchiveVersionNumbers::v5_refined_nodes };

        return version >= archiveVersion;
    }

    template <typename ValueType, typename FunctionType>
    void BroadcastOperationNode<ValueType, FunctionType>::WriteToArchive(utilities::Archiver& archiver) const
    {
        model::CompilableCodeNode::WriteToArchive(archiver);
        auto outputLayout = GetOutputMemoryLayout();
        archiver["outputLayout"] << outputLayout;
        archiver["padding"] << _paddingValue;
    }

    template <typename ValueType, typename FunctionType>
    void BroadcastOperationNode<ValueType, FunctionType>::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        model::CompilableCodeNode::ReadFromArchive(archiver);
        model::PortMemoryLayout outputLayout;
        archiver["outputLayout"] >> outputLayout;
        _output.SetMemoryLayout(outputLayout);
        archiver["padding"] >> _paddingValue;
    }

    //
    // BroadcastUnaryOperationNode
    //
    template <typename ValueType>
    BroadcastUnaryOperationNode<ValueType>::BroadcastUnaryOperationNode() :
        BroadcastOperationNode<ValueType, FunctionType>({ &_input }, {}, static_cast<ValueType>(0)),
        _input(this, {}, model::CompilableCodeNode::defaultInputPortName),
        _operation(OperationType::none)
    {
    }

    template <typename ValueType>
    BroadcastUnaryOperationNode<ValueType>::BroadcastUnaryOperationNode(const model::OutputPort<ValueType>& input, OperationType operation, ValueType paddingValue) :
        BroadcastOperationNode<ValueType, FunctionType>({ &_input }, { &input }, paddingValue),
        _input(this, input, model::CompilableCodeNode::defaultInputPortName),
        _operation(operation)
    {
    }

    template <typename ValueType>
    BroadcastUnaryOperationNode<ValueType>::BroadcastUnaryOperationNode(const model::OutputPort<ValueType>& input, const model::PortMemoryLayout& outputLayout, OperationType operation, ValueType paddingValue) :
        BroadcastOperationNode<ValueType, FunctionType>({ &_input }, { &input }, outputLayout, paddingValue),
        _input(this, input, model::CompilableCodeNode::defaultInputPortName),
        _operation(operation)
    {
    }

    template <typename ValueType>
    void BroadcastUnaryOperationNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInput = transformer.GetCorrespondingInputs(_input);
        auto newNode = transformer.AddNode<BroadcastUnaryOperationNode<ValueType>>(newInput,
                                                                                   this->GetOutputMemoryLayout(),
                                                                                   _operation);
        transformer.MapNodeOutput(GetOutput(), newNode->output);
    }

    template <typename ValueType>
    void BroadcastUnaryOperationNode<ValueType>::WriteToArchive(utilities::Archiver& archiver) const
    {
        BroadcastOperationNode<ValueType, FunctionType>::WriteToArchive(archiver);
        archiver[model::CompilableCodeNode::defaultInputPortName] << _input;
        archiver["operation"] << ToString(_operation);
    }

    template <typename ValueType>
    void BroadcastUnaryOperationNode<ValueType>::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        BroadcastOperationNode<ValueType, FunctionType>::ReadFromArchive(archiver);
        archiver[model::CompilableCodeNode::defaultInputPortName] >> _input;
        std::string operation;
        archiver["operation"] >> operation;
        _operation = FromString<UnaryOperationType>(operation);
    }

    //
    // BroadcastBinaryOperationNode
    //
    template <typename ValueType>
    BroadcastBinaryOperationNode<ValueType>::BroadcastBinaryOperationNode() :
        BroadcastOperationNode<ValueType, FunctionType>({ &_input1, &_input2 }, {}),
        _input1(this, {}, model::CompilableCodeNode::defaultInput1PortName),
        _input2(this, {}, model::CompilableCodeNode::defaultInput2PortName),
        _operation(OperationType::none)
    {
    }

    template <typename ValueType>
    BroadcastBinaryOperationNode<ValueType>::BroadcastBinaryOperationNode(const model::OutputPort<ValueType>& input1,
                                                                          const model::OutputPort<ValueType>& input2,
                                                                          OperationType operation,
                                                                          ValueType paddingValue) :
        BroadcastOperationNode<ValueType, FunctionType>({ &_input1, &_input2 }, { &input1, &input2 }, paddingValue),
        _input1(this, input1, model::CompilableCodeNode::defaultInput1PortName),
        _input2(this, input2, model::CompilableCodeNode::defaultInput2PortName),
        _operation(operation)
    {
    }

    template <typename ValueType>
    BroadcastBinaryOperationNode<ValueType>::BroadcastBinaryOperationNode(const model::OutputPort<ValueType>& input1,
                                                                          const model::OutputPort<ValueType>& input2,
                                                                          const model::PortMemoryLayout& outputLayout,
                                                                          OperationType operation,
                                                                          ValueType paddingValue) :
        BroadcastOperationNode<ValueType, FunctionType>({ &_input1, &_input2 }, { &input1, &input2 }, outputLayout, paddingValue),
        _input1(this, input1, model::CompilableCodeNode::defaultInput1PortName),
        _input2(this, input2, model::CompilableCodeNode::defaultInput2PortName),
        _operation(operation)
    {
    }

    template <typename ValueType>
    void BroadcastBinaryOperationNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInput1 = transformer.GetCorrespondingInputs(_input1);
        const auto& newInput2 = transformer.GetCorrespondingInputs(_input2);
        auto newNode = transformer.AddNode<BroadcastBinaryOperationNode<ValueType>>(newInput1,
                                                                                    newInput2,
                                                                                    this->GetOutputMemoryLayout(),
                                                                                    _operation);
        transformer.MapNodeOutput(GetOutput(), newNode->output);
    }

    template <typename ValueType>
    void BroadcastBinaryOperationNode<ValueType>::WriteToArchive(utilities::Archiver& archiver) const
    {
        BroadcastOperationNode<ValueType, FunctionType>::WriteToArchive(archiver);
        archiver[model::CompilableCodeNode::defaultInput1PortName] << _input1;
        archiver[model::CompilableCodeNode::defaultInput2PortName] << _input2;
        archiver["operation"] << ToString(_operation);
    }

    template <typename ValueType>
    void BroadcastBinaryOperationNode<ValueType>::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        BroadcastOperationNode<ValueType, FunctionType>::ReadFromArchive(archiver);
        archiver[model::CompilableCodeNode::defaultInput1PortName] >> _input1;
        archiver[model::CompilableCodeNode::defaultInput2PortName] >> _input2;
        std::string operation;
        archiver["operation"] >> operation;
        _operation = FromString<BinaryOperationType>(operation);
    }

    //
    // BroadcastTernaryOperationNode
    //
    template <typename ValueType>
    BroadcastTernaryOperationNode<ValueType>::BroadcastTernaryOperationNode() :
        BroadcastOperationNode<ValueType, FunctionType>({ &_input1, &_input2, &_input3 }, {}),
        _input1(this, {}, model::CompilableCodeNode::defaultInput1PortName),
        _input2(this, {}, model::CompilableCodeNode::defaultInput2PortName),
        _input3(this, {}, model::CompilableCodeNode::defaultInput3PortName),
        _operation(OperationType::none)
    {
    }

    template <typename ValueType>
    BroadcastTernaryOperationNode<ValueType>::BroadcastTernaryOperationNode(const model::OutputPort<ValueType>& input1,
                                                                            const model::OutputPort<ValueType>& input2,
                                                                            const model::OutputPort<ValueType>& input3,
                                                                            OperationType operation,
                                                                            ValueType paddingValue) :
        BroadcastOperationNode<ValueType, FunctionType>({ &_input1, &_input2, &_input3 }, { &input1, &input2, &input3 }, paddingValue),
        _input1(this, input1, model::CompilableCodeNode::defaultInput1PortName),
        _input2(this, input2, model::CompilableCodeNode::defaultInput2PortName),
        _input3(this, input3, model::CompilableCodeNode::defaultInput3PortName),
        _operation(operation)
    {
    }

    template <typename ValueType>
    BroadcastTernaryOperationNode<ValueType>::BroadcastTernaryOperationNode(const model::OutputPort<ValueType>& input1,
                                                                            const model::OutputPort<ValueType>& input2,
                                                                            const model::OutputPort<ValueType>& input3,
                                                                            const model::PortMemoryLayout& outputLayout,
                                                                            OperationType operation,
                                                                            ValueType paddingValue) :
        BroadcastOperationNode<ValueType, FunctionType>({ &_input1, &_input2, &_input3 }, { &input1, &input2, &input3 }, outputLayout, paddingValue),
        _input1(this, input1, model::CompilableCodeNode::defaultInput1PortName),
        _input2(this, input2, model::CompilableCodeNode::defaultInput2PortName),
        _input3(this, input3, model::CompilableCodeNode::defaultInput3PortName),
        _operation(operation)
    {
    }

    template <typename ValueType>
    void BroadcastTernaryOperationNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInput1 = transformer.GetCorrespondingInputs(_input1);
        const auto& newInput2 = transformer.GetCorrespondingInputs(_input2);
        const auto& newInput3 = transformer.GetCorrespondingInputs(_input3);
        auto newNode = transformer.AddNode<BroadcastTernaryOperationNode<ValueType>>(newInput1,
                                                                                     newInput2,
                                                                                     newInput3,
                                                                                     this->GetOutputMemoryLayout(),
                                                                                     _operation);
        transformer.MapNodeOutput(GetOutput(), newNode->output);
    }

    template <typename ValueType>
    void BroadcastTernaryOperationNode<ValueType>::WriteToArchive(utilities::Archiver& archiver) const
    {
        BroadcastOperationNode<ValueType, FunctionType>::WriteToArchive(archiver);
        archiver[model::CompilableCodeNode::defaultInput1PortName] << _input1;
        archiver[model::CompilableCodeNode::defaultInput2PortName] << _input2;
        archiver[model::CompilableCodeNode::defaultInput3PortName] << _input3;
        archiver["operation"] << ToString(_operation);
    }

    template <typename ValueType>
    void BroadcastTernaryOperationNode<ValueType>::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        BroadcastOperationNode<ValueType, FunctionType>::ReadFromArchive(archiver);
        archiver[model::CompilableCodeNode::defaultInput1PortName] >> _input1;
        archiver[model::CompilableCodeNode::defaultInput2PortName] >> _input2;
        archiver[model::CompilableCodeNode::defaultInput3PortName] >> _input3;
        std::string operation;
        archiver["operation"] >> operation;
        _operation = FromString<TernaryOperationType>(operation);
    }

    template <typename ValueType>
    auto BroadcastUnaryOperationNode<ValueType>::GetKernelFunction() const -> KernelFunctionType
    {
        switch (_operation)
        {
        case UnaryOperationType::abs:
            return MakeKernel(value::Abs);
            break;
        case UnaryOperationType::exp:
            return MakeKernel(value::Exp);
            break;
        case UnaryOperationType::log:
            return MakeKernel(value::Log);
            break;
        case UnaryOperationType::log10:
            return MakeKernel(value::Log10);
            break;
        case UnaryOperationType::sqrt:
            return MakeKernel(value::Sqrt);
            break;
        case UnaryOperationType::logicalNot:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Not implemented");
        case UnaryOperationType::tanh:
            return MakeKernel(value::Tanh);
            break;
        case UnaryOperationType::square:
            return MakeKernel(value::Square);
            break;
        case UnaryOperationType::sin:
            return MakeKernel(value::Sin);
            break;
        case UnaryOperationType::cos:
            return MakeKernel(value::Cos);
            break;
        default:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Illegal operation");
        }
    }

    template <typename ValueType>
    auto BroadcastBinaryOperationNode<ValueType>::GetKernelFunction() const -> KernelFunctionType
    {
        switch (_operation)
        {
        case BinaryOperationType::add:
            return MakeKernel(value::Add);
            break;
        case BinaryOperationType::subtract:
            return MakeKernel(value::Subtract);
            break;
        case BinaryOperationType::multiply:
            return MakeKernel(value::Multiply);
            break;
        case BinaryOperationType::divide:
            return MakeKernel(value::Divide);
            break;
        case BinaryOperationType::modulo:
            return MakeKernel(value::Modulo);
            break;
        case BinaryOperationType::logicalAnd:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Not implemented");
        case BinaryOperationType::logicalOr:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Not implemented");
        case BinaryOperationType::logicalXor:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Not implemented");
        default:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Illegal operation");
        }
    }

    template <typename ValueType>
    auto BroadcastTernaryOperationNode<ValueType>::GetKernelFunction() const -> KernelFunctionType
    {
        switch (_operation)
        {
        case TernaryOperationType::fma:
            return MakeKernel(value::FusedMultiplyAdd);
            break;
        default:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Illegal operation");
        }
    }
} // namespace nodes
} // namespace ell

#pragma endregion implementation
