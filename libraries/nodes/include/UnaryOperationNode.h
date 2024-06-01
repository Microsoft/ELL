////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     UnaryOperationNode.h (nodes)
//  Authors:  Chuck Jacobs, Chris Lovett
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <model/include/CompilableCodeNode.h>
#include <model/include/IRMapCompiler.h>
#include <model/include/InputPort.h>
#include <model/include/OutputPort.h>

#include <value/include/FunctionDeclaration.h>

#include "NodeOperations.h"

#include <string>
#include <vector>

namespace ell
{
namespace nodes
{
    /// <summary> A node that represents a unary function of its input </summary>
    template <typename ValueType>
    class UnaryOperationNode : public model::CompilableCodeNode
    {
    public:
        /// @name Input and Output Ports
        /// @{
        const model::InputPort<ValueType>& input = _input;
        const model::OutputPort<ValueType>& output = _output;
        /// @}

        /// <summary> Default Constructor </summary>
        UnaryOperationNode();

        /// <summary> Constructor </summary>
        ///
        /// <param name="input"> The signal to process. </param>
        /// <param name="operation"> The function to use to process the signal. </param>
        UnaryOperationNode(const model::OutputPort<ValueType>& input, UnaryOperationType operation);

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return utilities::GetCompositeTypeName<ValueType>("UnaryOperationNode"); }

        /// <summary> Gets the operation performed by this node </summary>
        ///
        /// <returns> The operation </returns>
        UnaryOperationType GetOperation() const { return _operation; }

    protected:
        void Define(ell::value::FunctionDeclaration& fn) override;
        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;
        bool HasState() const override { return true; } // stored state: operation
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

    private:
        void Copy(model::ModelTransformer& transformer) const override;

        // Inputs
        model::InputPort<ValueType> _input;

        // Output
        model::OutputPort<ValueType> _output;

        // Operation
        UnaryOperationType _operation;
    };

    /// <summary> Convenience function for adding a node to a model. </summary>
    ///
    /// <param name="input"> The port to get the input data from </param>
    /// <param name="operation"> The function to use to process the signal. </param>
    ///
    /// <returns> The output of the new node. </returns>
    template <typename ValueType>
    const model::OutputPort<ValueType>& UnaryOperation(const model::OutputPort<ValueType>& input, UnaryOperationType operation);

    /// @{
    /// <summary> Convenience functions for adding a node to a model. </summary>
    /// <param name="input"> The port to get the input data from </param>
    /// <returns> The output of the new node. </returns>
    template <typename ValueType, utilities::IsNotBoolean<ValueType> = true>
    const model::OutputPort<ValueType>& Abs(const model::OutputPort<ValueType>& input);

    template <typename ValueType, utilities::IsNotBoolean<ValueType> = true>
    const model::OutputPort<ValueType>& Cos(const model::OutputPort<ValueType>& input);

    template <typename ValueType, utilities::IsNotBoolean<ValueType> = true>
    const model::OutputPort<ValueType>& Exp(const model::OutputPort<ValueType>& input);

    template <typename ValueType, utilities::IsNotBoolean<ValueType> = true>
    const model::OutputPort<ValueType>& HardSigmoid(const model::OutputPort<ValueType>& input);

    template <typename ValueType, utilities::IsNotBoolean<ValueType> = true>
    const model::OutputPort<ValueType>& HardTanh(const model::OutputPort<ValueType>& input);

    template <typename ValueType, utilities::IsNotBoolean<ValueType> = true>
    const model::OutputPort<ValueType>& Log(const model::OutputPort<ValueType>& input);

    template <typename ValueType, utilities::IsNotBoolean<ValueType> = true>
    const model::OutputPort<ValueType>& Log10(const model::OutputPort<ValueType>& input);

    template <typename ValueType, utilities::IsBoolean<ValueType> = true>
    const model::OutputPort<ValueType>& LogicalNot(const model::OutputPort<ValueType>& input);

    template <typename ValueType, utilities::IsNotBoolean<ValueType> = true>
    const model::OutputPort<ValueType>& Sigmoid(const model::OutputPort<ValueType>& input);

    template <typename ValueType, utilities::IsNotBoolean<ValueType> = true>
    const model::OutputPort<ValueType>& Sign(const model::OutputPort<ValueType>& input);

    template <typename ValueType, utilities::IsNotBoolean<ValueType> = true>
    const model::OutputPort<ValueType>& Sin(const model::OutputPort<ValueType>& input);

    template <typename ValueType, utilities::IsNotBoolean<ValueType> = true>
    const model::OutputPort<ValueType>& Sqrt(const model::OutputPort<ValueType>& input);

    template <typename ValueType, utilities::IsNotBoolean<ValueType> = true>
    const model::OutputPort<ValueType>& Square(const model::OutputPort<ValueType>& input);

    template <typename ValueType, utilities::IsNotBoolean<ValueType> = true>
    const model::OutputPort<ValueType>& Tanh(const model::OutputPort<ValueType>& input);
    /// @}

} // namespace nodes
} // namespace ell

#pragma region implementation

namespace ell
{
namespace nodes
{
    template <typename ValueType>
    const model::OutputPort<ValueType>& UnaryOperation(const model::OutputPort<ValueType>& input, UnaryOperationType operation)
    {
        model::Model* model = input.GetNode()->GetModel();
        if (model == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input not part of a model");
        }
        auto node = model->AddNode<UnaryOperationNode<ValueType>>(input, operation);
        return node->output;
    }

    template <typename ValueType, utilities::IsNotBoolean<ValueType>>
    const model::OutputPort<ValueType>& Abs(const model::OutputPort<ValueType>& input)
    {
        return UnaryOperation(input, UnaryOperationType::abs);
    }

    template <typename ValueType, utilities::IsNotBoolean<ValueType>>
    const model::OutputPort<ValueType>& Cos(const model::OutputPort<ValueType>& input)
    {
        return UnaryOperation(input, UnaryOperationType::cos);
    }

    template <typename ValueType, utilities::IsNotBoolean<ValueType>>
    const model::OutputPort<ValueType>& Exp(const model::OutputPort<ValueType>& input)
    {
        return UnaryOperation(input, UnaryOperationType::exp);
    }

    template <typename ValueType, utilities::IsNotBoolean<ValueType>>
    const model::OutputPort<ValueType>& HardSigmoid(const model::OutputPort<ValueType>& input)
    {
        return UnaryOperation(input, UnaryOperationType::hardSigmoid);
    }

    template <typename ValueType, utilities::IsNotBoolean<ValueType>>
    const model::OutputPort<ValueType>& HardTanh(const model::OutputPort<ValueType>& input)
    {
        return UnaryOperation(input, UnaryOperationType::hardTanh);
    }

    template <typename ValueType, utilities::IsNotBoolean<ValueType>>
    const model::OutputPort<ValueType>& Log(const model::OutputPort<ValueType>& input)
    {
        return UnaryOperation(input, UnaryOperationType::log);
    }

    template <typename ValueType, utilities::IsNotBoolean<ValueType>>
    const model::OutputPort<ValueType>& Log10(const model::OutputPort<ValueType>& input)
    {
        return UnaryOperation(input, UnaryOperationType::log10);
    }

    template <typename ValueType, utilities::IsBoolean<ValueType>>
    const model::OutputPort<ValueType>& LogicalNot(const model::OutputPort<ValueType>& input)
    {
        return UnaryOperation(input, UnaryOperationType::logicalNot);
    }

    template <typename ValueType, utilities::IsNotBoolean<ValueType>>
    const model::OutputPort<ValueType>& Sigmoid(const model::OutputPort<ValueType>& input)
    {
        return UnaryOperation(input, UnaryOperationType::sigmoid);
    }

    template <typename ValueType, utilities::IsNotBoolean<ValueType>>
    const model::OutputPort<ValueType>& Sign(const model::OutputPort<ValueType>& input)
    {
        return UnaryOperation(input, UnaryOperationType::sign);
    }

    template <typename ValueType, utilities::IsNotBoolean<ValueType>>
    const model::OutputPort<ValueType>& Sin(const model::OutputPort<ValueType>& input)
    {
        return UnaryOperation(input, UnaryOperationType::sin);
    }

    template <typename ValueType, utilities::IsNotBoolean<ValueType>>
    const model::OutputPort<ValueType>& Sqrt(const model::OutputPort<ValueType>& input)
    {
        return UnaryOperation(input, UnaryOperationType::sqrt);
    }

    template <typename ValueType, utilities::IsNotBoolean<ValueType>>
    const model::OutputPort<ValueType>& Square(const model::OutputPort<ValueType>& input)
    {
        return UnaryOperation(input, UnaryOperationType::square);
    }

    template <typename ValueType, utilities::IsNotBoolean<ValueType>>
    const model::OutputPort<ValueType>& Tanh(const model::OutputPort<ValueType>& input)
    {
        return UnaryOperation(input, UnaryOperationType::tanh);
    }
} // namespace nodes
} // namespace ell

#pragma endregion implementation
