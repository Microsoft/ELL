////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     NodeOperations.h (nodes)
//  Authors:  Chris Lovett, Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <emitters/include/CompilableFunction.h>
#include <emitters/include/EmitterTypes.h>
#include <emitters/include/IRMath.h>

namespace ell
{
namespace nodes
{
    inline namespace operations
    {
        /// <summary> Unary operations supported by UnaryOperationNode / BroadcastUnaryOperationNode. </summary>
        enum class UnaryOperationType
        {
            none,
            abs,
            cos,
            exp,
            hardSigmoid,
            hardTanh,
            log,
            log10,
            logicalNot,
            min,
            max,
            sigmoid,
            sign,
            sin,
            softmax,
            sqrt,
            square,
            tanh
        };

        template <typename OperationType>
        std::string ToString(OperationType op);

        template <typename OperationType>
        OperationType FromString(std::string name);

        template <typename ValueType>
        using UnaryFunctionType = emitters::CompilableFunction<ValueType(ValueType)>;

        emitters::UnaryOperatorType ToEmitterType(UnaryOperationType t);

        /// <summary> Binary operations supported by BinaryOperationNode and BroadcastBinaryOperationNode. </summary>
        enum class BinaryOperationType
        {
            none,
            add,
            subtract,
            multiply,
            divide,
            logicalAnd,
            logicalOr,
            logicalXor,
            modulo,
            maximum,
            minimum
        };

        template <typename ValueType>
        using BinaryFunctionType = emitters::CompilableFunction<ValueType(ValueType, ValueType)>;

        emitters::BinaryOperatorType ToEmitterType(BinaryOperationType t);

        /// <summary> Binary predicates (functions returning a boolean value) supported by BinaryPredicateNode. </summary>
        enum class BinaryPredicateType
        {
            none,
            equal,
            less,
            greater,
            notEqual,
            lessOrEqual,
            greaterOrEqual
        };

        emitters::BinaryPredicateType ToEmitterType(BinaryPredicateType t);

        /// <summary> Ternary operations supported by BroadcastTernaryOperationNode. </summary>
        enum class TernaryOperationType
        {
            none,
            fma // fused multiply-add: (a*b) + c
        };

        template <typename ValueType>
        using TernaryFunctionType = emitters::CompilableFunction<ValueType(ValueType, ValueType, ValueType)>;

        emitters::TernaryOperationType ToEmitterType(TernaryOperationType t);

        // CompilableFunction subclasses for the operations

        // Unary functions
        template <typename ValueType>
        class AbsFunction;

        template <typename ValueType>
        class ExpFunction;

        template <typename ValueType>
        class SigmoidFunction;

        template <typename ValueType>
        class HardSigmoidFunction;

        template <typename ValueType>
        class HardTanhFunction;

        template <typename ValueType>
        class LogFunction;

        template <typename ValueType>
        class LogicalNotFunction;

        template <typename ValueType>
        class SqrtFunction;

        template <typename ValueType>
        class SquareFunction;

        template <typename ValueType>
        class TanhFunction;

        template <typename ValueType>
        class SinFunction;

        template <typename ValueType>
        class CosFunction;

        // Binary functions
        template <typename ValueType>
        class AddFunction;

        template <typename ValueType>
        class SubtractFunction;

        template <typename ValueType>
        class MultiplyFunction;

        template <typename ValueType>
        class DivideFunction;

        template <typename ValueType>
        class LogicalAndFunction;

        template <typename ValueType>
        class LogicalOrFunction;

        template <typename ValueType>
        class LogicalXorFunction;

        // Ternary functions
        template <typename ValueType>
        class FMAFunction;
    } // namespace operations
} // namespace nodes
} // namespace ell

#pragma region implementation

namespace ell
{
namespace nodes
{
    inline namespace operations
    {
#define ADD_TO_STRING_ENTRY(NAMESPACE, OPERATOR) \
    case NAMESPACE::OPERATOR:                    \
        return #OPERATOR;
#define BEGIN_FROM_STRING if (false)
#define ADD_FROM_STRING_ENTRY(NAMESPACE, OPERATOR) else if (name == #OPERATOR) return NAMESPACE::OPERATOR

        template <>
        inline std::string ToString(UnaryOperationType op)
        {
            switch (op)
            {
                ADD_TO_STRING_ENTRY(UnaryOperationType, none);
                ADD_TO_STRING_ENTRY(UnaryOperationType, abs);
                ADD_TO_STRING_ENTRY(UnaryOperationType, cos);
                ADD_TO_STRING_ENTRY(UnaryOperationType, exp);
                ADD_TO_STRING_ENTRY(UnaryOperationType, hardSigmoid);
                ADD_TO_STRING_ENTRY(UnaryOperationType, hardTanh);
                ADD_TO_STRING_ENTRY(UnaryOperationType, log);
                ADD_TO_STRING_ENTRY(UnaryOperationType, log10);
                ADD_TO_STRING_ENTRY(UnaryOperationType, logicalNot);
                ADD_TO_STRING_ENTRY(UnaryOperationType, min);
                ADD_TO_STRING_ENTRY(UnaryOperationType, max);
                ADD_TO_STRING_ENTRY(UnaryOperationType, sigmoid);
                ADD_TO_STRING_ENTRY(UnaryOperationType, sign);
                ADD_TO_STRING_ENTRY(UnaryOperationType, sin);
                ADD_TO_STRING_ENTRY(UnaryOperationType, softmax);
                ADD_TO_STRING_ENTRY(UnaryOperationType, sqrt);
                ADD_TO_STRING_ENTRY(UnaryOperationType, square);
                ADD_TO_STRING_ENTRY(UnaryOperationType, tanh);

            default:
                throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange, "Unknown unary operation");
            }
        }

        template <>
        inline UnaryOperationType FromString(std::string name)
        {
            BEGIN_FROM_STRING;
            ADD_FROM_STRING_ENTRY(UnaryOperationType, none);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, abs);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, cos);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, exp);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, hardSigmoid);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, hardTanh);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, log);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, log10);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, logicalNot);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, min);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, max);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, sigmoid);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, sign);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, sin);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, softmax);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, sqrt);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, square);
            ADD_FROM_STRING_ENTRY(UnaryOperationType, tanh);

            throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange, "Unknown unary operation: '" + name + "'");
        }

        inline emitters::UnaryOperatorType ToEmitterType(UnaryOperationType t)
        {
            switch (t)
            {
            case UnaryOperationType::logicalNot:
                return emitters::UnaryOperatorType::logicalNot;
            default:
                throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange, "unary operation is not directly emittable");
            }
        }

        template <>
        inline std::string ToString(BinaryOperationType op)
        {
            switch (op)
            {
                ADD_TO_STRING_ENTRY(BinaryOperationType, none);
                ADD_TO_STRING_ENTRY(BinaryOperationType, add);
                ADD_TO_STRING_ENTRY(BinaryOperationType, subtract);
                ADD_TO_STRING_ENTRY(BinaryOperationType, multiply);
                ADD_TO_STRING_ENTRY(BinaryOperationType, divide);
                ADD_TO_STRING_ENTRY(BinaryOperationType, modulo);
                ADD_TO_STRING_ENTRY(BinaryOperationType, logicalAnd);
                ADD_TO_STRING_ENTRY(BinaryOperationType, logicalOr);
                ADD_TO_STRING_ENTRY(BinaryOperationType, logicalXor);
                ADD_TO_STRING_ENTRY(BinaryOperationType, minimum);
                ADD_TO_STRING_ENTRY(BinaryOperationType, maximum);
            default:
                throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange, "Unknown binary operation");
            }
        }

        template <>
        inline BinaryOperationType FromString(std::string name)
        {
            BEGIN_FROM_STRING;
            ADD_FROM_STRING_ENTRY(BinaryOperationType, none);
            ADD_FROM_STRING_ENTRY(BinaryOperationType, add);
            ADD_FROM_STRING_ENTRY(BinaryOperationType, subtract);
            ADD_FROM_STRING_ENTRY(BinaryOperationType, multiply);
            ADD_FROM_STRING_ENTRY(BinaryOperationType, divide);
            ADD_FROM_STRING_ENTRY(BinaryOperationType, modulo);
            ADD_FROM_STRING_ENTRY(BinaryOperationType, logicalAnd);
            ADD_FROM_STRING_ENTRY(BinaryOperationType, logicalOr);
            ADD_FROM_STRING_ENTRY(BinaryOperationType, logicalXor);
            ADD_FROM_STRING_ENTRY(BinaryOperationType, minimum);
            ADD_FROM_STRING_ENTRY(BinaryOperationType, maximum);

            throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange, "Unknown binary operation");
        }

        inline emitters::BinaryOperatorType ToEmitterType(BinaryOperationType t)
        {
            switch (t)
            {
            case BinaryOperationType::none:
                return emitters::BinaryOperatorType::none;
            case BinaryOperationType::add:
                return emitters::BinaryOperatorType::add;
            case BinaryOperationType::subtract:
                return emitters::BinaryOperatorType::subtract;
            case BinaryOperationType::multiply:
                return emitters::BinaryOperatorType::multiply;
            case BinaryOperationType::divide:
                return emitters::BinaryOperatorType::divide;
            case BinaryOperationType::modulo:
                return emitters::BinaryOperatorType::modulo;
            case BinaryOperationType::logicalAnd:
                return emitters::BinaryOperatorType::logicalAnd;
            case BinaryOperationType::logicalOr:
                return emitters::BinaryOperatorType::logicalOr;
            case BinaryOperationType::logicalXor:
                return emitters::BinaryOperatorType::logicalXor;
            case BinaryOperationType::minimum:
                return emitters::BinaryOperatorType::minimum;
            case BinaryOperationType::maximum:
                return emitters::BinaryOperatorType::maximum;
            default:
                throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange, "Unknown binary operation");
            }
        }

        template <>
        inline std::string ToString(BinaryPredicateType op)
        {
            switch (op)
            {
                ADD_TO_STRING_ENTRY(BinaryPredicateType, none);
                ADD_TO_STRING_ENTRY(BinaryPredicateType, equal);
                ADD_TO_STRING_ENTRY(BinaryPredicateType, less);
                ADD_TO_STRING_ENTRY(BinaryPredicateType, greater);
                ADD_TO_STRING_ENTRY(BinaryPredicateType, notEqual);
                ADD_TO_STRING_ENTRY(BinaryPredicateType, lessOrEqual);
                ADD_TO_STRING_ENTRY(BinaryPredicateType, greaterOrEqual);
            default:
                throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange, "Unknown binary predicate");
            }
        }

        template <>
        inline BinaryPredicateType FromString(std::string name)
        {
            BEGIN_FROM_STRING;
            ADD_FROM_STRING_ENTRY(BinaryPredicateType, none);
            ADD_FROM_STRING_ENTRY(BinaryPredicateType, equal);
            ADD_FROM_STRING_ENTRY(BinaryPredicateType, less);
            ADD_FROM_STRING_ENTRY(BinaryPredicateType, greater);
            ADD_FROM_STRING_ENTRY(BinaryPredicateType, notEqual);
            ADD_FROM_STRING_ENTRY(BinaryPredicateType, lessOrEqual);
            ADD_FROM_STRING_ENTRY(BinaryPredicateType, greaterOrEqual);

            throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange, "Unknown binary predicate");
        }

        inline emitters::BinaryPredicateType ToEmitterType(BinaryPredicateType t)
        {
            switch (t)
            {
            case BinaryPredicateType::none:
                return emitters::BinaryPredicateType::none;
            case BinaryPredicateType::equal:
                return emitters::BinaryPredicateType::equal;
            case BinaryPredicateType::less:
                return emitters::BinaryPredicateType::less;
            case BinaryPredicateType::greater:
                return emitters::BinaryPredicateType::greater;
            case BinaryPredicateType::notEqual:
                return emitters::BinaryPredicateType::notEqual;
            case BinaryPredicateType::lessOrEqual:
                return emitters::BinaryPredicateType::lessOrEqual;
            case BinaryPredicateType::greaterOrEqual:
                return emitters::BinaryPredicateType::greaterOrEqual;
            default:
                throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange, "Unknown binary predicate");
            }
        }

        template <>
        inline std::string ToString(TernaryOperationType op)
        {
            switch (op)
            {
                ADD_TO_STRING_ENTRY(TernaryOperationType, none);
                ADD_TO_STRING_ENTRY(TernaryOperationType, fma);
            default:
                throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange, "Unknown ternary operation");
            }
        }

        template <>
        inline TernaryOperationType FromString(std::string name)
        {
            BEGIN_FROM_STRING;
            ADD_FROM_STRING_ENTRY(TernaryOperationType, none);
            ADD_FROM_STRING_ENTRY(TernaryOperationType, fma);

            throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange, "Unknown ternary operation");
        }
        inline emitters::TernaryOperationType ToEmitterType(TernaryOperationType t)
        {
            switch (t)
            {
            case TernaryOperationType::none:
                return emitters::TernaryOperationType::none;
            case TernaryOperationType::fma:
                return emitters::TernaryOperationType::fma;
            default:
                throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange, "Unknown ternary operation");
            }
        }

        // Unary functions
        template <typename ValueType>
        class AbsFunction : public emitters::CompilableFunction<ValueType(ValueType)>
        {
        public:
            ValueType Compute(ValueType x) const override { return std::abs(x); }
            emitters::IRLocalScalar Compile(emitters::IRFunctionEmitter& function, emitters::IRLocalScalar x) const override { return emitters::Abs(x); };
            static std::string GetTypeName() { return "abs"; }
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }
        };

        template <typename ValueType>
        class ExpFunction : public emitters::CompilableFunction<ValueType(ValueType)>
        {
        public:
            ValueType Compute(ValueType x) const override { return std::exp(x); }
            emitters::IRLocalScalar Compile(emitters::IRFunctionEmitter& function, emitters::IRLocalScalar x) const override { return emitters::Exp(x); };
            static std::string GetTypeName() { return "exp"; }
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }
        };

        template <typename ValueType>
        class LogFunction : public emitters::CompilableFunction<ValueType(ValueType)>
        {
        public:
            ValueType Compute(ValueType x) const override { return std::log(x); }
            emitters::IRLocalScalar Compile(emitters::IRFunctionEmitter& function, emitters::IRLocalScalar x) const override { return emitters::Log(x); };
            static std::string GetTypeName() { return "log"; }
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }
        };

        template <typename ValueType>
        class SqrtFunction : public emitters::CompilableFunction<ValueType(ValueType)>
        {
        public:
            ValueType Compute(ValueType x) const override { return std::sqrt(x); }
            emitters::IRLocalScalar Compile(emitters::IRFunctionEmitter& function, emitters::IRLocalScalar x) const override { return emitters::Sqrt(x); };
            static std::string GetTypeName() { return "sqrt"; }
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }
        };

        template <typename ValueType>
        class SquareFunction : public emitters::CompilableFunction<ValueType(ValueType)>
        {
        public:
            ValueType Compute(ValueType x) const override { return x * x; }
            emitters::IRLocalScalar Compile(emitters::IRFunctionEmitter& function, emitters::IRLocalScalar x) const override { return x * x; };
            static std::string GetTypeName() { return "sqrt"; }
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }
        };

        template <typename ValueType>
        class TanhFunction : public emitters::CompilableFunction<ValueType(ValueType)>
        {
        public:
            ValueType Compute(ValueType x) const override { return std::tanh(x); }
            emitters::IRLocalScalar Compile(emitters::IRFunctionEmitter& function, emitters::IRLocalScalar x) const override { return emitters::Tanh<ValueType>(x); };
            static std::string GetTypeName() { return "tanh"; }
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }
        };

        template <typename ValueType>
        class SinFunction : public emitters::CompilableFunction<ValueType(ValueType)>
        {
        public:
            ValueType Compute(ValueType x) const override { return std::sin(x); }
            emitters::IRLocalScalar Compile(emitters::IRFunctionEmitter& function, emitters::IRLocalScalar x) const override { return emitters::Sin(x); };
            static std::string GetTypeName() { return "sin"; }
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }
        };

        template <typename ValueType>
        class CosFunction : public emitters::CompilableFunction<ValueType(ValueType)>
        {
        public:
            ValueType Compute(ValueType x) const override { return std::cos(x); }
            emitters::IRLocalScalar Compile(emitters::IRFunctionEmitter& function, emitters::IRLocalScalar x) const override { return emitters::Cos(x); };
            static std::string GetTypeName() { return "cos"; }
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }
        };

        // Binary functions
        template <typename ValueType>
        class AddFunction : public emitters::CompilableFunction<ValueType(ValueType, ValueType)>
        {
        public:
            ValueType Compute(ValueType x, ValueType y) const override { return x + y; }
            emitters::IRLocalScalar Compile(emitters::IRFunctionEmitter& function, emitters::IRLocalScalar x, emitters::IRLocalScalar y) const override { return x + y; };
            static std::string GetTypeName() { return "add"; }
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }
        };

        template <typename ValueType>
        class SubtractFunction : public emitters::CompilableFunction<ValueType(ValueType, ValueType)>
        {
        public:
            ValueType Compute(ValueType x, ValueType y) const override { return x - y; }
            emitters::IRLocalScalar Compile(emitters::IRFunctionEmitter& function, emitters::IRLocalScalar x, emitters::IRLocalScalar y) const override { return x - y; };
            static std::string GetTypeName() { return "subtract"; }
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }
        };

        template <typename ValueType>
        class MultiplyFunction : public emitters::CompilableFunction<ValueType(ValueType, ValueType)>
        {
        public:
            ValueType Compute(ValueType x, ValueType y) const override { return x * y; }
            emitters::IRLocalScalar Compile(emitters::IRFunctionEmitter& function, emitters::IRLocalScalar x, emitters::IRLocalScalar y) const override { return x * y; };
            static std::string GetTypeName() { return "multiply"; }
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }
        };

        template <typename ValueType>
        class DivideFunction : public emitters::CompilableFunction<ValueType(ValueType, ValueType)>
        {
        public:
            ValueType Compute(ValueType x, ValueType y) const override { return x / y; }
            emitters::IRLocalScalar Compile(emitters::IRFunctionEmitter& function, emitters::IRLocalScalar x, emitters::IRLocalScalar y) const override { return x / y; };
            static std::string GetTypeName() { return "divide"; }
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }
        };

        // Ternary functions
        template <typename ValueType>
        class FMAFunction : public emitters::CompilableFunction<ValueType(ValueType, ValueType, ValueType)>
        {
        public:
            ValueType Compute(ValueType x, ValueType y, ValueType z) const override { return (x * y) + z; }
            emitters::IRLocalScalar Compile(emitters::IRFunctionEmitter& function, emitters::IRLocalScalar x, emitters::IRLocalScalar y, emitters::IRLocalScalar z) const override { return (x * y) + z; }
            static std::string GetTypeName() { return "fma"; }
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }
        };

    } // namespace operations
} // namespace nodes
} // namespace ell
#pragma endregion
