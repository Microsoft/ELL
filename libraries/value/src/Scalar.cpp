////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Scalar.cpp (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Scalar.h"
#include "EmitterContext.h"

#include <utilities/include/Exception.h>

namespace ell
{
namespace value
{
    using namespace utilities;

    Scalar::Scalar() = default;

    Scalar::Scalar(Value value, const std::string& name) :
        _value(std::move(value))
    {
        if (!_value.IsDefined() || !_value.IsConstrained() || _value.GetLayout().NumDimensions() != 0)
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }
        if (!name.empty())
        {
            SetName(name);
        }
    }

    Scalar::~Scalar() = default;
    Scalar::Scalar(const Scalar&) = default;
    Scalar::Scalar(Scalar&&) noexcept = default;

    Scalar& Scalar::operator=(const Scalar& other)
    {
        if (this != &other)
        {
            _value = other._value;
        }
        return *this;
    }

    Scalar& Scalar::operator=(Scalar&& other)
    {
        if (this != &other)
        {
            _value = std::move(other._value);
            other._value = Value();
        }
        return *this;
    }

    Value Scalar::GetValue() const
    {
        return _value;
    }

    Scalar Scalar::Copy() const
    {
        Scalar s(Allocate(GetType(), MemoryLayout{}));
        s = *this;
        return s;
    }

    ValueType Scalar::GetType() const
    {
        return _value.GetBaseType();
    }

    void Scalar::SetName(const std::string& name)
    {
        _value.SetName(name);
    }

    std::string Scalar::GetName() const
    {
        return _value.GetName();
    }

    Scalar& Scalar::operator+=(Scalar s)
    {
        _value = GetContext().BinaryOperation(ValueBinaryOperation::add, _value, s._value);
        return *this;
    }

    Scalar& Scalar::operator-=(Scalar s)
    {
        _value = GetContext().BinaryOperation(ValueBinaryOperation::subtract, _value, s._value);
        return *this;
    }

    Scalar& Scalar::operator*=(Scalar s)
    {
        _value = GetContext().BinaryOperation(ValueBinaryOperation::multiply, _value, s._value);
        return *this;
    }

    Scalar& Scalar::operator/=(Scalar s)
    {
        _value = GetContext().BinaryOperation(ValueBinaryOperation::divide, _value, s._value);
        return *this;
    }

    Scalar& Scalar::operator%=(Scalar s)
    {
        _value = GetContext().BinaryOperation(ValueBinaryOperation::modulus, _value, s._value);
        return *this;
    }

    // Free function operator overloads
    Scalar operator+(Scalar s1, Scalar s2)
    {
        return Add(s1, s2);
    }

    Scalar operator-(Scalar s1, Scalar s2)
    {
        return Subtract(s1, s2);
    }

    Scalar operator*(Scalar s1, Scalar s2)
    {
        return Multiply(s1, s2);
    }

    Scalar operator/(Scalar s1, Scalar s2)
    {
        return Divide(s1, s2);
    }

    Scalar operator%(Scalar s1, Scalar s2)
    {
        return Modulo(s1, s2);
    }

    Scalar operator-(Scalar s)
    {
        return Cast(0, s.GetType()) - s;
    }

    Scalar operator++(Scalar s)
    {
        if (!s.GetValue().IsIntegral())
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        return s += Cast(1, s.GetType());
    }

    Scalar operator++(Scalar s, int)
    {
        if (!s.GetValue().IsIntegral())
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        Scalar copy = s.Copy();
        s += Cast(1, s.GetType());
        return copy;
    }

    Scalar operator--(Scalar s)
    {
        if (!s.GetValue().IsIntegral())
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        return s -= Cast(1, s.GetType());
    }

    Scalar operator--(Scalar s, int)
    {
        if (!s.GetValue().IsIntegral())
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        Scalar copy = s.Copy();
        s -= Cast(1, s.GetType());
        return copy;
    }

    Scalar operator==(Scalar s1, Scalar s2)
    {
        return GetContext().LogicalOperation(ValueLogicalOperation::equality, s1.GetValue(), s2.GetValue());
    }

    Scalar operator!=(Scalar s1, Scalar s2)
    {
        return GetContext().LogicalOperation(ValueLogicalOperation::inequality, s1.GetValue(), s2.GetValue());
    }

    Scalar operator<=(Scalar s1, Scalar s2)
    {
        return GetContext().LogicalOperation(ValueLogicalOperation::lessthanorequal, s1.GetValue(), s2.GetValue());
    }

    Scalar operator<(Scalar s1, Scalar s2)
    {
        return GetContext().LogicalOperation(ValueLogicalOperation::lessthan, s1.GetValue(), s2.GetValue());
    }

    Scalar operator>=(Scalar s1, Scalar s2)
    {
        return GetContext().LogicalOperation(ValueLogicalOperation::greaterthanorequal, s1.GetValue(), s2.GetValue());
    }

    Scalar operator>(Scalar s1, Scalar s2)
    {
        return GetContext().LogicalOperation(ValueLogicalOperation::greaterthan, s1.GetValue(), s2.GetValue());
    }

    Scalar operator&&(Scalar s1, Scalar s2)
    {
        return GetContext().BinaryOperation(ValueBinaryOperation::logicalAnd, s1.GetValue(), s2.GetValue());
    }

    Scalar operator||(Scalar s1, Scalar s2)
    {
        return GetContext().BinaryOperation(ValueBinaryOperation::logicalOr, s1.GetValue(), s2.GetValue());
    }

    Scalar MakeScalar(ValueType type, const std::string& name)
    {
        return Scalar(Allocate(type, ScalarLayout), name);
    }

} // namespace value
} // namespace ell
