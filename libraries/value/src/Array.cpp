////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Array.cpp (value)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Array.h"
#include "EmitterContext.h"

#include <utilities/include/Exception.h>

#include <cassert>
#include <functional>

namespace ell
{
using namespace utilities;

namespace value
{
    Array::Array() = default;

    Array::Array(Value value, const std::string& name) :
        _value(value)
    {
        if (!_value.IsDefined() || !_value.IsConstrained())
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Value passed in must be defined and have a memory layout");
        }
        if (_value.GetLayout() == utilities::ScalarLayout)
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Value passed in must not be scalar");
        }
        if (!name.empty())
        {
            SetName(name);
        }
    }

    Array::~Array() = default;
    Array::Array(const Array&) = default;
    Array::Array(Array&&) noexcept = default;

    Array& Array::operator=(const Array& other)
    {
        if (this != &other)
        {
            _value = other._value;
        }
        return *this;
    }

    Array& Array::operator=(Array&& other)
    {
        if (this != &other)
        {
            _value = std::move(other._value);
            other._value = Value();
        }
        return *this;
    }

    Value Array::GetValue() const
    {
        return _value;
    }

    Array Array::Copy() const
    {
        auto newValue = Allocate(_value.GetBaseType(), _value.GetLayout());
        newValue = _value;
        return newValue;
    }

    Scalar Array::operator()(const std::vector<Scalar>& indices)
    {
        if (static_cast<int>(indices.size()) != GetValue().GetLayout().NumDimensions())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidSize);
        }
        Value indexedValue = GetContext().Offset(_value, indices);
        indexedValue.SetLayout(utilities::ScalarLayout);

        return indexedValue;
    }

    Scalar Array::operator()(const std::vector<Scalar>& indices) const
    {
        if (static_cast<int>(indices.size()) != GetValue().GetLayout().NumDimensions())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidSize);
        }
        Value indexedValue = GetContext().Offset(_value, indices);
        indexedValue.SetLayout(utilities::ScalarLayout);

        return Scalar(indexedValue).Copy();
    }

    size_t Array::Size() const
    {
        return _value.GetLayout().NumElements();
    }

    ValueType Array::Type() const
    {
        return _value.GetBaseType();
    }

    void Array::SetName(const std::string& name)
    {
        _value.SetName(name);
    }

    std::string Array::GetName() const
    {
        return _value.GetName();
    }

    void For(Array array, std::function<void(const std::vector<Scalar>&)> fn)
    {
        auto layout = array.GetValue().GetLayout();
        GetContext().For(layout, [fn = std::move(fn), &layout](std::vector<Scalar> coordinates) {
            if (layout.NumDimensions() != static_cast<int>(coordinates.size()))
            {
                throw utilities::InputException(utilities::InputExceptionErrors::invalidSize);
            }

            fn(coordinates);
        });
    }

} // namespace value
} // namespace ell
