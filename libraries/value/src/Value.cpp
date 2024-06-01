////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Value.cpp (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Value.h"
#include "EmitterContext.h"
#include "Scalar.h"

#include <utilities/include/Hash.h>
#include <utilities/include/TypeAliases.h>
#include <utilities/include/TypeTraits.h>

namespace ell
{
namespace value
{
    using namespace detail;
    using namespace utilities;

    Value::Value() = default;

    Value::~Value() = default;

    Value::Value(const Value&) = default;

    Value::Value(Value&& other) noexcept
    {
        // GCC 8's stdlib doesn't have std::swap for std::variant
        auto temp = _data;
        _data = other._data;
        other._data = temp;
        std::swap(_type, other._type);
        std::swap(_layout, other._layout);
        std::swap(_hasName, other._hasName);
    }

    // clang-format off
    /****************************************************************************
    Copy assignment has to be context aware. The terms "defined", "empty", and
    "constrained" match their definitions of the respective Value member functions.
    Shallow copies of data is done in this function. Deep copies are done by the
    global EmitterContext object.
    * if lhs is not defined
      * and rhs is not defined: don't do anything
      * and rhs is defined: copy everything
    * if lhs is defined and not constrained and empty
      * and rhs is not defined: throw
      * and rhs is defined: if types don't match, throw. copy everything
    * if lhs is defined and constrained and empty
      * and rhs is not defined: throw
      * and rhs is defined and not constrained and empty: if types don't match, throw.
      * and rhs is defined and constrained and empty: if types or layout don't match, throw
      * and rhs is defined and not constrained and not empty: if types don't match, throw. shallow copy of data
      * and rhs is defined and constrained and not empty: if types or layout don't match, throw. shallow copy of data
    * if lhs is defined and not constrained and not empty:
      * throw
    * if lhs is defined and constrained and not empty:
      * and rhs does not match type or layout: throw
      * and rhs matches type and layout: deep copy of data
    ****************************************************************************/
    // clang-format on
    Value& Value::operator=(const Value& other)
    {
        if (this != &other)
        {
            if (!IsDefined())
            {
                // lhs is not defined
                if (other.IsDefined())
                {
                    _layout = other._layout;
                    _data = other._data;
                    _type = other._type;
                    _hasName = other._hasName;
                }
            }
            else
            {
                // lhs is defined
                if (!other.IsDefined())
                {
                    throw InputException(InputExceptionErrors::invalidArgument);
                }
                if (GetBaseType() != other.GetBaseType())
                {
                    throw InputException(InputExceptionErrors::typeMismatch);
                }

                if (!IsConstrained())
                {
                    // lhs is not constrained
                    if (IsEmpty())
                    {
                        _layout = other._layout;
                        _data = other._data;
                        _type = other._type;
                        _hasName = other._hasName;
                    }
                    else
                    {
                        throw LogicException(LogicExceptionErrors::illegalState);
                    }
                }
                else
                {
                    // lhs is constrained
                    if (IsEmpty())
                    {
                        if (other.IsConstrained() && GetLayout() != other.GetLayout())
                        {
                            throw InputException(InputExceptionErrors::sizeMismatch);
                        }
                        _data = other._data;
                        _type = other._type;
                        _hasName = other._hasName;
                    }
                    else
                    {
                        if (!other.IsConstrained() || GetLayout() != other.GetLayout())
                        {
                            throw InputException(InputExceptionErrors::sizeMismatch);
                        }
                        GetContext().CopyData(other, *this);
                    }
                }
            }
        }

        return *this;
    }

    // Behaves similarly to the copy assignment, except rhs is reset afterwards
    Value& Value::operator=(Value&& other)
    {
        if (this != &other)
        {
            if (!IsDefined())
            {
                // lhs is not defined
                if (other.IsDefined())
                {
                    _data = std::move(other._data);
                    _layout = std::move(other._layout);
                    _type = std::move(other._type);
                    _hasName = std::move(other._hasName);
                }
            }
            else
            {
                // lhs is defined
                if (!other.IsDefined())
                {
                    throw InputException(InputExceptionErrors::invalidArgument);
                }
                if (GetBaseType() != other.GetBaseType())
                {
                    throw InputException(InputExceptionErrors::typeMismatch);
                }

                if (!IsConstrained())
                {
                    // lhs is not constrained
                    if (IsEmpty())
                    {
                        _data = std::move(other._data);
                        _layout = std::move(other._layout);
                        _type = std::move(other._type);
                        _hasName = std::move(other._hasName);
                    }
                    else
                    {
                        throw LogicException(LogicExceptionErrors::illegalState);
                    }
                }
                else
                {
                    // lhs is constrained
                    if (IsEmpty())
                    {
                        if (other.IsConstrained() && GetLayout() != other.GetLayout())
                        {
                            throw InputException(InputExceptionErrors::sizeMismatch);
                        }
                        _data = std::move(other._data);
                        _type = other._type;
                        _hasName = std::move(other._hasName);
                    }
                    else
                    {
                        if (!other.IsConstrained() || GetLayout() != other.GetLayout())
                        {
                            throw InputException(InputExceptionErrors::sizeMismatch);
                        }
                        GetContext().MoveData(other, *this);
                    }
                }
            }
            other.Reset();
        }

        return *this;
    }

    Value::Value(Emittable data, std::optional<MemoryLayout> layout) :
        _data(data),
        _type(GetContext().GetType(data)),
        _layout(layout)
    {}

    Value::Value(ValueType type, std::optional<MemoryLayout> layout) :
        _type({ type, 1 }),
        _layout(layout)
    {}

    Value::Value(detail::ValueTypeDescription typeDescription, std::optional<MemoryLayout> layout) noexcept :
        _type(typeDescription),
        _layout(layout)
    {}

    void Value::Reset()
    {
        _type = { ValueType::Undefined, 0 };
        _layout.reset();
        _data = {};
        _hasName = false;
    }

    void Value::SetData(Value value, bool force)
    {
        if (!force && value.IsConstrained() && value.GetLayout() != GetLayout())
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }

        std::visit(VariantVisitor{ [this, force](Emittable emittable) {
                                      auto type = GetContext().GetType(emittable);
                                      if (!force && type.first != _type.first)
                                      {
                                          throw InputException(InputExceptionErrors::typeMismatch);
                                      }

                                      _data = emittable;
                                      if (!force)
                                      {
                                          _type = type;
                                      }
                                  },
                                   [this, force](auto&& arg) {
                                       if (!force && GetValueType<std::decay_t<decltype(arg)>>() != _type.first)
                                       {
                                           throw InputException(InputExceptionErrors::typeMismatch);
                                       }

                                       _data = arg;
                                   } },
                   value._data);
    }

    bool Value::IsDefined() const
    {
        return _type.first != ValueType::Undefined;
    }

    bool Value::IsEmpty() const
    {
        return std::visit(
            VariantVisitor{
                [](Emittable data) -> bool { return data.GetDataAs<void*>() == nullptr; },
                [](auto&& data) -> bool { return data == nullptr; } },
            _data);
    }

    bool Value::IsConstant() const
    {
        return !std::holds_alternative<Emittable>(_data);
    }

    bool Value::IsIntegral() const
    {
        if (IsReference())
        {
            return false;
        }

        switch (_type.first)
        {
        case ValueType::Boolean:
            [[fallthrough]];
        case ValueType::Byte:
            [[fallthrough]];
        case ValueType::Char8:
            [[fallthrough]];
        case ValueType::Int16:
            [[fallthrough]];
        case ValueType::Int32:
            [[fallthrough]];
        case ValueType::Int64:
            return true;
        default:
            return false;
        }
    }

    bool Value::IsBoolean() const
    {
        return _type.first == ValueType::Boolean && !IsReference();
    }

    bool Value::IsInt16() const
    {
        return _type.first == ValueType::Int16 && !IsReference();
    }

    bool Value::IsInt32() const
    {
        return _type.first == ValueType::Int32 && !IsReference();
    }

    bool Value::IsInt64() const
    {
        return _type.first == ValueType::Int64 && !IsReference();
    }

    bool Value::IsFloatingPoint() const
    {
        return (_type.first == ValueType::Float || _type.first == ValueType::Double) && !IsReference();
    }

    bool Value::IsFloat32() const
    {
        return _type.first == ValueType::Float && !IsReference();
    }

    bool Value::IsDouble() const
    {
        return _type.first == ValueType::Double && !IsReference();
    }

    bool Value::IsReference() const
    {
        return _type.second > 1;
    }

    bool Value::IsIntegralReference() const
    {
        if (!IsReference())
        {
            return false;
        }

        switch (_type.first)
        {
        case ValueType::Boolean:
            [[fallthrough]];
        case ValueType::Byte:
            [[fallthrough]];
        case ValueType::Int16:
            [[fallthrough]];
        case ValueType::Char8:
            [[fallthrough]];
        case ValueType::Int32:
            [[fallthrough]];
        case ValueType::Int64:
            return true;
        default:
            return false;
        }
    }

    bool Value::IsBooleanReference() const
    {
        return _type.first == ValueType::Boolean && IsReference();
    }

    bool Value::IsShortReference() const
    {
        return _type.first == ValueType::Int16 && IsReference();
    }

    bool Value::IsInt32Reference() const
    {
        return _type.first == ValueType::Int32 && IsReference();
    }

    bool Value::IsInt64Reference() const
    {
        return _type.first == ValueType::Int64 && IsReference();
    }

    bool Value::IsFloatingPointReference() const
    {
        return (_type.first == ValueType::Float || _type.first == ValueType::Double) && IsReference();
    }

    bool Value::IsFloat32Reference() const
    {
        return _type.first == ValueType::Float && IsReference();
    }

    bool Value::IsDoubleReference() const
    {
        return _type.first == ValueType::Double && IsReference();
    }

    bool Value::IsConstrained() const
    {
        return _layout.has_value();
    }

    const MemoryLayout& Value::GetLayout() const
    {
        return _layout.value();
    }

    Value Value::Reference() const
    {
        return GetContext().Reference(*this);
    }

    Value Value::Dereference() const
    {
        return GetContext().Dereference(*this);
    }

    Value Value::Offset(Value index) const
    {
        return GetContext().Offset(*this, index);
    }

    Value Value::Offset(Scalar index) const
    {
        return Offset(index.GetValue());
    }

    Value Value::Offset(const std::vector<Scalar>& indices) const
    {
        if (GetLayout().NumDimensions() != static_cast<int>(indices.size()))
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Must pass in one index per dimension in the value layout");
        }
        return GetContext().Offset(*this, indices);
    }

    ValueType Value::GetBaseType() const
    {
        return _type.first;
    }

    void Value::SetLayout(MemoryLayout layout)
    {
        _layout = layout;
    }

    void Value::ClearLayout()
    {
        _layout.reset();
    }

    void Value::ClearData()
    {
        _data = Emittable{ nullptr };
    }

    int Value::PointerLevel() const
    {
        return _type.second;
    }

    Value::UnderlyingDataType& Value::GetUnderlyingData()
    {
        return _data;
    }

    const Value::UnderlyingDataType& Value::GetUnderlyingData() const
    {
        return _data;
    }

    void Value::SetName(const std::string& name)
    {
        GetContext().SetName(*this, name);
        _hasName = true;
    }

    std::string Value::GetName() const
    {
        return GetContext().GetName(*this);
    }

    bool Value::HasCustomName() const
    {
        return _hasName;
    }

    namespace detail
    {
        Value StoreConstantData(ConstantData data)
        {
            return GetContext().StoreConstantData(data);
        }
    } // namespace detail

#define ADD_TO_STRING_ENTRY(NAMESPACE, OPERATOR) \
    case NAMESPACE::OPERATOR:                    \
        return #OPERATOR;
#define BEGIN_FROM_STRING if (false)
#define ADD_FROM_STRING_ENTRY(NAMESPACE, OPERATOR) else if (name == #OPERATOR) return NAMESPACE::OPERATOR

    std::string ToString(ValueType vt)
    {
        switch (vt)
        {
            ADD_TO_STRING_ENTRY(ValueType, Undefined);
            ADD_TO_STRING_ENTRY(ValueType, Void);
            ADD_TO_STRING_ENTRY(ValueType, Boolean);
            ADD_TO_STRING_ENTRY(ValueType, Char8);
            ADD_TO_STRING_ENTRY(ValueType, Byte);
            ADD_TO_STRING_ENTRY(ValueType, Int16);
            ADD_TO_STRING_ENTRY(ValueType, Int32);
            ADD_TO_STRING_ENTRY(ValueType, Int64);
            ADD_TO_STRING_ENTRY(ValueType, Float);
            ADD_TO_STRING_ENTRY(ValueType, Double);

        default:
            return "Undefined";
        }
    }

    ValueType FromString(std::string name)
    {
        BEGIN_FROM_STRING;
        ADD_FROM_STRING_ENTRY(ValueType, Undefined);
        ADD_FROM_STRING_ENTRY(ValueType, Void);
        ADD_FROM_STRING_ENTRY(ValueType, Boolean);
        ADD_FROM_STRING_ENTRY(ValueType, Char8);
        ADD_FROM_STRING_ENTRY(ValueType, Byte);
        ADD_FROM_STRING_ENTRY(ValueType, Int16);
        ADD_FROM_STRING_ENTRY(ValueType, Int32);
        ADD_FROM_STRING_ENTRY(ValueType, Int64);
        ADD_FROM_STRING_ENTRY(ValueType, Float);
        ADD_FROM_STRING_ENTRY(ValueType, Double);

        return ValueType::Undefined;
    }
} // namespace value
} // namespace ell

size_t std::hash<::ell::value::Value>::operator()(const ::ell::value::Value& value) const noexcept
{
    using ::ell::utilities::HashCombine;
    using ::ell::utilities::IntPtrT;
    using ::ell::utilities::VariantVisitor;
    using ::ell::value::Emittable;

    size_t hash = 0;

    HashCombine(hash,
                std::visit(
                    VariantVisitor{
                        [](Emittable emittable) { return std::hash<IntPtrT>{}(emittable.GetDataAs<IntPtrT>()); },
                        [](auto&& data) { return std::hash<std::decay_t<decltype(data)>>{}(data); } },
                    value.GetUnderlyingData()));
    HashCombine(hash, value.GetBaseType());
    HashCombine(hash, value.PointerLevel());

    if (value.IsConstrained())
    {
        HashCombine(hash, value.GetLayout());
    }
    else
    {
        // Special random value for an unconstrained Value
        HashCombine(hash, 0x87654321);
    }

    return hash;
}
