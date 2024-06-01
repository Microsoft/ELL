////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Value.h (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Emittable.h"
#include "ValueOperations.h"
#include "ValueType.h"

#include <utilities/include/Boolean.h>
#include <utilities/include/Exception.h>
#include <utilities/include/MemoryLayout.h>
#include <utilities/include/StringUtil.h>
#include <utilities/include/TypeTraits.h>

#include <initializer_list>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>

namespace ell
{
namespace value
{

    class Value;
    class Scalar;

    namespace detail
    {
        using ConstantData =
            std::variant<std::vector<utilities::Boolean>, std::vector<char>, std::vector<uint8_t>, std::vector<int16_t>, std::vector<int32_t>, std::vector<int64_t>, std::vector<float>, std::vector<double>>;

        // ValueType is the base type, the int represents how many
        // pointer levels there are
        using ValueTypeDescription = std::pair<ValueType, int>;

        Value StoreConstantData(ConstantData);

        template <typename ViewType>
        Value GetValue(ViewType value);
    } // namespace detail

    /// <summary> The basic type in the Value library upon which most operations are based. Wraps either C++ data (constant data) or data that is
    /// specific to the EmitterContext, specified by the Emittable type. </summary>
    class Value
    {
        using Boolean = utilities::Boolean;

        template <typename T>
        inline static constexpr bool IsAcceptableDataType = std::is_same_v<std::decay_t<T>, T> &&
                                                            (std::is_arithmetic_v<T> ||
                                                             std::is_same_v<std::decay_t<T>, Boolean>);

        template <typename T>
        inline static constexpr bool IsAcceptableConstantPointerType =
            !std::is_same_v<std::decay_t<T>, Value> && std::is_pointer_v<T> &&
            IsAcceptableDataType<utilities::RemoveAllPointersT<T>>;

        template <typename T, typename = std::enable_if_t<IsAcceptableDataType<T>>>
        using DataType = T;

        template <typename Ty>
        struct ReturnValueHelper
        {
        private:
            template <typename T>
            static constexpr auto Helper()
            {
                using namespace std;
                using namespace utilities;

                if constexpr (is_same_v<decay_t<T>, Emittable> || is_same_v<decay_t<T>, const Emittable>)
                {
                    return IdentityType<T>{};
                }
                else if constexpr (is_pointer_v<T>)
                {
                    static_assert(IsAcceptableDataType<RemoveAllPointersT<T>>);
                    return IdentityType<T>{};
                }
            }

        public:
            using Type = typename decltype(Helper<Ty>())::Type;
        };

        template <typename T>
        using GetT = typename ReturnValueHelper<T>::Type;

        template <typename T>
        static constexpr detail::ValueTypeDescription GetValueTypeAndPointerLevel()
        {
            return { GetValueType<T>(), utilities::CountOfPointers<T> };
        }

        using UnderlyingDataType = std::variant<Emittable, Boolean*, char*, uint8_t*, int16_t*, int32_t*, int64_t*, float*, double*>;

        using MemoryLayout = utilities::MemoryLayout;

    public:
        /// <summary> Default constructor </summary>
        Value();

        /// <summary> Destructor </summary>
        ~Value();

        /// <summary> Copy constructor </summary>
        Value(const Value&);

        /// <summary> Move constructor </summary>
        Value(Value&&) noexcept;

        /// <summary> Copy assignment operator </summary>
        /// <param name="other"> The instance whose data should be copied </param>
        /// <remarks> Copy assignment has to be context aware. The terms "defined", "empty", and
        /// "constrained" match their definitions of the respective Value member functions.
        /// Shallow copies of data is done in this function. Deep copies are done by the
        /// global EmitterContext object.
        /// * if this is not defined
        ///   * and other is not defined: don't do anything
        ///   * and other is defined: copy everything
        /// * if this is defined and not constrained and empty
        ///   * and other is not defined: throw
        ///   * and other is defined: if types don't match, throw. copy everything
        /// * if this is defined and constrained and empty
        ///   * and other is not defined: throw
        ///   * and other is defined and not constrained and empty: if types don't match, throw.
        ///   * and other is defined and constrained and empty: if types or layout don't match, throw
        ///   * and other is defined and not constrained and not empty: if types don't match, throw. shallow copy of data
        ///   * and other is defined and constrained and not empty: if types or layout don't match, throw. shallow copy of data
        /// * if this is defined and not constrained and not empty:
        ///   * throw
        /// * if this is defined and constrained and not empty:
        ///   * and other does not match type or layout: throw
        ///   * and other matches type and layout: deep copy of data
        /// </remarks>
        Value& operator=(const Value& other);

        /// <summary> Move assignment operator </summary>
        /// <param name="other"> The instance whose data should be moved </param>
        /// <remarks> Behaves similarly to the copy assignment, except other is reset afterwards </remarks>
        Value& operator=(Value&& other);

        /// <summary> Constructor that creates an instance which serves as a placeholder for data that matches the type and layout specified </summary>
        /// <typeparam name="T"> The C++ fundamental type to be the basis of this instance </typeparam>
        /// <param name="layout"> An optional MemoryLayout instance that describes the memory structure of the eventual data to be stored. If
        /// MemoryLayout is not provided, the Value instance is considered unconstrained </param>
        template <typename T>
        Value(std::optional<MemoryLayout> layout = {}) :
            Value(GetValueType<T>(), layout)
        {}

        /// <summary> Constructor that creates an instance which serves as a placeholder for data that matches the type and layout specified </summary>
        /// <param name="type"> The type to be the basis of this instance </param>
        /// <param name="layout"> An optional MemoryLayout instance that describes the memory structure of the eventual data to be stored. If
        /// MemoryLayout is not provided, the Value instance is considered unconstrained </param>
        Value(ValueType type, std::optional<MemoryLayout> layout = {});

        /// <summary> Constructor that creates an instance that wraps an Emittable instance </summary>
        /// <param name="emittable"> Context-specific data that is to be wrapped </param>
        /// <param name="layout"> An optional MemoryLayout instance that describes the memory structure of the data. If MemoryLayout is
        /// not provided, the Value instance is considered unconstrained </param>
        Value(Emittable emittable, std::optional<MemoryLayout> layout = {});

        /// <summary> Constructor that creates an instance from one C++ fundamental type's value and gives it a scalar layout </summary>
        /// <typeparam name="T"> The C++ fundamental type to be the basis of this instance </typeparam>
        /// <param name="t"> The value to be wrapped </param>
        template <typename T>
        Value(DataType<T> t) :
            Value(std::vector<T>{ t }, utilities::ScalarLayout)
        {}

        /// <summary> Constructor that creates an instance wrapping a set of constant values </summary>
        /// <typeparam name="T"> The C++ fundamental type to be the basis of this instance </typeparam>
        /// <param name="data"> The constant data </param>>
        /// <param name="layout"> An optional MemoryLayout instance that describes the memory structure of the data. If MemoryLayout is
        /// not provided, the Value instance is considered unconstrained </param>
        template <typename T>
        Value(std::initializer_list<DataType<T>> data, std::optional<MemoryLayout> layout = {}) :
            Value(std::vector<T>(data), layout)
        {}

        /// <summary> Constructor that creates an instance wrapping a set of constant values </summary>
        /// <typeparam name="T"> The C++ fundamental type to be the basis of this instance </typeparam>
        /// <param name="data"> The constant data </param>>
        /// <param name="layout"> An optional MemoryLayout instance that describes the memory structure of the data. If MemoryLayout is
        /// not provided, the Value instance is considered unconstrained </param>
        template <typename T>
        Value(std::vector<DataType<T>> data, std::optional<MemoryLayout> layout = {}) noexcept :
            Value(detail::StoreConstantData(std::move(data)))
        {
            if (layout)
            {
                SetLayout(layout.value());
            }
        }

        /// <summary> Constructor that creates an instance wrapping a set of constant values </summary>
        /// <typeparam name="T"> The C++ fundamental type to be the basis of this instance </typeparam>
        /// <param name="data"> Pointer to the constant data </param>>
        /// <param name="layout"> An optional MemoryLayout instance that describes the memory structure of the data. If MemoryLayout
        /// is not provided, the Value instance is considered unconstrained </param>
        template <typename T, std::enable_if_t<IsAcceptableConstantPointerType<T>, void*> = nullptr>
        Value(T t, std::optional<MemoryLayout> layout = {}) noexcept :
            _data(reinterpret_cast<std::add_pointer_t<utilities::RemoveAllPointersT<T>>>(t)),
            _type(GetValueTypeAndPointerLevel<T>()),
            _layout(layout)
        {}

        /// <summary> Constructor that creates an instance which serves as a placeholder for data that matches the full type description and layout specified </summary>
        /// <param name="typeDescription"> The full type description to be the basis of this instance </param>
        /// <param name="layout"> An optional MemoryLayout instance that describes the memory structure of the eventual data to be stored. If
        /// MemoryLayout is not provided, the Value instance is considered unconstrained </param>
        Value(detail::ValueTypeDescription typeDescription, std::optional<MemoryLayout> layout = {}) noexcept;

        /// <summary> Sets the data on an empty Value instance </summary>
        /// <param name="value"> The Value instance from which to get the data </param>
        /// <param name="force"> Disable all type checks when setting the underlying data </param>
        void SetData(Value value, bool force = false);

        /// <summary> Resets the instance to an undefined, unconstrained, and empty state </summary>
        void Reset();

        /// <summary> Gets the underlying data as the specified type, if possible </summary>
        /// <typeparam name="T"> The type of the data that is being retrieved </typeparam>
        /// <returns> An instance of T if it matches the type stored within the instance, else an exception is thrown </returns>
        template <typename T>
        GetT<std::add_const_t<T>> Get() const
        {
            auto ptr = TryGet<T>();
            if (!ptr.has_value())
            {
                throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented,
                                                ell::utilities::FormatString("Cannot get const value of type %s from Value of type %s", typeid(T).name(), ToString(GetBaseType()).c_str()));
            }

            return *ptr;
        }

        /// <summary> Gets the underlying data as the specified type, if possible </summary>
        /// <typeparam name="T"> The type of the data that is being retrieved </typeparam>
        /// <returns> An instance of T if it matches the type stored within the instance, else an exception is thrown </returns>
        template <typename T>
        GetT<T> Get()
        {
            auto ptr = TryGet<T>();
            if (!ptr.has_value())
            {
                throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented,
                                                ell::utilities::FormatString("Cannot get value of type %s from Value of type %s", typeid(T).name(), ToString(GetBaseType()).c_str()));
            }

            return *ptr;
        }

        /// <summary> Gets the underlying data as the specified type, if possible </summary>
        /// <typeparam name="T"> The type of the data that is being retrieved </typeparam>
        /// <returns> A std::optional, holding an instance of T if it matches the type stored within the instance, otherwise empty </returns>
        template <typename T>
        std::optional<GetT<std::add_const_t<T>>> TryGet() const noexcept
        {
            auto ptr = std::get_if<T>(&_data);
            return ptr != nullptr ? std::optional<GetT<std::add_const_t<T>>>{ *ptr }
                                  : std::optional<GetT<std::add_const_t<T>>>{};
        }

        /// <summary> Gets the underlying data as the specified type, if possible </summary>
        /// <typeparam name="T"> The type of the data that is being retrieved </typeparam>
        /// <returns> A std::optional, holding an instance of T if it matches the type stored within the instance, otherwise empty </returns>
        template <typename T>
        std::optional<GetT<T>> TryGet() noexcept
        {
            auto ptr = std::get_if<T>(&_data);
            return ptr != nullptr ? std::optional<GetT<T>>{ *ptr } : std::optional<GetT<T>>{};
        }

        /// <summary> Returns true if the instance is defined </summmary>
        bool IsDefined() const;

        /// <summary> Returns true if the instance holds data </summary>
        bool IsEmpty() const;

        /// <summary> Returns true if the instance holds constant data </summary>
        bool IsConstant() const;

        /// <summary> Returns true if the instance's type is an integral type (non-floating point) </summary>
        bool IsIntegral() const;

        /// <summary> Returns true if the instance's type is a boolean </summary>
        bool IsBoolean() const;

        /// <summary> Returns true if the instance's type is a short </summary>
        bool IsInt16() const;

        /// <summary> Returns true if the instance's type is a 32-bit int </summary>
        bool IsInt32() const;

        /// <summary> Returns true if the instance's type is a 64-bit int </summary>
        bool IsInt64() const;

        /// <summary> Returns true if the instance's type is a floating point type </summary>
        bool IsFloatingPoint() const;

        /// <summary> Returns true if the instance's type is a 32-bit float </summary>
        bool IsFloat32() const;

        /// <summary> Returns true if the instance's type is a double </summary>
        bool IsDouble() const;

        /// <summary> Returns true if the instance's type is a reference </summary>
        bool IsReference() const;

        /// <summary> Returns true if the instance's pointer level is greater than one and
        /// type is an integral type (non-floating point) </summary>
        bool IsIntegralReference() const;

        /// <summary> Returns true if the instance's pointer level is greater than one and
        /// type is a boolean </summary>
        bool IsBooleanReference() const;

        /// <summary> Returns true if the instance's pointer level is greater than one and
        /// type is a short </summary>
        bool IsShortReference() const;

        /// <summary> Returns true if the instance's pointer level is greater than one and
        /// type is a 32-bit int </summary>
        bool IsInt32Reference() const;

        /// <summary> Returns true if the instance's pointer level is greater than one and
        /// type is a 64-bit int </summary>
        bool IsInt64Reference() const;

        /// <summary> Returns true if the instance's pointer level is greater than one and
        /// type is a floating point type </summary>
        bool IsFloatingPointReference() const;

        /// <summary> Returns true if the instance's pointer level is greater than one and
        /// type is a 32-bit float </summary>
        bool IsFloat32Reference() const;

        /// <summary> Returns true if the instance's pointer level is greater than one and
        /// type is a double </summary>
        bool IsDoubleReference() const;

        /// <summary> Returns true if the instance has a MemoryLayout </summary>
        bool IsConstrained() const;

        /// <summary> Returns the MemoryLayout if the instance has one, throws otherwise </summary>
        const MemoryLayout& GetLayout() const;

        /// <summary> Returns a reference to this instance </summary>
        Value Reference() const;

        /// <summary> Dereferences this instance and returns the resulting value </summary>
        Value Dereference() const;

        /// <summary> Returns a new Value instance that's offset by the specified index </summary>
        /// <param name="index"> The value to offset by </param>
        /// <returns> A new Value instance offset from this instance's memory location by the specified index </returns>
        /// <remarks> This operation is not bounds checked against the layout </remarks>
        Value Offset(Value index) const;

        /// <summary> Returns a new Value instance that's offset by the specified index </summary>
        /// <param name="index"> The value to offset by </param>
        /// <returns> A new Value instance offset from this instance's memory location by the specified index </returns>
        /// <remarks> This operation is not bounds checked against the layout </remarks>
        Value Offset(Scalar index) const;

        /// <summary> Returns a new Value instance that's offset by the specified indices </summary>
        /// <param name="indices"> The index values to offset by in each dimension </param>
        /// <returns> A new Value instance offset from this instance's memory location by the specified indices </returns>
        /// <remarks> This operation is not bounds checked against the layout </remarks>
        Value Offset(const std::vector<Scalar>& indices) const;

        /// <summary> Returns the type of data represented by this instance </summary>
        ValueType GetBaseType() const;

        /// <summary> Sets the MemoryLayout for this instance </summary>
        /// <param name="layout"> The MemoryLayout to be set on this instance </param>
        void SetLayout(MemoryLayout layout);

        /// <summary> Clear the MemoryLayout, if any, on this instance </summary>
        void ClearLayout();

        /// <summary> Clear the data, if any, on this instance </summary>
        void ClearData();

        /// <summary> Returns the number of pointer indirections on the data referred to by this instance </summary>
        /// <returns> The number of pointer indirections </returns>
        int PointerLevel() const;

        /// <summary> Gets a reference to the underlying data storage </summary>
        /// <returns> The underlying data storage </returns>
        UnderlyingDataType& GetUnderlyingData();

        /// <summary> Gets a reference to the underlying data storage </summary>
        /// <returns> The underlying data storage </returns>
        const UnderlyingDataType& GetUnderlyingData() const;

        /// <summary> Gets a reference to the complete type description being held </summary>
        const detail::ValueTypeDescription& GetType() const { return _type; }

        /// <summary> Set the name for this instance with the current emitter context </summary>
        /// <param name="name"> The name </param>
        void SetName(const std::string& name);

        /// <summary> Gets the name for this instance from the current emitter context.
        /// If `SetName` has not been called, this will return the name chosen by the emitter context, if any. </summary>
        std::string GetName() const;

        /// <summary> Returns true if a custom name has been set by calling `SetName` </summary>
        bool HasCustomName() const;

    private:
        UnderlyingDataType _data;

        detail::ValueTypeDescription _type{ ValueType::Undefined, 0 };
        std::optional<MemoryLayout> _layout = {};
        bool _hasName = false;
    };

    /// <summary> A helper type that can hold any View type, which is any type that has a member function
    /// `Value GetValue()` </summary>
    struct ViewAdapter
    {
        template <typename View>
        ViewAdapter(View view) :
            _value(detail::GetValue(view))
        {}

        inline operator const Value&() const { return _value; }
        inline operator Value&() { return _value; }

        inline Value& GetValue() { return _value; }
        inline const Value& GetValue() const { return _value; }

    private:
        Value _value;
    };

} // namespace value
} // namespace ell

namespace std
{
/// <summary> Hash specializaiton for Value </summary>
/// <remarks> Two Value containers of the same type and layout will have the same hash, regardless of actual contents </remarks>
template <>
struct hash<::ell::value::Value>
{
    using Type = ::ell::value::Value;

    [[nodiscard]] size_t operator()(const Type& value) const noexcept;
};

template <>
struct equal_to<::ell::value::Value>
{
    using Type = ::ell::value::Value;
    inline bool operator()(const Type& first, const Type& second) const noexcept
    {
        auto hash = std::hash<Type>{};
        return hash(first) == hash(second);
    }
};

} // namespace std

#pragma region implementation

namespace ell
{
namespace value
{
    namespace detail
    {
        template <typename ViewType>
        Value GetValue(ViewType value)
        {
            if constexpr (std::is_same_v<Value, utilities::RemoveCVRefT<ViewType>>)
            {
                return value;
            }
            else
            {
                static_assert(std::is_same_v<decltype(std::declval<ViewType>().GetValue()), Value>,
                              "Parameter type isn't a valid view type of Value. Must have member function GetValue that returns Value instance.");

                return value.GetValue();
            }
        }
    } // namespace detail

    template <typename T>
    Value Cast(Value value)
    {
        return Cast(value, GetValueType<T>());
    }

} // namespace value
} // namespace ell

#pragma endregion implementation
