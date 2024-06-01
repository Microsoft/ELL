////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     TypeTraits.h (utilities)
//  Authors:  Chuck Jacobs, Ofer Dekel, Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <type_traits>
#include <vector>

namespace ell
{
namespace utilities
{
    namespace detail
    {
        template <typename T>
        struct IsVectorType : std::false_type
        {};

        template <typename T, typename A>
        struct IsVectorType<std::vector<T, A>> : std::true_type
        {};
    } // namespace detail

    /// <summary> Templated type that always has a value of `false`. </summary>
    template <typename... Ts>
    struct FalseType : std::false_type
    {};

    /// <summary> Enabled if ValueType is a boolean. </summary>
    template <typename ValueType>
    using IsBoolean = std::enable_if_t<std::is_same<std::decay_t<ValueType>, bool>::value, bool>;

    /// <summary> Enabled if ValueType is not a boolean. </summary>
    template <typename ValueType>
    using IsNotBoolean = std::enable_if_t<!std::is_same<std::decay_t<ValueType>, bool>::value, bool>;

    /// <summary> Enabled if the two classes are different. </summary>
    template <typename T, typename S>
    using IsDifferent = std::enable_if_t<!std::is_same<T, S>::value, bool>;

    /// <summary> Enabled if ValueType is an enum. </summary>
    template <typename ValueType>
    using IsEnum = std::enable_if_t<std::is_enum<ValueType>::value, bool>;

    /// <summary> Enabled if ValueType is an floating point type. </summary>
    template <typename ValueType>
    using IsFloatingPoint = std::enable_if_t<std::is_floating_point<std::decay_t<ValueType>>::value, bool>;

    /// <summary> Enabled if ValueType is not a floating point type. </summary>
    template <typename ValueType>
    using IsNotFloatingPoint = std::enable_if_t<!std::is_floating_point<std::decay_t<ValueType>>::value, bool>;

    /// <summary> Enabled if ValueType is a fundamental type. </summary>
    template <typename ValueType>
    using IsFundamental = std::enable_if_t<std::is_fundamental<std::decay_t<ValueType>>::value, bool>;

    /// <summary> Enabled if ValueType is not a fundamental type. </summary>
    template <typename ValueType>
    using IsNotFundamental = std::enable_if_t<!std::is_fundamental<std::decay_t<ValueType>>::value, bool>;

    /// <summary> Enabled if ValueType is an integer type. </summary>
    template <typename ValueType>
    using IsIntegral = std::enable_if_t<std::is_integral<std::decay_t<ValueType>>::value, bool>;

    /// <summary> Enabled if ValueType is an integer value that's not a boolean. </summary>
    template <typename ValueType>
    using IsNonBooleanIntegral = std::enable_if_t<std::is_integral<std::decay_t<ValueType>>::value && !std::is_same<std::decay_t<ValueType>, bool>::value, bool>;

    /// <summary> Enabled if ValueType is a signed integer type. </summary>
    template <typename ValueType>
    using IsSignedIntegral = std::enable_if_t<std::is_integral<std::decay_t<ValueType>>::value && std::is_signed<std::decay_t<ValueType>>::value, bool>;

    /// <summary> Enabled if ValueType is an unsigned integer type. </summary>
    template <typename ValueType>
    using IsUnsignedIntegral = std::enable_if_t<std::is_integral<std::decay_t<ValueType>>::value && std::is_unsigned<std::decay_t<ValueType>>::value, bool>;

    /// <summary> Enabled if the two classes are the same. </summary>
    template <typename T, typename S>
    using IsSame = std::enable_if_t<std::is_same<T, S>::value, bool>;

    /// <summary> Enabled if ValueType is a specialization of std::vector. </summary>
    template <typename ValueType>
    using IsVector = std::enable_if_t<detail::IsVectorType<std::decay_t<ValueType>>{}, bool>;

    /// <summary> Enabled if ValueType is not a specialization of std::vector. </summary>
    template <typename ValueType>
    using IsNotVector = std::enable_if_t<!detail::IsVectorType<std::decay_t<ValueType>>{}, bool>;

    /// <summary> Identity type that encapsulates type T </summary>
    template <typename T>
    struct IdentityType
    {
        using Type = T;
    };

    template <typename T>
    using IdentityTypeT = typename IdentityType<T>::Type;

    namespace detail
    {
        template <typename T, bool = std::is_pointer_v<T>>
        struct RemoveAllPointers
        {
            static constexpr size_t NumPointers = 1 + RemoveAllPointers<std::remove_pointer_t<T>>::NumPointers;
            using Type = typename RemoveAllPointers<std::remove_pointer_t<T>>::Type;
        };

        template <typename T>
        struct RemoveAllPointers<T, false>
        {
            static constexpr size_t NumPointers = 0;
            using Type = T;
        };
    } // namespace detail

    /// <summary> Type alias that removes all pointers and returns the final underlying type </summary>
    template <typename T>
    using RemoveAllPointersT = typename detail::RemoveAllPointers<T>::Type;

    /// <summary> Provides a count of the number of pointers a given type has </summary>
    template <typename T>
    inline constexpr size_t CountOfPointers = detail::RemoveAllPointers<T>::NumPointers;

    /// <summary> Used to wrap a number of lambdas to create an overloaded set for std::visitor </summary>
    template <typename... T>
    struct VariantVisitor : T...
    {
        using T::operator()...;
    };

    // deduction guide
    template <typename... T>
    VariantVisitor(T...) -> VariantVisitor<T...>;

    namespace detail
    {
        template <typename T, typename... Ts>
        struct IsOneOf : std::disjunction<std::is_same<T, Ts>...>
        {};
    } // namespace detail

    /// <summary> Compile time check for ensuring type `T` is in the set of `TypesToCheckAgainst` </summary>
    template <typename T, typename... TypesToCheckAgainst>
    inline constexpr bool IsOneOf = detail::IsOneOf<T, TypesToCheckAgainst...>::value;

    namespace detail
    {
        template <typename T1, typename... Ts>
        struct AllSame : std::conjunction<std::is_same<T1, Ts>...>
        {};
    } // namespace detail

    /// <summary> Compile time check for ensuring that all types provided are the same </summary>
    template <typename T, typename... RestOfTypes>
    inline constexpr bool AllSame = detail::AllSame<T, RestOfTypes...>::value;

    namespace detail
    {
        template <typename T>
        struct RemoveCVRef
        {
            using type = std::remove_cv_t<std::remove_reference_t<T>>;
        };
    } // namespace detail

    // Convenience type alias to remove all references and const/volatile qualifiers
    // from a type.
    // NB: Should be removed when we move to C++20 and be replaced by std::remove_cvref_t<T>
    template <typename T>
    using RemoveCVRefT = typename detail::RemoveCVRef<T>::type;

} // namespace utilities
} // namespace ell
