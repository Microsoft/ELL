////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     TypeName.h (utilities)
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Boolean.h"
#include "TypeTraits.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ell
{
namespace utilities
{
    // Forward declaration of IArchivable
    class IArchivable;

    // struct to help determine if a class has GetTypeName() defined
    template <typename CheckType>
    class HasGetTypeName
    {
        // Fallback returns false if we don't have GetTypeName
        template <typename>
        static constexpr std::false_type FindGetTypeName(...);

        // returns true if we have GetTypeName()->string
        template <typename T>
        static constexpr auto FindGetTypeName(T*) -> typename std::is_same<std::string, decltype(std::declval<T>().GetTypeName())>::type;

        typedef decltype(FindGetTypeName<CheckType>(nullptr)) type;

    public:
        static constexpr bool value = type::value;
    };

    /// <summary> Utility function to get type name from a template parameter </summary>
    ///
    /// <typeparam name="Type"> The type (e.g., 'double') </typeparam>
    template <typename Type>
    std::string GetTypeName();

    /// <summary> Utility function to get templated type names (e.g., Vector<double>) </summary>
    ///
    /// <param name="baseType"> The base type (e.g., 'Vector') </param>
    /// <typeparam name="Types"> The templated type (e.g., 'double') </typeparam>
    template <typename... Types>
    std::string GetCompositeTypeName(std::string baseType);

    /// <summary> Utility function to get templated type names (e.g., Vector<double>) </summary>
    ///
    /// <param name="baseType"> The base type (e.g., 'Vector') </param>
    /// <param name="subtypes"> The list of templated types (e.g., 'double') </param>
    std::string GetCompositeTypeName(std::string baseType, const std::vector<std::string>& subtypes);

    template <typename T, typename Enable = void>
    struct TypeName; // undefined

    template <typename T>
    struct TypeName<T, std::enable_if_t<HasGetTypeName<std::decay_t<T>>::value>>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName();
    };

    template <typename T>
    struct TypeName<T, std::enable_if_t<std::is_enum<std::decay_t<T>>::value>>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName();
    };

    /// <summary> Class used to get information about pointer types. </summary>
    ///
    /// <typeparam name="T"> Generic type parameter. </typeparam>
    template <typename T>
    struct TypeName<T*>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName();
    };

    /// <summary> Class used to get information about unique_ptr types. </summary>
    ///
    /// <typeparam name="T"> Generic type parameter. </typeparam>
    template <typename T>
    struct TypeName<std::unique_ptr<T>>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName();
    };

    /// <summary> Class used to get information about std::vector types. </summary>
    ///
    /// <typeparam name="T"> Generic type parameter. </typeparam>
    template <typename T>
    struct TypeName<std::vector<T>>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName();
    };

    /// <summary> Class used to get information about std::vector types. </summary>
    ///
    /// <typeparam name="T"> Generic type parameter. </typeparam>
    template <typename T>
    struct TypeName<const std::vector<T>&>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName();
    };

    /// <summary> Class used to get information about tuple types. </summary>
    ///
    /// <typeparam name="T"> Generic type parameter. </typeparam>
    template <typename... T>
    struct TypeName<std::tuple<T...>>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName()
        {
            return GetCompositeTypeName<std::decay_t<T>...>("tuple");
        }
    };

    /// <summary> Class used to get information about the bool type. </summary>
    template <>
    struct TypeName<bool>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "bool"; }
    };

    /// <summary> Class used to get information about the Boolean type. </summary>
    template <>
    struct TypeName<Boolean>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "Boolean"; }
    };

    /// <summary> Class used to get information about the 8-bit integer type. </summary>
    template <>
    struct TypeName<char>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "char"; }
    };

    /// <summary> Class used to get information about the 8-bit integer type. </summary>
    template <>
    struct TypeName<int8_t>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "int8"; }
    };

    /// <summary> Class used to get information about the unsigned 8-bit integer type. </summary>
    template <>
    struct TypeName<uint8_t>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "uint8"; }
    };

    /// <summary> Class used to get information about the 16-bit integer type. </summary>
    template <>
    struct TypeName<short>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "short"; }
    };

    /// <summary> Class used to get information about the unsigned 16-bit integer type. </summary>
    template <>
    struct TypeName<unsigned short>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "ushort"; }
    };

    /// <summary> Class used to get information about the 32-bit integer type. </summary>
    template <>
    struct TypeName<int>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "int"; }
    };

    /// <summary> Class used to get information about the unsigned 32-bit integer type. </summary>
    template <>
    struct TypeName<unsigned int>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "uint"; }
    };

    /// <summary> Class used to get information about the 64-bit integer type. </summary>
    template <>
    struct TypeName<long>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "long"; }
    };

    /// <summary> Class used to get information about the unsigned 64-bit integer type. </summary>
    template <>
    struct TypeName<unsigned long>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "ulong"; }
    };

    /// <summary> Class used to get information about the 64-bit integer type. </summary>
    template <>
    struct TypeName<long long>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "longlong"; }
    };

    /// <summary> Class used to get information about the unsigned 64-bit integer type. </summary>
    template <>
    struct TypeName<unsigned long long>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "ulonglong"; }
    };

    /// <summary> Class used to get information about the float type. </summary>
    template <>
    struct TypeName<float>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "float"; }
    };

    /// <summary> Class used to get information about the double type. </summary>
    template <>
    struct TypeName<double>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "double"; }
    };

    /// <summary> Class used to get information about the double type. </summary>
    template <>
    struct TypeName<std::string>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "std::string"; }
    };

    /// <summary> Class used to get information about the void type. </summary>
    template <>
    struct TypeName<void>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "void"; }
    };

    /// <summary> Class used to get information about the void* type. </summary>
    template <>
    struct TypeName<void*>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "void*"; }
    };

    /// <summary> Class used to get information about the std::true_type type. </summary>
    template <>
    struct TypeName<std::true_type>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "std::true_type"; }
    };

    /// <summary> Class used to get information about the std::false_type type. </summary>
    template <>
    struct TypeName<std::false_type>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "std::false_type"; }
    };

    /// <summary> Class used to get information about the double type. </summary>
    template <>
    struct TypeName<IArchivable>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "IArchivable"; }
    };

    /// <summary> Class used to get information about the double type. </summary>
    template <>
    struct TypeName<const IArchivable>
    {
        /// <summary> Gets the serialization name of the type. </summary>
        ///
        /// <returns> The serialization name. </returns>
        static std::string GetName() { return "IArchivable"; }
    };
} // namespace utilities
} // namespace ell

#pragma region implementation

namespace ell
{
namespace utilities
{
    template <typename T>
    std::string TypeName<T, std::enable_if_t<HasGetTypeName<std::decay_t<T>>::value>>::GetName()
    {
        return std::string(std::decay_t<T>::GetTypeName());
    };

    template <typename T>
    std::string TypeName<T, std::enable_if_t<std::is_enum<std::decay_t<T>>::value>>::GetName()
    {
        return "enum";
    };

    template <typename T>
    std::string TypeName<T*>::GetName()
    {
        return GetCompositeTypeName<T>("ptr");
    }

    template <typename T>
    std::string TypeName<std::unique_ptr<T>>::GetName()
    {
        return GetCompositeTypeName<T>("unique_ptr");
    }

    template <typename T>
    std::string TypeName<std::vector<T>>::GetName()
    {
        return GetCompositeTypeName<T>("vector");
    }

    template <typename T>
    std::string TypeName<const std::vector<T>&>::GetName()
    {
        return GetCompositeTypeName<T>("vector");
    }

    template <typename Type>
    std::string GetTypeName()
    {
        return TypeName<typename std::decay_t<Type>>::GetName();
    }

    template <typename... Types>
    std::string GetCompositeTypeName(std::string baseType)
    {
        auto typeStrings = std::vector<std::string>{ TypeName<Types>::GetName()... };
        return GetCompositeTypeName(baseType, typeStrings);
    }
} // namespace utilities
} // namespace ell

#pragma endregion implementation
