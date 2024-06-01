////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Hash.h (utilities)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Unused.h"

#include <tuple>
#include <utility>
#include <vector>

namespace ell
{
namespace utilities
{

    namespace detail
    {
        // from boost.hash
        inline void HashCombineImpl(size_t& seed, size_t value)
        {
            seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
    } // namespace detail

    /// <summary> Combines a given seed with the result of `std::hash<T>(t)` </summary>
    /// <param name="seed"> The initial seed </param>
    /// <param name="t"> The object to hash and combine with the seed </param>
    template <typename T>
    inline void HashCombine(size_t& seed, const T& t);

    /// <summary> Calculate the combined hash value of the elements of an iterator range </summary>
    /// <param name="seed"> The initial seed </param>
    template <class Iterator>
    inline void HashRange(size_t& seed, Iterator first, Iterator last);

    /// <summary> Calculate the combined hash value of the elements of an iterator range </summary>
    template <class Iterator>
    [[nodiscard]] inline size_t HashRange(Iterator first, Iterator last);

    /// <summary> Returns a hash for an array </summary>
    template <typename T>
    [[nodiscard]] auto HashValue(const T& t) -> decltype(std::hash<T>{}(t))
    {
        return std::hash<T>{}(t);
    }

    /// <summary> Returns a hash for an array </summary>
    template <typename T, size_t N>
    [[nodiscard]] auto HashValue(const T (&a)[N])
    {
        return HashRange(a, a + N);
    }

    /// <summary> Returns a hash for an array </summary>
    template <typename T, size_t N>
    [[nodiscard]] auto HashValue(T (&a)[N])
    {
        return HashRange(a, a + N);
    }

    /// <summary> Returns a hash for a vector instance </summary>
    template <typename T, typename Alloc>
    [[nodiscard]] auto HashValue(const std::vector<T, Alloc>& v)
    {
        size_t seed = 0;
        HashCombine(seed, v.size());
        HashRange(seed, v.begin(), v.end());
        return seed;
    }

    namespace detail
    {
        template <size_t Idx, typename Tuple>
        void HashTupleImpl(size_t& seed, const Tuple& tuple)
        {
            if constexpr (Idx < std::tuple_size_v<Tuple>)
            {
                HashCombine(seed, std::get<Idx>(tuple));
                HashTupleImpl<Idx + 1>(seed, tuple);
            }
            else
            {
                UNUSED(seed, tuple);
            }
        }

        template <typename Tuple>
        [[nodiscard]] size_t HashTuple(const Tuple& tuple)
        {
            size_t seed = 0;
            HashTupleImpl<0>(seed, tuple);
            return seed;
        }
    } // namespace detail

    /// <summary> Returns a hash for a tuple instance </summary>
    template <typename... Ts>
    [[nodiscard]] auto HashValue(const std::tuple<Ts...>& tuple)
    {
        return detail::HashTuple(tuple);
    }
} // namespace utilities
} // namespace ell

#pragma region implementation

namespace ell
{
namespace utilities
{
    template <typename T>
    inline void HashCombine(size_t& seed, const T& t)
    {
        detail::HashCombineImpl(seed, HashValue(t));
    }

    template <class Iterator>
    inline void HashRange(size_t& seed, Iterator first, Iterator last)
    {
        for (; first != last; ++first)
        {
            HashCombine<typename std::iterator_traits<Iterator>::value_type>(seed, *first);
        }
    }

    template <class Iterator>
    [[nodiscard]] inline size_t HashRange(Iterator first, Iterator last)
    {
        size_t seed = 0;
        HashRange(seed, first, last);
        return seed;
    }
} // namespace utilities
} // namespace ell

#pragma endregion implementation
