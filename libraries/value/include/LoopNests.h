////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     LoopNests.h (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "CachingProvider.h"
#include "EmitterContext.h"
#include "FunctionDeclaration.h"
#include "Value.h"

#include "loopnests/Index.h"
#include "loopnests/Kernel.h"
#include "loopnests/KernelPredicate.h"

#include <utilities/include/FunctionUtils.h>

#include <functional>
#include <memory>
#include <tuple>
#include <vector>

namespace ell
{
namespace value
{
    namespace loopnests
    {
        class LoopNest;
    }

    using loopnests::Index;
    using loopnests::Kernel;

    enum class ArgumentType
    {
        Input,
        InputOutput,
        Output,
        Temporary
    };

    class LoopNestImpl;

    class Schedule
    {
    public:
        Schedule(const Schedule&);
        Schedule& operator=(const Schedule&);

        Schedule(Schedule&&) noexcept = delete;
        Schedule& operator=(Schedule&&) noexcept = delete;

        Index Split(Index& index, int factor);

        /// <summary> Parallelizes the loop represented by the index, if parallelization is enabled. No effect if parallelization is disabled </summary>
        /// <param name="index"> Represents the loop to parallelize </param>
        void Parallelize(Index index);

        /// <summary> Parallelizes the loop represented by the index, if parallelization is enabled. No effect if parallelization is disabled </summary>
        /// <param name="index"> Represents the loop to parallelize. On return, this index points to the inner loop created by the split </param>
        /// <param name="factor"> The factor by which to parallelize. Ideally, represents the number of threads to use </param>
        /// <returns> The index which represents the outer loop, now parallelized </returns>
        Index Parallelize(Index index, int factor);

        /// <summary> Unrolls the loop represented by the index </summary>
        /// <param name="index"> Represents the loop to unroll </param>
        void Unroll(Index index);

        /// <summary> Unrolls the loop represented by the index </summary>
        /// <param name="index"> Represents the loop to unroll. On return, this index points to the inner loop created by the split </param>
        /// <param name="factor"> The factor by which to unroll </param>
        /// <returns> The index which represents the outer loop, now unrolled </returns>
        Index Unroll(Index index, int factor);

        void Cache(std::unique_ptr<CachingProvider> provider);

        template <typename CachingStrategyType>
        void Cache(
            ViewAdapter view,
            std::vector<Index> kernelIndices,
            utilities::MemoryShape size = {},
            std::vector<Index> atIndices = {},
            std::optional<utilities::DimensionOrder> order = std::nullopt,
            std::any extras = {});

        void Cache(
            CachingProvider& provider,
            ViewAdapter view,
            std::vector<Index> kernelIndices,
            utilities::MemoryShape size,
            std::vector<Index> atIndices = {},
            std::optional<utilities::DimensionOrder> order = std::nullopt,
            std::any extras = {});

        void SetOrder(std::vector<Index> indices);

    private:
        Schedule(LoopNest&);
        friend class LoopNest;
        utilities::MemoryShape GetShapeFromIndicesIncrement(std::vector<Index>& kernelIndices);

        std::reference_wrapper<LoopNest> _nest;
        std::reference_wrapper<LoopNestImpl> _impl;
    };

    class LoopNest
    {
    public:
        LoopNest();
        LoopNest(const LoopNest&);
        LoopNest(LoopNest&&) noexcept;
        LoopNest& operator=(const LoopNest&);
        LoopNest& operator=(LoopNest&&) noexcept;
        ~LoopNest();

        LoopNest& Using(std::initializer_list<ViewAdapter> inputs, ArgumentType argType);
        LoopNest& ForAll(Index index, int begin, int end);

        template <typename Fn>
        LoopNest& Do(Fn&& fn, std::vector<Index> kernelOuterIndices = {}, std::string kernelId = "");

        template <typename Fn>
        LoopNest& Do(Fn&& fn, std::string kernelId);

        LoopNest& Do(std::function<void(std::vector<Value>)> fn, std::vector<Index> kernelOuterIndices = {}, std::string kernelId = "");

        LoopNest& Do(std::function<void(std::vector<Value>)> fn, std::string kernelId);

        LoopNest& Do(Kernel kernel, std::vector<Index> kernelOuterIndex = {});

        LoopNest& Do(Kernel kernel, const loopnests::KernelPredicate& predicate, const loopnests::KernelPredicate& placement = {});

        Schedule& GetSchedule();

        void Run() const;

        loopnests::LoopNest& GetUnderlyingLoopNest();
        const loopnests::LoopNest& GetUnderlyingLoopNest() const;

    private:
        template <typename... Args>
        LoopNest& DoImpl(std::function<void(Args...)> fn, std::vector<Index> kernelOuterIndices, std::string kernelId);

        friend void swap(LoopNest& nest1, LoopNest& nest2) noexcept;
        friend class Schedule;

        std::unique_ptr<LoopNestImpl> _impl;
        Schedule _schedule;
    };

    LoopNest Using(std::initializer_list<ViewAdapter> inputs, ArgumentType argType);
} // namespace value
} // namespace ell

#pragma region implementation

namespace ell
{
namespace value
{
#if defined(__APPLE__)
#define FUNCTION_TYPE detail::Function // defined in implementation region of FunctionDeclaration.h
#else
#define FUNCTION_TYPE std::function
#endif // defined(__APPLE__)

    template <typename Fn>
    LoopNest& LoopNest::Do(Fn&& fn, std::vector<Index> kernelOuterIndices, std::string kernelId)
    {
        return DoImpl(FUNCTION_TYPE(std::forward<Fn>(fn)), kernelOuterIndices, kernelId);
    }

    template <typename Fn>
    LoopNest& LoopNest::Do(Fn&& fn, std::string kernelId)
    {
        return Do(std::move(fn), {}, kernelId);
    }

    template <typename... Args>
    LoopNest& LoopNest::DoImpl(std::function<void(Args...)> fn, std::vector<Index> kernelOuterIndices, std::string kernelId)
    {
        static_assert(std::conjunction_v<std::is_convertible<Args, ViewAdapter>...>);
        return Do(
            std::function<void(std::vector<Value>)>{
                [fn = std::move(fn)](std::vector<Value> args) {
                    std::tuple<Args...> tupleArgs = utilities::VectorToTuple<Args...>(args);
                    std::apply(fn, tupleArgs);
                } },
            kernelOuterIndices,
            kernelId);
    }

    template <typename CachingStrategyType>
    void Schedule::Cache(
        ViewAdapter view,
        std::vector<Index> kernelIndices,
        utilities::MemoryShape size,
        std::vector<Index> atIndices,
        std::optional<utilities::DimensionOrder> order,
        std::any extras)
    {
        static_assert(std::is_base_of_v<CachingProvider, CachingStrategyType>, "CachingStrategyType must inherit from CachingProvider!");

        CachingStrategyType provider{};
        Cache(
            provider,
            view,
            kernelIndices,
            size,
            atIndices,
            order,
            extras);
    }

#undef FUNCTION_TYPE
} // namespace value
} // namespace ell

#pragma endregion implementation
