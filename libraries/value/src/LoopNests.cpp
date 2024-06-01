////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     LoopNests.cpp (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "LoopNests.h"

#include "loopnests/CodeGenerator.h"
#include "loopnests/CodePositionConstraints.h"
#include "loopnests/IndexRange.h"
#include "loopnests/Kernel.h"
#include "loopnests/LoopNest.h"

#include <map>
#include <optional>
#include <tuple>

using namespace ell::utilities;

namespace ell
{
namespace value
{
    class LoopNestImpl
    {
    public:
        ~LoopNestImpl() = default;

        void Using(std::initializer_list<ViewAdapter> inputs, ArgumentType argType)
        {
            for (auto input : inputs)
            {
                _arguments.emplace_back(input, argType);
            }
        }

        void ForAll(Index index, int begin, int end)
        {
            _ranges.emplace_back(index, loopnests::Range{ begin, end });
        }

        void EnsureCreated()
        {
            if (!_nest.has_value())
            {
                _nest = loopnests::LoopNest(_ranges);
            }
        }

        void Do(std::function<void(std::vector<Value>)> fn, std::vector<Index> kernelOuterIndices, std::string kernelId)
        {
            std::vector<Value> arguments;
            arguments.reserve(_arguments.size());
            for (const auto& arg : _arguments)
            {
                arguments.push_back(arg.first);
            }
            std::vector<Index> indices;
            for (const auto& index : _ranges)
            {
                indices.push_back(index.GetIndex());
            }
            auto name = UniqueName("kernel");
            if (kernelId.empty())
            {
                kernelId = name;
            }
            auto kernel = loopnests::Kernel(name, kernelId)
                              .Inputs(arguments)
                              .Indices(indices)
                              .Define(fn);

            Do(kernel, kernelOuterIndices);
        }

        void Do(Kernel kernelFn, std::vector<Index> kernelOuterIndices)
        {
            _kernels.push_back(std::move(kernelFn));
            auto& kernel = _kernels.back();

            EnsureCreated();
            if (kernelOuterIndices.empty())
            {
                _nest->AddKernel(kernel, loopnests::LoopNest::ConstraintType::constraint);
            }
            else
            {
                loopnests::CodePositionConstraints constraints{ loopnests::LoopFragmentType::body, kernelOuterIndices, {} };
                _nest->AddKernel(kernel, constraints);
            }
        }

        void Do(Kernel kernelFn, const loopnests::KernelPredicate& predicate, const loopnests::KernelPredicate& placement)
        {
            _kernels.push_back(std::move(kernelFn));
            auto& kernel = _kernels.back();

            EnsureCreated();
            _nest->AddKernel(kernel, predicate, placement);
        }

        Index Split(Index& index, int factor)
        {
            EnsureCreated();
            // TODO: this might not be needed in the high level api
            auto it = _splits.find({ index, factor });
            if (it == _splits.end())
            {
                auto splitResult = _nest->Split(index, factor);
                it = _splits.insert(it, { { index, factor }, splitResult });
            }

            index = it->second.inner;
            return it->second.outer;
        }

        void Parallelize(Index index)
        {
            EnsureCreated();
            _nest->Parallelize(index);
        }

        void Unroll(Index index)
        {
            EnsureCreated();
            _nest->Unroll(index);
        }

        void SetOrder(std::vector<Index> indices)
        {
            EnsureCreated();
            _nest->SetLoopOrder(indices);
        }

        void Run() const
        {
            loopnests::CodeGenerator{}.Run(*_nest);
        }

        loopnests::LoopNest& GetUnderlyingLoopNest()
        {
            EnsureCreated();
            return *_nest;
        }

    private:
        std::vector<std::pair<Value, ArgumentType>> _arguments;
        std::vector<loopnests::IndexRange> _ranges;
        std::vector<Kernel> _kernels;
        std::optional<loopnests::LoopNest> _nest;
        std::map<std::pair<Index, int>, loopnests::SplitIndex> _splits;
    };

    LoopNest Using(std::initializer_list<ViewAdapter> inputs, ArgumentType argType)
    {
        return LoopNest{}.Using(inputs, argType);
    }

    //
    // LoopNest
    //

    LoopNest::LoopNest() :
        _impl(std::make_unique<LoopNestImpl>()),
        _schedule(*this)
    {}
    LoopNest::LoopNest(const LoopNest& other) :
        _impl(std::make_unique<LoopNestImpl>(*other._impl)),
        _schedule(*this)
    {}
    LoopNest::LoopNest(LoopNest&& other) noexcept :
        _impl(std::move(other._impl)),
        _schedule(*this)
    {}

    LoopNest& LoopNest::operator=(const LoopNest& other)
    {
        if (this != &other)
        {
            *_impl = *other._impl;
        }

        return *this;
    }

    LoopNest& LoopNest::operator=(LoopNest&& other) noexcept
    {
        if (this != &other)
        {
            _impl = std::move(other._impl);
        }

        return *this;
    }

    LoopNest::~LoopNest() = default;

    LoopNest& LoopNest::Using(std::initializer_list<ViewAdapter> inputs, ArgumentType argType)
    {
        _impl->Using(inputs, argType);

        return *this;
    }

    LoopNest& LoopNest::ForAll(Index index, int begin, int end)
    {
        _impl->ForAll(index, begin, end);

        return *this;
    }

    LoopNest& LoopNest::Do(std::function<void(std::vector<Value>)> fn, std::vector<Index> kernelOuterIndices, std::string kernelId)
    {
        _impl->Do(fn, kernelOuterIndices, kernelId);

        return *this;
    }

    LoopNest& LoopNest::Do(std::function<void(std::vector<Value>)> fn, std::string kernelId)
    {
        return Do(fn, {}, kernelId);
    }

    LoopNest& LoopNest::Do(Kernel kernel, std::vector<Index> kernelOuterIndices)
    {
        _impl->Do(kernel, kernelOuterIndices);

        return *this;
    }

    LoopNest& LoopNest::Do(Kernel kernel, const loopnests::KernelPredicate& predicate, const loopnests::KernelPredicate& placement)
    {
        _impl->Do(kernel, predicate, placement);

        return *this;
    }

    void LoopNest::Run() const
    {
        _impl->Run();
    }

    loopnests::LoopNest& LoopNest::GetUnderlyingLoopNest()
    {
        return _impl->GetUnderlyingLoopNest();
    }

    const loopnests::LoopNest& LoopNest::GetUnderlyingLoopNest() const
    {
        return _impl->GetUnderlyingLoopNest();
    }

    Schedule& LoopNest::GetSchedule()
    {
        return _schedule;
    }

    void swap(LoopNest& nest1, LoopNest& nest2) noexcept
    {
        using std::swap;
        swap(nest1._impl, nest2._impl);
    }

    //
    // Schedule
    //

    Schedule::Schedule(LoopNest& nest) :
        _nest(nest),
        _impl(*nest._impl)
    {}

    Schedule::Schedule(const Schedule& other) = default;
    Schedule& Schedule::operator=(const Schedule& other) = default;

    Index Schedule::Split(Index& index, int factor)
    {
        return _impl.get().Split(index, factor);
    }

    void Schedule::Parallelize(Index index)
    {
        _impl.get().Parallelize(index);
    }

    Index Schedule::Parallelize(Index index, int factor)
    {
        auto outer = Split(index, factor);
        Parallelize(outer);
        return outer;
    }

    void Schedule::Unroll(Index index)
    {
        _impl.get().Unroll(index);
    }

    Index Schedule::Unroll(Index index, int factor)
    {
        auto outer = Split(index, factor);
        Unroll(outer);
        return outer;
    }

    void Schedule::Cache(std::unique_ptr<CachingProvider> provider)
    {
        provider->HandleCaching(_nest.get());
    }

    utilities::MemoryShape Schedule::GetShapeFromIndicesIncrement(std::vector<Index>& kernelIndices)
    {
        std::vector<int> sizes;
        for (auto index : kernelIndices)
        {
            auto range = _nest.get().GetUnderlyingLoopNest().GetIndexRange(index);
            sizes.push_back(range.Increment());
        }

        return { sizes };
    }

    void Schedule::Cache(
        CachingProvider& provider,
        ViewAdapter view,
        std::vector<Index> kernelIndices,
        utilities::MemoryShape size,
        std::vector<Index> atIndices,
        std::optional<utilities::DimensionOrder> order,
        std::any extras)
    {
        if (size.NumDimensions() == 0)
        {
            // Figure out size based on increment of the indices
            size = GetShapeFromIndicesIncrement(kernelIndices);
        };

        provider.Initialize(
            view,
            size,
            order.value_or(DimensionOrder(size.NumDimensions())),
            kernelIndices,
            atIndices.empty() ? kernelIndices : atIndices,
            extras);

        provider.HandleCaching(_nest.get());
    }

    void Schedule::SetOrder(std::vector<Index> indices)
    {
        _impl.get().SetOrder(indices);
    }

} // namespace value
} // namespace ell
