////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CachingProvider.h (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "EmitterContext.h"

#include "loopnests/Index.h"

#include <any>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

namespace ell
{
namespace value
{
    using loopnests::Index;

    class LoopNest;

    class CachingProvider
    {
    public:
        CachingProvider() = default;
        virtual ~CachingProvider() = default;

        void Initialize(ViewAdapter view, utilities::MemoryShape cacheShape, utilities::DimensionOrder order, std::vector<Index> kernelIndices, std::vector<Index> atIndices, std::any extra);

        void HandleCaching(LoopNest&);

    protected:
        Value _value;
        utilities::MemoryShape _shape;
        utilities::DimensionOrder _order;
        std::vector<Index> _kernelIndices;
        std::vector<Index> _atIndices;
        std::any _extra;

    private:
        virtual void HandleCachingImpl(LoopNest&) = 0;
    };

    namespace
    {
        class CachingHelper
        {
        public:
            CachingHelper(const CachingHelper&) = delete;
            CachingHelper& operator=(const CachingHelper&) = delete;
            CachingHelper(CachingHelper&& other)
            {
                *this = std::move(other);
            }

            CachingHelper& operator=(CachingHelper&& other)
            {
                if (this != &other)
                {
                    std::swap(_value, other._value);
                    std::swap(_atIndices, other._atIndices);
                    std::swap(_kernelIndices, other._kernelIndices);
                    std::swap(_shape, other._shape);
                    std::swap(_order, other._order);
                    std::swap(_provider, other._provider);
                    std::swap(_extra, other._extra);
                }

                return *this;
            }

            CachingHelper(ViewAdapter view) :
                _value(view)
            {
                if (!_value.IsDefined())
                {
                    throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState,
                                                    "View to be cached is not defined");
                }
            }

            CachingHelper Using(std::vector<Index> indices) &&
            {
                if (indices.empty())
                {
                    throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState,
                                                    "Indices that specify caching indices cannot be empty");
                }

                _kernelIndices = indices;
                if (_atIndices.empty())
                {
                    _atIndices = _kernelIndices;
                }

                return std::move(*this);
            }

            CachingHelper At(std::vector<Index> indices) &&
            {
                if (indices.empty())
                {
                    throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState,
                                                    "Indices that specify caching kernel location cannot be empty");
                }

                _atIndices = indices;
                return std::move(*this);
            }

            CachingHelper Size(utilities::MemoryShape shape, utilities::DimensionOrder order) &&
            {
                _shape = shape;
                _order = order;

                return std::move(*this);
            }

            CachingHelper Size(utilities::MemoryShape shape) &&
            {
                return std::move(*this).Size(shape, utilities::DimensionOrder(shape.NumDimensions()));
            }

            template <typename... Ts>
            CachingHelper Extra(Ts&&... ts) &&
            {
                _extra = std::make_tuple<Ts...>(std::forward<Ts>(ts)...);
                return std::move(*this);
            }

            template <typename T, typename ProviderType = typename T::ProviderType>
            CachingHelper Type(T&&) &&
            {
                if (!_value.IsDefined())
                {
                    throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState,
                                                    "View to be cached is not defined");
                }
                if (_kernelIndices.empty())
                {
                    throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState,
                                                    "Indices that specify caching location cannot be empty");
                }
                if (!_shape)
                {
                    throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState,
                                                    "Memory shape of cached location must be provided");
                }

                auto provider = std::make_unique<ProviderType>();
                provider->Initialize(std::move(_value), std::move(*_shape), std::move(*_order), std::move(_kernelIndices), std::move(_atIndices), std::move(_extra));
                _provider = std::move(provider);
                return std::move(*this);
            }

            operator std::unique_ptr<CachingProvider>() &&
            {
                return std::move(_provider);
            }

        private:
            Value _value;
            std::vector<Index> _atIndices;
            std::vector<Index> _kernelIndices;
            std::optional<utilities::MemoryShape> _shape;
            std::optional<utilities::DimensionOrder> _order;
            std::unique_ptr<CachingProvider> _provider;
            std::any _extra;
        };
    } // namespace

    inline CachingHelper CreateCacheFor(ViewAdapter view)
    {
        return CachingHelper(view);
    }

    template <typename CachingProviderType>
    struct CachingStrategyType
    {
        using ProviderType = CachingProviderType;
    };

} // namespace value
} // namespace ell
