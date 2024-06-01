////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ComputeContext.cpp (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ComputeContext.h"
#include "FunctionDeclaration.h"
#include "Scalar.h"
#include "Value.h"

#include <utilities/include/TypeAliases.h>
#include <utilities/include/TypeName.h>
#include <utilities/include/TypeTraits.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <string>
#include <thread>

namespace ell
{
namespace value
{
    using namespace detail;
    using namespace utilities;

    namespace
    {
        struct
        {
            int Current()
            {
                std::lock_guard lock{ _mutex };

                auto it = _idMap.find(std::this_thread::get_id());
                if (it == _idMap.end())
                {
                    it = _idMap.emplace_hint(it, std::this_thread::get_id(), ++_nextThreadId);
                }

                return it->second;
            }

            void Clear()
            {
                std::lock_guard lock{ _mutex };
                _idMap.clear();
                _nextThreadId = 0;
            }

            std::mutex _mutex;
            std::unordered_map<std::thread::id, int> _idMap;
            int _nextThreadId = 0;
        } ThreadIds;

        // TODO: Make this the basis of an iterator for MemoryLayout
        bool IncrementMemoryCoordinateImpl(int dimension, std::vector<int>& coordinate, const std::vector<int>& maxCoordinate)
        {
            // base case
            if (dimension < 0)
            {
                return false;
            }

            if (++coordinate[dimension] >= maxCoordinate[dimension])
            {
                coordinate[dimension] = 0;
                return IncrementMemoryCoordinateImpl(dimension - 1, coordinate, maxCoordinate);
            }

            return true;
        }

        bool IncrementMemoryCoordinate(std::vector<int>& coordinate, const std::vector<int>& maxCoordinate)
        {
            assert(coordinate.size() == maxCoordinate.size());
            return IncrementMemoryCoordinateImpl(static_cast<int>(maxCoordinate.size()) - 1, coordinate, maxCoordinate);
        }

        Value ConstantDataToValue(ConstantData& data, std::optional<MemoryLayout> layout = {})
        {
            return std::visit(
                [&layout](auto&& vector) -> Value {
                    return Value(vector.data(), layout.value_or(MemoryLayout({ (int)vector.size() })));
                },
                data);
        }

        template <typename T>
        auto AllocateConstantDataImpl(size_t size)
        {
            return std::vector<T>(size);
        }

        ConstantData AllocateConstantData(ValueType type, size_t size)
        {
            switch (type)
            {
            case ValueType::Boolean:
                return AllocateConstantDataImpl<Boolean>(size);
            case ValueType::Char8:
                return AllocateConstantDataImpl<char>(size);
            case ValueType::Byte:
                return AllocateConstantDataImpl<uint8_t>(size);
            case ValueType::Int16:
                return AllocateConstantDataImpl<int16_t>(size);
            case ValueType::Int32:
                return AllocateConstantDataImpl<int32_t>(size);
            case ValueType::Int64:
                return AllocateConstantDataImpl<int64_t>(size);
            case ValueType::Float:
                return AllocateConstantDataImpl<float>(size);
            case ValueType::Double:
                return AllocateConstantDataImpl<double>(size);
            default:
                throw LogicException(LogicExceptionErrors::notImplemented);
            }
        }

        template <typename T, typename Data>
        ConstantData CastVector(T, const Data& data)
        {
            using VectorType = std::vector<typename T::Type>;
            return VectorType(data.begin(), data.end());
        }

        // Intrinsics helper function objects
        struct MaxFn
        {
            template <typename T>
            constexpr auto operator()(const T& n1, const IdentityTypeT<T>& n2) const noexcept
            {
                return std::max(n1, n2);
            }
        };

        struct MinFn
        {
            template <typename T>
            constexpr auto operator()(const T& n1, const IdentityTypeT<T>& n2) const noexcept
            {
                return std::min(n1, n2);
            }
        };

        // Essentially a refactored lambda which returns a function that defines
        // the functionality for a particular intrinsic.
        // If the intrinsic is called with too many args, throw
        // If the arg is empty, boolean, or is Emittable, throw
        // Finally, if all conditions are met, a vector is created for the return values
        // and is filled in by calling std::transform with the proper STL function being
        // the transformation function
        struct SimpleNumericalFunctionIntrinsic
        {
            template <typename Fn>
            auto operator()(Fn&& fn) const -> std::function<Value(std::vector<Value>)>
            {
                return [fn](std::vector<Value> args) -> Value {
                    if (args.size() != 1)
                    {
                        throw InputException(InputExceptionErrors::invalidSize);
                    }

                    const auto& value = args[0];
                    return std::visit(
                        [&value, fn](auto&& data) -> Value {
                            using Type = std::decay_t<decltype(data)>;
                            using DataType = std::remove_pointer_t<Type>;

                            if constexpr (IsOneOf<DataType, Emittable, Boolean>)
                            {
                                throw InputException(InputExceptionErrors::invalidArgument);
                            }
                            else
                            {
                                auto dataBegin = data;
                                auto dataEnd = data + (value.IsConstrained() ? value.GetLayout().GetMemorySize() : 1u);
                                std::vector<DataType> returnData(dataBegin, dataEnd);

                                std::transform(returnData.begin(), returnData.end(), returnData.begin(), fn);

                                return Value(returnData, value.IsConstrained() ? std::optional<MemoryLayout>{ value.GetLayout() } : std::optional<MemoryLayout>{ std::nullopt });
                            }
                        },
                        value.GetUnderlyingData());
                };
            }
        };

        struct SimpleMinMaxNumFunctionIntrinsic
        {
            template <typename Fn>
            auto operator()(Fn&& fn) const -> std::function<Value(std::vector<Value>)>
            {
                return [fn](std::vector<Value> args) -> Value {
                    if (args.size() == 1)
                    {
                        const auto& value = args[0];
                        return std::visit(
                            [&value, fn](auto&& data) -> Value {
                                using Type = std::decay_t<decltype(data)>;
                                using DataType = std::remove_pointer_t<Type>;
                                if constexpr (IsOneOf<DataType, Emittable, Boolean>)
                                {
                                    throw InputException(InputExceptionErrors::invalidArgument);
                                }
                                else
                                {
                                    auto& valueLayout = value.GetLayout();
                                    auto maxCoordinate = valueLayout.GetActiveSize().ToVector();
                                    decltype(maxCoordinate) coordinate(maxCoordinate.size());

                                    DataType returnValue =
                                        fn(std::numeric_limits<DataType>::max(), std::numeric_limits<DataType>::lowest()) ==
                                                std::numeric_limits<DataType>::max()
                                            ? std::numeric_limits<DataType>::lowest()
                                            : std::numeric_limits<DataType>::max();

                                    do
                                    {
                                        auto logicalCoordinates = valueLayout.GetLogicalCoordinates(coordinate);
                                        auto valueOffset = valueLayout.GetLogicalEntryOffset(logicalCoordinates);
                                        returnValue = fn(returnValue, data[valueOffset]);
                                    } while (IncrementMemoryCoordinate(coordinate, maxCoordinate));

                                    return Value(returnValue);
                                }
                            },
                            value.GetUnderlyingData());
                    }
                    else if (args.size() == 2)
                    {
                        const auto& value1 = args[0];
                        const auto& value2 = args[1];
                        if ((value1.IsConstrained() && value1.GetLayout() != ScalarLayout) ||
                            (value2.IsConstrained() && value2.GetLayout() != ScalarLayout))
                        {
                            throw InputException(InputExceptionErrors::invalidSize);
                        }
                        return std::visit(
                            [fn](auto&& data1, auto&& data2) -> Value {
                                using Type1 = std::decay_t<decltype(data1)>;
                                using Type2 = std::decay_t<decltype(data2)>;
                                using DataType1 = std::remove_pointer_t<Type1>;
                                using DataType2 = std::remove_pointer_t<Type2>;

                                if constexpr (IsOneOf<DataType1, Emittable, Boolean> ||
                                              IsOneOf<DataType2, Emittable, Boolean>)
                                {
                                    throw InputException(InputExceptionErrors::invalidArgument);
                                }
                                else
                                {
                                    return Value(fn(*data1, *data2));
                                }
                            },
                            value1.GetUnderlyingData(),
                            value2.GetUnderlyingData());
                    }
                    else
                    {
                        throw InputException(InputExceptionErrors::invalidSize);
                    }
                };
            }
        };

        struct AbsFunctionIntrinsic
        {
            template <typename T>
            constexpr auto operator()(T n) const noexcept
            {
                if constexpr (std::is_unsigned_v<T>)
                {
                    return n;
                }
                else
                {
                    return std::abs(n);
                }
            }
        };

        struct PowFunctionIntrinsic
        {
            auto operator()(std::vector<Value> args) const -> Value
            {
                if (args.size() != 2)
                {
                    throw InputException(InputExceptionErrors::invalidSize);
                }

                auto& base = args[0];
                auto& exp = args[1];
                if (exp.IsConstrained() && exp.GetLayout() != ScalarLayout)
                {
                    throw InputException(InputExceptionErrors::notScalar);
                }
                return std::visit(
                    [&base](auto&& data1, auto&& data2) -> Value {
                        using Type1 = std::decay_t<decltype(data1)>;
                        using Type2 = std::decay_t<decltype(data2)>;
                        using DataType1 = std::remove_pointer_t<Type1>;
                        using DataType2 = std::remove_pointer_t<Type2>;

                        if constexpr (IsOneOf<DataType1, Emittable, Boolean> ||
                                      IsOneOf<DataType2, Emittable, Boolean>)
                        {
                            throw InputException(InputExceptionErrors::invalidArgument);
                        }
                        else if constexpr (!std::is_same_v<DataType1, DataType2>)
                        {
                            throw InputException(InputExceptionErrors::typeMismatch);
                        }
                        else
                        {
                            auto data1Begin = data1;
                            auto data1End = data1 + (base.IsConstrained() ? base.GetLayout().GetMemorySize() : 1u);
                            std::vector<DataType1> returnData(data1Begin, data1End);

                            std::transform(returnData.begin(), returnData.end(), returnData.begin(), [exp = *data2](auto n) {
                                return std::pow(n, exp);
                            });

                            return Value(returnData, base.IsConstrained() ? std::optional<MemoryLayout>{ base.GetLayout() } : std::optional<MemoryLayout>{ std::nullopt });
                        }
                    },
                    base.GetUnderlyingData(),
                    exp.GetUnderlyingData());
            }
        };

        struct InitializeVectorFunctionIntrinsic
        {
            auto operator()(std::vector<Value> args) const -> Value
            {
                if (args.size() != 2)
                {
                    throw InputException(InputExceptionErrors::invalidSize);
                }

                auto& targetVector = args[0];
                auto& copyValue = args[1];
                if (copyValue.IsConstrained() && copyValue.GetLayout() != ScalarLayout)
                {
                    throw InputException(InputExceptionErrors::notScalar);
                }
                return std::visit(
                    [&targetVector](auto&& data1, auto&& data2) -> Value {
                        using Type1 = std::decay_t<decltype(data1)>;
                        using Type2 = std::decay_t<decltype(data2)>;
                        using DataType1 = std::remove_pointer_t<Type1>;
                        using DataType2 = std::remove_pointer_t<Type2>;

                        if constexpr (IsOneOf<DataType1, Emittable, Boolean> ||
                                      IsOneOf<DataType2, Emittable, Boolean>)
                        {
                            throw InputException(InputExceptionErrors::invalidArgument);
                        }
                        else if constexpr (!std::is_same_v<DataType1, DataType2>)
                        {
                            throw InputException(InputExceptionErrors::typeMismatch);
                        }
                        else
                        {
                            auto data1Begin = data1;
                            auto data1End = data1 + (targetVector.IsConstrained() ? targetVector.GetLayout().GetMemorySize() : 1u);
                            std::vector<DataType1> returnData(data1Begin, data1End);
                            std::transform(returnData.begin(), returnData.end(), returnData.begin(), [target = *data2](auto n) {
                                return target;
                            });

                            return Value(returnData, targetVector.IsConstrained() ? std::optional<MemoryLayout>{ targetVector.GetLayout() } : std::optional<MemoryLayout>{ std::nullopt });
                        }
                    },
                    targetVector.GetUnderlyingData(),
                    copyValue.GetUnderlyingData());
            }
        };

        struct CopySignFunctionIntrinsic
        {
            auto operator()(std::vector<Value> args) const -> Value
            {
                if (args.size() != 2)
                {
                    throw InputException(InputExceptionErrors::invalidSize);
                }

                const auto& value1 = args[0];
                const auto& value2 = args[1];
                if ((value1.IsConstrained() && value1.GetLayout() != ScalarLayout) ||
                    (value2.IsConstrained() && value2.GetLayout() != ScalarLayout))
                {
                    throw InputException(InputExceptionErrors::invalidSize);
                }
                return std::visit(
                    [](auto&& data1, auto&& data2) -> Value {
                        using Type1 = std::decay_t<decltype(data1)>;
                        using Type2 = std::decay_t<decltype(data2)>;
                        using DataType1 = std::remove_pointer_t<Type1>;
                        using DataType2 = std::remove_pointer_t<Type2>;

                        if constexpr (IsOneOf<DataType1, Emittable, Boolean> ||
                                      IsOneOf<DataType2, Emittable, Boolean>)
                        {
                            throw InputException(InputExceptionErrors::invalidArgument);
                        }
                        else if constexpr (!std::is_same_v<DataType1, DataType2>)
                        {
                            throw InputException(InputExceptionErrors::typeMismatch);
                        }
                        else
                        {
                            return Value(std::copysign(*data1, *data2));
                        }
                    },
                    value1.GetUnderlyingData(),
                    value2.GetUnderlyingData());
            }
        };

        struct FmaFunctionIntrinsic
        {
            auto operator()(std::vector<Value> args) const -> Value
            {
                if (args.size() != 3)
                {
                    throw InputException(InputExceptionErrors::invalidSize);
                }

                if (std::any_of(args.begin(), args.end(), [](Value& value) { return value.IsConstrained() && value.GetLayout() != ScalarLayout; }))
                {
                    throw InputException(InputExceptionErrors::invalidSize);
                }

                const auto& value1 = args[0];
                const auto& value2 = args[1];
                const auto& value3 = args[2];

                return std::visit(
                    [](auto&& data1, auto&& data2, auto&& data3) -> Value {
                        using Type1 = std::decay_t<decltype(data1)>;
                        using Type2 = std::decay_t<decltype(data2)>;
                        using Type3 = std::decay_t<decltype(data3)>;
                        using DataType1 = std::remove_pointer_t<Type1>;
                        using DataType2 = std::remove_pointer_t<Type2>;
                        using DataType3 = std::remove_pointer_t<Type3>;

                        if constexpr (IsOneOf<DataType1, Emittable, Boolean> ||
                                      IsOneOf<DataType2, Emittable, Boolean> ||
                                      IsOneOf<DataType3, Emittable, Boolean>)
                        {
                            throw InputException(InputExceptionErrors::invalidArgument);
                        }
                        else if constexpr (!utilities::AllSame<DataType1, DataType2, DataType3>)
                        {
                            throw InputException(InputExceptionErrors::typeMismatch);
                        }
                        else
                        {
                            return Value(static_cast<DataType1>(std::fma(*data1, *data2, *data3)));
                        }
                    },
                    value1.GetUnderlyingData(),
                    value2.GetUnderlyingData(),
                    value3.GetUnderlyingData());
            }
        };

        enum class MemIntrinsicOp
        {
            Copy,
            Move,
            Set
        };
        template <MemIntrinsicOp op>
        struct MemOpFunctionIntrinsic
        {
            auto operator()(std::vector<Value> args) const -> Value
            {
                if (args.size() != 3)
                {
                    throw InputException(InputExceptionErrors::invalidSize);
                }

                if (!std::all_of(args.begin(), args.end(), [](const Value& value) { return value.IsConstant(); }))
                {
                    throw InputException(InputExceptionErrors::invalidArgument);
                }

                const auto& value1 = args[0];
                const auto& value2 = args[1];
                const auto& value3 = args[2];

                if (!value3.IsConstrained() || value3.GetLayout() != ScalarLayout)
                {
                    throw InputException(InputExceptionErrors::invalidArgument);
                }

                if constexpr (MemIntrinsicOp::Set == op)
                {
                    assert((value2.IsConstrained() && value2.GetLayout() == ScalarLayout && value2.GetType() == std::pair{ ValueType::Char8, 1 }));
                }

                std::visit(
                    [](auto&& data1, auto&& data2, auto&& data3) {
                        using Type1 = std::decay_t<decltype(data1)>;
                        using Type2 = std::decay_t<decltype(data2)>;
                        using Type3 = std::decay_t<decltype(data3)>;
                        if constexpr (utilities::IsOneOf<Emittable, Type1, Type2, Type3>)
                        {
                            assert(false);
                            return;
                        }
                        else
                        {
                            // Once we move away from VS 2017, this code can be uncommented and the code following can be simplified (lines 496-523)

                            //constexpr auto memFn = [] {
                            //    // static_casts needed because MSVC in VS 2017 can't handle the code without it
                            //    if constexpr (static_cast<int>(MemIntrinsicOp::Set) == static_cast<int>(op))
                            //    {
                            //        return &std::memset;
                            //    }
                            //    else if constexpr (static_cast<int>(MemIntrinsicOp::Copy) == static_cast<int>(op))
                            //    {
                            //        return &std::memcpy;
                            //    }
                            //    else if constexpr (static_cast<int>(MemIntrinsicOp::Move) == static_cast<int>(op))
                            //    {
                            //        return &std::memmove;
                            //    }
                            //    else
                            //    {
                            //        static_assert(utilities::FalseType<Type1, Type2, Type3>{}, "Unknown enum value");
                            //    }
                            //}();

                            constexpr bool isSet = op == MemIntrinsicOp::Set;
                            std::decay_t<std::conditional_t<isSet, decltype(data2[0]), decltype(data2)>> real2ndParam;
                            std::conditional_t<isSet, decltype(&std::memset), decltype(&std::memcpy)> memFn;
                            switch (op)
                            {
                            case MemIntrinsicOp::Set:
                                if constexpr (isSet)
                                {
                                    memFn = &std::memset;
                                    real2ndParam = *data2;
                                }
                                break;
                            case MemIntrinsicOp::Copy:
                                if constexpr (!isSet)
                                {
                                    memFn = &std::memcpy;
                                    real2ndParam = data2;
                                }
                                break;
                            case MemIntrinsicOp::Move:
                                if constexpr (!isSet)
                                {
                                    memFn = &std::memmove;
                                    real2ndParam = data2;
                                }
                                break;
                            default:
                                assert(false);
                            }

                            memFn(data1, real2ndParam, *data3 * sizeof(data1[0]));
                        }
                    },
                    value1.GetUnderlyingData(),
                    value2.GetUnderlyingData(),
                    value3.GetUnderlyingData());

                return {}; // ignored
            }
        };
    } // namespace

    struct ComputeContext::FunctionScope
    {
        FunctionScope(ComputeContext& context, std::string fnName) :
            context(context)
        {
            std::lock_guard lock{ context._mutex };
            context._stack.push({ fnName, {} });
        }

        ~FunctionScope()
        {
            std::lock_guard lock{ context._mutex };
            context._stack.pop();
        }

        ComputeContext& context;
    };

    ComputeContext::ComputeContext(std::string moduleName) :
        EmitterContext(emitters::GetTargetDevice("host")),
        _moduleName(std::move(moduleName))
    {
        // we always have at least one stack entry, in case the top level function needs to return something
        _stack.push({});
    }

    const ConstantData& ComputeContext::GetConstantData(Value value) const
    {
        if (!ValidateValue(value))
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }

        using Iterator = ConstantDataList::const_iterator;

        auto it = std::visit(
            VariantVisitor{
                [](Emittable) -> Iterator { return {}; },
                [this](auto&& data) -> Iterator {
                    using Type = std::decay_t<decltype(data)>;
                    using RealType = std::remove_pointer_t<Type>;
                    using VectorType = std::vector<RealType>;

                    const auto& frame = GetTopFrame();
                    auto it =
                        std::find_if(frame.second.begin(),
                                     frame.second.end(),
                                     [data](const ConstantData& constData) {
                                         if (auto ptr = std::get_if<VectorType>(&constData))
                                         {
                                             return ptr->data() <= data &&
                                                    data < (ptr->data() + ptr->size());
                                         }

                                         return false;
                                     });

                    return it;
                } },
            value.GetUnderlyingData());

        return *it;
    }

    Value ComputeContext::AllocateImpl(ValueType type, MemoryLayout layout, size_t /* alignment */, AllocateFlags flags)
    {
        if (flags != AllocateFlags::None)
        {
            throw LogicException(LogicExceptionErrors::notImplemented);
        }

        // special case the scalar case
        auto size = layout == ScalarLayout ? 1u : layout.GetMemorySize();

        auto constantData = AllocateConstantData(type, size);
        Value value = StoreConstantData(std::move(constantData));
        value.SetLayout(layout);

        return value;
    }

    std::optional<Value> ComputeContext::GetGlobalValue(GlobalAllocationScope scope, std::string name)
    {
        std::string adjustedName = GetScopeAdjustedName(scope, name);

        if (auto it = _globals.find(adjustedName); it != _globals.end())
        {
            return ConstantDataToValue(it->second.first, it->second.second);
        }

        return std::nullopt;
    }

    Value ComputeContext::GlobalAllocateImpl(GlobalAllocationScope scope, std::string name, ConstantData data, MemoryLayout layout, AllocateFlags flags)
    {
        if ((flags & AllocateFlags::ThreadLocal) == AllocateFlags::ThreadLocal)
        {
            throw LogicException(LogicExceptionErrors::illegalState, "Thread local storage cannot be specified for constant data");
        }

        std::string adjustedName = GetScopeAdjustedName(scope, name);

        if (_globals.find(adjustedName) != _globals.end())
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Unexpected collision in global data allocation");
        }

        auto& globalData = [&]() -> decltype(auto) {
            std::lock_guard lock{ _mutex };
            auto& globalData = _globals[adjustedName];
            globalData.first = std::move(data);
            globalData.second = std::move(layout);
            return globalData;
        }();

        return ConstantDataToValue(globalData.first, globalData.second);
    }

    Value ComputeContext::GlobalAllocateImpl(GlobalAllocationScope scope, std::string name, ValueType type, MemoryLayout layout, AllocateFlags flags)
    {
        // special case the scalar case
        auto size = layout == ScalarLayout ? 1u : layout.GetMemorySize();
        auto constantData = AllocateConstantData(type, size);
        if ((flags & AllocateFlags::ThreadLocal) == AllocateFlags::ThreadLocal)
        {
            name += std::to_string(ThreadIds.Current());

            if (auto globalValue = EmitterContext::GetGlobalValue(scope, name, layout))
            {
                return *globalValue;
            }

            flags &= ~AllocateFlags::ThreadLocal;
        }
        return GlobalAllocateImpl(scope, name, constantData, layout, flags);
    }

    Value ComputeContext::StoreConstantDataImpl(ConstantData data)
    {
        Value value = ConstantDataToValue(data);

        {
            std::lock_guard lock{ _mutex };
            GetTopFrame().second.push_front(std::move(data));
        }

        return value;
    }

    Value ComputeContext::OffsetImpl(Value begin, Value index)
    {
        if (!ValidateValue(begin))
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }

        if (!ValidateValue(index))
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }

        return std::visit(
            [&begin](auto&& index) -> Value {
                using IndexType = std::remove_pointer_t<std::decay_t<decltype(index)>>;
                if constexpr (std::is_arithmetic_v<IndexType> && !std::is_floating_point_v<IndexType>)
                {
                    auto indexValue = *index;
                    return std::visit(
                        [indexValue](auto&& begin) -> Value {
                            using BeginType = std::decay_t<decltype(begin)>;
                            if constexpr (std::is_same_v<BeginType, Emittable>)
                            {
                                throw InputException(InputExceptionErrors::invalidArgument);
                            }
                            else
                            {
                                return Value(begin + indexValue);
                            }
                        },
                        begin.GetUnderlyingData());
                }
                else
                {
                    throw InputException(InputExceptionErrors::invalidArgument);
                }
            },
            index.GetUnderlyingData());
    }

    ComputeContext::ConstantData ComputeContext::ExtractConstantData(Value value)
    {
        ConstantData movedOutOfScope;

        std::lock_guard lock{ _mutex };

        std::visit(
            VariantVisitor{
                [](Emittable) {},
                [&movedOutOfScope,
                 size = (value.IsConstrained() ? value.GetLayout().GetMemorySize() : 1)](auto&& data) {
                    using Type = std::decay_t<decltype(data)>;
                    using RealType = std::remove_pointer_t<Type>;
                    using VectorType = std::vector<RealType>;

                    movedOutOfScope = VectorType(data, data + size);
                } },
            value.GetUnderlyingData());

        return movedOutOfScope;
    }

    bool ComputeContext::IsGlobalValue(Value value) const
    {
        std::lock_guard lock{ _mutex };

        return std::visit(
            VariantVisitor{
                [](Emittable) -> bool {
                    throw LogicException(LogicExceptionErrors::illegalState);
                },

                [this](auto&& data) -> bool {
                    using Type = std::decay_t<decltype(data)>;
                    using RealType = std::remove_pointer_t<Type>;
                    using VectorType = std::vector<RealType>;

                    return std::find_if(
                               _globals.begin(),
                               _globals.end(),
                               [data](const auto& kvp) {
                                   if (auto ptr = std::get_if<VectorType>(
                                           &kvp.second.first))
                                   {
                                       return ptr->data() <= data &&
                                              data < (ptr->data() + ptr->size());
                                   }
                                   return false;
                               }) != _globals.end();
                } },
            value.GetUnderlyingData());
    }

    detail::ValueTypeDescription ComputeContext::GetTypeImpl(Emittable)
    {
        throw LogicException(LogicExceptionErrors::illegalState);
    }

    EmitterContext::DefinedFunction ComputeContext::CreateFunctionImpl(FunctionDeclaration decl, EmitterContext::DefinedFunction fn)
    {
        if (const auto& intrinsics = GetIntrinsics();
            std::find(intrinsics.begin(), intrinsics.end(), decl) != intrinsics.end())
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Specified function is an intrinsic");
        }

        std::lock_guard lock{ _mutex };

        if (auto it = _definedFunctions.find(decl); it != _definedFunctions.end())
        {
            return it->second;
        }

        DefinedFunction returnFn = [fn = std::move(fn),
                                    decl,
                                    this](std::vector<Value> args) -> std::optional<Value> {
            const auto& expectedArgs = decl.GetParameterTypes();
            const auto& fnName = decl.GetFunctionName();
            assert(expectedArgs.size() == args.size());

            auto fnArgs = NormalizeReferenceLevels(args, expectedArgs);

            if (const auto& returnType = decl.GetReturnType(); returnType.has_value())
            {
                Value expectedReturn = *returnType;

                ConstantData movedOutOfScope;
                std::optional<Value> maybeGlobal;
                {
                    FunctionScope scope(*this, fnName);

                    Value returnValue = expectedReturn;
                    auto fnReturn = fn(fnArgs);
                    if (!fnReturn.has_value())
                    {
                        throw LogicException(LogicExceptionErrors::illegalState, "Function definition was expected to return a value, but optional was empty");
                    }
                    returnValue = *fnReturn;
                    if (IsGlobalValue(returnValue))
                    {
                        maybeGlobal = returnValue;
                    }

                    if (!maybeGlobal)
                    {
                        // BUG: The error in the "compute" version of Vector_test3 may be in `ExtractConstantData()`. Possibly we're extracting from the wrong part of the constant array?
                        // The test in question has a variable called `dest` with the value `[10, 0]`. It (the test) then returns `dest(1)`, which should be `0`.
                        // Here, `movedOutOfScope` gets the value `[10, 0]`, which represents the entire `dest` value, not just element `1`.
                        movedOutOfScope = ExtractConstantData(returnValue);
                    }
                }
                if (!maybeGlobal)
                {
                    GetTopFrame().second.push_front(std::move(movedOutOfScope));
                    // BUG: The error in the "compute" version of Vector_test3 may instead be here. It may be that `ConstantDataToValue()` isn't returning the correct part of the "moved out of scope" data.
                    return ConstantDataToValue(GetTopFrame().second.front(), expectedReturn.GetLayout());
                }
                else
                {
                    return *maybeGlobal;
                }
            }
            else
            {
                FunctionScope scope(*this, fnName);

                // equivalent of a void return type
                (void)fn(fnArgs);

                return std::nullopt;
            }
        };

        _definedFunctions[decl] = returnFn;
        return returnFn;
    }

    bool ComputeContext::IsFunctionDefinedImpl(FunctionDeclaration decl) const
    {
        if (const auto& intrinsics = GetIntrinsics();
            std::find(intrinsics.begin(), intrinsics.end(), decl) != intrinsics.end())
        {
            return true;
        }

        std::lock_guard lock{ _mutex };
        return _definedFunctions.find(decl) != _definedFunctions.end();
    }

    void ComputeContext::CopyDataImpl(const Value& source, Value& destination)
    {
        std::visit(
            VariantVisitor{
                [](Emittable) {},
                [&destination, &source](auto&& sourceData) {
                    using SourceDataType = std::decay_t<decltype(sourceData)>;

                    if (source.PointerLevel() == destination.PointerLevel())
                    {
                        if (source.PointerLevel() == 1)
                        {
                            auto& destinationData = std::get<SourceDataType>(destination.GetUnderlyingData());
                            if (source.GetLayout().IsContiguous() && destination.GetLayout().IsContiguous())
                            {
                                auto numElements = destination.GetLayout().NumElements();
                                std::copy(sourceData, sourceData + numElements, destinationData);
                            }
                            else
                            {
                                auto& sourceLayout = source.GetLayout();
                                auto maxCoordinate = sourceLayout.GetActiveSize().ToVector();
                                decltype(maxCoordinate) coordinate(maxCoordinate.size());

                                do
                                {
                                    auto logicalCoordinates = sourceLayout.GetLogicalCoordinates(coordinate);
                                    auto sourceOffset =
                                        sourceLayout.GetLogicalEntryOffset(logicalCoordinates);
                                    auto destinationOffset =
                                        destination.GetLayout().GetLogicalEntryOffset(logicalCoordinates);
                                    *(destinationData + destinationOffset) = *(sourceData + sourceOffset);
                                } while (IncrementMemoryCoordinate(coordinate, maxCoordinate));
                            }
                        }
                        else
                        {
                            std::get<IntPtrT*>(destination.GetUnderlyingData())[0] = std::get<IntPtrT*>(source.GetUnderlyingData())[0];
                            if (source.IsConstrained())
                            {
                                destination.SetLayout(source.GetLayout());
                            }
                            else
                            {
                                destination.ClearLayout();
                            }
                        }
                    }
                    else
                    {
                        throw LogicException(LogicExceptionErrors::illegalState);
                    }
                } },
            source.GetUnderlyingData());
    }

    void ComputeContext::MoveDataImpl(Value& source, Value& destination)
    {
        // we treat a move the same as a copy, except we clear out the source
        CopyDataImpl(source, destination);

        // data has been "moved", so clear the source
        source.Reset();
    }

    void ComputeContext::ForImpl(MemoryLayout layout, std::function<void(std::vector<Scalar>)> fn, [[maybe_unused]] const std::string& name)
    {
        auto maxCoordinate = layout.GetActiveSize().ToVector();
        decltype(maxCoordinate) coordinate(maxCoordinate.size());

        do
        {
            auto logicalCoordinates = layout.GetLogicalCoordinates(coordinate).ToVector();
            fn(std::vector<Scalar>(logicalCoordinates.begin(), logicalCoordinates.end()));
        } while (IncrementMemoryCoordinate(coordinate, maxCoordinate));
    }

    void ComputeContext::ForImpl(Scalar start, Scalar stop, Scalar step, std::function<void(Scalar)> fn, [[maybe_unused]] const std::string& name)
    {
        if (!(start.GetValue().IsConstant() && stop.GetValue().IsConstant() && step.GetValue().IsConstant()))
        {
            throw InputException(InputExceptionErrors::invalidArgument, "start/stop/step values must be constant for ComputeContext");
        }

        std::visit(
            [&](auto&& data) {
                using Type = std::remove_pointer_t<std::decay_t<decltype(data)>>;
                if constexpr (IsOneOf<Type, Emittable, Boolean>)
                {
                    // no op
                }
                else
                {
                    auto startNum = start.Get<Type>();
                    auto stopNum = stop.Get<Type>();
                    auto stepNum = step.Get<Type>();

                    for (; startNum < stopNum; startNum += stepNum)
                    {
                        fn(startNum);
                    }
                }
            },
            start.GetValue().GetUnderlyingData());
    }

    Value ComputeContext::ReferenceImpl(Value source)
    {
        return std::visit(
            [&](auto&& data) -> Value {
                using DecayedType = std::decay_t<decltype(data)>;
                using Type = std::remove_pointer_t<DecayedType>;
                if constexpr (std::is_same_v<Type, Emittable>)
                {
                    throw LogicException(LogicExceptionErrors::illegalState);
                }
                else
                {
                    auto pointerCount = source.PointerLevel() + 1;
                    auto type = source.GetBaseType();
                    auto layout = source.GetLayout();
                    detail::ValueTypeDescription typeDesc{ type, pointerCount };
                    Value value{ typeDesc, layout };
                    value.SetData(reinterpret_cast<IntPtrT>(data), true);
                    return value;
                }
            },
            source.GetUnderlyingData());
    }

    Value ComputeContext::DereferenceImpl(Value source)
    {
        return std::visit(
            [&](auto&& data) -> Value {
                using DecayedType = std::decay_t<decltype(data)>;
                using Type = std::remove_pointer_t<DecayedType>;
                if constexpr (std::is_same_v<Type, Emittable>)
                {
                    throw LogicException(LogicExceptionErrors::illegalState);
                }
                else if constexpr (std::is_same_v<Type, IntPtrT>)
                {
                    [[maybe_unused]] auto type = source.GetBaseType();
                    auto layout = source.GetLayout();
                    if (auto pointerCount = source.PointerLevel(); pointerCount > 2)
                    {
                        auto address = *data;

                        detail::ValueTypeDescription typeDesc{ type, pointerCount - 1 };
                        Value value{ typeDesc, layout };

                        value.SetData(*reinterpret_cast<IntPtrT*>(address), true);
                        return value;
                    }
                    else
                    {
                        auto address = *data;
                        switch (type)
                        {
                        case ValueType::Boolean:
                            return { reinterpret_cast<bool*>(address), layout };
                        case ValueType::Char8:
                            return { reinterpret_cast<char*>(address), layout };
                        case ValueType::Byte:
                            return { reinterpret_cast<uint8_t*>(address), layout };
                        case ValueType::Int16:
                            return { reinterpret_cast<int16_t*>(address), layout };
                        case ValueType::Int32:
                            return { reinterpret_cast<int32_t*>(address), layout };
                        case ValueType::Int64:
                            return { reinterpret_cast<int64_t*>(address), layout };
                        case ValueType::Float:
                            return { reinterpret_cast<float*>(address), layout };
                        case ValueType::Double:
                            return { reinterpret_cast<double*>(address), layout };
                        default:
                            throw LogicException(LogicExceptionErrors::notImplemented);
                        }
                    }
                }
                else
                {
                    detail::ValueTypeDescription typeDesc{ source.GetBaseType(), 0 };
                    Value value{ typeDesc, utilities::ScalarLayout };

                    value.SetData(*data);
                    return value;
                }
            },
            source.GetUnderlyingData());
    }

    Value ComputeContext::UnaryOperationImpl(ValueUnaryOperation op, Value destination)
    {
        throw LogicException(LogicExceptionErrors::notImplemented);
    }

    Value ComputeContext::BinaryOperationImpl(ValueBinaryOperation op, Value destination, Value source)
    {
        if (!ValidateValue(source))
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }
        if (!ValidateValue(destination))
        {
            destination = Allocate(source.GetBaseType(), source.GetLayout());
        }

        if (!TypeCompatible(destination, source))
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        if (destination.GetLayout() != source.GetLayout())
        {
            throw InputException(InputExceptionErrors::sizeMismatch);
        }

        std::visit(
            VariantVisitor{
                [](Emittable) {},
                [&destination, &source, op](auto&& destinationData) {
                    using DestinationDataType =
                        std::remove_pointer_t<std::decay_t<decltype(destinationData)>>;

                    std::function<DestinationDataType(DestinationDataType, DestinationDataType)>
                        opFn;
                    if constexpr (!std::is_same_v<DestinationDataType, Boolean>)
                    {
                        switch (op)
                        {
                        case ValueBinaryOperation::add:
                            opFn = [](auto dst, auto src) { return dst + src; };
                            break;
                        case ValueBinaryOperation::subtract:
                            opFn = [](auto dst, auto src) { return dst - src; };
                            break;
                        case ValueBinaryOperation::multiply:
                            opFn = [](auto dst, auto src) { return dst * src; };
                            break;
                        case ValueBinaryOperation::divide:
                            opFn = [](auto dst, auto src) { return dst / src; };
                            break;

                        default:
                            if constexpr (std::is_integral_v<DestinationDataType>)
                            {
                                switch (op)
                                {
                                case ValueBinaryOperation::modulus:
                                    opFn = [](auto dst, auto src) { return dst % src; };
                                    break;
                                default:
                                    throw LogicException(LogicExceptionErrors::illegalState);
                                }
                            }
                            else
                            {
                                throw LogicException(LogicExceptionErrors::illegalState);
                            }
                        }
                    }
                    else
                    {
                        switch (op)
                        {
                        case ValueBinaryOperation::logicalAnd:
                            opFn = [](auto dst, auto src) { return dst && src; };
                            break;
                        case ValueBinaryOperation::logicalOr:
                            opFn = [](auto dst, auto src) { return dst || src; };
                            break;
                        default:
                            throw LogicException(LogicExceptionErrors::illegalState);
                        }
                    }

                    auto& sourceData = std::get<DestinationDataType*>(source.GetUnderlyingData());
                    if (source.GetLayout().IsContiguous() && destination.GetLayout().IsContiguous())
                    {
                        auto numElements = destination.GetLayout().NumElements();
                        std::transform(destinationData,
                                       destinationData + numElements,
                                       sourceData,
                                       destinationData,
                                       opFn);
                    }
                    else
                    {
                        auto& sourceLayout = source.GetLayout();
                        auto maxCoordinate = sourceLayout.GetActiveSize().ToVector();
                        decltype(maxCoordinate) coordinate(maxCoordinate.size());

                        do
                        {
                            auto logicalCoordinates = sourceLayout.GetLogicalCoordinates(coordinate);
                            auto sourceOffset =
                                sourceLayout.GetLogicalEntryOffset(logicalCoordinates);
                            auto destinationOffset =
                                destination.GetLayout().GetLogicalEntryOffset(logicalCoordinates);
                            *(destinationData + destinationOffset) =
                                opFn(*(destinationData + destinationOffset),
                                     *(sourceData + sourceOffset));
                        } while (IncrementMemoryCoordinate(coordinate, maxCoordinate));
                    }
                } },
            destination.GetUnderlyingData());

        return destination;
    }

    Value ComputeContext::LogicalOperationImpl(ValueLogicalOperation op, Value source1, Value source2)
    {
        if (source1.GetLayout() != source2.GetLayout())
        {
            throw InputException(InputExceptionErrors::sizeMismatch);
        }

        Value returnValue =
            std::visit(
                VariantVisitor{
                    [](Emittable) -> Boolean {
                        throw LogicException(LogicExceptionErrors::illegalState);
                    },
                    [op,
                     &source2Data = source2.GetUnderlyingData(),
                     &source1Layout = source1.GetLayout(),
                     &source2Layout = source2.GetLayout()](auto&& source1) -> Boolean {
                        using Type = std::remove_pointer_t<std::decay_t<decltype(source1)>>;

                        std::function<bool(Type, Type)> opFn;
                        switch (op)
                        {
                        case ValueLogicalOperation::equality:
                            opFn = std::equal_to<Type>{};
                            break;
                        case ValueLogicalOperation::inequality:
                            opFn = std::not_equal_to<Type>{};
                            break;
                        default:
                            if constexpr (std::is_same_v<Boolean, Type>)
                            {
                                throw LogicException(LogicExceptionErrors::illegalState);
                            }
                            else
                            {
                                switch (op)
                                {
                                case ValueLogicalOperation::greaterthan:
                                    opFn = std::greater<Type>{};
                                    break;
                                case ValueLogicalOperation::greaterthanorequal:
                                    opFn = std::greater_equal<Type>{};
                                    break;
                                case ValueLogicalOperation::lessthan:
                                    opFn = std::less<Type>{};
                                    break;
                                case ValueLogicalOperation::lessthanorequal:
                                    opFn = std::less_equal<Type>{};
                                    break;
                                default:
                                    throw LogicException(LogicExceptionErrors::illegalState);
                                }
                            }
                        }

                        auto maxCoordinate = source1Layout.GetActiveSize().ToVector();
                        decltype(maxCoordinate) coordinate(maxCoordinate.size());

                        bool b = true;
                        auto source2 = std::get<Type*>(source2Data);
                        do
                        {
                            auto logicalCoordinates =
                                source1Layout.GetLogicalCoordinates(coordinate);
                            auto source1Offset =
                                source1Layout.GetLogicalEntryOffset(logicalCoordinates);
                            auto source2Offset =
                                source2Layout.GetLogicalEntryOffset(logicalCoordinates);
                            b &= opFn(source1[source1Offset], source2[source2Offset]);
                        } while (IncrementMemoryCoordinate(coordinate, maxCoordinate));

                        return b;
                    } },
                source1.GetUnderlyingData());

        return returnValue;
    }

    Value ComputeContext::CastImpl(Value value, ValueType destType)
    {
        if (!ValidateValue(value))
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }

        ConstantData castedData;

        std::visit(
            VariantVisitor{
                [](Emittable) {},
                [&castedData, destType, &value](auto&& data) {
                    auto ptrBegin = data;
                    auto ptrEnd = data + value.GetLayout().GetMemorySize();

                    switch (destType)
                    {
                    case ValueType::Boolean:
                        castedData = std::vector<Boolean>(ptrBegin, ptrEnd);
                        break;
                    case ValueType::Char8:
                        castedData = std::vector<char>(ptrBegin, ptrEnd);
                        break;
                    case ValueType::Byte:
                        castedData = std::vector<uint8_t>(ptrBegin, ptrEnd);
                        break;
                    case ValueType::Int16:
                        castedData = std::vector<int16_t>(ptrBegin, ptrEnd);
                        break;
                    case ValueType::Int32:
                        castedData = std::vector<int32_t>(ptrBegin, ptrEnd);
                        break;
                    case ValueType::Int64:
                        castedData = std::vector<int64_t>(ptrBegin, ptrEnd);
                        break;
                    case ValueType::Float:
                        castedData = std::vector<float>(ptrBegin, ptrEnd);
                        break;
                    case ValueType::Double:
                        castedData = std::vector<double>(ptrBegin, ptrEnd);
                        break;
                    default:
                        throw LogicException(LogicExceptionErrors::notImplemented);
                    }
                } },
            value.GetUnderlyingData());

        Value castedValue = StoreConstantData(std::move(castedData));
        castedValue.SetLayout(value.GetLayout());
        return castedValue;
    }

    class ComputeContext::IfContextImpl : public EmitterContext::IfContextImpl
    {
    public:
        IfContextImpl(bool state) :
            _state(state) {}

        void ElseIf(Scalar test, std::function<void()> fn) override
        {
            if (!test.GetValue().IsConstant())
            {
                throw LogicException(LogicExceptionErrors::illegalState);
            }

            if (!_state && test.Get<Boolean>())
            {
                fn();
                _state = !_state;
            }
        }

        void Else(std::function<void()> fn) override
        {
            if (!_state)
            {
                fn();
                _state = !_state;
            }
        }

    private:
        bool _state;
    };

    EmitterContext::IfContext ComputeContext::IfImpl(Scalar test, std::function<void()> fn)
    {
        if (!test.GetValue().IsConstant())
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        bool state = test.Get<Boolean>();
        if (state)
        {
            fn();
        }

        return { std::make_unique<ComputeContext::IfContextImpl>(state) };
    }

    void ComputeContext::WhileImpl(Scalar test, std::function<void()> fn)
    {
        if (!(test.GetValue().IsConstant()))
        {
            throw InputException(InputExceptionErrors::invalidArgument, "start/stop/step values must be constant for ComputeContext");
        }

        std::visit(
            [&](auto&& data) {
                using Type = std::remove_pointer_t<std::decay_t<decltype(data)>>;
                if constexpr (IsOneOf<Type, Boolean>)
                {
                    auto testVal = test.Get<Boolean>();

                    while (testVal)
                    {
                        fn();
                        testVal = test.Get<Boolean>();
                    }
                }
                else
                {
                    // error?
                }
            },
            test.GetValue().GetUnderlyingData());
    }

    std::optional<Value> ComputeContext::CallImpl(FunctionDeclaration func, std::vector<Value> args)
    {
        if (!std::all_of(args.begin(), args.end(), [this](const Value& value) { return ValidateValue(value); }))
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        const auto& intrinsics = GetIntrinsics();
        if (intrinsics.end() != std::find(intrinsics.begin(), intrinsics.end(), func))
        {
            return IntrinsicCall(func, args);
        }

        if (func.IsPointerSet())
        {
            auto ptr = func.GetPointer();
            return (*reinterpret_cast<DefinedFunction*>(reinterpret_cast<void*>(ptr.Get<IntPtrT>())))(args);
        }

        {
            std::lock_guard lock{ _mutex };
            if (auto it = _definedFunctions.find(func); it != _definedFunctions.end())
            {
                return it->second(args);
            }
        }

        throw InputException(InputExceptionErrors::invalidArgument, "Specified function is not defined for this context");
    }

    void ComputeContext::PrefetchImpl(Value, PrefetchType, PrefetchLocality)
    {}

    void ComputeContext::ParallelizeImpl(int numTasks, std::vector<Value> captured, std::function<void(Scalar, std::vector<Value>)> fn)
    {
        ThreadIds.Clear();

        std::vector<std::future<void>> futures;
        futures.reserve(numTasks);
        for (int i = 0; i < numTasks; ++i)
        {
            futures.push_back(std::async(fn, Scalar{ i }, captured));
        }
        for (auto& future : futures)
        {
            future.wait();
        }
    }

    namespace
    {
        void PrintValue(const Value& value, std::ostream& stream)
        {
            if (!value.IsEmpty())
            {
                std::visit(
                    [&stream](auto&& data) {
                        using Type = std::decay_t<decltype(data)>;
                        if constexpr (std::is_same_v<Type, Emittable>)
                        {
                            return;
                        }
                        else
                        {
                            using DataTypeTemp = std::remove_pointer_t<Type>;
                            using DataType = std::conditional_t<std::is_same_v<DataTypeTemp, utilities::Boolean>, bool, DataTypeTemp>;
                            stream << utilities::TypeName<DataType>::GetName();
                        }
                    },
                    value.GetUnderlyingData());
            }
            else
            {
                stream << ToString(value.GetBaseType());
            }
            stream << "@(" << value.GetLayout() << ")";
        }
    } // namespace

    void ComputeContext::DebugBreakImpl()
    {
        throw 0; // TODO: throw a real exception (of type value::Exception::DebugTrapException, perhaps)
    }

    void ComputeContext::DebugDumpImpl(Value value, std::string tag, std::ostream& stream) const
    {
        PrintValue(value, stream);
        if (!tag.empty())
        {
            stream << " [tag = " << tag << "]";
        }
        stream << "\n";
    }

    void ComputeContext::DebugDumpImpl(FunctionDeclaration fn, std::string tag, std::ostream& stream) const
    {
        stream << (IsFunctionDefined(fn) ? "Defined " : "Undefined ") << "function: " << fn.GetFunctionName() << "(";
        if (const auto& paramTypes = fn.GetParameterTypes(); !paramTypes.empty())
        {
            for (auto it = paramTypes.begin(); it != paramTypes.end() - 1; ++it)
            {
                PrintValue(*it, stream);
                stream << ", ";
            }
            PrintValue(paramTypes.back(), stream);
        }
        stream << ") -> ";
        if (const auto& returnType = fn.GetReturnType(); returnType)
        {
            PrintValue(*returnType, stream);
        }
        else
        {
            stream << "void";
        }
        stream << "\n";
    }

    void ComputeContext::DebugPrintImpl(std::string message)
    {
        std::cout << message;
    }

    void ComputeContext::SetNameImpl(const Value& value, const std::string& name)
    {
        std::lock_guard lock{ _mutex };
        _namedValues[value] = name;
    }

    std::string ComputeContext::GetNameImpl(const Value& value) const
    {
        std::lock_guard lock{ _mutex };
        if (auto it = _namedValues.find(value); it != _namedValues.end())
        {
            return it->second;
        }

        return {};
    }

    void ComputeContext::ImportCodeFileImpl(std::string)
    {
        throw LogicException(LogicExceptionErrors::notImplemented);
    }

    Scalar ComputeContext::GetFunctionAddressImpl(const FunctionDeclaration& fn)
    {
        {
            std::lock_guard lock{ _mutex };
            if (auto it = _definedFunctions.find(fn); it != _definedFunctions.end())
            {
                return reinterpret_cast<IntPtrT>(reinterpret_cast<void*>(&(it->second)));
            }
        }

        throw InputException(InputExceptionErrors::invalidArgument, "ComputeContext can't take address of function that hasn't been defined");
    }

    Value ComputeContext::IntrinsicCall(FunctionDeclaration intrinsic, std::vector<Value> args)
    {
        static std::unordered_map<FunctionDeclaration, std::function<Value(std::vector<Value>)>> intrinsics = {
            { AbsFunctionDeclaration, SimpleNumericalFunctionIntrinsic{}(AbsFunctionIntrinsic{}) },
            { CosFunctionDeclaration, SimpleNumericalFunctionIntrinsic{}([](auto n) { return std::cos(n); }) },
            { ExpFunctionDeclaration, SimpleNumericalFunctionIntrinsic{}([](auto n) { return std::exp(n); }) },
            { LogFunctionDeclaration, SimpleNumericalFunctionIntrinsic{}([](auto n) { return std::log(n); }) },
            { Log10FunctionDeclaration, SimpleNumericalFunctionIntrinsic{}([](auto n) { return std::log10(n); }) },
            { Log2FunctionDeclaration, SimpleNumericalFunctionIntrinsic{}([](auto n) { return std::log2(n); }) },
            { MaxNumFunctionDeclaration, SimpleMinMaxNumFunctionIntrinsic{}(MaxFn{}) },
            { MinNumFunctionDeclaration, SimpleMinMaxNumFunctionIntrinsic{}(MinFn{}) },
            { PowFunctionDeclaration, PowFunctionIntrinsic{} },
            { InitializeVectorFunctionDeclaration, InitializeVectorFunctionIntrinsic{} },
            { SinFunctionDeclaration, SimpleNumericalFunctionIntrinsic{}([](auto n) { return std::sin(n); }) },
            { SqrtFunctionDeclaration, SimpleNumericalFunctionIntrinsic{}([](auto n) { return std::sqrt(n); }) },
            { TanhFunctionDeclaration, SimpleNumericalFunctionIntrinsic{}([](auto n) { return std::tanh(n); }) },
            { RoundFunctionDeclaration, SimpleNumericalFunctionIntrinsic{}([](auto n) { return std::round(n); }) },
            { FloorFunctionDeclaration, SimpleNumericalFunctionIntrinsic{}([](auto n) { return std::floor(n); }) },
            { CeilFunctionDeclaration, SimpleNumericalFunctionIntrinsic{}([](auto n) { return std::ceil(n); }) },
            { CopySignFunctionDeclaration, CopySignFunctionIntrinsic{} },
            { FmaFunctionDeclaration, FmaFunctionIntrinsic{} },
            { MemCopyFunctionDeclaration, MemOpFunctionIntrinsic<MemIntrinsicOp::Copy>{} },
            { MemMoveFunctionDeclaration, MemOpFunctionIntrinsic<MemIntrinsicOp::Move>{} },
            { MemSetFunctionDeclaration, MemOpFunctionIntrinsic<MemIntrinsicOp::Set>{} },
        };

        if (auto it = intrinsics.find(intrinsic); it != intrinsics.end())
        {
            return it->second(args);
        }
        else
        {
            throw LogicException(LogicExceptionErrors::notImplemented);
        }
    }

    bool ComputeContext::ValidateValue(Value value) const
    {
        return value.IsDefined() && !value.IsEmpty() && value.IsConstant();
    }

    bool ComputeContext::TypeCompatible(Value value1, Value value2) const
    {
        auto pointerLevel1 = value1.PointerLevel();
        auto pointerLevel2 = value2.PointerLevel();

        if (pointerLevel1 == pointerLevel2 &&
            pointerLevel1 == 1)
        {
            if (value1.GetBaseType() != value2.GetBaseType())
            {
                throw InputException(InputExceptionErrors::typeMismatch);
            }
            return true;
        }

        if (pointerLevel1 > 1 && pointerLevel2 > 1)
        {
            return false;
        }

        constexpr auto intptrType = GetValueType<IntPtrT>();
        if (pointerLevel1 > 1)
        {
            assert(pointerLevel2 == 1);
            return value2.GetBaseType() == intptrType;
        }

        if (pointerLevel2 > 1)
        {
            assert(pointerLevel1 == 1);
            return value1.GetBaseType() == intptrType;
        }

        assert(false);
        return false;
    }

    std::string ComputeContext::GetScopeAdjustedName(GlobalAllocationScope scope, std::string name) const
    {
        switch (scope)
        {
        case GlobalAllocationScope::Global:
            return GetGlobalScopedName(name);
        case GlobalAllocationScope::Function:
            return GetCurrentFunctionScopedName(name);
        }

        throw LogicException(LogicExceptionErrors::illegalState);
    }

    std::string ComputeContext::GetGlobalScopedName(std::string name) const
    {
        return _moduleName + "_" + name;
    }

    std::string ComputeContext::GetCurrentFunctionScopedName(std::string name) const
    {
        // Our stack always has one empty "scope" pushed to it, which we
        // can use to create our global prefix.

        std::lock_guard lock{ _mutex };
        return GetGlobalScopedName(GetTopFrame().first + "_" + name);
    }

    ComputeContext::Frame& ComputeContext::GetTopFrame()
    {
        return _stack.top();
    }

    const ComputeContext::Frame& ComputeContext::GetTopFrame() const
    {
        return _stack.top();
    }

    void swap(ComputeContext& l, ComputeContext& r) noexcept
    {
        using std::swap;

        swap(static_cast<EmitterContext&>(l), static_cast<EmitterContext&>(r));
        swap(l._stack, r._stack);
        swap(l._globals, r._globals);
        swap(l._definedFunctions, r._definedFunctions);
        swap(l._namedValues, r._namedValues);
        swap(l._moduleName, r._moduleName);
    }
} // namespace value
} // namespace ell
