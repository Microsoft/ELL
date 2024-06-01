////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     EmitterContext.cpp (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "EmitterContext.h"
#include "FunctionDeclaration.h"
#include "Scalar.h"
#include "Value.h"
#include "Vector.h"

#include <emitters/include/IREmitter.h>

#include <utilities/include/Exception.h>

#include <iostream>
#include <tuple>
#include <utility>

namespace ell
{
namespace value
{
    using namespace utilities;

    namespace detail
    {
        Scalar CalculateOffset(const MemoryLayout& layout, std::vector<Scalar> coordinates)
        {
            if (layout == ScalarLayout)
            {
                assert(coordinates.empty());
                return { 0 };
            }
            else
            {
                const auto& offset = layout.GetOffset();
                const auto& increment = layout.GetCumulativeIncrement();
                const auto& order = layout.GetLogicalDimensionOrder();
                const auto numDimensions = layout.NumDimensions();

                Scalar result;
                for (int index = 0; index < numDimensions; ++index)
                {
                    result += increment[index] * (coordinates[order[index]] + offset[index]);
                }

                return result;
            }
        }

    } // namespace detail

    EmitterContext::IfContextImpl::~IfContextImpl() = default;

    EmitterContext::IfContext::IfContext(std::unique_ptr<EmitterContext::IfContextImpl> impl) :
        _impl(std::move(impl))
    {}

    EmitterContext::IfContext&& EmitterContext::IfContext::ElseIf(Scalar test, std::function<void()> fn) &&
    {
        if (test.GetType() != ValueType::Boolean)
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        _impl->ElseIf(test, fn);

        return std::move(*this);
    }

    void EmitterContext::IfContext::ElseIf(Scalar test, std::function<void()> fn) &
    {
        if (test.GetType() != ValueType::Boolean)
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        _impl->ElseIf(test, fn);
    }

    void EmitterContext::IfContext::Else(std::function<void()> fn) &&
    {
        _impl->Else(fn);
    }

    void EmitterContext::IfContext::Else(std::function<void()> fn) &
    {
        _impl->Else(fn);
    }

    EmitterContext::~EmitterContext() = default;

    Value EmitterContext::Allocate(ValueType type, size_t size, size_t align, AllocateFlags flags)
    {
        return Allocate(type, MemoryLayout({ (int)size }), align, flags);
    }

    Value EmitterContext::Allocate(ValueType type, MemoryLayout layout, size_t align, AllocateFlags flags)
    {
        return AllocateImpl(type, layout, align, flags);
    }

    Value EmitterContext::StaticAllocate(std::string name, ValueType type, utilities::MemoryLayout layout, AllocateFlags flags)
    {
        if (auto globalValue = GetGlobalValue(GlobalAllocationScope::Function, name, layout))
        {
            return *globalValue;
        }

        return GlobalAllocateImpl(GlobalAllocationScope::Function, name, type, layout, flags);
    }

    Value EmitterContext::GlobalAllocate(std::string name, ValueType type, utilities::MemoryLayout layout, AllocateFlags flags)
    {
        if (auto globalValue = GetGlobalValue(GlobalAllocationScope::Global, name, layout))
        {
            return *globalValue;
        }

        return GlobalAllocateImpl(GlobalAllocationScope::Global, name, type, layout, flags);
    }

    std::optional<Value> EmitterContext::GetGlobalValue(GlobalAllocationScope scope, std::string name, MemoryLayout layout)
    {
        if (auto globalValue = GetGlobalValue(scope, name))
        {
            Value value = globalValue.value();
            if (layout.GetMemorySize() > value.GetLayout().GetMemorySize())
            {
                throw InputException(InputExceptionErrors::invalidSize);
            }
            value.SetLayout(layout);

            return value;
        }

        return std::nullopt;
    }

    detail::ValueTypeDescription EmitterContext::GetType(Emittable emittable)
    {
        return GetTypeImpl(emittable);
    }

    EmitterContext::DefinedFunction EmitterContext::CreateFunction(FunctionDeclaration decl, EmitterContext::DefinedFunction fn)
    {
        return CreateFunctionImpl(decl, fn);
    }

    bool EmitterContext::IsFunctionDefined(FunctionDeclaration decl) const
    {
        return IsFunctionDefinedImpl(decl);
    }

    Value EmitterContext::StoreConstantData(ConstantData data)
    {
        return StoreConstantDataImpl(data);
    }

    void EmitterContext::For(MemoryLayout layout, std::function<void(std::vector<Scalar>)> fn, const std::string& name)
    {
        if (layout.NumElements() == 0)
        {
            return;
        }

        return ForImpl(layout, fn, name);
    }

    void EmitterContext::For(Scalar start, Scalar stop, Scalar step, std::function<void(Scalar)> fn, const std::string& name)
    {
        if (!(start.GetType() == stop.GetType() && start.GetType() == step.GetType()))
        {
            throw InputException(InputExceptionErrors::typeMismatch, "start/stop/step types must match");
        }

        if (start.GetType() == ValueType::Boolean)
        {
            throw InputException(InputExceptionErrors::invalidArgument, "start/stop/step must not be boolean");
        }

        return ForImpl(start, stop, step, fn, name);
    }

    void EmitterContext::MoveData(Value& source, Value& destination)
    {
        return MoveDataImpl(source, destination);
    }

    void EmitterContext::CopyData(const Value& source, Value& destination)
    {
        return CopyDataImpl(source, destination);
    }

    Value EmitterContext::Reference(Value source)
    {
        return ReferenceImpl(source);
    }

    Value EmitterContext::Dereference(Value source)
    {
        if (source.PointerLevel() < 0)
        {
            throw LogicException(LogicExceptionErrors::illegalState, "Pointer level is less than the minimum of 0");
        }
        else if (source.PointerLevel() == 0)
        {
            throw LogicException(LogicExceptionErrors::illegalState, "Attempted to dereference Value that is not a reference");
        }

        return DereferenceImpl(source);
    }

    Value EmitterContext::Offset(Value begin, Value index)
    {
        return OffsetImpl(begin, index);
    }

    Value EmitterContext::Offset(Value begin, std::vector<Scalar> coordinates)
    {
        const auto& layout = begin.GetLayout();

        Scalar result = detail::CalculateOffset(layout, coordinates);
        return Offset(begin, result.GetValue());
    }

    Value EmitterContext::UnaryOperation(ValueUnaryOperation op, Value value)
    {
        return UnaryOperationImpl(op, value);
    }

    Value EmitterContext::BinaryOperation(ValueBinaryOperation op, Value destination, Value source)
    {
        return BinaryOperationImpl(op, destination, source);
    }

    Value EmitterContext::LogicalOperation(ValueLogicalOperation op, Value source1, Value source2)
    {
        if (!source1.IsDefined() || !source2.IsDefined())
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }

        if (source1.GetBaseType() != source2.GetBaseType())
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }

        return LogicalOperationImpl(op, source1, source2);
    }

    Value EmitterContext::Cast(Value value, ValueType type)
    {
        if (!value.IsConstrained())
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }

        return CastImpl(value, type);
    }

    EmitterContext::IfContext EmitterContext::If(Scalar test, std::function<void()> fn)
    {
        if (test.GetType() != ValueType::Boolean)
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        return IfImpl(test, fn);
    }

    void EmitterContext::While(Scalar test, std::function<void()> fn)
    {
        if (test.GetType() != ValueType::Boolean)
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        return WhileImpl(test, fn);
    }

    std::optional<Value> EmitterContext::Call(FunctionDeclaration func, std::vector<ViewAdapter> args)
    {
        std::vector<Value> valueArgs(args.begin(), args.end());
        return CallImpl(func, valueArgs);
    }

    void EmitterContext::Prefetch(Value data, PrefetchType type, PrefetchLocality locality)
    {
        PrefetchImpl(data, type, locality);
    }

    void EmitterContext::Parallelize(int numTasks, std::vector<Value> captured, std::function<void(Scalar, std::vector<Value>)> fn)
    {
        if (numTasks == 0) return;

        return ParallelizeImpl(numTasks, captured, fn);
    }

    void EmitterContext::DebugBreak()
    {
        DebugBreakImpl();
    }

    void EmitterContext::DebugDump(Value value, std::string tag, std::ostream* stream) const
    {
        std::ostream& outStream = stream != nullptr ? *stream : std::cerr;

        if (value.IsDefined())
        {
            DebugDumpImpl(value, tag, outStream);
        }
        else
        {
            outStream << "Value is undefined";
            if (!tag.empty())
            {
                outStream << "[tag = " << tag << "]";
            }
            outStream << "\n";
        }
    }

    void EmitterContext::DebugDump(FunctionDeclaration fn, std::string tag, std::ostream* stream) const
    {
        std::ostream& outStream = stream != nullptr ? *stream : std::cerr;

        DebugDumpImpl(fn, tag, outStream);
    }

    void EmitterContext::DebugPrint(std::string message)
    {
        DebugPrintImpl(message);
    }

    void EmitterContext::SetName(const Value& value, const std::string& name)
    {
        SetNameImpl(value, name);
    }

    std::string EmitterContext::GetName(const Value& value) const
    {
        return GetNameImpl(value);
    }

    void EmitterContext::ImportCodeFile(std::string file)
    {
        ImportCodeFileImpl(file);
    }

    const std::vector<std::reference_wrapper<FunctionDeclaration>>& EmitterContext::GetIntrinsics() const
    {
        static std::vector intrinsics = {
            std::ref(AbsFunctionDeclaration),
            std::ref(CopySignFunctionDeclaration),
            std::ref(CosFunctionDeclaration),
            std::ref(ExpFunctionDeclaration),
            std::ref(LogFunctionDeclaration),
            std::ref(Log10FunctionDeclaration),
            std::ref(Log2FunctionDeclaration),
            std::ref(MaxNumFunctionDeclaration),
            std::ref(MinNumFunctionDeclaration),
            std::ref(PowFunctionDeclaration),
            std::ref(SinFunctionDeclaration),
            std::ref(SqrtFunctionDeclaration),
            std::ref(TanhFunctionDeclaration),
            std::ref(RoundFunctionDeclaration),
            std::ref(FloorFunctionDeclaration),
            std::ref(CeilFunctionDeclaration),
            std::ref(FmaFunctionDeclaration),
            std::ref(InitializeVectorFunctionDeclaration),
            std::ref(MemCopyFunctionDeclaration),
            std::ref(MemMoveFunctionDeclaration),
            std::ref(MemSetFunctionDeclaration),
        };

        return intrinsics;
    }

    std::vector<Value> EmitterContext::NormalizeReferenceLevels(const std::vector<Value>& args, const std::vector<Value>& expected) const
    {
        if (args.size() != expected.size())
        {
            throw new InputException(InputExceptionErrors::sizeMismatch);
        }
        std::vector<Value> normalizedArgs;
        normalizedArgs.reserve(args.size());
        for (unsigned index = 0; index < args.size(); ++index)
        {
            auto& expectedValue = expected[index];
            auto& arg = args[index];
            Value value{
                { expectedValue.GetBaseType(), expectedValue.PointerLevel() },
                expectedValue.IsConstrained() ? std::optional{ expectedValue.GetLayout() } : std::optional<MemoryLayout>{ std::nullopt }
            };
            if (expectedValue.PointerLevel() == arg.PointerLevel())
            {
                value.SetData(arg, true);
            }
            else if (expectedValue.PointerLevel() == (arg.PointerLevel() - 1))
            {
                value.SetData(arg.Dereference(), true);
            }
            else
            {
                throw LogicException(LogicExceptionErrors::illegalState);
            }
            normalizedArgs.push_back(value);
        }
        return normalizedArgs;
    }

    std::string EmitterContext::UniqueName(const std::string& prefix)
    {
        auto uniqueId = _uniqueNames[prefix]++;
        return prefix + "_" + std::to_string(uniqueId);
    }

    Scalar EmitterContext::GetFunctionAddress(const FunctionDeclaration& decl)
    {
        if (const auto& intrinsics = GetIntrinsics();
            std::find(intrinsics.begin(), intrinsics.end(), decl) != intrinsics.end())
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Cannot get function address of intrinsic");
        }

        return GetFunctionAddressImpl(decl);
    }

    void swap(EmitterContext& l, EmitterContext& r) noexcept
    {
        std::swap(l._uniqueNames, r._uniqueNames);
    }

    namespace
    {
        EmitterContext* s_context = nullptr;
    }

    EmitterContext& GetContext()
    {
        if (s_context == nullptr)
        {
            throw LogicException(LogicExceptionErrors::illegalState, "EmitterContext is not set!");
        }

        return *s_context;
    }

    void SetContext(EmitterContext& context)
    {
        s_context = &context;
    }

    void ClearContext() noexcept
    {
        s_context = nullptr;
    }

    ContextGuard<>::ContextGuard(EmitterContext& context) :
        _oldContext(s_context)
    {
        SetContext(context);
    }

    ContextGuard<>::~ContextGuard()
    {
        _oldContext ? SetContext(*_oldContext) : ClearContext();
    }

    Value Allocate(ValueType type, size_t size, size_t align, AllocateFlags flags)
    {
        return GetContext().Allocate(type, size, align, flags);
    }

    Value Allocate(ValueType type, MemoryLayout layout, size_t align, AllocateFlags flags)
    {
        return GetContext().Allocate(type, layout, align, flags);
    }

    Value StaticAllocate(std::string name, ValueType type, utilities::MemoryLayout layout, AllocateFlags flags)
    {
        return GetContext().StaticAllocate(name, type, layout, flags);
    }

    Value GlobalAllocate(std::string name, ValueType type, utilities::MemoryLayout layout, AllocateFlags flags)
    {
        return GetContext().GlobalAllocate(name, type, layout, flags);
    }

    EmitterContext::IfContext If(Scalar test, std::function<void()> fn)
    {
        return GetContext().If(test, fn);
    }

    void While(Scalar test, std::function<void()> fn)
    {
        return GetContext().While(test, fn);
    }

    void ForRange(Scalar end, std::function<void(Scalar)> fn)
    {
        ForRange(std::string{}, end, fn);
    }

    void ForRange(const std::string& name, Scalar end, std::function<void(Scalar)> fn)
    {
        ForRange(name, Scalar{ 0 }, end, fn);
    }

    void ForRange(Scalar start, Scalar end, std::function<void(Scalar)> fn)
    {
        ForRange(std::string{}, start, end, fn);
    }

    void ForRange(const std::string& name, Scalar start, Scalar end, std::function<void(Scalar)> fn)
    {
        ForRange(name, start, end, 1, fn);
    }

    void ForRange(Scalar start, Scalar end, Scalar step, std::function<void(Scalar)> fn)
    {
        ForRange(std::string{}, start, end, step, fn);
    }

    void ForRange(const std::string& name, Scalar start, Scalar end, Scalar step, std::function<void(Scalar)> fn)
    {
        GetContext().For(start, end, step, fn, name);
    }

    void DebugBreak()
    {
        GetContext().DebugBreak();
    }

    void DebugDump(FunctionDeclaration fn, std::string tag, std::ostream* stream)
    {
        GetContext().DebugDump(fn, tag, stream);
    }

    void DebugDump(Value value, std::string tag, std::ostream* stream)
    {
        GetContext().DebugDump(value, tag, stream);
    }

    void DebugPrint(std::string message)
    {
        GetContext().DebugPrint(message);
    }

    void Parallelize(int numTasks, std::vector<Value> captured, std::function<void(Scalar, std::vector<Value>)> fn)
    {
        GetContext().Parallelize(numTasks, captured, fn);
    }

    Scalar Abs(Scalar s)
    {
        return *GetContext().Call(AbsFunctionDeclaration, { s.GetValue() });
    }

    Scalar Cos(Scalar s)
    {
        return *GetContext().Call(CosFunctionDeclaration, { s.GetValue() });
    }

    Scalar CopySign(Scalar s1, Scalar s2)
    {
        return *GetContext().Call(CopySignFunctionDeclaration, { s1.GetValue(), s2.GetValue() });
    }

    Scalar Exp(Scalar s)
    {
        return *GetContext().Call(ExpFunctionDeclaration, { s.GetValue() });
    }

    Scalar Log(Scalar s)
    {
        return *GetContext().Call(LogFunctionDeclaration, { s.GetValue() });
    }

    Scalar Log10(Scalar s)
    {
        return *GetContext().Call(Log10FunctionDeclaration, { s.GetValue() });
    }

    Scalar Log2(Scalar s)
    {
        return *GetContext().Call(Log2FunctionDeclaration, { s.GetValue() });
    }

    Scalar Max(Scalar s1, Scalar s2)
    {
        return *GetContext().Call(MaxNumFunctionDeclaration, { s1.GetValue(), s2.GetValue() });
    }

    Scalar Min(Scalar s1, Scalar s2)
    {
        return *GetContext().Call(MinNumFunctionDeclaration, { s1.GetValue(), s2.GetValue() });
    }

    Scalar Pow(Scalar base, Scalar exp)
    {
        return *GetContext().Call(PowFunctionDeclaration, { base.GetValue(), exp.GetValue() });
    }

    Scalar Sin(Scalar s)
    {
        return *GetContext().Call(SinFunctionDeclaration, { s.GetValue() });
    }

    Scalar Sqrt(Scalar s)
    {
        return *GetContext().Call(SqrtFunctionDeclaration, { s.GetValue() });
    }

    Scalar Tanh(Scalar s)
    {
        return *GetContext().Call(TanhFunctionDeclaration, { s.GetValue() });
    }

    Scalar Round(Scalar v)
    {
        return *GetContext().Call(RoundFunctionDeclaration, { v.GetValue() });
    }

    Scalar Floor(Scalar s)
    {
        return *GetContext().Call(FloorFunctionDeclaration, { s.GetValue() });
    }

    Scalar Ceil(Scalar s)
    {
        return *GetContext().Call(CeilFunctionDeclaration, { s.GetValue() });
    }

    Scalar Fma(Scalar a, Scalar b, Scalar c)
    {
        return *GetContext().Call(FmaFunctionDeclaration, { a.GetValue(), b.GetValue(), c.GetValue() });
    }

    Scalar Sign(Scalar s)
    {
        return *GetContext().Call(CopySignFunctionDeclaration, { Cast(1, s.GetType()).GetValue(), s.GetValue() });
    }

    Vector Abs(Vector v)
    {
        return *GetContext().Call(AbsFunctionDeclaration, { v.GetValue() });
    }

    Vector Cos(Vector v)
    {
        return *GetContext().Call(CosFunctionDeclaration, { v.GetValue() });
    }

    Vector Exp(Vector v)
    {
        return *GetContext().Call(ExpFunctionDeclaration, { v.GetValue() });
    }

    Vector Log(Vector v)
    {
        return *GetContext().Call(LogFunctionDeclaration, { v.GetValue() });
    }

    Vector Log10(Vector v)
    {
        return *GetContext().Call(Log10FunctionDeclaration, { v.GetValue() });
    }

    Vector Log2(Vector v)
    {
        return *GetContext().Call(Log2FunctionDeclaration, { v.GetValue() });
    }

    Scalar Max(Vector v)
    {
        return *GetContext().Call(MaxNumFunctionDeclaration, { v.GetValue() });
    }

    Scalar Min(Vector v)
    {
        return *GetContext().Call(MinNumFunctionDeclaration, { v.GetValue() });
    }

    Vector Maximize(Vector v)
    {
        Scalar s = Max(v);
        auto r = InitVector(v, s);
        return r;
    }

    Vector Minimize(Vector v)
    {
        Scalar s = Min(v);
        return InitVector(v, s);
    }

    Vector Pow(Vector bases, Scalar exp)
    {
        return *GetContext().Call(PowFunctionDeclaration, { bases.GetValue(), exp.GetValue() });
    }

    Vector InitVector(Vector v, Scalar value)
    {
        return *GetContext().Call(InitializeVectorFunctionDeclaration, { v.GetValue(), value.GetValue() });
    }

    Vector Sin(Vector v)
    {
        return *GetContext().Call(SinFunctionDeclaration, { v.GetValue() });
    }

    Vector Sqrt(Vector v)
    {
        return *GetContext().Call(SqrtFunctionDeclaration, { v.GetValue() });
    }

    Scalar Square(const Scalar x)
    {
        auto y = x.Copy();
        return x * y;
    }

    Vector Tanh(Vector v)
    {
        return *GetContext().Call(TanhFunctionDeclaration, { v.GetValue() });
    }

    Scalar LogicalNot(const Scalar v)
    {
        Scalar r;
        if (v.GetType() == ValueType::Boolean)
        {
            ell::utilities::Boolean t(true);
            r = (v != t);
        }
        else
        {
            If(v == Cast(0, v.GetType()), [&] {
                r = Cast(1, v.GetType());
            }).Else([&] {
                r = Cast(0, v.GetType());
            });
        }
        return r;
    }

    Vector Round(Vector v)
    {
        return *GetContext().Call(RoundFunctionDeclaration, { v.GetValue() });
    }
    Vector Floor(Vector v)
    {
        return *GetContext().Call(FloorFunctionDeclaration, { v.GetValue() });
    }

    Vector Ceil(Vector v)
    {
        return *GetContext().Call(CeilFunctionDeclaration, { v.GetValue() });
    }

    void MemCopy(ViewAdapter dest, ViewAdapter source, std::optional<Scalar> length)
    {
        (void)GetContext().Call(MemCopyFunctionDeclaration, { dest, source, length.value_or(static_cast<int64_t>(source.GetValue().GetLayout().GetMemorySize())).GetValue() });
    }

    void MemMove(ViewAdapter dest, ViewAdapter source, std::optional<Scalar> length)
    {
        (void)GetContext().Call(MemMoveFunctionDeclaration, { dest, source, length.value_or(static_cast<int64_t>(source.GetValue().GetLayout().GetMemorySize())).GetValue() });
    }

    void MemSet(ViewAdapter dest, Scalar data, std::optional<Scalar> length)
    {
        if (data.GetType() != ValueType::Char8)
        {
            throw InputException(InputExceptionErrors::typeMismatch, "Memory pattern specified by data is expected to be of type Char8");
        }

        (void)GetContext().Call(MemSetFunctionDeclaration, { dest, data.GetValue(), length.value_or(static_cast<int64_t>(dest.GetValue().GetLayout().GetMemorySize())).GetValue() });
    }

    void ZeroMemory(ViewAdapter dest, std::optional<Scalar> length)
    {
        // As of 9/11/2019, when compiling with C++17, `Value{ char{} }` in place of `Value(char{})`
        // triggers the `std::initializer_list<T>` ctor for Value, instead of the `Value<T>(T)` ctor.
        // This might change in C++20.
        MemSet(dest, Value(char{}), length);
    }

    std::string UniqueName(const std::string& prefix)
    {
        return GetContext().UniqueName(prefix);
    }

} // namespace value
} // namespace ell
