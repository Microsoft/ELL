////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CppEmitterContext.cpp (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CppEmitterContext.h"
#include "FunctionDeclaration.h"

#include <utilities/include/Exception.h>
#include <utilities/include/StringUtil.h>
#include <utilities/include/TypeTraits.h>

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cmath>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

using namespace ell::utilities;

namespace ell
{
namespace value
{
    struct CppEmitterContext::FunctionScope
    {
        FunctionScope(CppEmitterContext& context, const std::string& fnName) :
            _context(context),
            _guard(*this)
        {
            _context._fnStacks.push({ {}, fnName });
            _context._promotedConstantStack.push({});
        }

        ~FunctionScope()
        {
            _context._fnStacks.pop();
            _context._promotedConstantStack.pop();
        }

    private:
        struct StreamGuard
        {
            StreamGuard(FunctionScope& context) :
                _context(context),
                _oldStream(_context._context._stream),
                _oldIndent(_context._context._indent)
            {
                _context._context._indent = 0;
                _context._context._stream = _context._sstr;
            }

            ~StreamGuard()
            {
                _context._context._stream = _oldStream;
                _context._context._indent = _oldIndent;

                // write our contents directly to the expression stream because the old stream
                // might point to someone else's stream
                _context._context._expressionStream << _context._sstr.str();
            }

        private:
            FunctionScope& _context;
            std::reference_wrapper<std::ostream> _oldStream;
            decltype(CppEmitterContext::_indent) _oldIndent;
        };

        CppEmitterContext& _context;
        std::stringstream _sstr;
        StreamGuard _guard;
    };

    CppEmitterContext::CppEmitterContext(std::string moduleName, std::ostream& stream) :
        CppEmitterContext(emitters::GetTargetDevice("host"), moduleName, stream)
    {
    }

    CppEmitterContext::CppEmitterContext(std::string moduleName, std::unique_ptr<std::ostream> stream) :
        CppEmitterContext(emitters::GetTargetDevice("host"), moduleName, std::move(stream))
    {
    }

    CppEmitterContext::CppEmitterContext(const TargetDevice& target, std::string moduleName, std::unique_ptr<std::ostream> stream) :
        CppEmitterContext(target, moduleName, *stream)
    {
        _ownedStream = std::move(stream);
    }

    CppEmitterContext::CppEmitterContext(const TargetDevice& target, std::string moduleName, std::ostream& stream) :
        EmitterContext(target),
        _computeContext(moduleName),
        _stream(_expressionStream),
        _outputStream(stream),
        _moduleName(std::move(moduleName))
    {
        Global() << "// Instantiating CppEmitterContext\n"
                    "// Writing "
                 << _moduleName << ".cpp\n"
                 << "\n"
                    "#include <algorithm>\n"
                    "#include <array>\n"
                    "#include <cmath>\n"
                    "#include <cstring>\n"
                    "#include <cstdint>\n"
                    "#include <future>\n"
                    "#include <iostream>\n"
                    "#include <vector>\n"
                    "\n"
                    "\n"
                    "#if !defined(VALUE_CPP_EMITTER_HELPERS_DEFINED)\n"
                    "#define VALUE_CPP_EMITTER_HELPERS_DEFINED\n"
                    "template <typename T> using Scalar = std::array<T, 1>;\n"
                    "#endif // VALUE_CPP_EMITTER_HELPERS_DEFINED\n"
                    "\n"
                    "namespace {\n";
    }

    CppEmitterContext::~CppEmitterContext()
    {
        Out() << "\n// Cleaning up CppEmitterContext" << std::endl;

        _outputStream.get() << _globalStream.str()
                            << "} // namespace \n"
                            << _fnDeclStream.str() << "\n"
                            << _expressionStream.str() << std::endl;

        _outputStream.get().flush();
    }

    namespace
    {
        std::string ValueTypeToCTypeString(ValueType type)
        {
            switch (type)
            {
            case ValueType::Void:
                return "void";
            case ValueType::Boolean:
                return "bool";
            case ValueType::Byte:
                return "uint8_t";
            case ValueType::Char8:
                return "int8_t";
            case ValueType::Double:
                return "double";
            case ValueType::Float:
                return "float";
            case ValueType::Int16:
                return "int16_t";
            case ValueType::Int32:
                return "int32_t";
            case ValueType::Int64:
                return "int64_t";
            default:
                throw LogicException(LogicExceptionErrors::illegalState);
            }
        }

        std::string ValueTypeToCTypeString(detail::ValueTypeDescription desc, size_t size, bool forcePointer = false)
        {
            std::string str;
            if (!forcePointer && desc.second == 1)
            {
                if (size == 1)
                {
                    str = "Scalar<" + ValueTypeToCTypeString(desc.first) + ">";
                }
                else
                {
                    str = "std::array<" + ValueTypeToCTypeString(desc.first) + ", " + std::to_string(size) + ">";
                }
            }
            else
            {
                str = ValueTypeToCTypeString(desc.first);
                str.insert(str.end(), desc.second, '*');
            }
            return str;
        }

        std::string ValueToCString(const Value& value, bool forcePointer = false)
        {
            size_t size{};
            if (!value.IsConstrained())
            {
                if (!forcePointer)
                {
                    throw LogicException(LogicExceptionErrors::illegalState, "Can't create concrete allocation for value with no known layout");
                }
            }
            else
            {
                size = value.GetLayout().GetMemorySize();
            }
            auto str = ValueTypeToCTypeString(value.GetType(), size, forcePointer);
            return str;
        }

        template <typename T>
        std::string TypeToCTypeString(T)
        {

#define BEGIN_TYPE_TO_CTYPE_STRING_MAP \
    if constexpr (false)               \
    {                                  \
    }
#define ADD_TYPE_TO_CTYPE_STRING_STRING(TYPE, STR) \
    else if constexpr (std::is_same_v<TYPE, T>)    \
    {                                              \
        return #STR;                               \
    }
#define ADD_TYPE_TO_CTYPE_STRING(TYPE) ADD_TYPE_TO_CTYPE_STRING_STRING(TYPE, TYPE)
#define END_TYPE_TO_CTYPE_STRING_MAP                              \
    else                                                          \
    {                                                             \
        static_assert(utilities::FalseType<T>{}, "Unknown type"); \
    }

            BEGIN_TYPE_TO_CTYPE_STRING_MAP
            ADD_TYPE_TO_CTYPE_STRING(bool)
            ADD_TYPE_TO_CTYPE_STRING_STRING(char, int8_t)
            ADD_TYPE_TO_CTYPE_STRING(uint8_t)
            ADD_TYPE_TO_CTYPE_STRING(int16_t)
            ADD_TYPE_TO_CTYPE_STRING(int32_t)
            ADD_TYPE_TO_CTYPE_STRING(int64_t)
            ADD_TYPE_TO_CTYPE_STRING(float)
            ADD_TYPE_TO_CTYPE_STRING(double)
            ADD_TYPE_TO_CTYPE_STRING(void)
            END_TYPE_TO_CTYPE_STRING_MAP
        }

        template <typename StreamType, typename T, typename AllocatorType = std::allocator<T>>
        void PrintVector(StreamType&& stream, const std::vector<T, AllocatorType>& v, const std::string& delim = ", ")
        {
            using RealT = std::conditional_t<sizeof(T) == 1, int, T>;
            if (!v.empty())
            {
                std::copy(v.begin(), v.end() - 1, std::ostream_iterator<RealT>{ stream, delim.c_str() });
                stream << static_cast<RealT>(v.back());
            }
        }
    } // namespace

    Value CppEmitterContext::AllocateImpl(ValueType type, MemoryLayout layout, size_t alignment, AllocateFlags flags)
    {
        if (alignment != 0 || flags != AllocateFlags::None)
        {
            throw LogicException(LogicExceptionErrors::notImplemented);
        }

        // TODO: add alignment directive
        return AllocateImpl({ type, 1 }, layout, "{}; // " + layout.ToString() + "\n");
    }

    Value CppEmitterContext::AllocateImpl(
        detail::ValueTypeDescription type,
        std::optional<MemoryLayout> layout,
        std::string initializationString,
        std::optional<std::string> name,
        bool forcePointer)
    {
        CppEmitterContext::ValueImpl data{ name.value_or(UniqueName("v")), type };

        auto& dataList = _fnStacks.top().dataList;

        dataList.push_front(std::move(data));
        auto& front = dataList.front();

        Emittable emittable{ &front };

        Value value(emittable, layout);

        Out() << ValueToCString(value, forcePointer) << " " << front.name << initializationString;

        return value;
    }

    std::optional<Value> CppEmitterContext::GetGlobalValue(GlobalAllocationScope scope, std::string name)
    {
        std::string adjustedName = GetScopeAdjustedName(scope, name);
        if (auto it = _globals.find(adjustedName); it != _globals.end())
        {
            return Value(it->second.first, it->second.second);
        }

        return std::nullopt;
    }

    Value CppEmitterContext::GlobalAllocateImpl(GlobalAllocationScope scope, std::string name, ConstantData data, MemoryLayout layout, AllocateFlags flags)
    {
        std::string adjustedName = GetScopeAdjustedName(scope, name);

        auto it = _globals.find(adjustedName);
        if (it != _globals.end())
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Unexpected collision in global data allocation");
        }

        auto [type, definitionString] = std::visit(
            [](auto&& vectorData) -> std::pair<ValueType, std::string> {
                using Type = std::decay_t<decltype(vectorData)>;
                using VectorElementType = typename Type::value_type;
                using ElementType = std::conditional_t<std::is_same_v<VectorElementType, utilities::Boolean>, bool, VectorElementType>;

                std::stringstream sstr;

                sstr << " = { ";
                PrintVector(sstr, vectorData);
                sstr << " };\n";

                return { GetValueType<ElementType>(), sstr.str() };
            },
            data);

        CppEmitterContext::ValueImpl valueDesc{ adjustedName, { type, 1 } };

        _globalsList.push_front(std::move(valueDesc));
        auto& front = _globalsList.front();

        Emittable emittable{ &front };
        _globals.insert(it, { adjustedName, { emittable, layout } });

        Value value(emittable, layout);

        std::string prefix = [flags] {
            switch (flags)
            {
            case AllocateFlags::ThreadLocal:
                return "thread_local ";
            default:
                return "";
            }
        }();

        Global() << prefix << ValueToCString(value) << " " << adjustedName << definitionString;

        return value;
    }

    Value CppEmitterContext::GlobalAllocateImpl(GlobalAllocationScope scope, std::string name, ValueType type, MemoryLayout layout, AllocateFlags flags)
    {
        std::string adjustedName = GetScopeAdjustedName(scope, name);

        auto it = _globals.find(adjustedName);
        if (it != _globals.end())
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Unexpected collision in global data allocation");
        }

        CppEmitterContext::ValueImpl valueDesc{ adjustedName, { type, 1 } };

        _globalsList.push_front(std::move(valueDesc));
        auto& front = _globalsList.front();

        Emittable emittable{ &front };
        _globals.insert(it, { adjustedName, { emittable, layout } });

        Value value(emittable, layout);

        std::string prefix = [flags] {
            switch (flags)
            {
            case AllocateFlags::ThreadLocal:
                return "thread_local ";
            default:
                return "";
            }
        }();

        Global() << prefix << ValueToCString(value) << " " << adjustedName << "{}; // " << layout << "\n";

        return value;
    }

    detail::ValueTypeDescription CppEmitterContext::GetTypeImpl(Emittable emittable)
    {
        return emittable.GetDataAs<CppEmitterContext::ValueImpl*>()->typeDesc;
    }

    void CppEmitterContext::DeclareFunction(FunctionDeclaration decl)
    {
        auto [it, inserted] = _declaredFunctions.emplace(decl.GetFunctionName());
        if (!inserted)
        {
            // already declared
            return;
        }

        WriteFunctionSignature(FnDecl(), decl) << ";\n";
    }

    std::ostream& CppEmitterContext::WriteFunctionSignature(std::ostream& stream, FunctionDeclaration decl)
    {
        const auto& argValues = decl.GetParameterTypes();
        const auto& returnValue = decl.GetReturnType();
        const auto& fnName = decl.GetFunctionName();
        const auto isPublic = decl.IsPublic();

        std::vector<std::string> functionArgs;
        functionArgs.reserve(argValues.size());
        for (auto index = 0u; index < argValues.size(); ++index)
        {
            auto& arg = argValues[index];
            functionArgs.push_back(
                ValueTypeToCTypeString(
                    arg.GetType(),
                    arg.IsConstrained() ? arg.GetLayout().GetMemorySize() : 0,
                    true) +
                " arg_" + std::to_string(index) + "/* " + (arg.IsConstrained() ? arg.GetLayout().ToString() : std::string{ "unconstrained" }) + " */");
        }

        std::string returnType = returnValue ? ValueToCString(*returnValue) : ValueTypeToCTypeString(ValueType::Void);

        stream << (isPublic ? "" : "static ") << returnType << " " << fnName << "(";
        PrintVector(stream, functionArgs);
        stream << ")";

        return stream;
    }

    EmitterContext::DefinedFunction
    CppEmitterContext::CreateFunctionImpl(FunctionDeclaration decl, DefinedFunction fn)
    {
        if (const auto& intrinsics = GetIntrinsics();
            std::find(intrinsics.begin(), intrinsics.end(), decl) != intrinsics.end())
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Specified function is an intrinsic");
        }

        if (auto it = _definedFunctions.find(decl); it != _definedFunctions.end())
        {
            return it->second;
        }

        DeclareFunction(decl);

        const auto& argValues = decl.GetParameterTypes();
        const auto& fnName = decl.GetFunctionName();

        {
            FunctionScope scope(*this, fnName);

            // create function sig
            WriteFunctionSignature(Out(), decl) << " {\n";

            auto params = argValues;
            for (auto index = 0u; index < params.size(); ++index)
            {
                auto& param = params[index];
                CppEmitterContext::ValueImpl data{ "arg_" + std::to_string(index), param.GetType() };

                auto& dataList = _fnStacks.top().dataList;

                dataList.push_front(std::move(data));
                auto& front = dataList.front();

                Emittable emittable{ &front };

                param.SetData(emittable);
            }

            Indented([&] {
                auto fnReturnValue = fn(params);
                if (fnReturnValue)
                {
                    auto emittableReturn = EnsureEmittable(*fnReturnValue);
                    Out() << "return " << emittableReturn.GetName() << ";\n";
                }
            });

            Out() << "} \n"
                  << std::endl;
        }

        DefinedFunction returnFn = [this, decl](std::vector<Value> args) -> std::optional<Value> {
            const auto& argValues = decl.GetParameterTypes();
            const auto& returnValue = decl.GetReturnType();
            const auto& fnName = decl.GetFunctionName();

            if (!std::equal(args.begin(),
                            args.end(),
                            argValues.begin(),
                            argValues.end(),
                            [](Value suppliedValue, Value fnValue) {
                                return suppliedValue.GetBaseType() == fnValue.GetBaseType() &&
                                       (suppliedValue.PointerLevel() == fnValue.PointerLevel() ||
                                        suppliedValue.PointerLevel() == fnValue.PointerLevel() + 1);
                            }))
            {
                throw InputException(InputExceptionErrors::invalidArgument);
            }

            std::vector<std::string> params;
            for (auto index = 0u; index < args.size(); ++index)
            {
                auto arg = EnsureEmittable(args[index]);
                auto& expected = argValues[index];

                std::string param = "&" + ScalarToString(arg);
                if (arg.PointerLevel() == expected.PointerLevel() + 1)
                {
                    param = "*(" + param + ")";
                }
                params.push_back(param);
            }

            std::stringstream funcCallStream;
            funcCallStream << fnName << "(";
            PrintVector(funcCallStream, params);
            funcCallStream << ")";
            auto funCallString = funcCallStream.str();

            std::optional<Value> fnReturnValue;
            if (returnValue)
            {
                auto typeDesc = returnValue->GetType();
                auto layout = returnValue->IsConstrained() ? std::optional{ returnValue->GetLayout() } : std::optional<MemoryLayout>{};
                bool originalScalar = false;
                if (typeDesc.second == 0)
                {
                    typeDesc.second = 1;
                    originalScalar = true;
                }
                std::string initStr = std::string{ " = " } +
                                      (originalScalar ? "{ " : "") + funCallString + (originalScalar ? " }" : "") +
                                      "; // " + (layout ? layout->ToString() : std::string{ "unconstrained" }) + "\n\n";

                fnReturnValue = AllocateImpl(typeDesc, originalScalar ? ScalarLayout : layout, initStr);
            }
            else
            {
                Out() << funCallString << ";\n\n";
            }

            return fnReturnValue;
        };

        _definedFunctions[decl] = returnFn;

        return returnFn;
    }

    bool CppEmitterContext::IsFunctionDefinedImpl(FunctionDeclaration decl) const
    {
        if (const auto& intrinsics = GetIntrinsics();
            std::find(intrinsics.begin(), intrinsics.end(), decl) != intrinsics.end())
        {
            return true;
        }

        return _definedFunctions.find(decl) != _definedFunctions.end();
    }

    Value CppEmitterContext::StoreConstantDataImpl(ConstantData data)
    {
        return _computeContext.StoreConstantData(data);
    }

    void CppEmitterContext::ForImpl(MemoryLayout layout, std::function<void(std::vector<Scalar>)> fn, const std::string& name)
    {
        struct Range
        {
            Scalar start;
            Scalar stop;
            Scalar step;
        };

        using LooperFn = std::function<void(std::vector<Range>, std::vector<Scalar>, std::function<void(std::vector<Scalar>)>)>;

        const auto& logicalOrder = layout.GetLogicalDimensionOrder();

        LooperFn looper = [this, &looper, &logicalOrder, &name](std::vector<Range> ranges, std::vector<Scalar> indices, std::function<void(std::vector<Scalar>)> bodyFn) {
            if (ranges.empty())
            {
                std::vector<Scalar> logicalIndices(indices.size());
                for (auto index = 0u; index < indices.size(); ++index)
                {
                    logicalIndices[logicalOrder[index]] = indices[index];
                }
                bodyFn(logicalIndices);
            }
            else
            {
                Range range = std::move(ranges.front());
                ranges.erase(ranges.begin());

                ForImpl(
                    range.start, range.stop, range.step, [=, &looper, &bodyFn](Scalar index) mutable {
                        indices.push_back(index);
                        looper(ranges, indices, bodyFn);
                    },
                    name);
            }
        };

        std::vector<Range> ranges;
        ranges.reserve(layout.NumDimensions());
        for (auto index = 0; index < layout.NumDimensions(); ++index)
        {
            ranges.push_back({ Value(0), Value(layout.GetActiveSize(index)), Value(1) });
        }

        looper(ranges, {}, fn);
    }

    void CppEmitterContext::ForImpl(Scalar start, Scalar stop, Scalar step, std::function<void(Scalar)> fn, const std::string& name)
    {
        auto startStr = ScalarToString(start);
        auto index = AllocateImpl({ ValueType::Int32, 1 }, ScalarLayout, "{ " + startStr + " };\n");
        auto indexStr = index.GetName();
        auto stopStr = ScalarToString(stop);
        auto stepStr = ScalarToString(step);
        std::string optionalTag;
        if (!name.empty())
        {
            optionalTag = " // " + UniqueName(name + " loop");
        }

        Out() << "for (;" << indexStr << "[0] < " << stopStr
              << "; " << indexStr << "[0] += " << stepStr << ") {" << optionalTag << "\n";

        Indented([&] { fn(index); });

        Out() << "}" << optionalTag << "\n\n";
    }

    void CppEmitterContext::MoveDataImpl(Value& source, Value& destination)
    {
        // we treat a move the same as a copy, except we clear out the source
        CopyDataImpl(source, destination);

        // data has been "moved", so clear the source
        source.Reset();
    }

    void CppEmitterContext::CopyDataImpl(const Value& source, Value& destination)
    {
        if (destination.IsConstant())
        {
            if (source.IsConstant())
            {
                return _computeContext.CopyData(source, destination);
            }
            else
            {
                destination.SetData(AllocateImpl(source.GetType(), source.GetLayout(), "{ " + source.GetName() + " };\n"));
            }
        }
        else
        {
            if (!source.IsConstant() && source.Get<Emittable>().GetDataAs<void*>() == destination.Get<Emittable>().GetDataAs<void*>())
            {
                return;
            }

            if (auto& layout = source.GetLayout(); layout == destination.GetLayout())
            {
                if (layout == ScalarLayout)
                {
                    Out() << ScalarToString(destination) << " = " << ScalarToString(source) << ";\n";
                }
                else
                {
                    auto realizedSource = EnsureEmittable(source);
                    Out() << "std::copy_n(&" << GetNameImpl(realizedSource) << "[0], "
                          << realizedSource.GetLayout().GetMemorySize() << ", &"
                          << GetNameImpl(destination) << "[0]);\n";
                }
            }
            else
            {
                throw LogicException(LogicExceptionErrors::notImplemented);
            }
        }
    }

    Value CppEmitterContext::ReferenceImpl(Value sourceValue)
    {
        auto source = Realize(sourceValue);
        if (source.IsConstant())
        {
            return _computeContext.Reference(source);
        }

        auto typeDesc = source.GetType();
        ++typeDesc.second;

        auto sourceName = source.GetName();
        auto tempOffsetValue = Offset(sourceValue, 0);

        auto value = AllocateImpl(
            typeDesc,
            sourceValue.IsConstrained() ? sourceValue.GetLayout() : ScalarLayout,
            " = &" + tempOffsetValue.GetName() + ";\n",
            UniqueName(sourceName + "_ref"),
            true);

        if (!sourceValue.IsConstrained())
        {
            value.ClearLayout();
        }

        return value;
    }

    Value CppEmitterContext::DereferenceImpl(Value sourceValue)
    {
        auto source = Realize(sourceValue);
        if (source.IsConstant())
        {
            return _computeContext.Dereference(source);
        }

        auto typeDesc = source.GetType();
        --typeDesc.second;

        auto sourceName = source.GetName();

        auto value = AllocateImpl(
            typeDesc,
            sourceValue.IsConstrained() ? sourceValue.GetLayout() : ScalarLayout,
            " = " + sourceName + "[0];\n",
            UniqueName(sourceName + "_ref"),
            true);

        if (!sourceValue.IsConstrained())
        {
            value.ClearLayout();
        }

        return value;
    }

    Value CppEmitterContext::OffsetImpl(Value source, Value offset)
    {
        if (offset.GetLayout() != ScalarLayout)
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        if (source.IsConstant() && offset.IsConstant())
        {
            return _computeContext.Offset(source, offset);
        }
        else
        {
            auto emittableSource = EnsureEmittable(source);
            auto sourceName = emittableSource.GetName();

            std::string initString = " = &" + sourceName + "[0]";
            if (auto offsetString = ScalarToString(offset); offsetString != "0")
            {
                initString += " + " + offsetString;
            }
            initString += ";\n";

            auto value = AllocateImpl(source.GetType(), std::nullopt, initString, UniqueName(sourceName + "_offset"), true);

            return value;
        }
    }

    Value CppEmitterContext::UnaryOperationImpl(ValueUnaryOperation op, Value destination)
    {
        throw LogicException(LogicExceptionErrors::notImplemented);
    }
    Value CppEmitterContext::BinaryOperationImpl(ValueBinaryOperation op, Value destination, Value source)
    {
        if (destination.IsConstant() && source.IsConstant())
        {
            return _computeContext.BinaryOperation(op, destination, source);
        }

        std::string opStr;
        bool canSelfAssign;
        std::tie(opStr, canSelfAssign) = [op]() -> std::pair<std::string, bool> {
            switch (op)
            {
            case ValueBinaryOperation::add:
                return { " += ", true };
            case ValueBinaryOperation::divide:
                return { " /= ", true };
            case ValueBinaryOperation::modulus:
                return { " %= ", true };
            case ValueBinaryOperation::multiply:
                return { " *= ", true };
            case ValueBinaryOperation::subtract:
                return { " -= ", true };
            case ValueBinaryOperation::logicalAnd:
                return { " && ", false };
            case ValueBinaryOperation::logicalOr:
                return { " || ", false };
            default:
                throw LogicException(LogicExceptionErrors::illegalState);
            }
        }();

        if (destination.IsDefined())
        {
            if (destination.GetLayout() != source.GetLayout())
            {
                throw LogicException(LogicExceptionErrors::illegalState);
            }
            if (destination.GetBaseType() != source.GetBaseType())
            {
                throw LogicException(LogicExceptionErrors::illegalState);
            }
        }
        else
        {
            if (!source.IsConstrained())
            {
                throw LogicException(LogicExceptionErrors::illegalState);
            }
            return source;
        }

        const auto& layout = destination.GetLayout();
        auto destStr = GetNameImpl(destination);
        auto srcStr = GetNameImpl(source);
        if (layout == ScalarLayout)
        {
            auto sourceString = ScalarToString(source);
            switch (op)
            {
            case ValueBinaryOperation::add:
                [[fallthrough]];
            case ValueBinaryOperation::subtract:
                if (sourceString == "0") // destination = destination { +, - } 0
                {
                    return destination;
                }
                break;
            case ValueBinaryOperation::divide:
                [[fallthrough]];
            case ValueBinaryOperation::multiply:
                if (sourceString == "1") // destination = destination { /, * } 1
                {
                    return destination;
                }
                break;
            default:
                break;
            }
            if (canSelfAssign)
            {
                Out() << destStr << "[0]" << opStr << sourceString << ";\n";
            }
            else
            {
                Out() << destStr << "[0] = " << destStr << "[0]" << opStr << sourceString << ";\n";
            }
        }
        else
        {
            auto emittableSource = EnsureEmittable(source);
            srcStr = emittableSource.GetName();

            auto iterationVariable = UniqueName("index");
            Out() << "for (size_t " << iterationVariable << " = 0; " << iterationVariable << " < " << layout.GetMemorySize() << "; " << iterationVariable << " += " << layout.GetCumulativeIncrement(layout.NumDimensions() - 1) << ") {\n";

            Indented([&] {
                if (canSelfAssign)
                {
                    Out() << destStr << "[" << iterationVariable << "]" << opStr << srcStr << "[" << iterationVariable << "];\n";
                }
                else
                {
                    Out() << destStr << "[" << iterationVariable << "] = " << destStr << "[" << iterationVariable << "]" << opStr << srcStr << "[" << iterationVariable << "];\n";
                }
            });

            Out() << "}\n\n";
        }
        return destination;
    }

    Value CppEmitterContext::LogicalOperationImpl(ValueLogicalOperation op, Value source1, Value source2)
    {
        if (source1.IsConstant() && source2.IsConstant())
        {
            return _computeContext.LogicalOperation(op, source1, source2);
        }

        auto opStr = [op]() -> std::string {
            switch (op)
            {
            case ValueLogicalOperation::equality:
                return " == ";
            case ValueLogicalOperation::inequality:
                return " != ";
            case ValueLogicalOperation::greaterthan:
                return " > ";
            case ValueLogicalOperation::greaterthanorequal:
                return " >= ";
            case ValueLogicalOperation::lessthan:
                return " < ";
            case ValueLogicalOperation::lessthanorequal:
                return " <= ";
            default:
                throw LogicException(LogicExceptionErrors::illegalState);
            }
        }();

        if (source1.GetLayout() == source2.GetLayout())
        {
            std::string initString;
            if (source1.GetLayout() == ScalarLayout)
            {
                initString = ScalarToString(source1) + opStr + ScalarToString(source2);
            }
            else
            {
                auto emittableSource1 = EnsureEmittable(source1);
                auto emittableSource2 = EnsureEmittable(source2);

                initString = GetNameImpl(emittableSource1) + opStr + GetNameImpl(emittableSource2);
            }

            return AllocateImpl({ ValueType::Boolean, 1 }, ScalarLayout, "{ " + initString + " };\n");
        }
        else
        {
            throw LogicException(LogicExceptionErrors::notImplemented);
        }
    }

    Value CppEmitterContext::CastImpl(Value value, ValueType type)
    {
        if (value.IsConstant())
        {
            return _computeContext.Cast(value, type);
        }

        if (value.PointerLevel() != 1)
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        if (value.GetBaseType() == type)
        {
            return value;
        }

        if (auto& layout = value.GetLayout(); layout == ScalarLayout)
        {
            std::string initString = "{ static_cast<" + ValueTypeToCTypeString(type) + ">(" + ScalarToString(value) + ") };\n";
            return AllocateImpl({ type, 1 }, layout, initString);
        }
        else
        {
            auto returnValue = Allocate(type, value.GetLayout());
            For(0,
                static_cast<int>(layout.GetMemorySize()),
                1,
                [&](Scalar index) {
                    Out() << returnValue.GetName() << "[" << ScalarToString(index) << "] = static_cast<" << ValueTypeToCTypeString(type)
                          << ">(" << value.GetName() << "[" << ScalarToString(index) << "]);\n";
                });
            return returnValue;
        }
    }

    class CppEmitterContext::IfContextImpl : public EmitterContext::IfContextImpl
    {
    public:
        IfContextImpl(CppEmitterContext& context, Scalar test, std::function<void()> fn) :
            _context(context)
        {
            StreamGuard guard{ *this };
            _context.Out() << "if (" << _context.ScalarToString(test) << ") {\n";
            _context.Indented(fn);
            _context.Out() << "}";
        }

        ~IfContextImpl()
        {
            _sstr << "\n";
            _context._stream.get() << _sstr.str();
        }

        void ElseIf(Scalar test, std::function<void()> fn) override
        {
            StreamGuard guard{ *this };
            _context._stream.get() << " else if (" << _context.ScalarToString(test) << ") {\n";
            _context.Indented(fn);
            _context.Out() << "}";
        }

        void Else(std::function<void()> fn) override
        {
            StreamGuard guard{ *this };
            _context._stream.get() << " else {\n";
            _context.Indented(fn);
            _context.Out() << "}\n";
        }

    private:
        struct StreamGuard
        {
            StreamGuard(IfContextImpl& context) :
                _context(context),
                _oldStream(_context._context._stream)
            {
                _context._context._stream = _context._sstr;
            }

            ~StreamGuard()
            {
                _context._context._stream = _oldStream;
            }

        private:
            IfContextImpl& _context;
            std::reference_wrapper<std::ostream> _oldStream;
        };

        CppEmitterContext& _context;
        std::stringstream _sstr;
    };

    EmitterContext::IfContext CppEmitterContext::IfImpl(Scalar test, std::function<void()> fn)
    {
        return EmitterContext::IfContext{ std::make_unique<CppEmitterContext::IfContextImpl>(*this, test, fn) };
    }

    void CppEmitterContext::WhileImpl(Scalar test, std::function<void()> fn)
    {
        auto testStr = ScalarToString(test);
        std::string optionalTag;
        std::string name;
        if (!name.empty())
        {
            optionalTag = " // " + UniqueName(name + " loop");
        }

        Out() << "while (" << testStr << ") {" << optionalTag << "\n";

        Indented(fn);

        Out() << "}" << optionalTag << "\n\n";
    }

    std::optional<Value> CppEmitterContext::CallImpl(FunctionDeclaration func, std::vector<Value> args)
    {
        if (std::any_of(args.begin(), args.end(), [](const auto& value) { return value.IsEmpty(); }))
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }

        const auto& intrinsics = GetIntrinsics();
        if (std::find(intrinsics.begin(), intrinsics.end(), func) != intrinsics.end())
        {
            return IntrinsicCall(func, args);
        }

        if (auto it = _definedFunctions.find(func); it != _definedFunctions.end())
        {
            return it->second(args);
        }

        return EmitExternalCall(func, args);
    }

    Value CppEmitterContext::SimpleNumericIntrinsic(FunctionDeclaration intrinsic, std::vector<Value> args)
    {
        if (args.size() != 1)
        {
            throw InputException(InputExceptionErrors::invalidSize);
        }

        const auto& value = args[0];
        if (value.GetBaseType() == ValueType::Boolean)
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        auto typeDesc = value.GetType();
        if (typeDesc.first != ValueType::Float)
        {
            typeDesc.first = ValueType::Double;
        }

        auto returnLayout = value.IsConstrained() ? value.GetLayout() : ScalarLayout;
        auto fnName = ToLowercase(intrinsic.GetFunctionName());

        if (returnLayout == ScalarLayout)
        {
            return AllocateImpl(typeDesc, returnLayout, "{ std::" + fnName + "(" + ScalarToString(value) + ") };\n");
        }
        else
        {
            auto result = Allocate(typeDesc.first, returnLayout);
            auto valueStr = ScalarToString(value);
            Out() << "std::transform(&" << valueStr
                  << ", &" << valueStr << " + " << returnLayout.GetMemorySize()
                  << ", &" << ScalarToString(result)
                  << ", [](decltype(" << valueStr << ") x) { return std::" << fnName << "(x); });\n";
            return result;
        }
    }

    Value CppEmitterContext::MaxMinIntrinsic(FunctionDeclaration intrinsic, std::vector<Value> args)
    {
        if (args.size() == 1)
        {
            const auto& value = args[0];
            if (value.GetBaseType() == ValueType::Boolean)
            {
                throw InputException(InputExceptionErrors::typeMismatch);
            }

            std::string fnName;
            if (intrinsic == MaxNumFunctionDeclaration)
            {
                fnName = "std::max_element";
            }
            else if (intrinsic == MinNumFunctionDeclaration)
            {
                fnName = "std::min_element";
            }
            else
            {
                throw LogicException(LogicExceptionErrors::illegalState);
            }

            auto valueStr = ScalarToString(value);
            return AllocateImpl(
                { value.GetBaseType(), 1 },
                ScalarLayout,
                "{ *" + fnName + "(&" + valueStr +
                    ", &" + valueStr + " + " + std::to_string(value.GetLayout().GetMemorySize()) +
                    ") };\n");
        }
        else if (args.size() == 2)
        {
            const auto& value1 = args[0];
            const auto& value2 = args[1];
            if (value1.GetBaseType() != value2.GetBaseType())
            {
                throw InputException(InputExceptionErrors::typeMismatch);
            }

            if (value1.GetBaseType() == ValueType::Boolean)
            {
                throw InputException(InputExceptionErrors::typeMismatch);
            }

            if ((value1.IsConstrained() && value1.GetLayout() != ScalarLayout) ||
                (value2.IsConstrained() && value2.GetLayout() != ScalarLayout))
            {
                throw InputException(InputExceptionErrors::invalidSize);
            }

            std::string fnName;
            if (intrinsic == MaxNumFunctionDeclaration)
            {
                fnName = "std::max";
            }
            else if (intrinsic == MinNumFunctionDeclaration)
            {
                fnName = "std::min";
            }
            else
            {
                throw LogicException(LogicExceptionErrors::illegalState);
            }

            return AllocateImpl(
                { value1.GetBaseType(), 1 },
                ScalarLayout,
                "{ " + fnName + "(" + ScalarToString(value1) +
                    ", " + ScalarToString(value2) + ") };\n");
        }
        else
        {
            throw InputException(InputExceptionErrors::invalidSize);
        }
    }

    Value CppEmitterContext::PowIntrinsic(FunctionDeclaration, std::vector<Value> args)
    {
        if (args.size() != 2)
        {
            throw InputException(InputExceptionErrors::invalidSize);
        }

        const auto& value1 = args[0];
        const auto& value2 = args[1];
        if (value1.GetBaseType() != value2.GetBaseType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        if (value1.GetBaseType() == ValueType::Boolean)
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        if (value2.IsConstrained() && value2.GetLayout() != ScalarLayout)
        {
            throw InputException(InputExceptionErrors::invalidSize);
        }

        auto typeDesc = value1.GetType();
        if (typeDesc.first != ValueType::Float)
        {
            typeDesc.first = ValueType::Double;
        }

        auto returnLayout = value1.IsConstrained() ? value1.GetLayout() : ScalarLayout;
        std::string fnName = "std::pow";

        if (returnLayout == ScalarLayout)
        {
            return AllocateImpl(typeDesc, returnLayout, "{ " + fnName + "(" + ScalarToString(value1) + ", " + ScalarToString(value2) + ") };\n");
        }
        else
        {
            auto result = Allocate(typeDesc.first, returnLayout);
            auto valueStr = ScalarToString(value1);
            Out() << "std::transform(&" << valueStr
                  << ", &" << valueStr << " + " << returnLayout.GetMemorySize()
                  << ", &" << ScalarToString(result)
                  << ", [&" << value2.GetName() << "](decltype(" << valueStr
                  << ") x) { return " << fnName << "(x, " << ScalarToString(value2) << "); });\n";
            return result;
        }
    }

    Value CppEmitterContext::CopySignIntrinsic(FunctionDeclaration, std::vector<Value> args)
    {
        if (args.size() != 2)
        {
            throw InputException(InputExceptionErrors::invalidSize);
        }

        const auto& value1 = args[0];
        const auto& value2 = args[1];
        if (value1.GetBaseType() != value2.GetBaseType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        if (value1.GetBaseType() == ValueType::Boolean)
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        if ((value1.IsConstrained() && value1.GetLayout() != ScalarLayout) ||
            (value2.IsConstrained() && value2.GetLayout() != ScalarLayout))
        {
            throw InputException(InputExceptionErrors::invalidSize);
        }

        auto typeDesc = value1.GetType();
        if (typeDesc.first != ValueType::Float)
        {
            typeDesc.first = ValueType::Double;
        }

        return AllocateImpl(typeDesc, ScalarLayout, "{ std::copysign(" + ScalarToString(value1) + ", " + ScalarToString(value2) + ") };\n");
    }

    Value CppEmitterContext::FmaIntrinsic(FunctionDeclaration, std::vector<Value> args)
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
        if (value1.GetBaseType() != value2.GetBaseType() || value1.GetBaseType() != value3.GetBaseType())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        if (value1.GetBaseType() == ValueType::Boolean)
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        auto typeDesc = value1.GetType();

        return AllocateImpl(
            typeDesc,
            ScalarLayout,
            "{ static_cast<" + ValueTypeToCTypeString(typeDesc.first) + ">(std::fma(" + ScalarToString(value1) + ", " + ScalarToString(value2) + ", " + ScalarToString(value3) + ")) };\n");
    }

    Value CppEmitterContext::MemFnIntrinsic(FunctionDeclaration intrinsic, std::vector<Value> args)
    {
        if (args.size() != 3)
        {
            throw InputException(InputExceptionErrors::invalidSize);
        }

        const auto& value1 = args[0];
        const auto& value2 = args[1];
        const auto& value3 = args[2];

        if (!value3.IsConstrained() || value3.GetLayout() != ScalarLayout)
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }

        std::string secondValuePrefix;
        if (intrinsic == MemSetFunctionDeclaration)
        {
            assert((value2.IsConstrained() && value2.GetLayout() == ScalarLayout && value2.GetType() == std::pair{ ValueType::Char8, 1 }));
        }
        else
        {
            secondValuePrefix = "&";
        }

        auto fnName = ToLowercase(intrinsic.GetFunctionName());
        auto value1Str = ScalarToString(value1);
        Out() << "std::" << fnName << "(&" << value1Str << ", "
              << secondValuePrefix << ScalarToString(value2)
              << ", sizeof(" << value1Str << ") * " << ScalarToString(value3) << ");\n";

        return {}; // unused
    }

    Value CppEmitterContext::IntrinsicCall(FunctionDeclaration intrinsic, std::vector<Value> args)
    {
        if (std::all_of(args.begin(), args.end(), [](const auto& value) { return value.IsConstant(); }))
        {
            // Compute context can handle intrinsic calls with constant data
            return *_computeContext.Call(intrinsic, std::vector<ViewAdapter>(args.begin(), args.end()));
        }

        static std::unordered_map<FunctionDeclaration, Value (CppEmitterContext::*)(FunctionDeclaration, std::vector<Value>)> intrinsics{
            { AbsFunctionDeclaration, &CppEmitterContext::SimpleNumericIntrinsic },
            { CosFunctionDeclaration, &CppEmitterContext::SimpleNumericIntrinsic },
            { ExpFunctionDeclaration, &CppEmitterContext::SimpleNumericIntrinsic },
            { LogFunctionDeclaration, &CppEmitterContext::SimpleNumericIntrinsic },
            { Log10FunctionDeclaration, &CppEmitterContext::SimpleNumericIntrinsic },
            { Log2FunctionDeclaration, &CppEmitterContext::SimpleNumericIntrinsic },
            { MaxNumFunctionDeclaration, &CppEmitterContext::MaxMinIntrinsic },
            { MinNumFunctionDeclaration, &CppEmitterContext::MaxMinIntrinsic },
            { PowFunctionDeclaration, &CppEmitterContext::PowIntrinsic },
            { SinFunctionDeclaration, &CppEmitterContext::SimpleNumericIntrinsic },
            { SqrtFunctionDeclaration, &CppEmitterContext::SimpleNumericIntrinsic },
            { TanhFunctionDeclaration, &CppEmitterContext::SimpleNumericIntrinsic },
            { RoundFunctionDeclaration, &CppEmitterContext::SimpleNumericIntrinsic },
            { FloorFunctionDeclaration, &CppEmitterContext::SimpleNumericIntrinsic },
            { CeilFunctionDeclaration, &CppEmitterContext::SimpleNumericIntrinsic },
            { CopySignFunctionDeclaration, &CppEmitterContext::CopySignIntrinsic },
            { FmaFunctionDeclaration, &CppEmitterContext::FmaIntrinsic },
            { MemCopyFunctionDeclaration, &CppEmitterContext::MemFnIntrinsic },
            { MemMoveFunctionDeclaration, &CppEmitterContext::MemFnIntrinsic },
            { MemSetFunctionDeclaration, &CppEmitterContext::MemFnIntrinsic },
        };

        std::vector<Value> emittableArgs;
        emittableArgs.reserve(args.size());
        std::transform(args.begin(), args.end(), std::back_inserter(emittableArgs), [this](const auto& value) { return EnsureEmittable(value); });

        return std::invoke(intrinsics.at(intrinsic), this, intrinsic, emittableArgs);
    }

    std::optional<Value> CppEmitterContext::EmitExternalCall(FunctionDeclaration externalFunc, std::vector<Value> args)
    {
        DeclareFunction(externalFunc);

        const auto& argTypes = externalFunc.GetParameterTypes();

        if (args.size() != argTypes.size())
        {
            throw InputException(InputExceptionErrors::sizeMismatch);
        }
        if (!std::equal(args.begin(),
                        args.end(),
                        argTypes.begin(),
                        argTypes.end(),
                        [](Value suppliedValue, Value fnValue) {
                            return suppliedValue.GetBaseType() == fnValue.GetBaseType() &&
                                   (suppliedValue.PointerLevel() == fnValue.PointerLevel() ||
                                    suppliedValue.PointerLevel() == fnValue.PointerLevel() + 1);
                        }))
        {
            throw InputException(InputExceptionErrors::invalidArgument);
        }

        // explicitly making a copy
        auto returnType = externalFunc.GetReturnType();
        const auto& fnName = externalFunc.GetFunctionName();

        std::vector<std::string> params;
        for (auto index = 0u; index < args.size(); ++index)
        {
            auto arg = EnsureEmittable(args[index]);
            auto& expected = argTypes[index];

            std::string param = "&" + ScalarToString(arg);
            if (arg.PointerLevel() == expected.PointerLevel() + 1)
            {
                param = "*(" + param + ")";
            }
            params.push_back(param);
        }

        std::stringstream funcCallStream;
        funcCallStream << fnName << "(";
        PrintVector(funcCallStream, params);
        funcCallStream << ")";

        std::string returnValueString;
        if (returnType)
        {
            auto typeDesc = returnType->GetType();
            auto layout = returnType->GetLayout();
            returnType.reset();
            bool originalScalar = false;

            if (typeDesc.second == 0)
            {
                typeDesc.second = 1;
                originalScalar = true;
            }
            std::string initStr = std::string{ " = " } +
                                  (originalScalar ? "{ " : "") + funcCallStream.str() + (originalScalar ? " }" : "") +
                                  "; // " + layout.ToString() + "\n\n";

            returnType = AllocateImpl(typeDesc, originalScalar ? ScalarLayout : layout, initStr);
        }
        else
        {
            Out() << funcCallStream.str() << ";\n\n";
        }

        return returnType;
    }

    void CppEmitterContext::PrefetchImpl(Value data, PrefetchType type, PrefetchLocality locality)
    {
        // no-op for now
    }

    void CppEmitterContext::ParallelizeImpl(int numTasks, std::vector<Value> captured, std::function<void(Scalar, std::vector<Value>)> fn)
    {
        auto futuresName = UniqueName("futures");
        Out() << "std::vector<std::future<void>> " << futuresName << ";\n";
        Out() << futuresName << ".reserve(" << numTasks << ");\n";
        std::vector<std::string> capturedParams;
        std::transform(captured.begin(), captured.end(), std::back_inserter(capturedParams), [this](const Value& value) {
            auto emittableValue = EnsureEmittable(value);
            return "&" + emittableValue.GetName();
        });

        ForRange(numTasks, [&](Scalar index) {
            auto& outStream = Out() << futuresName << ".emplace_back(std::async([";
            PrintVector(outStream, capturedParams);
            auto parallelizedIndexName = UniqueName("parallelized_index");
            outStream << "](int " << parallelizedIndexName << ") {\n";
            Indented([&] {
                Scalar parallelizedIndex = AllocateImpl({ ValueType::Int32, 1 }, ScalarLayout, " = { " + parallelizedIndexName + " };\n\n");
                fn(parallelizedIndex, captured);
            });
            Out() << "}, " << ScalarToString(index) << "));\n";
        });

        Out() << "for (auto& " << futuresName << "_temp : " << futuresName << ") {\n";
        Indented([&] {
            Out() << futuresName << "_temp.wait();\n";
        });
        Out() << "}\n\n";
    }

    void CppEmitterContext::DebugBreakImpl()
    {
        // no-op for now
    }

    void CppEmitterContext::DebugDumpImpl(Value value, std::string tag, std::ostream& stream) const
    {
        // no-op for now
    }

    void CppEmitterContext::DebugDumpImpl(FunctionDeclaration fn, std::string tag, std::ostream& stream) const
    {
        // no-op for now
    }

    void CppEmitterContext::DebugPrintImpl(std::string message)
    {
        std::string::iterator it;
        char tempBuffer[10] = {};
        while ((it = std::find_if_not(message.begin(), message.end(), [](auto c) {
                    return static_cast<bool>(::isprint(static_cast<unsigned char>(c)));
                })) != message.end())
        {
            snprintf(tempBuffer, sizeof(tempBuffer), "\\x%02x", static_cast<int>(*it));
            message.replace(it, it + 1, tempBuffer);
        }
        Out() << "std::cout << \"" << message << "\";\n";
    }

    void CppEmitterContext::SetNameImpl(const Value& value, const std::string& name)
    {
        // TODO: fix
        // value.Get<Emittable>().GetDataAs<CppEmitterContext::ValueImpl*>()->name = name;
    }

    std::string CppEmitterContext::GetNameImpl(const Value& value) const
    {
        return value.IsConstant() ? _computeContext.GetName(value) : value.Get<Emittable>().GetDataAs<CppEmitterContext::ValueImpl*>()->name;
    }

    void CppEmitterContext::ImportCodeFileImpl(std::string)
    {
        throw LogicException(LogicExceptionErrors::notImplemented);
    }

    Scalar CppEmitterContext::GetFunctionAddressImpl(const FunctionDeclaration& fn)
    {
        throw LogicException(LogicExceptionErrors::notImplemented);
    }

    std::string CppEmitterContext::GetScopeAdjustedName(GlobalAllocationScope scope, std::string name) const
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

    std::string CppEmitterContext::GetGlobalScopedName(std::string name) const
    {
        return _moduleName + "_" + name;
    }

    std::string CppEmitterContext::GetCurrentFunctionScopedName(std::string name) const
    {
        if (_fnStacks.empty())
        {
            throw LogicException(LogicExceptionErrors::illegalState);
        }

        return GetGlobalScopedName(_fnStacks.top().name + "_" + name);
    }

    std::ostream& CppEmitterContext::Out()
    {
        return _stream.get() << std::string(2 * _indent, ' ');
    }

    std::ostream& CppEmitterContext::Global()
    {
        return _globalStream;
    }

    std::ostream& CppEmitterContext::FnDecl()
    {
        return _fnDeclStream;
    }

    Value CppEmitterContext::PromoteConstantData(Value value)
    {
        assert(value.IsConstant() && value.IsDefined() && !value.IsEmpty());

        const auto& constantData = _computeContext.GetConstantData(value);

        auto [offset, size] = std::visit(
            [&value](auto&& data) -> std::pair<ptrdiff_t, int> {
                using Type = std::decay_t<decltype(data)>;
                using DataType = typename Type::value_type;

                auto ptrData = std::get<DataType*>(value.GetUnderlyingData());
                ptrdiff_t offset = ptrData - data.data();
                return { offset, static_cast<int>(data.size()) };
            },
            constantData);

        auto type = value.GetBaseType();
        auto promotedBaseValue = GlobalAllocateImpl(_fnStacks.empty() ? GlobalAllocationScope::Global : GlobalAllocationScope::Function,
                                                    UniqueName("_"),
                                                    constantData,
                                                    MemoryLayout{ { size } },
                                                    AllocateFlags::None);

        _promotedConstantStack.top().push_back({ &constantData, promotedBaseValue.Get<Emittable>() });

        if (offset == 0)
        {
            promotedBaseValue.SetLayout(value.GetLayout());
            return promotedBaseValue;
        }
        else
        {
            CppEmitterContext::ValueImpl valueDesc{ UniqueName(promotedBaseValue.GetName() + "_offset"), { type, 1 } };

            _globalsList.push_front(std::move(valueDesc));
            auto& front = _globalsList.front();

            Emittable emittable{ &front };
            const auto& layout = value.GetLayout();

            _globals[front.name] = { emittable, layout };

            Value value(emittable, layout);

            Global() << ValueTypeToCTypeString(type) << "* " << front.name
                     << " = &"
                     << promotedBaseValue.GetName() << "[" << offset << "];\n";

            return value;
        }
    }

    std::optional<CppEmitterContext::PromotedConstantDataDescription> CppEmitterContext::HasBeenPromoted(Value value) const
    {
        if (!value.IsDefined() || value.IsEmpty() || !value.IsConstant())
        {
            return std::nullopt;
        }

        const auto& constantData = _computeContext.GetConstantData(value);
        const auto& promotedStack = _promotedConstantStack.top();

        if (auto it = std::find_if(promotedStack.begin(),
                                   promotedStack.end(),
                                   [&constantData](const auto& desc) { return desc.data == &constantData; });
            it != promotedStack.end())
        {
            return *it;
        }
        else
        {
            return std::nullopt;
        }
    }

    Value CppEmitterContext::Realize(Value value)
    {
        if (auto desc = HasBeenPromoted(value); !desc)
        {
            return value;
        }
        else
        {
            const auto& promotionalDesc = *desc;
            auto offset = std::visit(
                [&value](auto&& data) -> ptrdiff_t {
                    using Type = std::decay_t<decltype(data)>;
                    using DataType = typename Type::value_type;

                    auto ptrData = std::get<DataType*>(value.GetUnderlyingData());

                    return ptrData - data.data();
                },
                *promotionalDesc.data);

            Value newValue = value;
            auto emittable = promotionalDesc.realValue;
            if (offset == 0)
            {
                newValue.SetData(emittable);
            }
            else
            {
                auto type = value.GetBaseType();
                auto valueImpl = emittable.GetDataAs<CppEmitterContext::ValueImpl*>();
                CppEmitterContext::ValueImpl valueDesc{ UniqueName(valueImpl->name + "_offset"), { type, 1 } };

                _globalsList.push_front(std::move(valueDesc));
                auto& front = _globalsList.front();

                Emittable offsetEmittable{ &front };
                const auto& layout = value.GetLayout();

                _globals[front.name] = { offsetEmittable, layout };

                newValue.SetData(offsetEmittable);

                Global() << ValueTypeToCTypeString(type) << "* " << front.name
                         << " = &"
                         << valueImpl->name << "[" << offset << "];\n";
            }

            return newValue;
        }
    }

    Value CppEmitterContext::EnsureEmittable(Value value)
    {
        if (!value.IsConstant())
        {
            return value;
        }
        else if (Value newValue = Realize(value); !newValue.IsConstant())
        {
            return newValue;
        }
        else
        {
            return PromoteConstantData(newValue);
        }
    }

    // Despite the name, the function does not actually try to ensure the param passed in is a Scalar
    std::string CppEmitterContext::ScalarToString(ViewAdapter scalar) const
    {
        return std::visit(
            [](auto&& data) {
                using Type = std::decay_t<decltype(data)>;
                if constexpr (std::is_same_v<Type, Emittable>)
                {
                    return data.template GetDataAs<CppEmitterContext::ValueImpl*>()->name + "[0]";
                }
                else
                {
                    using RealType = RemoveAllPointersT<Type>;
                    if constexpr (std::is_same_v<RealType, Boolean>)
                    {
                        return std::to_string(static_cast<bool>(data[0]));
                    }
                    else if constexpr (std::is_floating_point_v<RealType>)
                    {
                        if (std::trunc(data[0]) == data[0])
                        {
                            return std::to_string(static_cast<int64_t>(data[0]));
                        }
                        return std::to_string(data[0]);
                    }
                    else
                    {
                        return std::to_string(data[0]);
                    }
                }
            },
            scalar.GetValue().GetUnderlyingData());
    }

    template <typename Fn>
    void CppEmitterContext::Indented(Fn&& fn)
    {
        ++_indent;
        fn();
        --_indent;
    }

} // namespace value
} // namespace ell
