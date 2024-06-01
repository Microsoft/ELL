////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CompilableCodeNode.cpp (model)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CompilableCodeNode.h"

#include "IRMapCompiler.h"
#include "InputPort.h"

#include <utilities/include/Exception.h>

#include <value/include/ValueType.h>

#include <numeric>

namespace ell
{
namespace model
{
    using namespace utilities;
    using namespace value;

    using PortType = Port::PortType;
    using ValueTypeEnum = value::ValueType;

    namespace
    {
        constexpr auto PortTypeToValueTypeEnum = [](Port::PortType type) -> ValueTypeEnum {
            switch (type)
            {
            case PortType::bigInt:
                return ValueTypeEnum ::Int64;
            case PortType::boolean:
                return ValueTypeEnum::Boolean;
            case PortType::integer:
                return ValueTypeEnum::Int32;
            case PortType::real:
                return ValueTypeEnum::Double;
            case PortType::smallReal:
                return ValueTypeEnum::Float;
            case PortType::categorical:
                [[fallthrough]];
            case PortType::none:
                [[fallthrough]];
            default:
                throw LogicException(LogicExceptionErrors::illegalState);
            }
        };
    }

    CompilableCodeNode::CompilableCodeNode(std::string name, const std::vector<InputPortBase*>& inputs, const std::vector<OutputPortBase*>& outputs) :
        CompilableNode(inputs, outputs),
        _name(name),
        _fn(name + (HasState() ? "Code" + IdString(*this) : "") + "_"),
        _resetFn(_fn.GetFunctionName() + "_Reset")
    {
    }

    std::string CompilableCodeNode::GetRuntimeTypeName() const
    {
        return _name + "CodeNode";
    }

    void CompilableCodeNode::SetFunctionParameters() const
    {
        constexpr auto PortToValue = [](auto* port) -> Value {
            auto type = port->GetType();
            auto layout = port->GetMemoryLayout();

            return Value(PortTypeToValueTypeEnum(type), layout);
        };

        const auto& inputs = GetInputPorts();
        const auto& outputs = GetOutputPorts();

        std::vector<ViewAdapter> parameters;
        parameters.reserve(inputs.size() + outputs.size());
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(parameters), PortToValue);
        std::transform(outputs.begin(), outputs.end(), std::back_inserter(parameters), PortToValue);
        (void)_fn.Parameters(parameters);
    }

    std::string CompilableCodeNode::GetCompiledFunctionName() const
    {
        SetFunctionParameters();

        return _fn.GetFunctionName();
    }

    void CompilableCodeNode::Reset()
    {
        if (_resetFn.IsDefined())
        {
            _resetFn.Call();
        }
    }

    bool CompilableCodeNode::HasOwnFunction() const
    {
        return true;
    }

    void CompilableCodeNode::EmitNodeFunction(IRMapCompiler& compiler)
    {
        if (!_fn.IsDefined())
        {
            Define(_fn);

            DefineReset(_resetFn);
            if (_resetFn.IsDefined())
            {
                auto& resetFn = compiler.GetModule().BeginResetFunction(_resetFn.GetFunctionName() + "_stub");
                resetFn.Call(_resetFn.GetFunctionName());
                compiler.GetModule().EndResetFunction();
            }
        }
    }

    namespace
    {
        template <typename T>
        auto InputPortToValue(InputPortBase* port) -> Value
        {
            auto castedPort = static_cast<InputPort<T>*>(port);
            auto data = castedPort->GetValue();
            if constexpr (std::is_same_v<T, bool>)
            {
                return Value(std::vector<utilities::Boolean>(data.begin(), data.end()), castedPort->GetMemoryLayout());
            }
            else
            {
                return Value(data, castedPort->GetMemoryLayout());
            }
        }

        template <typename T>
        void ValueToOutputPort(Value value, OutputPortBase* port)
        {
            auto castedPort = static_cast<OutputPort<T>*>(port);
            auto dataPtrBegin = value.Get<std::conditional_t<std::is_same_v<T, bool>, utilities::Boolean, T>*>();
            auto dataPtrEnd = dataPtrBegin + castedPort->GetMemoryLayout().GetMemorySize();
            if constexpr (std::is_same_v<T, bool>)
            {
                std::vector<bool> vec(dataPtrBegin, dataPtrEnd);
                castedPort->SetOutput(std::move(vec));
            }
            else
            {
                castedPort->SetOutput(dataPtrBegin, dataPtrEnd);
            }
        }

        Value PortToValue(InputPortBase* port)
        {
            switch (port->GetType())
            {
            case PortType::bigInt:
                return InputPortToValue<int64_t>(port);
            case PortType::boolean:
                return InputPortToValue<bool>(port);
            case PortType::integer:
                return InputPortToValue<int>(port);
            case PortType::real:
                return InputPortToValue<double>(port);
            case PortType::smallReal:
                return InputPortToValue<float>(port);
            case PortType::categorical:
                [[fallthrough]];
            case PortType::none:
                [[fallthrough]];
            default:
                throw LogicException(LogicExceptionErrors::illegalState);
            }
        }

        int ValueToPort(Value value, OutputPortBase* port)
        {
            switch (port->GetType())
            {
            case PortType::bigInt:
                ValueToOutputPort<int64_t>(value, port);
                break;
            case PortType::boolean:
                ValueToOutputPort<bool>(value, port);
                break;
            case PortType::integer:
                ValueToOutputPort<int>(value, port);
                break;
            case PortType::real:
                ValueToOutputPort<double>(value, port);
                break;
            case PortType::smallReal:
                ValueToOutputPort<float>(value, port);
                break;
            case PortType::categorical:
                [[fallthrough]];
            case PortType::none:
                [[fallthrough]];
            default:
                throw LogicException(LogicExceptionErrors::illegalState);
            }

            return 0;
        }

    } // namespace

    void CompilableCodeNode::Compute() const
    {
        SetFunctionParameters();
        if (!_fn.IsDefined())
        {
            const_cast<CompilableCodeNode*>(this)->Define(_fn);

            const_cast<CompilableCodeNode*>(this)->DefineReset(_resetFn);
        }

        const auto& inputs = GetInputPorts();
        const auto& outputs = GetOutputPorts();

        std::vector<ViewAdapter> args;
        args.reserve(inputs.size() + outputs.size());
        std::transform(inputs.begin(), inputs.end(), std::back_inserter(args), PortToValue);

        std::transform(outputs.begin(), outputs.end(), std::back_inserter(args), [](auto* port) {
            return Allocate(PortTypeToValueTypeEnum(port->GetType()), port->GetMemoryLayout());
        });

        _fn.Call(args);

        const auto numInputs = inputs.size();
        // we just need a binary predicate to map over the two vectors, so the accumulator functor is basically a no-op
        (void)std::inner_product(
            args.begin() + numInputs, args.end(), outputs.begin(), 0, [](int, int) { return 0; }, ValueToPort);
    }

} // namespace model
} // namespace ell
