////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CompilableNode.cpp (model)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CompilableNode.h"
#include "CompilableNodeUtilities.h"
#include "IRMapCompiler.h"
#include "MapCompiler.h"

#include <emitters/include/EmitterException.h>
#include <emitters/include/LLVMUtilities.h>

#include <utilities/include/Logger.h>
#include <utilities/include/UniqueId.h>
#include <utilities/include/UniqueNameList.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>

namespace ell
{
namespace model
{
    using namespace logging;

    void CompilableNode::CompileNode(MapCompiler& compiler)
    {
        auto irCompiler = dynamic_cast<IRMapCompiler*>(&compiler);
        if (irCompiler == nullptr)
        {
            throw emitters::EmitterException(emitters::EmitterError::notSupported, "Unknown compiler type");
        }

        emitters::IRModuleEmitter& moduleEmitter = irCompiler->GetModule();
        auto& enclosingFunction = moduleEmitter.GetCurrentFunction();
        if (ShouldCompileInline() || compiler.GetMapCompilerOptions(*this).inlineNodes)
        {
            Log() << "Inlining node " << DiagnosticString(*this) << " into function " << enclosingFunction.GetFunctionName() << ", currently in block " << enclosingFunction.GetCurrentBlock()->getName().str() << EOL;

            irCompiler->NewNodeRegion(*this);
            auto oldOptions = enclosingFunction.GetCompilerOptions();
            enclosingFunction.SetCompilerOptions(compiler.GetMapCompilerOptions(*this).compilerSettings);
            Compile(*irCompiler, enclosingFunction);
            enclosingFunction.SetCompilerOptions(oldOptions);
            irCompiler->TryMergeNodeRegion(*this);
        }
        else
        {
            Log() << "Not inlining code for node " << DiagnosticString(*this) << EOL;

            // Emit code for function if it doesn't exist yet
            auto functionName = GetCompiledFunctionName();
            if (!moduleEmitter.HasFunction(functionName))
            {
                Log() << "Creating new function for " << DiagnosticString(*this) << EOL;

                compiler.PushScope();
                emitters::NamedVariableTypeList args = GetNodeFunctionParameterList(*irCompiler);

                // TODO: combine precompiled-IR case with use-own-function case
                if (HasPrecompiledIR())
                {
                    Log() << DiagnosticString(*this) << " has precompiled IR" << EOL;
                    auto functionCode = GetPrecompiledIR();
                    moduleEmitter.LoadIR(functionCode);
                }
                else if (HasOwnFunction())
                {
                    Log() << DiagnosticString(*this) << " has its own function" << EOL;
                    auto oldOptions = enclosingFunction.GetCompilerOptions();
                    enclosingFunction.SetCompilerOptions(compiler.GetMapCompilerOptions(*this).compilerSettings);
                    EmitNodeFunction(*irCompiler);
                    enclosingFunction.SetCompilerOptions(oldOptions);
                }
                else
                {
                    auto function = moduleEmitter.BeginFunction(functionName, emitters::VariableType::Void, args);
                    function.SetCompilerOptions(compiler.GetMapCompilerOptions(*this).compilerSettings);
                    function.SetAttributeForArguments(emitters::IRFunctionEmitter::Attributes::NoAlias);

                    irCompiler->NewNodeRegion(*this);
                    Compile(*irCompiler, function);
                    irCompiler->TryMergeNodeRegion(*this);
                    moduleEmitter.EndFunction();
                }
                compiler.PopScope();
            }
            else
            {
                Log() << "Function " << functionName << " already exists for " << DiagnosticString(*this) << EOL;
            }

            // Call function for node
            irCompiler->NewNodeRegion(*this);
            CallNodeFunction(*irCompiler, enclosingFunction);
            irCompiler->TryMergeNodeRegion(*this);
        }
    }

    void CompilableNode::Compile(IRMapCompiler& compiler, emitters::IRFunctionEmitter& function)
    {
        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
    }

    bool CompilableNode::ShouldCompileInline() const
    {
        return false;
    }

    std::string CompilableNode::GetCompiledFunctionName() const
    {
        auto baseName = _nodeFunctionPrefix + GetRuntimeTypeName();

        auto accumSizeString = [](const std::string& prefix, const Port* port) { return prefix + "_" + std::to_string(port->Size()); };
        auto inputPortStr = std::accumulate(GetInputPorts().begin(), GetInputPorts().end(), std::string("_in"), accumSizeString);
        auto outputPortStr = std::accumulate(GetOutputPorts().begin(), GetOutputPorts().end(), std::string("_out"), accumSizeString);
        auto stateId = GetInternalStateIdentifier();
        auto functionName = baseName + inputPortStr + outputPortStr;

        if (stateId != "")
        {
            functionName += std::string("_") + GetInternalStateIdentifier();
        }

        for (auto ch : _badIdentifierChars)
        {
            std::replace(functionName.begin(), functionName.end(), ch, '_');
        }

        return functionName;
    }

    bool CompilableNode::HasOwnFunction() const
    {
        return false;
    }

    void CompilableNode::EmitNodeFunction(IRMapCompiler& compiler)
    {
        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
    }

    bool CompilableNode::HasPrecompiledIR() const
    {
        return false;
    }

    std::string CompilableNode::GetPrecompiledIR() const
    {
        return "";
    }

    std::string CompilableNode::GetInternalStateIdentifier() const
    {
        if (HasState())
        {
            return IdString(*this);
        }
        else
        {
            return "";
        }
    }

    // Get parameters used in the node function's signature
    emitters::NamedVariableTypeList CompilableNode::GetNodeFunctionParameterList(IRMapCompiler& compiler) const
    {
        emitters::IRModuleEmitter& module = compiler.GetModule();
        emitters::NamedVariableTypeList args;
        utilities::UniqueNameList uniqueNameScope;
        for (auto port : GetInputPorts())
        {
            auto varType = PortTypeToVariableType(port->GetType());
            auto ptrType = emitters::GetPointerType(varType);

            auto var = compiler.AllocatePortFunctionArgument(module, port->GetReferencedPort(), emitters::ArgumentFlags::Input, uniqueNameScope);
            auto varName = var->EmittedName();
            args.emplace_back(varName, ptrType);
        };

        // add node state
        auto stateParams = GetNodeFunctionStateParameterList(compiler);
        for (const auto& param : stateParams)
        {
            args.emplace_back(param);
        }

        for (auto port : GetOutputPorts())
        {
            auto varType = PortTypeToVariableType(port->GetType());
            auto ptrType = emitters::GetPointerType(varType);

            auto var = compiler.AllocatePortFunctionArgument(module, *port, emitters::ArgumentFlags::Output, uniqueNameScope);
            auto varName = var->EmittedName();
            args.emplace_back(varName, ptrType);
        };

        return args;
    }

    emitters::NamedVariableTypeList CompilableNode::GetNodeFunctionStateParameterList(IRMapCompiler& compiler) const
    {
        return {};
    }

    // Get the actual arguments use to call the node function
    std::vector<emitters::LLVMValue> CompilableNode::GetNodeFunctionArguments(IRMapCompiler& compiler, emitters::IRFunctionEmitter& currentFunction) const
    {
        int numArgs = GetInputPorts().size() + GetOutputPorts().size();
        std::vector<emitters::LLVMValue> args;
        args.reserve(numArgs);

        for (auto port : GetInputPorts())
        {
            auto inputArg = compiler.EnsurePortEmitted(*port);
            assert(inputArg->getType()->isPointerTy());
            auto inputArgPtr = currentFunction.PointerOffset(inputArg, 0);
            args.push_back(inputArgPtr);
        };

        // add node state
        auto stateArgs = GetNodeFunctionStateArguments(compiler, currentFunction);
        for (const auto& arg : stateArgs)
        {
            args.emplace_back(arg);
        }

        for (auto port : GetOutputPorts())
        {
            auto outputArg = compiler.EnsurePortEmitted(*port);
            auto outputArgPtr = currentFunction.PointerOffset(outputArg, 0);
            args.push_back(outputArgPtr);
        };

        return args;
    }

    std::vector<emitters::LLVMValue> CompilableNode::GetNodeFunctionStateArguments(IRMapCompiler& compiler, emitters::IRFunctionEmitter& currentFunction) const
    {
        return {};
    }

    void CompilableNode::CallNodeFunction(IRMapCompiler& compiler, emitters::IRFunctionEmitter& currentFunction)
    {
        auto functionName = GetCompiledFunctionName();
        auto function = compiler.GetModule().GetFunction(functionName);
        if (function == nullptr)
        {
            throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState, "Error: node function not found.");
        }

        auto args = GetNodeFunctionArguments(compiler, currentFunction);
        currentFunction.Call(function, args);
        Log() << "Emitting call to node function " << functionName << EOL;
    }
} // namespace model
} // namespace ell
