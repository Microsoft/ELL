////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRHeaderWriter.cpp (emitters)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRHeaderWriter.h"
#include "EmitterException.h"
#include "IRMetadata.h"
#include "IRModuleEmitter.h"

#include <llvm/IR/Attributes.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Support/raw_os_ostream.h>

#include <utilities/include/Debug.h>
#include <utilities/include/StringUtil.h>
#include <utilities/include/UniqueNameList.h>

#include <cctype>
#include <sstream>
#include <string>
#include <vector>

namespace ell
{
namespace emitters
{
    namespace
    {
        void WriteStructType(std::ostream& os, llvm::StructType* t)
        {
            if (t->hasName()) // && !t->isLiteral() ?
            {
                std::string typeName = t->getName();
                os << typeName;
            }
        }

        void WriteArrayType(std::ostream& os, llvm::ArrayType* t)
        {
            auto size = t->getNumElements();
            auto elemType = t->getElementType();
            WriteLLVMType(os, elemType);
            os << "[" << size << "]";
        }

        void WritePointerType(std::ostream& os, llvm::PointerType* t)
        {
            auto elemType = t->getElementType();
            WriteLLVMType(os, elemType);
            os << "*";
        }

        void WriteIntegerType(std::ostream& os, llvm::IntegerType* t)
        {
            auto size = t->getBitWidth();
            os << "int" << size << "_t";
        }

        void WriteFunctionType(std::ostream& os, llvm::FunctionType* t)
        {
            auto returnType = t->getReturnType();
            WriteLLVMType(os, returnType);
            os << " (";
            bool first = true;
            for (auto pt : t->params())
            {
                if (!first)
                {
                    os << ", ";
                }
                first = false;
                WriteLLVMType(os, pt);
            }
            os << ");";
        }

        void WriteLLVMVariableDeclaration(std::ostream& os, LLVMType t, std::string name)
        {
            if (t->isArrayTy())
            {
                auto arrType = llvm::cast<llvm::ArrayType>(t);
                auto size = arrType->getNumElements();
                auto elemType = arrType->getTypeAtIndex(0u);
                WriteLLVMType(os, elemType);
                os << " " << name << "[" << size << "]";
            }
            else
            {
                WriteLLVMType(os, t);
                os << " " << name;
            }
        }

        void WriteStructDefinition(std::ostream& os, llvm::StructType* t, const std::vector<std::string>& fieldNames)
        {
            if (t->hasName()) // && !t->isLiteral() ?
            {
                std::string typeName = t->getName();
                DeclareIfDefDefine guard(os, "ELL_" + typeName);
                os << "typedef struct " << typeName << "\n";
                os << "{\n";
                size_t index = 0;
                for (auto& fieldType : t->elements())
                {
                    os << "    ";
                    std::string fieldName = (index >= fieldNames.size()) ? std::string("param") + std::to_string(index) : fieldNames[index];
                    WriteLLVMVariableDeclaration(os, fieldType, fieldName);
                    os << ";\n";
                    ++index;
                }
                os << "} " << typeName << ";\n\n";
            }
        }
    } // namespace

    void WriteLLVMType(std::ostream& os, LLVMType t)
    {
        if (t->isStructTy())
        {
            WriteStructType(os, llvm::cast<llvm::StructType>(t));
        }
        else if (t->isArrayTy())
        {
            WriteArrayType(os, llvm::cast<llvm::ArrayType>(t));
        }
        else if (t->isPointerTy())
        {
            WritePointerType(os, llvm::cast<llvm::PointerType>(t));
        }
        else if (t->isIntegerTy())
        {
            WriteIntegerType(os, llvm::cast<llvm::IntegerType>(t));
        }
        else if (t->isFloatTy())
        {
            os << "float";
        }
        else if (t->isDoubleTy())
        {
            os << "double";
        }
        else if (t->isVoidTy())
        {
            os << "void";
        }
        else if (t->isFunctionTy())
        {
            WriteFunctionType(os, llvm::cast<llvm::FunctionType>(t));
        }
        else
        {
            os << "[[UNKNOWN]]";
            // look up in table
            // ???
        }
    }

    void WriteLLVMType(std::ostream& os, LLVMType t, VariableType vt)
    {
        switch (vt)
        {
        case VariableType::Void:
            os << "void";
            break;
        case VariableType::VoidPointer:
            os << "void*";
            break;
        case VariableType::Char8:
            os << "char";
            break;
        case VariableType::Char8Pointer:
            os << "char*";
            break;
        default:
            WriteLLVMType(os, t);
        }
    }

    void WriteFunctionDeclaration(std::ostream& os, IRModuleEmitter& moduleEmitter, llvm::Function& function)
    {
        auto hasName = function.hasName();
        if (hasName)
        {

            std::string name = function.getName();

            auto functionDeclaration = moduleEmitter.GetFunctionDeclaration(name);

            // Check if we've added comments for this function
            if (functionDeclaration.HasComments())
            {
                const auto& comments = functionDeclaration.GetComments();
                for (const auto& comment : comments)
                {
                    os << "// " << comment << "\n";
                }
            }

            // Now write the function signature
            auto returnType = function.getReturnType();
            WriteLLVMType(os, returnType, functionDeclaration.GetReturnType());
            os << " " << name << "(";
            bool first = true;
            size_t i = 0;

            for (const auto& arg : functionDeclaration.GetArguments())
            {
                if (!first)
                {
                    os << ", ";
                }
                first = false;

                std::string argName = arg.GetName();
                VariableType argType = arg.GetType();
                LLVMType llvmType = arg.GetLLVMType();
                if (llvmType == nullptr)
                {
                    llvmType = moduleEmitter.GetIREmitter().Type(argType);
                }
                WriteLLVMType(os, llvmType, argType);

                if (!argName.empty())
                {
                    os << " " << argName;
                }
                i++;
            }

            os << ");";
        }
    }

    void WriteModuleHeader(std::ostream& os, IRModuleEmitter& moduleEmitter)
    {
        auto pModule = moduleEmitter.GetLLVMModule();

        // Header comment
        std::string moduleName = pModule->getName();
        std::string fileName = moduleEmitter.GetCompilerOptions().modelFile;
        os << "//\n";
        os << "// ELL header for module " << moduleName << "\n";
        if (!fileName.empty())
        {
            os << "// Compiled from input model " << fileName << "\n";
        }
        os << "//\n\n";
        os << "#pragma once\n\n";
        os << "#include <stdint.h>\n\n";

        {
            DeclareExternC externC(os);

            // preprocessor definitions
            auto defines = moduleEmitter.GetPreprocessorDefinitions();
            for (const auto& def : defines)
            {
                DeclareIfDefDefine define(os, def.first, def.second);
            }

            {
                // First write out type definitions
                os << "//\n// Types\n//\n\n";

                // Look for the module-level "declare in header" tag
                if (moduleEmitter.HasMetadata(c_declareTypeInHeaderTagName))
                {
                    auto typeNames = GetSingletonModuleTagValues(moduleEmitter, c_declareTypeInHeaderTagName);

                    auto structTypes = pModule->getIdentifiedStructTypes();
                    for (const auto& t : structTypes)
                    {
                        if (t->hasName() && (typeNames.cend() != typeNames.find(t->getName())))
                        {
                            // Get struct field names
                            auto tagName = GetStructFieldsTagName(t);
                            std::vector<std::string> fieldNames;
                            if (moduleEmitter.HasMetadata(tagName))
                            {
                                auto fieldNameMetadata = moduleEmitter.GetMetadata(tagName);
                                if (!fieldNameMetadata.empty())
                                {
                                    fieldNames = fieldNameMetadata[0];
                                }
                            }
                            WriteStructDefinition(os, t, fieldNames);
                        }
                    }
                }
            }

            os << "//\n// Functions\n//\n\n";
            // Now write out function signatures
            auto tagValues = GetFunctionsWithTag(moduleEmitter, c_declareFunctionInHeaderTagName);
            for (auto& tv : tagValues)
            {
                WriteFunctionDeclaration(os, moduleEmitter, *(tv.function));
                os << "\n\n";
            }
        }
    }

    std::string TrimPrefix(std::string s, std::string prefix)
    {
        if (s.find(prefix) == 0)
        {
            return s.substr(prefix.size());
        }
        return s;
    }

    struct CppWrapperInfo
    {
        std::string moduleName;
        std::string className;
        std::string predictFunctionName;
        std::string predictMethodName;
        std::string predictReturnType;
        std::string predictReturnMember;
        std::vector<std::string> predictMethodArgs;
        std::vector<std::string> predictCallArgs;
        std::stringstream constructorInit;
        std::stringstream predictPreBody;
        std::stringstream predictPostBody;
        std::stringstream memberDecls;
        std::stringstream cdecls;
        std::stringstream helperMethods;
        std::stringstream resetMethodBody;
    };

    static void WriteSourceNodeCallbacks(ModuleCallbackDefinitions& moduleCallbacks, CppWrapperInfo& info)
    {
        utilities::UniqueNameList list;
        int sourceIndex = 0;
        for (auto cb : moduleCallbacks.sources)
        {
            std::string inputType = cb.inputType;
            std::string callbackFunction = cb.functionName;
            std::string callbackMethod = TrimPrefix(callbackFunction, info.moduleName + "_");
            std::string argName = cb.argNames[1];

            // setup an internal method for the callback in our Wrapper class and delegate to
            // a virtual method that can be implemented in another language (via SWIG)
            info.helperMethods << "    void Internal_" << callbackMethod << "(" << inputType << "* buffer)\n";
            info.helperMethods << "    {\n";
            info.helperMethods << "        " << callbackMethod << "(_" << argName << ");\n"; // fill the input buffer
            info.helperMethods << "        int32_t size = GetInputSize(" << sourceIndex << ");\n";
            info.helperMethods << "        ::memcpy(buffer, _" << argName << ".data(), size * sizeof(" << inputType << "));\n";
            info.helperMethods << "    }\n\n";
            info.helperMethods << "    virtual void " << callbackMethod << "(std::vector<" << inputType << ">& " << argName << ")\n";
            info.helperMethods << "    {\n";
            info.helperMethods << "        // todo: override this method to fill the request input data.\n";
            info.helperMethods << "    }\n\n";

            info.constructorInit << "        _" << argName << ".resize(GetInputSize(" << sourceIndex << "));\n";

            info.memberDecls << "    std::vector<" << inputType << "> _" << argName << ";\n";

            // Delegate the "C" callback function to the above virtual method on the Wrapper class.
            info.cdecls << "    void " << callbackFunction << "(void* context, " << inputType << "* " << argName << ")\n";
            info.cdecls << "    {\n";
            info.cdecls << "        if (context != nullptr)\n";
            info.cdecls << "        {\n";
            info.cdecls << "            auto predictor = reinterpret_cast<" << info.className << "*>(context);\n";
            info.cdecls << "            predictor->Internal_" << callbackMethod << "(" << argName << ");\n";
            info.cdecls << "        }\n";
            info.cdecls << "    }\n";
            ++sourceIndex;
        }
    }

    static void WriteLagNotificationCallbacks(ModuleCallbackDefinitions& moduleCallbacks, CppWrapperInfo& info)
    {
        for (auto cb : moduleCallbacks.lagNotifications)
        {
            std::string callbackFunction = cb.functionName;
            std::string callbackMethod = TrimPrefix(callbackFunction, info.moduleName + "_");

            // setup a virtual method for the callback in our Wrapper class
            info.helperMethods << "    virtual void " << callbackMethod << "(double lag)\n";
            info.helperMethods << "    {\n";
            info.helperMethods << "        // todo: add your implementation to this method if you care about lag notifications\n";
            info.helperMethods << "        // or override this method and implement it that way\n";
            info.helperMethods << "    }\n\n";
            info.helperMethods << "    double GetStepInterval() const\n";
            info.helperMethods << "    {\n";
            info.helperMethods << "        return " << info.moduleName << "_GetStepInterval();\n";
            info.helperMethods << "    }\n\n";
            info.helperMethods << "    double GetLagThreshold() const\n";
            info.helperMethods << "    {\n";
            info.helperMethods << "        return " << info.moduleName << "_GetLagThreshold();\n";
            info.helperMethods << "    }\n\n";
            info.helperMethods << "    double GetTicksUntilNextInterval(double currentTime) const\n";
            info.helperMethods << "    {\n";
            info.helperMethods << "        return " << info.moduleName << "_GetTicksUntilNextInterval(currentTime);\n";
            info.helperMethods << "    }\n\n";

            // Delegate the "C" callback function to the above virtual method on the Wrapper class.
            info.cdecls << "    void " << callbackFunction << "(void* context, double lag)\n";
            info.cdecls << "    {\n";
            info.cdecls << "        if (context != nullptr)\n";
            info.cdecls << "        {\n";
            info.cdecls << "            auto predictor = reinterpret_cast<" << info.className << "*>(context);\n";
            info.cdecls << "            predictor->" << callbackMethod << "(lag);\n";
            info.cdecls << "        }\n";
            info.cdecls << "    }\n";
        }
    }

    void WriteSinkNotificationCallbacks(ModuleCallbackDefinitions& moduleCallbacks, CppWrapperInfo& info)
    {
        bool hasSourceNodes = !moduleCallbacks.sources.empty();

        if (!moduleCallbacks.sinks.empty())
        {
            // add the GetSinkShape helper methods
            info.helperMethods << "    TensorShape GetSinkShape(int index = 0) const\n";
            info.helperMethods << "    {\n";
            info.helperMethods << "        TensorShape inputShape;\n";
            info.helperMethods << "        " << info.moduleName << "_GetSinkOutputShape(index, &inputShape);\n";
            info.helperMethods << "        return inputShape;\n";
            info.helperMethods << "    }\n\n";
            info.helperMethods << "    int GetSinkOutputSize(int index = 0) const\n";
            info.helperMethods << "    {\n";
            info.helperMethods << "        return " << info.moduleName << "_GetSinkOutputSize(index);\n";
            info.helperMethods << "    }\n\n";

            int sinkIndex = 0;
            for (auto cb : moduleCallbacks.sinks)
            {
                std::string outputType = cb.inputType;
                std::string callbackFunction = cb.functionName;
                std::string callbackMethod = TrimPrefix(callbackFunction, info.moduleName + "_");
                std::string argName = cb.argNames[1];
                std::string memberName = "_" + callbackFunction + "_" + argName;
                memberName[1] = ::tolower(memberName[1]);

                // setup an internal method for the callback in our Wrapper class and delegate to
                // a virtual method that can be implemented in another language (via SWIG)
                info.helperMethods << "    void Internal_" << callbackMethod << "(" << outputType << "* buffer)\n";
                info.helperMethods << "    {\n";
                info.helperMethods << "        int32_t size = GetSinkOutputSize(" << sinkIndex << ");\n";
                info.helperMethods << "        " << memberName << ".assign(buffer, buffer + size);\n";
                info.helperMethods << "        " << callbackMethod << "(" << memberName << ");\n"; // pass it to virtual method
                info.helperMethods << "    }\n\n";
                info.helperMethods << "    virtual void " << callbackMethod << "(std::vector<" << outputType << ">& " << argName << ")\n";
                info.helperMethods << "    {\n";
                info.helperMethods << "        // override this method to get the sink callback data as a vector\n";
                info.helperMethods << "    }\n\n";

                if (hasSourceNodes && sinkIndex == 0)
                {
                    // in the SourceNode case any OutputNodes have to be returned via SinkNode, but we
                    // can expose the first Sink the return value of our Predict method just for convenience,
                    // We just return the value of the member that the Internal sink callback saved, see above.
                    info.predictReturnType = "std::vector<" + outputType + ">&";
                    info.predictReturnMember = memberName;
                }
                info.memberDecls << "    std::vector<" << outputType << "> " << memberName << ";\n";

                // Delegate the "C" callback function to the above virtual method on the Wrapper class.
                info.cdecls << "    void " << callbackFunction << "(void* context, " << outputType << "* " << argName << ")\n";
                info.cdecls << "    {\n";
                info.cdecls << "        if (context != nullptr)\n";
                info.cdecls << "        {\n";
                info.cdecls << "            auto predictor = reinterpret_cast<" << info.className << "*>(context);\n";
                info.cdecls << "            predictor->Internal_" << callbackMethod << "(" << argName << ");\n";
                info.cdecls << "        }\n";
                info.cdecls << "    }\n";
                ++sinkIndex;
            }
        }
    }

    void WriteSimplePredictMethod(LLVMFunction predictFunction, CppWrapperInfo& info, IRModuleEmitter& moduleEmitter)
    {
        size_t index = 0;
        int outputCount = 0;
        FunctionDeclaration funDecl = moduleEmitter.GetFunctionDeclaration(predictFunction->getName());
        auto argDecls = funDecl.GetArguments();

        // then there are no SourceNodes, so we have a regular predict function with direct inputs.
        for (auto arg = predictFunction->arg_begin(), end = predictFunction->arg_end(); arg != end; ++arg)
        {
            ArgumentFlags flags = ArgumentFlags::InOut;
            if (index < argDecls.size())
            {
                flags = argDecls[index++].GetFlags();
            }
            std::string argName = arg->getName();
            if (argName == "context")
            {
                // we really want void* on these puppies, but LLVM won't let us...(which is why the argType is int8_t*,
                // and for our wrapper class, the context will be 'this' so the "C" callbacks can find this object.
                info.predictCallArgs.push_back("this");
            }
            else
            {
                std::stringstream ss;
                WriteLLVMType(ss, arg->getType()->getPointerElementType());
                std::string argType = ss.str();
                bool passArgument = false;
                if ((flags & ArgumentFlags::Output) && outputCount == 0)
                {
                    // first argument is special, it becomes a class member and the return value for this function.
                    std::string memberName = "_" + funDecl.GetFunctionName() + "_" + argName;
                    info.predictCallArgs.push_back(memberName + ".data()");
                    info.predictReturnType = "std::vector<" + argType + ">&";
                    info.predictReturnMember = memberName;

                    // the predict output arg is a cached member vector so we only have to allocate it once.
                    info.constructorInit << "        " << memberName << ".resize(GetOutputSize(" << outputCount << "));\n";
                    info.memberDecls << "    std::vector<" << argType << "> " << memberName << ";\n";
                    outputCount++;
                }
                else
                {
                    // inputs are always passed in and passed through to C predict function
                    info.predictCallArgs.push_back(argName + ".data()"); // convert vector to raw buffer.
                    passArgument = true;
                }
                if (passArgument)
                {
                    info.predictMethodArgs.push_back("std::vector<" + argType + ">& " + argName);
                }
            }
        }
    }

    void WritePredictMethod(ModuleCallbackDefinitions& moduleCallbacks, CppWrapperInfo& info)
    {
        bool hasSourceNodes = !moduleCallbacks.sources.empty();
        // Predict helper method has two different forms dependsing on whether we have SourceNodes or not.
        if (hasSourceNodes)
        {
            // This version of predict takes context, time, and output, but output is null because we assume
            // we will also have a SinkNode in this case.
            info.helperMethods << "    " << info.predictReturnType << " " << info.predictMethodName << "(" << utilities::Join(info.predictMethodArgs, ", ") << ")\n";
            info.helperMethods << "    {\n";
            info.helperMethods << info.predictPreBody.str();
            info.helperMethods << "        double time = _timer.GetMilliseconds();\n";
            info.helperMethods << "        " << info.predictFunctionName << "(this, &time, nullptr);\n";
            info.helperMethods << info.predictPostBody.str();
            if (info.predictReturnType != "void")
            {
                info.helperMethods << "        return " << info.predictReturnMember << ";\n";
            }
            info.helperMethods << "    }\n\n";

            info.memberDecls << "    HighResolutionTimer _timer;\n";
            info.resetMethodBody << "        _timer.Reset();\n";
        }
        else
        {
            // This is the easy version to use where inputs are passed in directly, and first output is returned as a
            // return value.  The second and subsequent outputs are 'out' arguments after all the inputs.
            info.helperMethods << "    " << info.predictReturnType << " " << info.predictMethodName << "(" << utilities::Join(info.predictMethodArgs, ", ") << ")\n";
            info.helperMethods << "    {\n";
            info.helperMethods << "        " << info.predictFunctionName << "(" << utilities::Join(info.predictCallArgs, ", ") << ");\n";
            if (info.predictReturnType != "void")
            {
                info.helperMethods << "        return " << info.predictReturnMember << ";\n";
            }
            info.helperMethods << "    }\n\n";
        }
    }

    void WriteModuleCppWrapper(std::ostream& os, IRModuleEmitter& moduleEmitter)
    {
        auto callbacks = GetFunctionsWithTag(moduleEmitter, c_callbackFunctionTagName);

        auto predictFunctions = GetFunctionsWithTag(moduleEmitter, c_predictFunctionTagName);
        if (predictFunctions.empty())
        {
            return;
        }

        std::string moduleName = moduleEmitter.GetLLVMModule()->getName();
        std::string className = moduleName + "Wrapper";
        className[0] = ::toupper(className[0]); // pascal case

        LLVMFunction predictFunction = predictFunctions[0].function;

        // the CppPredictWrapper.in is just the first part, now do the optional call backs functions
        // which is variable sized.
        ModuleCallbackDefinitions moduleCallbacks(callbacks);

        // Build up some variable sized content for the ModelWrapperr C++ helper class.  It is variable sized
        // because we have optional call backs and variable number of callbacks, and variable number of
        // parameters for predict methods.
        CppWrapperInfo info;
        info.moduleName = moduleName;
        info.className = className;
        info.predictReturnType = "void";
        info.predictFunctionName = predictFunction->getName();
        info.predictMethodName = TrimPrefix(info.predictFunctionName, moduleName + "_");
        info.predictMethodName[0] = ::toupper(info.predictMethodName[0]); // pascal case

        bool hasSourceNodes = !moduleCallbacks.sources.empty();

        if (!hasSourceNodes)
        {
            WriteSimplePredictMethod(predictFunction, info, moduleEmitter);
        }
        else
        {
            WriteSourceNodeCallbacks(moduleCallbacks, info);
            WriteLagNotificationCallbacks(moduleCallbacks, info);
        }

        WriteSinkNotificationCallbacks(moduleCallbacks, info);

        WritePredictMethod(moduleCallbacks, info);

        // now write out the final completed code.

        // (Note: newlines are part of the syntax for #include)
        // clang-format off
        std::string predictWrapperCode(
            #include "CppPredictWrapper.in"
        );
        // clang-format on

        // fill in the tempate parameters
        ReplaceDelimiter(predictWrapperCode, "MODULE", moduleName);
        ReplaceDelimiter(predictWrapperCode, "CLASSNAME", className);
        ReplaceDelimiter(predictWrapperCode, "CONSTRUCTOR_IMPL", info.constructorInit.str());
        ReplaceDelimiter(predictWrapperCode, "CLASS_GUARD", utilities::ToUppercase(moduleName) + "_WRAPPER_DEFINED");
        ReplaceDelimiter(predictWrapperCode, "MEMBER_DECLS", info.memberDecls.str());
        ReplaceDelimiter(predictWrapperCode, "HELPER_METHODS", info.helperMethods.str());
        ReplaceDelimiter(predictWrapperCode, "CDECLS_GUARD", utilities::ToUppercase(className) + "_CDECLS");
        ReplaceDelimiter(predictWrapperCode, "CDECLS_IMPL", info.cdecls.str());
        ReplaceDelimiter(predictWrapperCode, "STEPPABLE", hasSourceNodes ? "true" : "false");
        ReplaceDelimiter(predictWrapperCode, "RESET_BODY", info.resetMethodBody.str());

        os << predictWrapperCode;
    }

    void ReplaceDelimiter(std::string& text, const std::string& delimiter, const std::string& replacement)
    {
        utilities::ReplaceAll(text, "@@" + delimiter + "@@", replacement);
    }

    //
    // DeclareExternC
    //
    DeclareExternC::DeclareExternC(std::ostream& os) :
        _os(&os)
    {
        DeclareIfDefGuard guard(os, "__cplusplus", DeclareIfDefGuard::Type::Positive);
        *_os << "extern \"C\"\n{\n";
    }

    DeclareExternC::~DeclareExternC()
    {
        DeclareIfDefGuard guard(*_os, "__cplusplus", DeclareIfDefGuard::Type::Positive);
        *_os << "} // extern \"C\"\n";
    }

    DeclareIfDefGuard::DeclareIfDefGuard(std::ostream& os, std::string symbol, Type type) :
        _os(os),
        _symbol(std::move(symbol)),
        _type(type)
    {
        _os << "#if " << (_type == Type::Negative ? "!" : "") << "defined(" << _symbol << ")\n";
    }

    DeclareIfDefGuard::~DeclareIfDefGuard()
    {
        _os << "#endif // " << (_type == Type::Negative ? "!" : "") << "defined(" << _symbol << ")\n\n";
    }

    DeclareIfDefDefine::DeclareIfDefDefine(std::ostream& os, std::string symbol, const std::string& value /* = "" */) :
        DeclareIfDefGuard(os, std::move(symbol), Type::Negative)
    {
        _os << "#define " << _symbol << (value.empty() ? "" : " ") << value << "\n\n";
    }

    //
    // ModuleCallbackDefinitions
    //
    ModuleCallbackDefinitions::ModuleCallbackDefinitions(const std::vector<FunctionTagValues>& callbacks)
    {
        for (const auto& c : callbacks)
        {
            if (!c.values.empty())
            {
                auto nodeType = c.values[0];
                if (nodeType == "SourceNode")
                {
                    sources.push_back(CallbackSignature(*c.function));
                }
                else if (nodeType == "SinkNode")
                {
                    sinks.push_back(CallbackSignature(*c.function));
                }
                else if (nodeType == "ClockNode")
                {
                    lagNotifications.push_back(CallbackSignature(*c.function));
                }
            }
        }

        // Eventually we'd support multiple sources and sinks.
        // For now, assert that we're only going to look at the first ones of each.
        DEBUG_THROW(sources.size() > 1, EmitterException(EmitterError::badFunctionDefinition, "Only one source callback function will be generated"));
        DEBUG_THROW(sinks.size() > 1, EmitterException(EmitterError::badFunctionDefinition, "Only one sink callback function will be generated"));
        DEBUG_THROW(lagNotifications.size() > 1, EmitterException(EmitterError::badFunctionDefinition, "Only one lag callback function will be generated"));
    }

    ModuleCallbackDefinitions::CallbackSignature::CallbackSignature(llvm::Function& function) :
        functionName(function.getName())
    {
        // Parameters
        // Callbacks have one input parameter and a return (which can be void)
        {
            std::ostringstream os;
            auto ptr = function.args().begin();
            argNames.push_back(ptr->getName());
            ptr++; // first argument is now a void* context, second argument has the type info we need.
            argNames.push_back(ptr->getName());
            auto& argument = *ptr;
            auto type = argument.getType();
            if (type->isPointerTy())
            {
                inputIsScalar = false;
                WriteLLVMType(os, argument.getType()->getPointerElementType());
            }
            else
            {
                inputIsScalar = true;
                WriteLLVMType(os, argument.getType());
            }
            inputType = os.str();
        }

        // Return type
        {
            std::ostringstream os;
            WriteLLVMType(os, function.getReturnType());
            returnType = os.str();
        }

        // Wrapper class name
        className = inputType + "CallbackBase";
        className[0] = ::toupper(className[0]); // pascal case
    }
} // namespace emitters
} // namespace ell
