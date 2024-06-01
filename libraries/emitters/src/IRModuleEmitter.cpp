////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRModuleEmitter.cpp (emitters)
//  Authors:  Umesh Madan
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRModuleEmitter.h"
#include "EmitterException.h"
#include "IRAssemblyWriter.h"
#include "IRDiagnosticHandler.h"
#include "IRHeaderWriter.h"
#include "IRLoader.h"
#include "IRMetadata.h"
#include "IRSwigInterfaceWriter.h"
#include "LLVMUtilities.h"
#include "TargetDevice.h"

#include <utilities/include/Files.h>
#include <utilities/include/Logger.h>

#include <llvm/AsmParser/Parser.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetRegistry.h>
#include <llvm/Support/ToolOutputFile.h>
#include <llvm/Support/raw_os_ostream.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>

#include <algorithm>
#include <vector>

namespace ell
{
namespace emitters
{
    using namespace utilities::logging;
    using utilities::logging::Log;
    using namespace ell::utilities;

    //
    // Constructors
    //

    IRModuleEmitter::IRModuleEmitter(const std::string& moduleName, const CompilerOptions& parameters) :
        _llvmContext(std::make_unique<llvm::LLVMContext>()),
        _llvmModule(std::make_unique<llvm::Module>(moduleName, *_llvmContext)),
        _emitter(new IREmitter(*_llvmContext, *_llvmModule)),
        _runtime(new IRRuntime(*this)),
        _threadPool(new IRThreadPool(*this)),
        _profiler(new IRProfiler(*this, parameters.profile))
    {
        InitializeLLVM();

        // Create a diagnostic handler to record if there was an error
        _diagnosticHandler = std::unique_ptr<IRDiagnosticHandler>(new IRDiagnosticHandler(*_llvmContext));

        SetCompilerOptions(parameters);
        if (GetCompilerOptions().includeDiagnosticInfo)
        {
            DeclarePrintf();
        }

        _profiler->Init();
    }

    void IRModuleEmitter::SetCompilerOptions(const CompilerOptions& parameters)
    {
        auto params = parameters;
        CompleteCompilerOptions(params);

        // Call base class implementation
        ModuleEmitter::SetCompilerOptions(params);

        // Set IR-specific parameters
        const auto& targetDevice = GetCompilerOptions().targetDevice;
        if (targetDevice.triple.empty())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Target device triple is empty or could not be determined");
        }
        SetTargetTriple(targetDevice.triple);
        llvm::DataLayout dataLayout(GetCompilerOptions().targetDevice.dataLayout);
        GetLLVMModule()->setDataLayout(dataLayout);
    }

    void IRModuleEmitter::CompleteCompilerOptions(CompilerOptions& parameters)
    {
        CompleteTargetDevice(parameters.targetDevice);
    }

    //
    // Creating functions
    //

    void IRModuleEmitter::BeginMapPredictFunction(const std::string& functionName, FunctionArgumentList& args)
    {
        IRFunctionEmitter& function = BeginFunction(functionName, VariableType::Void, args);
        function.SetAttributeForArguments(IRFunctionEmitter::Attributes::NoAlias);

        // store context variable so the callbacks can find it later.
        auto context = function.GetFunctionArgument("context");
        auto globalContext = GlobalPointer(GetModuleName() + "_context", emitters::VariableType::Byte);
        function.Store(globalContext, context);
    }

    void IRModuleEmitter::EndMapPredictFunction()
    {
        EndFunction();

        // now generate the public reset function that combines all the node level reset functions.
        IRFunctionEmitter& resetFunction = BeginFunction(GetModuleName() + "_Reset", VariableType::Void);
        resetFunction.IncludeInHeader();
        for (auto name : _resetFunctions)
        {
            resetFunction.Call(name);
        }
        EndFunction();
    }

    /// <summary> Begin a new function for resetting a given node. Each node that needs to implement
    /// reset calls this and implements their own reset logic.  The IRModuleEmitter wraps all that
    /// in a master model_Reset function which is exposed in the API. </summary>
    IRFunctionEmitter& IRModuleEmitter::BeginResetFunction(const std::string& functionName)
    {
        IRFunctionEmitter& function = BeginFunction(functionName, VariableType::Void);
        _resetFunctions.push_back(functionName);
        return function;
    }

    IRFunctionEmitter& IRModuleEmitter::BeginFunction(const std::string& functionName, VariableType returnType)
    {
        _functions[functionName] = FunctionDeclaration(functionName, returnType);
        return BeginFunction(functionName, GetIREmitter().Type(returnType));
    }

    IRFunctionEmitter& IRModuleEmitter::BeginFunction(const std::string& functionName, LLVMType returnType)
    {
        return BeginFunction(functionName, returnType, NamedVariableTypeList{});
    }

    IRFunctionEmitter& IRModuleEmitter::BeginFunction(const std::string& functionName, VariableType returnType, const VariableTypeList& args)
    {
        NamedVariableTypeList fake;
        for (auto t : args)
        {
            fake.push_back({ "", t });
        }
        _functions[functionName] = FunctionDeclaration(functionName, returnType, fake);
        return BeginFunction(functionName, GetIREmitter().Type(returnType), GetIREmitter().GetLLVMTypes(args));
    }

    IRFunctionEmitter& IRModuleEmitter::BeginFunction(const std::string& functionName, VariableType returnType, const NamedVariableTypeList& args)
    {
        _functions[functionName] = FunctionDeclaration(functionName, returnType, args);
        return BeginFunction(functionName, GetIREmitter().Type(returnType), args);
    }

    IRFunctionEmitter& IRModuleEmitter::BeginFunction(const std::string& functionName, VariableType returnType, const FunctionArgumentList& args)
    {
        _functions[functionName] = FunctionDeclaration(functionName, returnType, args);
        Log() << "Begin emitting IR for function " << functionName << EOL;
        auto currentPos = GetIREmitter().GetCurrentInsertPoint();
        IRFunctionEmitter newFunction = Function(functionName, GetIREmitter().Type(returnType), args, false);
        _functionStack.emplace(newFunction, currentPos);
        return _functionStack.top().first;
    }

    IRFunctionEmitter& IRModuleEmitter::BeginFunction(const std::string& functionName, LLVMType returnType, const NamedVariableTypeList& args)
    {
        if (_functions.find(functionName) == _functions.end())
        {
            _functions[functionName] = FunctionDeclaration(functionName, VariableType::Custom, args);
        }
        Log() << "Begin emitting IR for function " << functionName << EOL;
        auto currentPos = GetIREmitter().GetCurrentInsertPoint();
        IRFunctionEmitter newFunction = Function(functionName, returnType, args, false);
        _functionStack.emplace(newFunction, currentPos);
        return _functionStack.top().first;
    }

    IRFunctionEmitter& IRModuleEmitter::BeginFunction(const std::string& functionName, LLVMType returnType, const std::vector<LLVMType>& argTypes)
    {
        if (_functions.find(functionName) == _functions.end())
        {
            FunctionArgumentList argInfo;
            for (auto t : argTypes)
            {
                argInfo.push_back({ "", ToVariableType(t), ArgumentFlags::Input, t });
            }
            _functions[functionName] = FunctionDeclaration(functionName, ToVariableType(returnType), argInfo);
        }
        Log() << "Begin emitting IR for function " << functionName << EOL;
        auto currentPos = GetIREmitter().GetCurrentInsertPoint();
        IRFunctionEmitter newFunction = Function(functionName, returnType, argTypes, false);
        _functionStack.emplace(newFunction, currentPos);
        return _functionStack.top().first;
    }

    IRFunctionEmitter& IRModuleEmitter::BeginFunction(const std::string& functionName, LLVMType returnType, const NamedLLVMTypeList& args)
    {
        if (_functions.find(functionName) == _functions.end())
        {
            FunctionArgumentList argInfo;
            for (auto t : args)
            {
                argInfo.push_back({ t.first, ToVariableType(t.second), ArgumentFlags::Input, t.second });
            }
            _functions[functionName] = FunctionDeclaration(functionName, ToVariableType(returnType), argInfo);
        }
        Log() << "Begin emitting IR for function " << functionName << EOL;
        auto currentPos = GetIREmitter().GetCurrentInsertPoint();
        IRFunctionEmitter newFunction = Function(functionName, returnType, args, false);
        _functionStack.emplace(newFunction, currentPos);
        return _functionStack.top().first;
    }

    IRFunctionEmitter IRModuleEmitter::BeginMainFunction()
    {
        return BeginFunction("main", VariableType::Void);
    }

    void IRModuleEmitter::EndFunction()
    {
        EndFunction(nullptr);
    }

    void IRModuleEmitter::EndFunction(LLVMValue pReturn)
    {
        if (_functionStack.empty())
        {
            throw EmitterException(EmitterError::indexOutOfRange);
        }

        auto currentFunctionInfo = _functionStack.top();
        _functionStack.pop();

        auto currentFunction = currentFunctionInfo.first;
        auto previousPos = currentFunctionInfo.second;
        if (currentFunction.GetFunction() != nullptr)
        {
            // only put a return here if this block doesn't already have one
            auto pTerm = currentFunction.GetCurrentBlock()->getTerminator();
            if (pTerm == nullptr)
            {
                Log() << "Function " << currentFunction.GetFunctionName() << " does not have a terminator. " << EOL;
                if (pReturn != nullptr)
                {
                    Log() << "Returning non-void value." << EOL;
                    currentFunction.Return(pReturn);
                }
                else
                {
                    Log() << "Returning void." << EOL;
                    currentFunction.Return();
                }
            }
            else
            {
                Log() << "Function " << currentFunction.GetFunctionName() << " already has a terminator" << EOL;
            }
            currentFunction.ConcatRegions();
            currentFunction.CompleteFunction();
        }
        GetIREmitter().SetCurrentInsertPoint(previousPos);

        Log() << "End emitting of function " << currentFunction.GetFunctionName() << EOL;
    }

    IRFunctionEmitter& IRModuleEmitter::GetCurrentFunction()
    {
        if (_functionStack.empty())
        {
            throw EmitterException(EmitterError::indexOutOfRange, "There is no current function on the stack");
        }
        return _functionStack.top().first;
    }

    FunctionDeclaration& IRModuleEmitter::GetFunctionDeclaration(const std::string& name)
    {
        return _functions[name];
    }

    std::vector<std::string> IRModuleEmitter::GetFunctionNames() const
    {
        std::vector<std::string> names;
        std::transform(_functions.begin(), _functions.end(), std::back_inserter(names), [](const auto& pair) {
            return pair.first;
        });
        return names;
    }

    std::vector<std::string> IRModuleEmitter::GetCallbackFunctionNames() const
    {
        std::vector<std::string> names;
        for (auto pair : _functions)
        {
            std::string name = pair.first;
            if (GetFunction(name) != nullptr)
            {
                if (HasFunctionMetadata(name, c_callbackFunctionTagName))
                {
                    names.push_back(name);
                }
            }
        }
        return names;
    }

    // Module metadata
    std::vector<std::vector<std::string>> IRModuleEmitter::GetMetadata(const std::string& tag)
    {
        if (!HasMetadata(tag))
        {
            throw EmitterException(EmitterError::metadataNotFound);
        }

        std::vector<std::vector<std::string>> values;
        const llvm::NamedMDNode* metadata = _llvmModule->getNamedMetadata(tag);
        for (const auto& op : metadata->operands()) // op is an MDNode*
        {
            std::vector<std::string> opValues;
            for (const auto& innerOp : op->operands())
            {
                opValues.push_back(llvm::cast<llvm::MDString>(innerOp)->getString());
            }
            values.push_back(opValues);
        }
        return values;
    }

    // Function metadata
    std::vector<std::string> IRModuleEmitter::GetFunctionMetadata(const std::string& functionName, const std::string& tag)
    {
        if (!HasFunctionMetadata(functionName, tag))
        {
            throw EmitterException(EmitterError::metadataNotFound);
        }

        std::vector<std::string> values;
        auto pFunction = GetFunction(functionName);
        const llvm::MDNode* metadata = pFunction->getMetadata(tag);
        for (const auto& op : metadata->operands())
        {
            // llvm::Function::setMetadata appears to only support single entries?
            values.push_back(llvm::cast<llvm::MDString>(op)->getString());
        }
        return values;
    }

    // Module metadata
    bool IRModuleEmitter::HasMetadata(const std::string& tag) const
    {
        return (_llvmModule->getNamedMetadata(tag) != nullptr);
    }

    // Function metadata
    bool IRModuleEmitter::HasFunctionMetadata(const std::string& functionName, const std::string& tag) const
    {
        auto pFunction = GetFunction(functionName);
        if (pFunction == nullptr)
        {
            throw EmitterException(EmitterError::functionNotFound);
        }
        return (pFunction->getMetadata(tag) != nullptr);
    }

    // Module metadata
    void IRModuleEmitter::InsertMetadata(const std::string& tag, const std::vector<std::string>& values)
    {
        std::vector<llvm::Metadata*> metadataElements;
        for (const auto& value : values)
        {
            metadataElements.push_back({ llvm::MDString::get(GetIREmitter().GetContext(), value) });
        }

        auto metadataNode = llvm::MDNode::get(GetIREmitter().GetContext(), metadataElements);
        auto metadata = _llvmModule->getOrInsertNamedMetadata(tag);
        metadata->addOperand(metadataNode);
    }

    // Function metadata
    void IRModuleEmitter::InsertFunctionMetadata(const std::string& functionName, const std::string& tag, const std::vector<std::string>& values)
    {
        auto pFunction = GetFunction(functionName);
        if (pFunction == nullptr)
        {
            throw EmitterException(EmitterError::functionNotFound);
        }

        InsertFunctionMetadata(pFunction, tag, values);
    }

    void IRModuleEmitter::InsertFunctionMetadata(LLVMFunction function, const std::string& tag, const std::vector<std::string>& values)
    {
        assert(function);

        std::vector<llvm::Metadata*> metadataElements;
        for (const auto& value : values)
        {
            metadataElements.push_back({ llvm::MDString::get(GetIREmitter().GetContext(), value) });
        }

        auto metadataNode = llvm::MDNode::get(GetIREmitter().GetContext(), metadataElements);
        function->setMetadata(tag, metadataNode);
    }

    //
    // Variable management
    //

    LLVMValue IRModuleEmitter::EnsureEmitted(Variable& var)
    {
        LLVMValue pVal = nullptr;
        if (var.HasEmittedName())
        {
            pVal = GetEmittedVariable(var.Scope(), var.EmittedName());
        }

        if (pVal == nullptr)
        {
            AllocateVariable(var);
            pVal = GetEmittedVariable(var.Scope(), var.EmittedName());
            if (pVal == nullptr)
            {
                pVal = EmitVariable(var);
            }
        }
        return pVal;
    }

    LLVMValue IRModuleEmitter::LoadVariable(Variable& var)
    {
        LLVMValue pVal = EnsureEmitted(var);
        if (!var.IsLiteral())
        {
            pVal = GetCurrentFunction().Load(pVal);
        }
        return pVal;
    }

    //
    // Variable and Constant creation
    //

    llvm::GlobalVariable* IRModuleEmitter::Constant(VariableType type, const std::string& name, double value)
    {
        return AddGlobal(name, GetIREmitter().Type(type), GetIREmitter().Literal(value), true);
    }

    llvm::GlobalVariable* IRModuleEmitter::Global(VariableType type, const std::string& name, bool isThreadLocal)
    {
        return AddGlobal(name, GetIREmitter().Type(type), GetIREmitter().Zero(type), false, isThreadLocal);
    }

    llvm::GlobalVariable* IRModuleEmitter::Global(LLVMType pType, const std::string& name, bool isThreadLocal)
    {
        auto initializer = ZeroInitializer(pType);
        return AddGlobal(name, pType, initializer, false, isThreadLocal);
    }

    llvm::GlobalVariable* IRModuleEmitter::GlobalPointer(const std::string& name, VariableType type, bool isThreadLocal)
    {
        llvm::PointerType* pointerType = GetIREmitter().Type(type)->getPointerTo();
        return AddGlobal(name, pointerType, GetIREmitter().NullPointer(pointerType), false, isThreadLocal);
    }

    llvm::GlobalVariable* IRModuleEmitter::GlobalArray(VariableType type, const std::string& name, const size_t size, bool isThreadLocal)
    {
        llvm::ArrayType* pArrayType = GetIREmitter().ArrayType(type, size);
        return AddGlobal(name, pArrayType, ZeroInitializer(pArrayType), false, isThreadLocal);
    }

    llvm::GlobalVariable* IRModuleEmitter::GlobalArray(const std::string& name, LLVMType pType, const size_t size, bool isThreadLocal)
    {
        assert(pType != nullptr);

        llvm::ArrayType* pArrayType = llvm::ArrayType::get(pType, size);
        return AddGlobal(name, pArrayType, ZeroInitializer(pArrayType), false, isThreadLocal);
    }

    // This function has the actual implementation for all the above Global/GlobalArray() methods
    llvm::GlobalVariable* IRModuleEmitter::AddGlobal(const std::string& name, LLVMType pType, llvm::Constant* pInitial, bool isConst, bool isThreadLocal)
    {
        CompilerOptions options = GetCompilerOptions();
        _llvmModule->getOrInsertGlobal(name, pType);
        auto global = _llvmModule->getNamedGlobal(name);
        global->setAlignment(options.globalValueAlignment);
        global->setInitializer(pInitial);
        global->setConstant(isConst);
        global->setExternallyInitialized(false);
        global->setLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage);
        global->setThreadLocal(isThreadLocal);
        assert(llvm::isa<llvm::GlobalVariable>(global));
        return llvm::cast<llvm::GlobalVariable>(global);
    }

    //
    // Functions
    //

    LLVMFunction IRModuleEmitter::DeclareFunction(const std::string& name, VariableType returnType)
    {
        _functions[name] = FunctionDeclaration(name, returnType);
        return GetIREmitter().DeclareFunction(GetLLVMModule(), name, returnType);
    }

    LLVMFunction IRModuleEmitter::DeclareFunction(const std::string& name, VariableType returnType, const VariableTypeList& arguments)
    {
        NamedVariableTypeList fake;
        for (auto t : arguments)
        {
            fake.push_back({ "", t });
        }
        // record this function definition in our local _functions so that these definitions can be found via GetFunctionNames
        // and GetCallbackFunctionNames
        _functions[name] = FunctionDeclaration(name, returnType, fake);
        return GetIREmitter().DeclareFunction(GetLLVMModule(), name, returnType, arguments);
    }

    LLVMFunction IRModuleEmitter::DeclareFunction(const std::string& name, VariableType returnType, const NamedVariableTypeList& arguments)
    {
        _functions[name] = FunctionDeclaration(name, returnType, arguments);
        return GetIREmitter().DeclareFunction(GetLLVMModule(), name, returnType, arguments);
    }

    LLVMFunction IRModuleEmitter::DeclareFunction(const std::string& name, llvm::FunctionType* functionType)
    {
        // TODO: add to _functions list
        return GetIREmitter().DeclareFunction(GetLLVMModule(), name, functionType);
    }

    IRFunctionEmitter IRModuleEmitter::Function(const std::string& name, VariableType returnType, bool isPublic)
    {
        return Function(name, returnType, nullptr, isPublic);
    }

    IRFunctionEmitter IRModuleEmitter::Function(const std::string& name, VariableType returnType, const VariableTypeList& arguments, bool isPublic)
    {
        return Function(name, returnType, &arguments, isPublic);
    }

    IRFunctionEmitter IRModuleEmitter::Function(const std::string& name, VariableType returnType, const std::initializer_list<VariableType>& arguments, bool isPublic)
    {
        VariableTypeList valueTypeList = arguments;
        return Function(name, returnType, &valueTypeList, isPublic);
    }

    IRFunctionEmitter IRModuleEmitter::Function(const std::string& name, VariableType returnType, const NamedVariableTypeList& arguments, bool isPublic)
    {
        // TODO: add this function to the _functions list??
        LLVMFunction pFunction = GetIREmitter().Function(GetLLVMModule(), name, returnType, Linkage(isPublic), arguments);
        if (pFunction == nullptr)
        {
            throw EmitterException(EmitterError::functionNotFound);
        }
        // TODO: put the above IREmitter call in the IRFunctionEmitter constructor

        return IRFunctionEmitter(this, pFunction, arguments, name);
    }

    IRFunctionEmitter IRModuleEmitter::Function(const std::string& name, LLVMType returnType, const NamedVariableTypeList& arguments, bool isPublic)
    {
        // TODO: add this function to the _functions list??
        LLVMFunction pFunction = GetIREmitter().Function(GetLLVMModule(), name, returnType, Linkage(isPublic), arguments);
        if (pFunction == nullptr)
        {
            throw EmitterException(EmitterError::functionNotFound);
        }
        // TODO: put the above IREmitter call in the IRFunctionEmitter constructor

        return IRFunctionEmitter(this, pFunction, arguments, name);
    }

    IRFunctionEmitter IRModuleEmitter::Function(const std::string& name, LLVMType returnType, const FunctionArgumentList& arguments, bool isPublic)
    {
        // TODO: add this function to the _functions list??
        LLVMFunction pFunction = GetIREmitter().Function(GetLLVMModule(), name, returnType, Linkage(isPublic), arguments);
        if (pFunction == nullptr)
        {
            throw EmitterException(EmitterError::functionNotFound);
        }
        // TODO: put the above IREmitter call in the IRFunctionEmitter constructor

        return IRFunctionEmitter(this, pFunction, arguments, name);
    }

    IRFunctionEmitter IRModuleEmitter::Function(const std::string& name, VariableType returnType, const VariableTypeList* pArguments, bool isPublic)
    {
        // TODO: add this function to the _functions list??
        LLVMFunction pFunction = GetIREmitter().Function(GetLLVMModule(), name, returnType, Linkage(isPublic), pArguments);
        if (pFunction == nullptr)
        {
            throw EmitterException(EmitterError::functionNotFound);
        }
        return IRFunctionEmitter(this, pFunction, name);
    }

    IRFunctionEmitter IRModuleEmitter::Function(const std::string& name, LLVMType returnType, const std::vector<LLVMType>& argTypes, bool isPublic)
    {
        // TODO: add this function to the _functions list??
        LLVMFunction pFunction = GetIREmitter().Function(GetLLVMModule(), name, returnType, Linkage(isPublic), argTypes);
        if (pFunction == nullptr)
        {
            throw EmitterException(EmitterError::functionNotFound);
        }
        return IRFunctionEmitter(this, pFunction, name);
    }

    IRFunctionEmitter IRModuleEmitter::Function(const std::string& name, LLVMType returnType, const NamedLLVMTypeList& arguments, bool isPublic)
    {
        // TODO: add this function to the _functions list??
        LLVMFunction pFunction = GetIREmitter().Function(GetLLVMModule(), name, returnType, Linkage(isPublic), arguments);
        if (pFunction == nullptr)
        {
            throw EmitterException(EmitterError::functionNotFound);
        }
        return IRFunctionEmitter(this, pFunction, arguments, name);
    }

    bool IRModuleEmitter::HasFunction(const std::string& name) const
    {
        return GetLLVMModule()->getFunction(name) != nullptr;
    }

    LLVMFunction IRModuleEmitter::GetFunction(const std::string& name) const
    {
        return GetLLVMModule()->getFunction(name);
    }

    LLVMFunction IRModuleEmitter::GetIntrinsic(llvm::Intrinsic::ID id)
    {
        return GetIREmitter().GetIntrinsic(GetLLVMModule(), id, VariableTypeList{});
    }

    LLVMFunction IRModuleEmitter::GetIntrinsic(llvm::Intrinsic::ID id, const std::initializer_list<VariableType>& arguments)
    {
        VariableTypeList valueTypeList = arguments;
        return GetIREmitter().GetIntrinsic(GetLLVMModule(), id, valueTypeList);
    }

    LLVMFunction IRModuleEmitter::GetIntrinsic(llvm::Intrinsic::ID id, const std::initializer_list<LLVMType>& arguments)
    {
        LLVMTypeList valueTypeList = arguments;
        return GetIREmitter().GetIntrinsic(GetLLVMModule(), id, valueTypeList);
    }

    //
    // Types
    //

    llvm::StructType* IRModuleEmitter::GetOrCreateStruct(const std::string& name, const NamedVariableTypeList& fields)
    {
        NamedLLVMTypeList llvmFields;
        for (auto& field : fields)
        {
            llvmFields.emplace_back(field.first, GetIREmitter().Type(field.second));
        }
        return GetOrCreateStruct(name, llvmFields);
    }

    llvm::StructType* IRModuleEmitter::GetOrCreateStruct(const std::string& name, const NamedLLVMTypeList& fields)
    {
        std::vector<std::string> fieldNames;
        std::vector<LLVMType> types;
        for (const auto& field : fields)
        {
            fieldNames.push_back(field.first);
            types.push_back(field.second);
        }

        auto structType = GetOrCreateStruct(name, types);
        auto tag = GetStructFieldsTagName(structType);
        InsertMetadata(tag, fieldNames);
        return structType;
    }

    llvm::StructType* IRModuleEmitter::GetOrCreateStruct(const std::string& name, const LLVMTypeList& fields)
    {
        if (auto structType = GetIREmitter().GetStruct(name))
        {
            // Check that existing struct fields match the ones we're trying to create
            auto structFields = structType->elements();
            if (structFields.size() != fields.size())
            {
                throw EmitterException(EmitterError::badStructDefinition, "Fields don't match existing struct of same name");
            }
            else
            {
                for (size_t fieldIndex = 0; fieldIndex < fields.size(); ++fieldIndex)
                {
                    if (fields[fieldIndex] != structFields[fieldIndex])
                    {
                        throw EmitterException(EmitterError::badStructDefinition, "Fields don't match existing struct of same name");
                    }
                }
            }
            return structType;
        }

        return GetIREmitter().DeclareStruct(name, fields);
    }

    llvm::StructType* IRModuleEmitter::GetAnonymousStructType(const LLVMTypeList& fieldTypes, bool packed)
    {
        return GetIREmitter().GetAnonymousStructType(fieldTypes, packed);
    }

    llvm::StructType* IRModuleEmitter::GetStruct(const std::string& name)
    {
        return GetIREmitter().GetStruct(name);
    }

    //
    // Code output / input
    //

    MachineCodeOutputOptions IRModuleEmitter::GetMachineCodeOutputOptions() const
    {
        MachineCodeOutputOptions options;
        auto compilerOptions = GetCompilerOptions();
        options.targetDevice = compilerOptions.targetDevice;
        if (compilerOptions.optimize)
        {
            options.optimizationLevel = OptimizationLevel::Aggressive;
        }

        options.verifyModule = true;
        options.floatFusionMode = compilerOptions.useFastMath ? FloatFusionMode::Fast : FloatFusionMode::Standard;
        options.unsafeFPMath = compilerOptions.useFastMath;
        options.noInfsFPMath = compilerOptions.useFastMath;
        options.noNaNsFPMath = compilerOptions.useFastMath;
        options.noSignedZerosFPMath = compilerOptions.useFastMath;

        if (compilerOptions.positionIndependentCode.HasValue())
        {
            options.relocModel = compilerOptions.positionIndependentCode.GetValue() ? OutputRelocationModel::PIC_ : OutputRelocationModel::Static;
        }
        else
        {
            // 'auto'
            options.relocModel = compilerOptions.targetDevice.IsWindows() ? OutputRelocationModel::Static : OutputRelocationModel::PIC_;
        }

        // Other params to possibly set:
        //   bool verboseOutput = false;
        //   bool verifyModule = false;
        //   FloatABIType floatABI = FloatABIType::Default;
        return options;
    }

    void IRModuleEmitter::WriteToFile(const std::string& filePath, ModuleOutputFormat format)
    {
        auto options = GetMachineCodeOutputOptions();

        switch (format)
        {
        case ModuleOutputFormat::assembly:
        case ModuleOutputFormat::bitcode:
        case ModuleOutputFormat::ir:
        case ModuleOutputFormat::objectCode: {
            WriteToFile(filePath, format, options);
            break;
        }
        case ModuleOutputFormat::cHeader: {
            auto os = utilities::OpenOfstream(filePath);
            WriteToStream(os, format, options);
            break;
        }
        case ModuleOutputFormat::swigInterface: {
            // Write the swig interface file
            auto headerFilePath = filePath + ".h";
            auto os = utilities::OpenOfstream(filePath);
            WriteModuleSwigInterface(os, *this, utilities::GetFileName(headerFilePath));
            break;
        }
        default:
            throw EmitterException(EmitterError::notSupported);
        }
    }

    void IRModuleEmitter::WriteToFile(const std::string& filePath, ModuleOutputFormat format, const MachineCodeOutputOptions& options)
    {
        auto openFlags = (ModuleOutputFormat::bitcode == format || ModuleOutputFormat::objectCode == format) ? llvm::sys::fs::F_None : llvm::sys::fs::F_Text;
        std::error_code error;
        llvm::ToolOutputFile out(filePath, error, openFlags);
        if (error)
        {
            throw LLVMException(error);
        }

        WriteToLLVMStream(out.os(), format, options);

        if (out.os().has_error())
        {
            throw EmitterException(EmitterError::writeStreamFailed);
        }
        out.keep();
    }

    void IRModuleEmitter::WriteToStream(std::ostream& stream, ModuleOutputFormat format)
    {
        auto options = GetMachineCodeOutputOptions();

        WriteToStream(stream, format, options);
    }

    void IRModuleEmitter::WriteToStream(std::ostream& os, ModuleOutputFormat format, const MachineCodeOutputOptions& options)
    {
        if (ModuleOutputFormat::cHeader == format)
        {
            WriteHeader(os);
        }
        else if (ModuleOutputFormat::swigInterface == format)
        {
            WriteHeader(os);
        }
        else
        {
            llvm::raw_os_ostream out(os);
            WriteToLLVMStream(out, format, options);
        }
    }

    void IRModuleEmitter::WriteToLLVMStream(llvm::raw_ostream& os, ModuleOutputFormat format, MachineCodeOutputOptions options)
    {
        const auto& params = GetCompilerOptions();

        if (options.targetDevice.triple.empty())
        {
            options.targetDevice.triple = params.targetDevice.triple;
        }

        if (options.targetDevice.cpu.empty())
        {
            options.targetDevice.cpu = params.targetDevice.cpu;
        }

        if (options.targetDevice.features.empty())
        {
            options.targetDevice.features = params.targetDevice.features;
        }

        GenerateMachineCode(os, *this, format, options);
    }

    void IRModuleEmitter::LoadIR(const std::string& text)
    {
        auto buffer = llvm::MemoryBuffer::getMemBuffer(text);
        llvm::SMDiagnostic errorHandler;
        bool hadError = llvm::parseAssemblyInto(*buffer, GetLLVMModule(), nullptr, errorHandler);
        if (hadError)
        {
            std::string message = errorHandler.getMessage();
            throw EmitterException(EmitterError::parserError, message);
        }
    }

    void IRModuleEmitter::LoadIR(std::istream& stream)
    {
        std::string irStr(std::istreambuf_iterator<char>(stream), {});
        LoadIR(irStr);
    }

    void IRModuleEmitter::LoadIRFromFile(const std::string& filename)
    {
        auto buffer = llvm::MemoryBuffer::getFile(filename);
        if (!buffer)
        {
            throw EmitterException(EmitterError::parserError, "Unable to open " + filename);
        }

        llvm::SMDiagnostic errorHandler;
        bool hadError = llvm::parseAssemblyInto(*(buffer->get()), GetLLVMModule(), nullptr, errorHandler);
        if (hadError)
        {
            std::string message = errorHandler.getMessage();
            throw EmitterException(EmitterError::parserError, message);
        }
    }

    void IRModuleEmitter::LoadAsm(const std::string& text)
    {
        GetLLVMModule()->appendModuleInlineAsm(text);
    }

    void IRModuleEmitter::LoadAsm(std::istream& stream)
    {
        std::string asmStr(std::istreambuf_iterator<char>(stream), {});
        LoadAsm(asmStr);
    }

    void IRModuleEmitter::LoadAsmFromFile(const std::string& filename)
    {
        auto stream = OpenIfstream(filename);
        LoadAsm(stream);
    }

    void IRModuleEmitter::WriteHeader(std::ostream& os)
    {
        WriteModuleHeader(os, *this);
        WriteModuleCppWrapper(os, *this);
    }

    //
    // Optimization
    //

    void IRModuleEmitter::Optimize(IROptimizer& optimizer)
    {
        auto compilerOptions = GetCompilerOptions();
        if (compilerOptions.optimize)
        {
            std::stringstream sstr;
            if (CheckForErrors(sstr))
            {
                auto errorString = sstr.str();
                throw EmitterException(EmitterError::unexpected, "Module verification failed.\n\n" + errorString);
            }

            optimizer.BeginOptimizeFunctions();
            auto module = GetLLVMModule();
            for (auto& function : *module)
            {
                optimizer.OptimizeFunction(&function);
            }
            optimizer.EndOptimizeFunctions();

            optimizer.OptimizeModule(module);
        }
    }

    //
    // Helpers, standard C Runtime functions, and debug support
    //

    LLVMFunction IRModuleEmitter::DeclareDebugPrint()
    {
        auto functionName = "DebugPrint";
        LLVMFunction function;
        if (_functions.find(functionName) == _functions.end())
        {
            function = DeclareFunction(functionName, VariableType::Void, NamedVariableTypeList{ { "message", VariableType::Char8Pointer } });
            function->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
            InsertFunctionMetadata(function, c_callbackFunctionTagName, { "global" });
        }
        else
        {
            function = GetFunction(functionName);
        }
        return function;
    }

    void IRModuleEmitter::DebugPrint(std::string message)
    {
        std::string globalName = GetModuleName() + "_debug_message_" + std::to_string(_globalStringIndex++);
        llvm::GlobalVariable* msg = ConstantArray(globalName, std::vector<char>(message.c_str(), message.c_str() + message.size() + 1));
        auto& context = GetIREmitter().GetContext();
        auto charPointerType = llvm::Type::getInt8Ty(context)->getPointerTo();
        llvm::Value* msgptr = GetIREmitter().CastPointer(msg, charPointerType);
        auto function = DeclareDebugPrint();
        GetIREmitter().Call(function, msgptr);
    }

    void IRModuleEmitter::DeclarePrintf()
    {
        auto& context = GetIREmitter().GetContext();
        auto type = llvm::FunctionType::get(
            llvm::Type::getInt32Ty(context),
            { llvm::Type::getInt8PtrTy(context) },
            true);
        DeclareFunction("printf", type);
    }

    void IRModuleEmitter::DeclareMalloc()
    {
        DeclareFunction("malloc", VariableType::BytePointer, { VariableType::Int64 });
    }

    void IRModuleEmitter::DeclareFree()
    {
        DeclareFunction("free", VariableType::Void, { VariableType::BytePointer });
    }

    IRFunctionEmitter IRModuleEmitter::BeginMainDebugFunction()
    {
        DeclarePrintf();
        return BeginMainFunction();
    }

    IRDiagnosticHandler& IRModuleEmitter::GetDiagnosticHandler()
    {
        return *_diagnosticHandler;
    }

    bool IRModuleEmitter::CheckForErrors()
    {
        return llvm::verifyModule(*_llvmModule);
    }

    bool IRModuleEmitter::CheckForErrors(std::ostream& stream)
    {
        llvm::raw_os_ostream out(stream);
        return llvm::verifyModule(*_llvmModule, &out);
    }

    void IRModuleEmitter::DebugDump() const
    {
        auto module = GetLLVMModule();
        if (module)
        {
            emitters::DebugDump(module);
        }
        else
        {
            throw EmitterException(EmitterError::unexpected, "Cannot dump module because TransferOwnership has been called.");
        }
    }

    //
    // low-level LLVM-related functionality
    //

    llvm::TargetMachine* IRModuleEmitter::GetTargetMachine()
    {
        std::string error;
        auto parameters = GetCompilerOptions();

        const llvm::Target* target = llvm::TargetRegistry::lookupTarget(parameters.targetDevice.triple, error);
        if (target == nullptr)
        {
            return nullptr;
        }

        auto relocModel = parameters.targetDevice.IsWindows() ? OutputRelocationModel::Static : OutputRelocationModel::PIC_;
        llvm::TargetOptions options;
        options.FloatABIType = llvm::FloatABI::Default;
        options.AllowFPOpFusion = parameters.useFastMath ? llvm::FPOpFusion::Standard : llvm::FPOpFusion::Fast;
        options.UnsafeFPMath = parameters.useFastMath ? 1 : 0;
        options.NoInfsFPMath = parameters.useFastMath ? 1 : 0;
        options.NoNaNsFPMath = parameters.useFastMath ? 1 : 0;
        options.NoSignedZerosFPMath = parameters.useFastMath ? 1 : 0;

        const llvm::CodeModel::Model codeModel = llvm::CodeModel::Small;
        auto tm = target->createTargetMachine(llvm::Triple::normalize(parameters.targetDevice.triple),
                                              parameters.targetDevice.cpu,
                                              parameters.targetDevice.features,
                                              options,
                                              relocModel,
                                              codeModel,
                                              parameters.optimize ? llvm::CodeGenOpt::Level::Aggressive : llvm::CodeGenOpt::Level::Default);
        return tm;
    }

    std::unique_ptr<llvm::Module> IRModuleEmitter::TransferOwnership()
    {
        auto result = std::move(_llvmModule);
        _llvmModule = nullptr;
        return result;
    }

    void IRModuleEmitter::SetTargetTriple(const std::string& triple)
    {
        GetLLVMModule()->setTargetTriple(llvm::Triple::normalize(triple));
    }

    const llvm::DataLayout& IRModuleEmitter::GetTargetDataLayout() const
    {
        assert(!GetLLVMModule()->getDataLayout().getStringRepresentation().empty());
        return GetLLVMModule()->getDataLayout();
    }

    void IRModuleEmitter::AddPreprocessorDefinition(const std::string& name, const std::string& value)
    {
        _preprocessorDefinitions.emplace_back(name, value);
    }

    //
    // Metadata
    //
    void IRModuleEmitter::IncludeTypeInHeader(const std::string& typeName)
    {
        InsertMetadata(c_declareTypeInHeaderTagName, { typeName });
    }

    void IRModuleEmitter::IncludeInCallbackInterface(const std::string& functionName, const std::string& nodeName)
    {
        // We shouldn't actually have to set this linkage to external because:
        // * All usages are preceded by IncludeInHeader, which already does external linkage
        // * We should be using this for just declarations that the user provides as a callback
        // That said, leaving it in for now until usage proves to be counter-productive.
        auto function = GetFunction(functionName);
        function->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);
        InsertFunctionMetadata(function, c_callbackFunctionTagName, { nodeName });
    }

    //
    // Module initialization / finalization
    //
    void IRModuleEmitter::AddInitializationFunction(LLVMFunction function, int priority, llvm::Constant* forData)
    {
        llvm::appendToGlobalCtors(*GetLLVMModule(), function, priority, forData);
    }

    void IRModuleEmitter::AddInitializationFunction(IRFunctionEmitter& function, int priority, llvm::Constant* forData)
    {
        AddInitializationFunction(function.GetFunction(), priority, forData);
    }

    void IRModuleEmitter::AddFinalizationFunction(LLVMFunction function, int priority, llvm::Constant* forData)
    {
        llvm::appendToGlobalDtors(*GetLLVMModule(), function, priority, forData);
    }

    void IRModuleEmitter::AddFinalizationFunction(IRFunctionEmitter& function, int priority, llvm::Constant* forData)
    {
        AddFinalizationFunction(function.GetFunction(), priority, forData);
    }

    //
    // Protected methods
    //

    //
    // Private methods
    //

    //
    // Internal variable and function creation implementation
    //

    LLVMValue IRModuleEmitter::GetEmittedVariable(const VariableScope scope, const std::string& name)
    {
        switch (scope)
        {
        case VariableScope::literal:
            return _literals.Get(name);

        case VariableScope::global:
            return _globals.Get(name);

        case VariableScope::local:
            return GetCurrentFunction().GetEmittedVariable(scope, name);

        case VariableScope::input:
        case VariableScope::output:
            return GetCurrentFunction().GetEmittedVariable(scope, name);

        default:
            throw EmitterException(EmitterError::variableScopeNotSupported);
        }
    }

    LLVMValue IRModuleEmitter::EmitVariable(Variable& var)
    {
        assert(var.HasEmittedName());
        switch (var.Type())
        {
        case VariableType::Byte:
            return EmitVariable<uint8_t>(var);
        case VariableType::Int32:
            return EmitVariable<int>(var);
        case VariableType::Int64:
            return EmitVariable<int64_t>(var);
        case VariableType::Float:
            return EmitVariable<float>(var);
        case VariableType::Double:
            return EmitVariable<double>(var);
        default:
            break;
        }
        throw EmitterException(EmitterError::variableTypeNotSupported);
    }

    llvm::Function::LinkageTypes IRModuleEmitter::Linkage(bool isPublic)
    {
        return isPublic ? llvm::Function::LinkageTypes::ExternalLinkage : llvm::Function::LinkageTypes::InternalLinkage;
    }

    llvm::ConstantAggregateZero* IRModuleEmitter::ZeroInitializer(LLVMType pType)
    {
        assert(pType != nullptr);
        return llvm::ConstantAggregateZero::get(pType);
    }

    template <>
    CallbackRegistry<float>& IRModuleEmitter::GetCallbackRegistry() const
    {
        return const_cast<CallbackRegistry<float>&>(_floatCallbacks);
    }

    template <>
    CallbackRegistry<double>& IRModuleEmitter::GetCallbackRegistry() const
    {
        return const_cast<CallbackRegistry<double>&>(_doubleCallbacks);
    }

    template <>
    CallbackRegistry<int>& IRModuleEmitter::GetCallbackRegistry() const
    {
        return const_cast<CallbackRegistry<int>&>(_intCallbacks);
    }

    template <>
    CallbackRegistry<int64_t>& IRModuleEmitter::GetCallbackRegistry() const
    {
        return const_cast<CallbackRegistry<int64_t>&>(_int64Callbacks);
    }

    template <>
    CallbackRegistry<bool>& IRModuleEmitter::GetCallbackRegistry() const
    {
        return const_cast<CallbackRegistry<bool>&>(_boolCallbacks);
    }

    /// <summary> Returns true if the CallbackRegistry objects contain some functions. </summary>
    bool IRModuleEmitter::HasCallbackFunctions() const
    {
        return _floatCallbacks.HasCallbackFunctions() || _doubleCallbacks.HasCallbackFunctions() || _intCallbacks.HasCallbackFunctions() ||
               _int64Callbacks.HasCallbackFunctions() || _boolCallbacks.HasCallbackFunctions();
    }

    //
    // Functions
    //

    IRModuleEmitter MakeHostModuleEmitter(const std::string moduleName)
    {
        CompilerOptions parameters;
        parameters.targetDevice.deviceName = "host";
        return { moduleName, parameters };
    }
} // namespace emitters
} // namespace ell
