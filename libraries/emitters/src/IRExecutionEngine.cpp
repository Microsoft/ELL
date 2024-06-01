////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRExecutionEngine.cpp (emitters)
//  Authors:  Umesh Madan
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRExecutionEngine.h"
#include "IRModuleEmitter.h"

#include <utilities/include/TypeAliases.h>

#include <llvm/Support/TargetSelect.h>

#include <iostream>
#include <memory>
#include <string>

extern "C" {
void DebugPrintImpl(char* message)
{
    if (message != nullptr)
    {
        std::cout << message;
    }
}
}

namespace ell
{
namespace emitters
{
    using namespace utilities;

    void FatalErrorHandler(void* userData, const std::string& reason, bool genCrashDiag)
    {
        std::string msg = "llvm fatal error: " + reason;
        throw emitters::EmitterException(emitters::EmitterError::unexpected, msg);
    }

    IRExecutionEngine::IRExecutionEngine(IRModuleEmitter&& module, bool verify, llvm::CodeGenOpt::Level optLevel) :
        IRExecutionEngine(module.TransferOwnership(), verify, optLevel)
    {
    }

    IRExecutionEngine::IRExecutionEngine(std::unique_ptr<llvm::Module> pModule, bool verify, llvm::CodeGenOpt::Level optLevel)
    {
        auto debugPrintFunction = pModule->getFunction("DebugPrint");

        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();

        _pBuilder = std::make_unique<llvm::EngineBuilder>(std::move(pModule));
        _pBuilder->setEngineKind(llvm::EngineKind::JIT).setVerifyModules(verify).setOptLevel(optLevel).setEmulatedTLS(true);

        static bool installed = false;
        if (!installed)
        {
            installed = true;
            llvm::remove_fatal_error_handler();
            llvm::install_fatal_error_handler(&FatalErrorHandler, nullptr);
        }

        // If DeclareDebugPrint was called, then define it here.
        if (debugPrintFunction)
        {
            DefineFunction(debugPrintFunction, reinterpret_cast<UIntPtrT>(&DebugPrintImpl));
        }
    }

    IRExecutionEngine::~IRExecutionEngine()
    {
        if (_pEngine)
        {
            PerformFinalization();
        }
    }

    void IRExecutionEngine::AddModule(std::unique_ptr<llvm::Module> pModule)
    {
        assert(pModule != nullptr);
        EnsureEngine();
        _pEngine->addModule(std::move(pModule));
    }

    void IRExecutionEngine::PerformInitialization()
    {
        _pEngine->runStaticConstructorsDestructors(false);
    }

    void IRExecutionEngine::PerformFinalization()
    {
        _pEngine->runStaticConstructorsDestructors(true);
    }

    uint64_t IRExecutionEngine::GetFunctionAddress(const std::string& name)
    {
        EnsureEngine();
        return _pEngine->getFunctionAddress(name);
    }

    uint64_t IRExecutionEngine::GetGlobalValueAddress(const std::string& name)
    {
        EnsureEngine();
        return _pEngine->getGlobalValueAddress(name);
    }

    uint64_t IRExecutionEngine::ResolveFunctionAddress(const std::string& name)
    {
        auto functionAddress = GetFunctionAddress(name);
        if (functionAddress == 0)
        {
            throw EmitterException(EmitterError::functionNotFound);
        }
        return functionAddress;
    }

    void IRExecutionEngine::DefineFunction(LLVMFunction func, UIntPtrT address)
    {
        EnsureEngine();
        _pEngine->addGlobalMapping(func, (void*)address);
    }

    void IRExecutionEngine::DefineFunction(const std::string& name, UIntPtrT address)
    {
        EnsureEngine();
        _pEngine->addGlobalMapping(name, address);
    }

    DynamicFunction IRExecutionEngine::GetMain()
    {
        return reinterpret_cast<DynamicFunction>(GetFunctionAddress("main"));
    }

    DynamicFunction IRExecutionEngine::ResolveMain()
    {
        auto mainFunction = GetMain();
        if (mainFunction == nullptr)
        {
            throw EmitterException(EmitterError::functionNotFound);
        }
        return mainFunction;
    }

    void IRExecutionEngine::RunMain()
    {
        auto mainFunction = ResolveMain();
        // Call the function
        mainFunction();
    }

    void IRExecutionEngine::EnsureEngine()
    {
        if (!_pEngine)
        {
            auto pEngine = _pBuilder->create();
            _pEngine.reset(pEngine);
            PerformInitialization();
        }
    }
} // namespace emitters
} // namespace ell
