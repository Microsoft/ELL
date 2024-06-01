////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRExecutionEngine.h (emitters)
//  Authors:  Umesh Madan
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "LLVMUtilities.h"

#include <utilities/include/Exception.h>
#include <utilities/include/TypeAliases.h>

#include <llvm/ExecutionEngine/ExecutionEngine.h>
#include <llvm/IR/Module.h>

#include <functional>
#include <type_traits>

namespace ell
{
namespace emitters
{
    class IRModuleEmitter;

    /// <summary> Function signature for a basic function that takes no input and returns no output </summary>
    typedef void (*DynamicFunction)(void);

    /// <summary> Wrapper class to setup and manage the LLVM Execution Engine. By default, we use the new "MCJIT" </summary>
    class IRExecutionEngine
    {
    public:
        /// <summary>
        /// Move the primary "owner" module into the execution engine.
        /// </summary>
        ///
        /// <param name="module"> The module. </param>
        /// <param name="verify"> Indicates if the execution engine should run a verification pass before running the code. </param>
        IRExecutionEngine(IRModuleEmitter&& module, bool verify = true, llvm::CodeGenOpt::Level optLevel = llvm::CodeGenOpt::Level::Default);

        /// <summary> Inject the primary "owner" module into the execution engine. </summary>
        ///
        /// <param name="pModule"> The module. </param>
        /// <param name="verify"> Indicates if the execution engine should run a verification pass before running the code. </param>
        IRExecutionEngine(std::unique_ptr<llvm::Module> pModule, bool verify = false, llvm::CodeGenOpt::Level optLevel = llvm::CodeGenOpt::Level::Default);

        /// <summary> Destructor </summary>
        ~IRExecutionEngine();

        /// <summary> Add an additional module to the execution engine. </summary>
        ///
        /// <param name="pModule"> The module to add. </param>
        void AddModule(std::unique_ptr<llvm::Module> pModule);

        /// <summary>
        /// Return the address of a named function, JITTing code as needed. Returns 0 if not found.
        /// </summary>
        ///
        /// <param name="name"> Name of the requested function. </param>
        ///
        /// <returns> The function address. </returns>
        uint64_t GetFunctionAddress(const std::string& name);

        /// <summary>
        /// Return a std::function to invoke a named function, JITTing code as needed. Throws an exception if not found.
        /// </summary>
        ///
        /// <param name="name"> Name of the requested function. </param>
        ///
        /// <returns> The function object. </returns>
        template <typename FunctionType>
        std::function<FunctionType> GetFunction(const std::string& name)
        {
            auto addr = GetFunctionAddress(name);
            if (!addr)
            {
                throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Function not found");
            }

            return reinterpret_cast<std::add_pointer_t<FunctionType>>(addr);
        } // STYLE discrepancy

        /// <summary>
        /// Return the address of a global variable, JITTing code as needed. Returns 0 if not found.
        /// </summary>
        ///
        /// <param name="name"> Name of the requested global variable. </param>
        ///
        /// <returns> The variable address. </returns>
        uint64_t GetGlobalValueAddress(const std::string& name);

        /// <summary> Return the address of a named function. Throws if not found. </summary>
        ///
        /// <param name="name"> Name of the requested function. </param>
        ///
        /// <returns> The function address. </returns>
        uint64_t ResolveFunctionAddress(const std::string& name);

        /// <summary> Set the address of a named function. </summary>
        ///
        /// <param name="func"> The function being defined. </param>
        /// <param name="address"> The address of the function being defined. </param>
        void DefineFunction(LLVMFunction func, utilities::UIntPtrT address);

        /// <summary> Set the address of a named function already defined elsewhere in the binary. </summary>
        ///
        /// <param name="name"> The function name being defined. </param>
        /// <param name="address"> The address of the function being referenced. </param>
        void DefineFunction(const std::string& name, utilities::UIntPtrT address);

        /// <summary>
        /// Return a main function that takes no arguments - if one exists. Returns nullptr if not found.
        /// </summary>
        ///
        /// <returns> The main function. </returns>
        DynamicFunction GetMain();

        /// <summary> Return a main function that takes no arguments. Throws if not found. </summary>
        ///
        /// <returns> The main function. </returns>
        DynamicFunction ResolveMain();

        /// <summary> Resolve and run the default Main function, if any. </summary>
        void RunMain();

    private:
        void EnsureEngine();
        void EnsureClockGetTime();
        void PerformInitialization();
        void PerformFinalization();

        std::unique_ptr<llvm::EngineBuilder> _pBuilder;
        std::unique_ptr<llvm::ExecutionEngine> _pEngine;
    };
} // namespace emitters
} // namespace ell
