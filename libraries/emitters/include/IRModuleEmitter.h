////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRModuleEmitter.h (emitters)
//  Authors:  Umesh Madan, Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "CompilerOptions.h"
#include "EmitterException.h"
#include "IRAssemblyWriter.h"
#include "IRDiagnosticHandler.h"
#include "IREmitter.h"
#include "IRFunctionEmitter.h"
#include "IRProfiler.h"
#include "IRRuntime.h"
#include "IRThreadPool.h"
#include "LLVMUtilities.h"
#include "ModuleEmitter.h"
#include "ScalarVariable.h"
#include "VectorVariable.h"

#include <llvm/IR/Constant.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/GlobalVariable.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>

#include <utilities/include/CallbackRegistry.h>

#include <initializer_list>
#include <iosfwd>
#include <memory>
#include <stack>
#include <string>
#include <utility>
#include <vector>

namespace ell
{
namespace emitters
{
    /// <summary> Object used to emit LLVM Module level instructions. </summary>
    class IRModuleEmitter : public ModuleEmitter
    {
    public:
        /// <summary> Construct a new module emitter to output a new named module. </summary>
        ///
        /// <param name="moduleName"> Name of the module. </param>
        /// <param name="parameters"> Options for the compiler </param>
        IRModuleEmitter(const std::string& moduleName, const CompilerOptions& parameters);

        IRModuleEmitter(const IRModuleEmitter&) = delete;
        IRModuleEmitter(IRModuleEmitter&&) = delete;
        IRModuleEmitter& operator=(const IRModuleEmitter&) = delete;
        IRModuleEmitter& operator=(IRModuleEmitter&&) = delete;

        //
        // Properties of the module
        //

        /// <summary> Returns the module's name. </summary>
        ///
        /// <returns> The module's name. </returns>
        std::string GetModuleName() const { return _llvmModule->getName(); }

        //
        // Getting state
        //

        /// <summary> The current function being emitted. </summary>
        ///
        /// <returns> Reference to an `IRFunctionEmitter` for the current function. </returns>
        IRFunctionEmitter& GetCurrentFunction();

        /// <summary> Returns the current block being emitted into. </summary>
        ///
        /// <returns> Pointer to an `IRBlockRegion` for the current region. </returns>
        IRBlockRegion* GetCurrentRegion() { return GetCurrentFunction().GetCurrentRegion(); }

        /// <summary> Returns the runtime object that manages functions. </summary>
        ///
        /// <returns> Reference to the `IRRuntime`. </returns>
        IRRuntime& GetRuntime() { return *_runtime; }

        /// <summary> Gets a reference to the profiler. </summary>
        ///
        /// <returns> Reference to the `IRProfiler` object for this module. </returns>
        IRProfiler& GetProfiler() { return *_profiler; }

        /// <summary> Gets a reference to the underlying IREmitter. </summary>
        ///
        /// <returns> Reference to the underlying IREmitter. </returns>
        IREmitter& GetIREmitter() { return *_emitter; }

        /// <summary> Can this module emitter still be used to add functions to the module? </summary>
        ///
        /// <returns> true if active, false if not. </returns>
        bool IsValid() const { return (_llvmModule != nullptr); }

        //
        // Creating functions
        //

        /// <summary> Begins a function and directs subsequent commands to it. </summary>
        ///
        /// <param name="functionName"> The name of the function. </param>
        /// <param name="args"> The arguments to the function. </param>
        void BeginMapPredictFunction(const std::string& functionName, FunctionArgumentList& args) override;

        /// <summary> Ends the current model prediction function. </summary>
        void EndMapPredictFunction() override;

        /// <summary> Begin a new function for resetting a given node. Each node that needs to implement
        /// reset calls this and implements their own reset logic.  The IRModuleEmitter wraps all that
        /// in a master model_Reset function which is exposed in the API. </summary>
        ///
        /// <param name="nodeName"> The name of the node being reset (should be unique to the model). </param>
        IRFunctionEmitter& BeginResetFunction(const std::string& nodeName);

        /// <summary> End your reset function created with BeginResetFunction. </summary>
        void EndResetFunction() { EndFunction(); }

        /// <summary> Begins an IR function with no arguments and directs subsequent commands to it. </summary>
        ///
        /// <param name="functionName"> The name of the function. </param>
        /// <param name="returnType"> The return type of the function. </param>
        IRFunctionEmitter& BeginFunction(const std::string& functionName, VariableType returnType);

        /// <summary> Begins an IR function with no arguments and directs subsequent commands to it. </summary>
        ///
        /// <param name="functionName"> The name of the function. </param>
        /// <param name="returnType"> The return type of the function. </param>
        IRFunctionEmitter& BeginFunction(const std::string& functionName, LLVMType returnType);

        /// <summary> Begins an IR function and directs subsequent commands to it. </summary>
        ///
        /// <param name="functionName"> The name of the function. </param>
        /// <param name="returnType"> The return type of the function. </param>
        /// <param name="args"> The arguments to the function. </param>
        IRFunctionEmitter& BeginFunction(const std::string& functionName, VariableType returnType, const VariableTypeList& args);

        /// <summary> Begins an IR function and directs subsequent commands to it. </summary>
        ///
        /// <param name="functionName"> The name of the function. </param>
        /// <param name="returnType"> The return type of the function. </param>
        /// <param name="args"> The arguments to the function. </param>
        IRFunctionEmitter& BeginFunction(const std::string& functionName, VariableType returnType, const NamedVariableTypeList& args);

        /// <summary> Begins an IR function and directs subsequent commands to it. </summary>
        ///
        /// <param name="functionName"> The name of the function. </param>
        /// <param name="returnType"> The return type of the function. </param>
        /// <param name="args"> The arguments to the function. </param>
        IRFunctionEmitter& BeginFunction(const std::string& functionName, VariableType returnType, const FunctionArgumentList& args);

        /// <summary> Begins an IR function and directs subsequent commands to it. </summary>
        ///
        /// <param name="functionName"> The name of the function. </param>
        /// <param name="returnType"> The return type of the function. </param>
        /// <param name="args"> The arguments to the function. </param>
        IRFunctionEmitter& BeginFunction(const std::string& functionName, LLVMType returnType, const NamedVariableTypeList& args);

        /// <summary> Begins an IR function and directs subsequent commands to it. </summary>
        ///
        /// <param name="functionName"> The name of the function. </param>
        /// <param name="returnType"> The return type of the function. </param>
        /// <param name="argsTypes"> The argument types for the function. </param>
        IRFunctionEmitter& BeginFunction(const std::string& functionName, LLVMType returnType, const std::vector<LLVMType>& argTypes);

        /// <summary> Begins an IR function and directs subsequent commands to it. </summary>
        ///
        /// <param name="functionName"> The name of the function. </param>
        /// <param name="returnType"> The return type of the function. </param>
        /// <param name="args"> The arguments to the function. </param>
        IRFunctionEmitter& BeginFunction(const std::string& functionName, LLVMType returnType, const NamedLLVMTypeList& args);

        /// <summary> Emit a "main" function, the entry point to an LLVM program. </summary>
        ///
        /// <returns> An IRFunctionEmitter. </returns>
        IRFunctionEmitter BeginMainFunction();

        /// <summary> Ends the current function. </summary>
        void EndFunction();

        /// <summary> Ends the current function with a return value. </summary>
        ///
        /// <param name="return"> The value the function returns. </param>
        void EndFunction(LLVMValue pReturn);

        //
        // Variable management
        //

        /// <summary> Ensure that the given variable has been declared in IR. </summary>
        LLVMValue EnsureEmitted(Variable& var);

        /// <summary> Ensure that the given variable is loaded into a register. </summary>
        LLVMValue LoadVariable(Variable& var);

        //
        // Variable and Constant creation
        //

        /// <summary> Emit a named, module scoped constant. </summary>
        ///
        /// <param name="type"> The type of the constant. </param>
        /// <param name="name"> The name of the constant. </param>
        /// <param name="value"> The value of the contant. </param>
        ///
        /// <returns> Pointer to the llvm::GlobalVariable that represents the constant. </returns>
        llvm::GlobalVariable* Constant(VariableType type, const std::string& name, double value);

        /// <summary> Emit a named, module scoped constant of a template type. </summary>
        ///
        /// <param name="name"> The name of the constant. </param>
        /// <param name="value"> The value of the constant. </param>
        ///
        /// <returns> Pointer to the llvm::GlobalVariable that represents the constant. </returns>
        template <typename ValueType>
        llvm::GlobalVariable* Constant(const std::string& name, ValueType value);

        /// <summary> Emit a named, module scoped array constant of a template type. </summary>
        ///
        /// <typeparam name="ValueType"> Type of each array entry. </typeparam>
        /// <param name="name"> The array constant name. </param>
        /// <param name="value"> The array constant value. </param>
        ///
        /// <returns> Pointer to the llvm::GlobalVariable that represents the constant. </returns>
        template <typename ValueType>
        llvm::GlobalVariable* ConstantArray(const std::string& name, const std::vector<ValueType>& value);

        /// <summary> Emit a named global variable of the given type. </summary>
        ///
        /// <param name="type"> The variable type. </param>
        /// <param name="name"> The name of the variable. </param>
        /// <param name="isThreadLocal"> Specifies whether the global memory address is unique to each thread </param>
        ///
        /// <returns> Pointer to the llvm::GlobalVariable that represents the variable. </returns>
        llvm::GlobalVariable* Global(VariableType type, const std::string& name, bool isThreadLocal = false);

        /// <summary> Emit a named global variable of the given type. </summary>
        ///
        /// <param name="pType"> Pointer to the runtime value that contains the variable type. </param>
        /// <param name="name"> The name of the variable. </param>
        /// <param name="isThreadLocal"> Specifies whether the global memory address is unique to each thread </param>
        ///
        /// <returns> Pointer to the llvm::GlobalVariable that represents the variable. </returns>
        llvm::GlobalVariable* Global(LLVMType pType, const std::string& name, bool isThreadLocal = false);

        /// <summary> Emit a named global variable of a template type. </summary>
        ///
        /// <typeparam name="ValueType"> The variable type. </typeparam>
        /// <param name="name"> The name of the variable. </param>
        /// <param name="value"> The initial value of the variable. </param>
        /// <param name="isThreadLocal"> Specifies whether the global memory address is unique to each thread </param>
        ///
        /// <returns> Pointer to the llvm::GlobalVariable that represents the variable. </returns>
        template <typename ValueType>
        llvm::GlobalVariable* Global(const std::string& name, ValueType value, bool isThreadLocal = false);

        /// <summary> Emit a named global variable of Pointer type, initialized to nullptr. </summary>
        ///
        /// <typeparam name="ValueType"> The variable type. </typeparam>
        /// <param name="name"> The name of the variable. </param>
        /// <param name="type"> The variable type. </param>
        /// <param name="isThreadLocal"> Specifies whether the global memory address is unique to each thread </param>
        ///
        /// <returns> Pointer to the llvm::GlobalVariable that represents the variable. </returns>
        llvm::GlobalVariable* GlobalPointer(const std::string& name, VariableType type, bool isThreadLocal = false);

        /// <summary> Emit a named global array of the given type and size. </summary>
        ///
        /// <param name="type"> The variable type. </param>
        /// <param name="name"> The name of the variable. </param>
        /// <param name="size"> The array size. </param>
        /// <param name="isThreadLocal"> Specifies whether the global memory address is unique to each thread </param>
        ///
        /// <returns> Pointer to the llvm::GlobalVariable that represents the variable. </returns>
        llvm::GlobalVariable* GlobalArray(VariableType type, const std::string& name, const size_t size, bool isThreadLocal = false);

        /// <summary> Emit a named global array of the given type and size. </summary>
        ///
        /// <param name="name"> The name of the variable. </param>
        /// <param name="pType"> Pointer to the runtime value that contains the variable type. </param>
        /// <param name="size"> The array size. </param>
        /// <param name="isThreadLocal"> Specifies whether the global memory address is unique to each thread </param>
        ///
        /// <returns> Pointer to the llvm::GlobalVariable that represents the variable. </returns>
        llvm::GlobalVariable* GlobalArray(const std::string& name, LLVMType pType, const size_t size, bool isThreadLocal = false);

        /// <summary> Emit a zero-initialized named, module scoped array of a template type. </summary>
        ///
        /// <typeparam name="ValueType"> Type of each array entry. </typeparam>
        /// <param name="name"> The name of the variable. </param>
        /// <param name="size"> The size of the array. </param>
        /// <param name="isThreadLocal"> Specifies whether the global memory address is unique to each thread </param>
        ///
        /// <returns> Pointer to the llvm::GlobalVariable that represents the variable. </returns>
        template <typename ValueType>
        llvm::GlobalVariable* GlobalArray(const std::string& name, size_t size, bool isThreadLocal = false);

        /// <summary> Emit a named, module scoped array of a template type. </summary>
        ///
        /// <typeparam name="ValueType"> Type of each array entry. </typeparam>
        /// <param name="name"> The name of the variable. </param>
        /// <param name="value"> The value of the array. </param>
        /// <param name="isThreadLocal"> Specifies whether the global memory address is unique to each thread </param>
        ///
        /// <returns> Pointer to the llvm::GlobalVariable that represents the variable. </returns>
        template <typename ValueType>
        llvm::GlobalVariable* GlobalArray(const std::string& name, const std::vector<ValueType>& value, bool isThreadLocal = false);

        //
        // Functions
        //

        /// <summary> Emit a declaration of an external function with the given return type. </summary>
        ///
        /// <param name="name"> The function name. </param>
        /// <param name="returnType"> The function return type. </param>
        LLVMFunction DeclareFunction(const std::string& name, VariableType returnType);

        /// <summary> Emit a declaration of an external function with the given return type and arguments. </summary>
        ///
        /// <param name="name"> The function name. </param>
        /// <param name="returnType"> The function return type. </param>
        /// <param name="arguments"> The function arguments. </param>
        LLVMFunction DeclareFunction(const std::string& name, VariableType returnType, const VariableTypeList& arguments);

        /// <summary>
        /// Emit a declaration of an external function with the given return type and named arguments.
        /// </summary>
        ///
        /// <param name="name"> The function name. </param>
        /// <param name="returnType"> The function return type. </param>
        /// <param name="arguments"> The named function arguments. </param>
        LLVMFunction DeclareFunction(const std::string& name, VariableType returnType, const NamedVariableTypeList& arguments);

        /// <summary> Emit a declaration of an external function with the given function signature. </summary>
        ///
        /// <param name="name"> The function name. </param>
        /// <param name="functionType"> The function type. </param>
        LLVMFunction DeclareFunction(const std::string& name, llvm::FunctionType* functionType);

        /// <summary> Check if a given function name exists. </summary>
        ///
        /// <param name="name"> The function name. </param>
        ///
        /// <returns> True if the function name exists, false if not. </returns>
        bool HasFunction(const std::string& name) const;

        /// <summary> Get an emitted or declared function with the given name. </summary>
        ///
        /// <param name="name"> The function name. </param>
        ///
        /// <returns> Pointer to an llvm::Function that represents the requested function, or nullptr if it doesn't exist. </returns>
        LLVMFunction GetFunction(const std::string& name) const;

        /// <summary> Get an LLVM intrinsic function taking no arguments with the given id. </summary>
        ///
        /// <param name="id"> The intrinsic function identifier. </param>
        ///
        /// <returns> Pointer to an llvm::Function that represents the requested function. </returns>
        LLVMFunction GetIntrinsic(llvm::Intrinsic::ID id);

        /// <summary> Get an LLVM intrinsic function with the given id and signature. </summary>
        ///
        /// <param name="id"> The intrinsic function identifier. </param>
        /// <param name="arguments"> The function arguments. </param>
        ///
        /// <returns> Pointer to an llvm::Function that represents the requested function. </returns>
        LLVMFunction GetIntrinsic(llvm::Intrinsic::ID id, const std::initializer_list<VariableType>& arguments);

        /// <summary> Get an LLVM intrinsic function with the given id and signature. </summary>
        ///
        /// <param name="id"> The intrinsic function identifier. </param>
        /// <param name="arguments"> The function arguments. </param>
        ///
        /// <returns> Pointer to an llvm::Function that represents the requested function. </returns>
        LLVMFunction GetIntrinsic(llvm::Intrinsic::ID id, const std::initializer_list<LLVMType>& arguments);

        //
        // Types
        //

        /// <summary> Emit a module-scoped struct with the given fields. </summary>
        ///
        /// <param name="name"> The struct name. </param>
        /// <param name="fields"> The struct fields. </param>
        ///
        /// <returns> Pointer to the llvm::StructType that represents the emitted structure. </returns>
        llvm::StructType* GetOrCreateStruct(const std::string& name, const NamedVariableTypeList& fields);

        /// <summary> Emit a module-scoped struct with the given fields. </summary>
        ///
        /// <param name="name"> The struct name. </param>
        /// <param name="fields"> The struct fields. </param>
        ///
        /// <returns> Pointer to the llvm::StructType that represents the emitted structure. </returns>
        llvm::StructType* GetOrCreateStruct(const std::string& name, const NamedLLVMTypeList& fields);

        /// <summary> Emit a module-scoped struct with the given fields. </summary>
        ///
        /// <param name="name"> The struct name. </param>
        /// <param name="fields"> The struct fields. </param>
        ///
        /// <returns> Pointer to the llvm::StructType that represents the emitted structure. </returns>
        llvm::StructType* GetOrCreateStruct(const std::string& name, const LLVMTypeList& fields);

        /// <summary> Emit a module-scoped anonymous struct with the given field types. </summary>
        ///
        /// <param name="fieldTypes"> The struct field types. </param>
        /// <param name="packed"> If `true`, the fields will be packed together without any padding. </param>
        ///
        /// <returns> Pointer to the llvm::StructType that represents the emitted structure. </returns>
        llvm::StructType* GetAnonymousStructType(const LLVMTypeList& fieldTypes, bool packed = false);

        /// <summary> Gets a declaration of a Struct with the given name.</summary>
        ///
        /// <param name="name"> The struct name. </param>
        ///
        /// <returns> Pointer to a llvm::StructType that represents the declared struct. </returns>
        llvm::StructType* GetStruct(const std::string& name);

        /// <summary> Find a FunctionDeclaration by name.  Use this to add comments.</summary>
        ///
        /// <param name="name"> The function name. </param>
        ///
        /// <returns> Pointer to FunctionDeclaration. </returns>
        FunctionDeclaration& GetFunctionDeclaration(const std::string& name) override;

        /// <summary> Get list of all functions defined in this module.</summary>
        ///
        /// <returns> List of names. </returns>
        std::vector<std::string> GetFunctionNames() const;

        /// <summary> Get list of all callback functions defined in this module.</summary>
        ///
        /// <returns> List of names. </returns>
        std::vector<std::string> GetCallbackFunctionNames() const;

        //
        // Code annotation
        //

        /// <summary> Gets any preprocessor definitions set for the module. </summary>
        ///
        /// <returns> The preprocessor definitions set for the module, as a vector of (name,value) pairs. </returns>
        const std::vector<std::pair<std::string, std::string>>& GetPreprocessorDefinitions() const { return _preprocessorDefinitions; };

        /// <summary> Adds a preprocessor definition to the header file. </summary>
        ///
        /// <param name="name"> The name of the preprocessor symbol to define. </param>
        /// <param name="value"> The value to set for the symbol. </param>
        void AddPreprocessorDefinition(const std::string& name, const std::string& value);

        /// <summary> Indicates if the module has the associated metadata. </summary>
        ///
        /// <param name="tag"> The metadata tag. </param>
        ///
        /// <returns> `true` if the module has the metadata associated with it. </returns>
        bool HasMetadata(const std::string& tag) const override;

        /// <summary> Indicates if a given function has the associated metadata. </summary>
        ///
        /// <param name="functionName"> The name of the function. </param>
        /// <param name="tag"> The metadata tag. </param>
        ///
        /// <returns> `true` if the function has the metadata associated with it. </returns>
        bool HasFunctionMetadata(const std::string& functionName, const std::string& tag) const override;

        /// <summary> Gets the metadata associated with the module. </summary>
        ///
        /// <param name="tag"> The metadata tag. </param>
        ///
        /// <returns> The metadata values, as a vector of strings. </returns>
        std::vector<std::vector<std::string>> GetMetadata(const std::string& tag) override;

        /// <summary> Gets the metadata associated with a given function. </summary>
        ///
        /// <param name="functionName"> The name of the function for function-level metadata. </param>
        /// <param name="tag"> The metadata tag. </param>
        ///
        /// <returns> The metadata values, as a vector of strings. </returns>
        std::vector<std::string> GetFunctionMetadata(const std::string& functionName, const std::string& tag) override;

        /// <summary> Associates metadata with the module. </summary>
        ///
        /// <param name="tag"> The metadata tag. </param>
        /// <param name="content"> Optional metadata value. </param>
        /// <remarks> To insert well-known metadata, prefer the "IncludeInXXX" metadata methods. </remarks>
        void InsertMetadata(const std::string& tag, const std::vector<std::string>& value = { "" }) override;

        /// <summary> Associates metadata with a given function. </summary>
        ///
        /// <param name="functionName"> The name of the function. </param>
        /// <param name="tag"> The metadata tag. </param>
        /// <param name="content"> Optional metadata value. </param>
        /// <remarks> To insert well-known metadata, prefer the "IncludeInXXX" metadata methods. </remarks>
        void InsertFunctionMetadata(const std::string& functionName, const std::string& tag, const std::vector<std::string>& value = { "" }) override;

        //
        // Code output / input
        //

        /// <summary> Output the compiled module to the given file, using file extension to determine output format. </summary>
        ///
        /// <param name="filePath"> The path of the file to write to. </param>
        using ModuleEmitter::WriteToFile;

        /// <summary> Output the compiled module to an output file with the given format. </summary>
        ///
        /// <param name="filePath"> Full pathname of the file. </param>
        /// <param name="format"> The format of the output. </param>
        void WriteToFile(const std::string& filePath, ModuleOutputFormat format) override;

        /// <summary> Output the compiled module to an output file with the given format. </summary>
        ///
        /// <param name="filePath"> Full pathname of the file. </param>
        /// <param name="format"> The format of the output. </param>
        /// <param name="options"> Options to control how machine code is generated during output. </params>
        void WriteToFile(const std::string& filePath, ModuleOutputFormat format, const MachineCodeOutputOptions& options);

        /// <summary> Output the compiled module to an output stream with the given format. </summary>
        ///
        /// <param name="stream"> The stream to write to. </param>
        /// <param name="format"> The format of the output. </param>
        void WriteToStream(std::ostream& stream, ModuleOutputFormat format) override;

        /// <summary> Output the compiled module to an output stream with the given format. </summary>
        ///
        /// <param name="stream"> The stream to write to. </param>
        /// <param name="format"> The format of the output. </param>
        /// <param name="options"> Options to control how machine code is generated during output. </params>
        void WriteToStream(std::ostream& stream, ModuleOutputFormat format, const MachineCodeOutputOptions& options);

        /// <summary> Load LLVM IR text into this module. </summary>
        ///
        /// <param name="text"> The IR text. </param>
        void LoadIR(const std::string& text);

        /// <summary> Load LLVM IR from the given stream </summary>
        ///
        /// <param name="stream"> The stream that serves as the input </param>
        void LoadIR(std::istream& stream);

        /// <summary> Load LLVM IR from the file into this module. </summary>
        ///
        /// <param name="filename"> The name of the file containing the IR </param>
        void LoadIRFromFile(const std::string& filename);

        /// <summary> Load Assembler text into this module. </summary>
        ///
        /// <param name="text"> The IR text. </param>
        void LoadAsm(const std::string& text);

        /// <summary> Load Assembler from the given stream </summary>
        ///
        /// <param name="stream"> The stream that serves as the input </param>
        void LoadAsm(std::istream& stream);

        /// <summary> Load Assembler from the file into this module. </summary>
        ///
        /// <param name="filename"> The name of the file containing the Assembler text </param>
        void LoadAsmFromFile(const std::string& filename);

        //
        // Optimization
        //

        /// <summary> Optimize this module using the given optimizer. </summary>
        ///
        /// <param name="optimizer"> The optimizer. </param>
        void Optimize(IROptimizer& optimizer);

        /// <summary>
        /// Get the target machine and arch for this module. The target machine aids the system in optimizations and
        /// Jitting etc.
        /// </summary>
        ///
        /// <returns>
        /// An llvm::TargetMachine object that describes the target machine. It may return `nullptr` if a specific
        /// target machine can't be found. It isn't an error if no target machine is found.
        ///</returns>
        llvm::TargetMachine* GetTargetMachine();

        //
        // Helpers, standard C Runtime functions, and debug support
        //

        /// <summary> For platforms that don't support printf, this is a lower level debug print function.
        /// The calling application needs to define this function, it's signature will be:
        ///    void DebugPrint(const char* message);
        /// If running in JIT mode the function will be automatically defined by the IRExecutionEngine.
        /// </summary>
        LLVMFunction DeclareDebugPrint();

        /// <summary> Call the DebugPrint function declared by DeclareDebugPrint. The message will become
        /// stored as a global literal with a uniquely generated name.
        /// </summary>
        void DebugPrint(std::string message);

        /// <summary> Emit declaration of extern printf. </summary>
        void DeclarePrintf();

        /// <summary> Emit declaration of extern malloc. </summary>
        void DeclareMalloc();

        /// <summary> Emit declaration of extern free. </summary>
        void DeclareFree();

        /// <summary> Add a main function into which you will inject debugging code. </summary>
        IRFunctionEmitter BeginMainDebugFunction();

        /// <summary> Get the diagnostic handler. </summary>
        IRDiagnosticHandler& GetDiagnosticHandler();

        /// <summary> Check the module for errors </summary>
        ///
        /// <returns> `true` if there are errors </returns>
        bool CheckForErrors();

        /// <summary> Check the module for errors and report them to an output stream </summary>
        ///
        /// <param name="stream"> The stream to write to. </param>
        ///
        /// <returns> `true` if there are errors </returns>
        bool CheckForErrors(std::ostream& stream);

        /// <summary> Emit LLVM IR to std::out for debugging. </summary>
        void DebugDump() const;

        //
        // low-level LLVM-related functionality
        //

        /// <summary> Gets an owning pointer to the underlying llvm::Module. </summary>
        ///
        /// <returns> Unique pointer to the underlying llvm::Module. </returns>
        /// <remarks> After calling this, the module will not be valid. </remarks>
        std::unique_ptr<llvm::Module> TransferOwnership();

        /// <summary> Gets a pointer to the underlying llvm::Module. </summary>
        ///
        /// <returns> Pointer to the underlying llvm::Module. </returns>
        llvm::Module* GetLLVMModule() const { return _llvmModule.get(); }

        /// <summary> Gets the LLVM data layout object for the current module. </summary>
        const llvm::DataLayout& GetTargetDataLayout() const;

        /// <summary> Can this module emitter still be used to add functions to the module? </summary>
        ///
        /// <returns> true if active, false if not. </returns>
        bool IsActive() const { return (_llvmModule != nullptr); }

        /// <summary> Gets a reference to the underlying llvm context. </summary>
        ///
        /// <returns> Reference to the underlying llvm context. </returns>
        llvm::LLVMContext& GetLLVMContext() { return *_llvmContext; }

        //
        // Metadata
        //

        /// <summary> Tags a type to be declared in a C/C++ header. </summary>
        ///
        /// <param name="typeName"> The name of the type. </param>
        void IncludeTypeInHeader(const std::string& typeName);

        /// <summary> Tags a callback function to be included in the SWIG interface. </summary>
        ///
        /// <param name="functionName"> The function name. </param>
        /// <param name="nodeName"> The node name. </param>
        void IncludeInCallbackInterface(const std::string& functionName, const std::string& nodeName);

        //
        // Module initialization / finalization
        //

        /// <summary> Adds an initialization function to run before any (non-initialization) application code. </summary>
        ///
        /// <param name="function"> The function to call. Must be of type `void()`. </param>
        /// <param name="priority"> The priority for this initialization function. Initialization functions are called in increasing order of priority. The default value is LLVM's default priority. </param>
        /// <param name="forData"> Optional global constant that this function initializes. If the data is optimized away, then the initialization function will be also. </param>
        void AddInitializationFunction(LLVMFunction function, int priority = 65536, llvm::Constant* forData = nullptr);

        /// <summary> Adds an initialization function to run before any (non-initialization) application code. </summary>
        ///
        /// <param name="function"> The function to call. Must be of type `void()`. </param>
        /// <param name="priority"> The priority for this initialization function. Initialization functions are called in increasing order of priority. The default value is LLVM's default priority.</param>
        /// <param name="forData"> Optional global constant that this function initializes. If the data is optimized away, then the initialization function will be also. </param>
        void AddInitializationFunction(IRFunctionEmitter& function, int priority = 65536, llvm::Constant* forData = nullptr);

        /// <summary> Adds a finalization function to run after any application code. </summary>
        ///
        /// <param name="function"> The function to call. Must be of type `void()`. </param>
        /// <param name="priority"> The priority for this finalization function. Finalization functions are called in increasing order of priority. The default value is LLVM's default priority. </param>
        /// <param name="forData"> Optional global constant that this function is for. If the data is optimized away, then the finalization function will be also. </param>
        void AddFinalizationFunction(LLVMFunction function, int priority = 65536, llvm::Constant* forData = nullptr);

        /// <summary> Adds a finalization function to run after any application code. </summary>
        ///
        /// <param name="function"> The function to call. Must be of type `void()`. </param>
        /// <param name="priority"> The priority for this finalization function. Finalization functions are called in increasing order of priority. The default value is LLVM's default priority. </param>
        /// <param name="forData"> Optional global constant that this function is for. If the data is optimized away, then the finalization function will be also. </param>
        void AddFinalizationFunction(IRFunctionEmitter& function, int priority = 65536, llvm::Constant* forData = nullptr);

        /// <summary> Return the typed CallbackRegistry object that is used to manage any std::functions defined
        /// on any SourceNodes or SinkNodes in the graph. </summary>
        template <typename ElementType>
        ell::utilities::CallbackRegistry<ElementType>& GetCallbackRegistry() const;

        /// <summary> Returns true if the CallbackRegistry objects contain some functions. </summary>
        bool HasCallbackFunctions() const;

    private:
        friend class IRFunctionEmitter;

        //
        // Internal variable and function creation implementation
        //

        /// Gets an emitted variable with the given name and scope.
        LLVMValue GetEmittedVariable(const VariableScope scope, const std::string& name);

        /// Emit IR for a variable.
        LLVMValue EmitVariable(Variable& var);

        // Templated version implementing above
        template <typename T>
        LLVMValue EmitVariable(Variable& var);

        /// Emit IR for a scalar variable.
        template <typename T>
        LLVMValue EmitScalar(Variable& var);

        /// Emit IR for a vector variable.
        template <typename T>
        LLVMValue EmitVector(Variable& var);

        /// Emit IR for a constant.
        template <typename T>
        LLVMValue EmitLiteral(LiteralVariable<T>& var);

        /// Emit IR for a Global scalar with an initial value.
        template <typename T>
        LLVMValue EmitGlobal(InitializedScalarVariable<T>& var);

        /// Emit IR for a stack scalar variable.
        template <typename T>
        LLVMValue EmitLocal(ScalarVariable<T>& var);

        /// Emit IR for a stack scalar variable with an initial value.
        template <typename T>
        LLVMValue EmitLocal(InitializedScalarVariable<T>& var);

        /// Emit IR for a constant vector.
        template <typename T>
        LLVMValue EmitLiteralVector(LiteralVectorVariable<T>& var);

        /// Emit IR for a global vector. Global is initialized to zero.
        template <typename T>
        LLVMValue EmitGlobalVector(VectorVariable<T>& var);

        /// Emit IR for a global vector with initial values.
        template <typename T>
        LLVMValue EmitGlobalVector(InitializedVectorVariable<T>& var);

        /// Emit IR for a reference to an element in a vector.
        template <typename T>
        LLVMValue EmitRef(VectorElementVariable<T>& var);

        IRFunctionEmitter Function(const std::string& name, VariableType returnType, bool isPublic = false);
        IRFunctionEmitter Function(const std::string& name, VariableType returnType, const VariableTypeList& arguments, bool isPublic = false);
        IRFunctionEmitter Function(const std::string& name, VariableType returnType, const NamedVariableTypeList& arguments, bool isPublic = false);
        IRFunctionEmitter Function(const std::string& name, LLVMType returnType, const NamedVariableTypeList& arguments, bool isPublic = false);
        IRFunctionEmitter Function(const std::string& name, VariableType returnType, const std::initializer_list<VariableType>& arguments, bool isPublic = false);
        IRFunctionEmitter Function(const std::string& name, LLVMType returnType, const std::vector<LLVMType>& argTypes, bool isPublic = false);
        IRFunctionEmitter Function(const std::string& name, LLVMType returnType, const NamedLLVMTypeList& arguments, bool isPublic = false);
        IRFunctionEmitter Function(const std::string& name, LLVMType returnType, const FunctionArgumentList& arguments, bool isPublic = false);

        // Associate metadata with a given function.
        // Note: to insert well-known metadata, prefer the "IncludeInXXX" metadata methods.
        void InsertFunctionMetadata(LLVMFunction function, const std::string& tag, const std::vector<std::string>& value = { "" });

        // Get a reference to the thread pool
        IRThreadPool& GetThreadPool() { return *_threadPool; }

        // Actual code output implementations
        void WriteHeader(std::ostream& stream);
        void WriteToLLVMStream(llvm::raw_ostream& stream, ModuleOutputFormat format, MachineCodeOutputOptions options);

        //
        // Lower-level internal functions
        //
        void SetCompilerOptions(const CompilerOptions& parameters) override;
        llvm::GlobalVariable* AddGlobal(const std::string& name, LLVMType pType, llvm::Constant* pInitial, bool isConst, bool isThreadLocal = false);
        IRFunctionEmitter Function(const std::string& name, VariableType returnType, const VariableTypeList* pArguments, bool isPublic);
        llvm::Function::LinkageTypes Linkage(bool isPublic);
        llvm::ConstantAggregateZero* ZeroInitializer(LLVMType pType);
        static void CompleteCompilerOptions(CompilerOptions& parameters);
        void SetTargetTriple(const std::string& triple);
        MachineCodeOutputOptions GetMachineCodeOutputOptions() const;

        //
        // Data members
        //
        std::unique_ptr<llvm::LLVMContext> _llvmContext; // LLVM global context
        std::unique_ptr<llvm::Module> _llvmModule; // The LLVM Module being emitted
        std::unique_ptr<IRDiagnosticHandler> _diagnosticHandler = nullptr;
        std::unique_ptr<IREmitter> _emitter;
        std::stack<std::pair<IRFunctionEmitter, llvm::IRBuilder<>::InsertPoint>> _functionStack; // contains the location we were emitting code into when we paused to emit a new function

        IRValueTable _literals; // Symbol table - name to literals
        IRValueTable _globals; // Symbol table - name to global variables
        std::unique_ptr<IRRuntime> _runtime; // Manages emission of runtime functions
        std::unique_ptr<IRThreadPool> _threadPool; // A pool of worker threads -- gets initialized the first time it's used (?)
        std::unique_ptr<IRProfiler> _profiler;
        int _globalStringIndex = 0;

        // Info to modify how code is written out
        std::vector<std::pair<std::string, std::string>> _preprocessorDefinitions;
        std::vector<std::string> _resetFunctions;
        std::map<std::string, FunctionDeclaration> _functions;

        ell::utilities::CallbackRegistry<float> _floatCallbacks;
        ell::utilities::CallbackRegistry<double> _doubleCallbacks;
        ell::utilities::CallbackRegistry<int> _intCallbacks;
        ell::utilities::CallbackRegistry<int64_t> _int64Callbacks;
        ell::utilities::CallbackRegistry<bool> _boolCallbacks;
    };

    //
    // Functions
    //

    /// <summary> Convenience function for creating an `IRModuleEmitter` with the default compiler parameters, set up for the host environment </summary>
    IRModuleEmitter MakeHostModuleEmitter(const std::string moduleName);
} // namespace emitters
} // namespace ell

#pragma region implementation

namespace ell
{
namespace emitters
{
    //
    // Public methods
    //
    template <typename ValueType>
    llvm::GlobalVariable* IRModuleEmitter::Constant(const std::string& name, ValueType value)
    {
        return AddGlobal(name, GetIREmitter().Type(GetVariableType<ValueType>()), GetIREmitter().Literal(value), true);
    }

    template <typename ValueType>
    llvm::GlobalVariable* IRModuleEmitter::Global(const std::string& name, ValueType value, bool isThreadLocal)
    {
        return AddGlobal(name, GetIREmitter().Type(GetVariableType<ValueType>()), GetIREmitter().Literal(value), false, isThreadLocal);
    }

    template <typename ValueType>
    llvm::GlobalVariable* IRModuleEmitter::ConstantArray(const std::string& name, const std::vector<ValueType>& value)
    {
        return AddGlobal(name, GetIREmitter().ArrayType(GetVariableType<ValueType>(), value.size()), GetIREmitter().Literal(value), true);
    }

    template <typename ValueType>
    llvm::GlobalVariable* IRModuleEmitter::GlobalArray(const std::string& name, size_t size, bool isThreadLocal)
    {
        return GlobalArray(GetVariableType<ValueType>(), name, size, isThreadLocal);
    }

    template <typename ValueType>
    llvm::GlobalVariable* IRModuleEmitter::GlobalArray(const std::string& name, const std::vector<ValueType>& value, bool isThreadLocal)
    {
        return AddGlobal(name, GetIREmitter().ArrayType(GetVariableType<ValueType>(), value.size()), GetIREmitter().Literal(value), false, isThreadLocal);
    }

    //
    // Private methods
    //

    template <typename T>
    LLVMValue IRModuleEmitter::EmitVariable(Variable& var)
    {
        // TODO: have a more specific check to see if the variable is mapped to a port, rather than if it's a function input/output
        if (var.IsScalar() && (var.Scope() != VariableScope::input && var.Scope() != VariableScope::output))
        {
            return EmitScalar<T>(var);
        }
        else if (var.IsVector())
        {
            return EmitVector<T>(var);
        }
        else
        {
            throw EmitterException(EmitterError::variableTypeNotSupported);
        }
    }

    template <typename T>
    LLVMValue IRModuleEmitter::EmitScalar(Variable& var)
    {
        LLVMValue pVal = nullptr;
        switch (var.Scope())
        {
        case VariableScope::literal:
            pVal = EmitLiteral<T>(static_cast<LiteralVariable<T>&>(var));
            _literals.Add(var.EmittedName(), pVal);
            break;

        case VariableScope::local:
            if (var.IsVectorRef())
            {
                pVal = EmitRef<T>(static_cast<VectorElementVariable<T>&>(var));
            }
            else if (var.HasInitValue())
            {
                pVal = EmitLocal<T>(static_cast<InitializedScalarVariable<T>&>(var));
            }
            else
            {
                pVal = EmitLocal<T>(static_cast<ScalarVariable<T>&>(var));
            }
            break;

        case VariableScope::global:
            pVal = EmitGlobal<T>(static_cast<InitializedScalarVariable<T>&>(var));
            break;

        default:
            throw EmitterException(EmitterError::variableScopeNotSupported);
        }
        return pVal;
    }

    template <typename T>
    LLVMValue IRModuleEmitter::EmitVector(Variable& var)
    {
        LLVMValue pVal = nullptr;
        switch (var.Scope())
        {
        case VariableScope::literal:
            pVal = EmitLiteralVector<T>(static_cast<LiteralVectorVariable<T>&>(var));
            _literals.Add(var.EmittedName(), pVal);
            break;

        case VariableScope::global:
            if (var.HasInitValue())
            {
                pVal = EmitGlobalVector<T>(static_cast<InitializedVectorVariable<T>&>(var));
            }
            else
            {
                pVal = EmitGlobalVector<T>(static_cast<VectorVariable<T>&>(var));
            }
            _globals.Add(var.EmittedName(), pVal);
            break;

        default:
            throw EmitterException(EmitterError::variableScopeNotSupported);
        }
        assert(pVal != nullptr);
        return pVal;
    }

    template <typename T>
    LLVMValue IRModuleEmitter::EmitLiteral(LiteralVariable<T>& var)
    {
        auto& currentFunction = GetCurrentFunction();
        LLVMValue pVar = currentFunction.Literal(var.Data());
        return pVar;
    }

    template <typename T>
    LLVMValue IRModuleEmitter::EmitGlobal(InitializedScalarVariable<T>& var)
    {
        auto& currentFunction = GetCurrentFunction();
        LLVMValue pVal = nullptr;
        if (var.IsMutable())
        {
            pVal = Global(var.Type(), var.EmittedName());
            currentFunction.Store(pVal, currentFunction.Literal(var.Data()));
        }
        else
        {
            pVal = Constant(var.Type(), var.EmittedName(), var.Data());
        }
        return pVal;
    }

    template <typename T>
    LLVMValue IRModuleEmitter::EmitLocal(ScalarVariable<T>& var)
    {
        auto& currentFunction = GetCurrentFunction();
        return currentFunction.EmittedVariable(var.Type(), var.EmittedName());
    }

    template <typename T>
    LLVMValue IRModuleEmitter::EmitLocal(InitializedScalarVariable<T>& var)
    {
        auto& currentFunction = GetCurrentFunction();
        LLVMValue pVar = currentFunction.EmittedVariable(var.Type(), var.EmittedName());
        currentFunction.Store(pVar, currentFunction.Literal(var.Data()));
        return pVar;
    }

    template <typename T>
    LLVMValue IRModuleEmitter::EmitLiteralVector(LiteralVectorVariable<T>& var)
    {
        return ConstantArray(var.EmittedName(), var.Data());
    }

    template <typename T>
    LLVMValue IRModuleEmitter::EmitGlobalVector(VectorVariable<T>& var)
    {
        return GlobalArray(GetVariableType<T>(), var.EmittedName(), var.Dimension());
    }

    template <typename T>
    LLVMValue IRModuleEmitter::EmitGlobalVector(InitializedVectorVariable<T>& var)
    {
        return GlobalArray(var.EmittedName(), var.Data());
    }

    template <typename T>
    LLVMValue IRModuleEmitter::EmitRef(VectorElementVariable<T>& var)
    {
        auto& currentFunction = GetCurrentFunction();
        LLVMValue pSrcVar = EnsureEmitted(var.Src());
        return currentFunction.PtrOffsetA(pSrcVar, currentFunction.Literal(var.Offset()), var.EmittedName());
    }
} // namespace emitters
} // namespace ell

#pragma endregion implementation
