////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRCompiledMap.h (model)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "CompiledMap.h"
#include "IRModelProfiler.h"
#include "InputNode.h"
#include "Map.h"
#include "Node.h"
#include "OutputPort.h"
#include "PortElements.h"

#include <emitters/include/IRExecutionEngine.h>
#include <emitters/include/IRModuleEmitter.h>
#include <emitters/include/ModuleEmitter.h>

#include <utilities/include/Boolean.h>
#include <utilities/include/TypeName.h>

#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

namespace ell
{
namespace model
{
    /// <summary> A map that can be compiled </summary>
    class IRCompiledMap : public CompiledMap
    {
        using Boolean = utilities::Boolean;

    public:
        IRCompiledMap(const IRCompiledMap&) = delete;
        IRCompiledMap(IRCompiledMap&& other);
        IRCompiledMap& operator=(const IRCompiledMap&) = delete;

        /// <summary> Clone the compiled model </summary>
        IRCompiledMap Clone();

        /// <summary> Output the compiled model to the given file </summary>
        ///
        /// <param name="filePath"> The file to write to </param>
        void WriteCode(const std::string& filePath) const override;

        /// <summary> Output the compiled model to the given file with the given format </summary>
        ///
        /// <param name="filePath"> The file to write to </param>
        /// <param name="format"> The format to write out </param>
        void WriteCode(const std::string& filePath, emitters::ModuleOutputFormat format) const override;

        /// <summary> Output a 'C'-style function prototype for the compiled function </summary>
        ///
        /// <param name="filePath"> The path to the file to write </param>
        /// <param name="format"> The format to write out (cHeader or Swig interface) </param>
        void WriteCodeHeader(const std::string& filePath, emitters::ModuleOutputFormat format) const override;

        /// <summary> Output a 'C'-style function prototype for the compiled function </summary>
        ///
        /// <param name="streawm"> The stream to write the prototype to </param>
        /// <param name="format"> The format to write out (cHeader or Swig interface) </param>
        void WriteCodeHeader(std::ostream& stream, emitters::ModuleOutputFormat format) const override;

        /// <summary> Output the compiled model to an output stream with the given format </summary>
        ///
        /// <param name="stream"> The stream to write to </param>
        /// <param name="format"> The format to write out </param>
        void WriteCode(std::ostream& stream, emitters::ModuleOutputFormat format) const override;

        /// <summary> Output a 'C'-style function prototype for the compiled function </summary>
        ///
        /// <returns> A string with the function prototype </returns>
        std::string GetCodeHeaderString() const override;

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "IRCompiledMap"; }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

        /// <summary> Can this compiled map be used? </summary>
        ///
        /// <returns> true if active, false if not. </returns>
        bool IsValid() const override;

        /// <summary> Gets a reference to the underlying IRModuleEmitter. </summary>
        ///
        /// <returns> Reference to an IRModuleEmitter. </returns>
        emitters::IRModuleEmitter& GetModule() { return _module; }

        /// <summary> Gets a reference to the underlying jitter. </summary>
        ///
        /// <returns> The jitter. </returns>
        emitters::IRExecutionEngine& GetJitter();

        //
        // Node profiling support
        //

        /// <summary> Get a pointer to the performance counters struct for the whole model. </summary>
        PerformanceCounters* GetModelPerformanceCounters();

        /// <summary> Print a summary of the performance for the model. </summary>
        void PrintModelProfilingInfo();

        /// <summary> Reset the performance summary for the model to zero. </summary>
        void ResetModelProfilingInfo();

        /// <summary> Get the number of nodes that have profiling information. </summary>
        int GetNumProfiledNodes();

        /// <summary> Get a pointer to the info struct for a node. </summary>
        ///
        /// <param name="nodeIndex"> the index of the node. </param>
        NodeInfo* GetNodeInfo(int nodeIndex);

        /// <summary> Get a pointer to the performance counters struct for a node. </summary>
        ///
        /// <param name="nodeIndex"> the index of the node. </param>
        PerformanceCounters* GetNodePerformanceCounters(int nodeIndex);

        /// <summary> Print a summary of the performance for the nodes. </summary>
        void PrintNodeProfilingInfo();

        /// <summary> Reset the performance counters for all the nodes to zero. </summary>
        void ResetNodeProfilingInfo();

        /// <summary> Get the number of node types that have profiling information. </summary>
        int GetNumProfiledNodeTypes();

        /// <summary> Get a pointer to the info struct for a node type. </summary>
        ///
        /// <param name="nodeIndex"> the index of the node type. </param>
        NodeInfo* GetNodeTypeInfo(int nodeIndex);

        /// <summary> Get a pointer to the aggregated performance counters struct for a node type. </summary>
        ///
        /// <param name="nodeIndex"> the index of the node type. </param>
        PerformanceCounters* GetNodeTypePerformanceCounters(int nodeIndex);

        /// <summary> Get a pointer to the named global array. </summary>
        ///
        /// <param name="name"> name of the global. </param>
        template <typename ElementType>
        ElementType* GetGlobalValuePointer(const std::string& name);

        /// <summary> Print a summary of the performance for the node types. </summary>
        void PrintNodeTypeProfilingInfo();

        /// <summary> Reset the performance counters for all the node types to zero. </summary>
        void ResetNodeTypeProfilingInfo();

        //
        // Low-level region profiling support
        //

        /// <summary> Get the number of regions that have profiling information. </summary>
        int GetNumProfileRegions();

        /// <summary> Get a pointer to the info struct for a region. </summary>
        ///
        /// <param name="regionIndex"> the index of the region. </param>
        emitters::ProfileRegionInfo* GetRegionProfilingInfo(int regionIndex);

        /// <summary> Reset the performance summary for the model to zero. </summary>
        void ResetRegionProfilingInfo();

        //
        // Just-in-time compilation functions
        //

        /// <summary> Force jitting to finish so you can time execution without jit cost. </summary>
        void FinishJitting();

        /// <summary> Set a context object to use in the predict call.  Do not call this if you have
        /// already provided std::functions on your SourceNodes and SinkNodes.  </summary>
        void SetContext(void* context);

        /// <summary> Get the context object to use in the predict call </summary>
        void* GetContext() const { return _context; }

        /// <summary> This exposes another way to call Compute when the model has multiple inputs and outputs
        /// These void* pointers must point to pre-allocated buffers of the right type (float*, double*, etc)
        /// and size. </summary>
        ///
        /// <param name="inputs"> A vector containing all the input buffers. </param>
        /// <param name="outputs"> A vector containing all the output buffers. </param>
        void ComputeMultiple(const std::vector<void*>& inputs, const std::vector<void*>& outputs) override;

        /// <summary> Reset any model state. </summary>
        void Reset() override;

    protected:
        void WriteCode(const std::string& filePath, emitters::ModuleOutputFormat format, emitters::MachineCodeOutputOptions options) const;
        void WriteCode(std::ostream& stream, emitters::ModuleOutputFormat format, emitters::MachineCodeOutputOptions options) const;

        void SetNodeInput(model::InputNode<bool>* node, const std::vector<bool>& inputValues) override;
        void SetNodeInput(model::InputNode<int>* node, const std::vector<int>& inputValues) override;
        void SetNodeInput(model::InputNode<int64_t>* node, const std::vector<int64_t>& inputValues) override;
        void SetNodeInput(model::InputNode<float>* node, const std::vector<float>& inputValues) override;
        void SetNodeInput(model::InputNode<double>* node, const std::vector<double>& inputValues) override;

        std::vector<bool> ComputeBoolOutput(const model::PortElementsBase& outputs) override;
        std::vector<int> ComputeIntOutput(const model::PortElementsBase& outputs) override;
        std::vector<int64_t> ComputeInt64Output(const model::PortElementsBase& outputs) override;
        std::vector<float> ComputeFloatOutput(const model::PortElementsBase& outputs) override;
        std::vector<double> ComputeDoubleOutput(const model::PortElementsBase& outputs) override;

    private:
        friend class IRMapCompiler;

        IRCompiledMap(Map map, const std::string& functionName, const MapCompilerOptions& options, emitters::IRModuleEmitter& module, bool verifyJittedModule);

        void EnsureExecutionEngine();
        void SetComputeFunction();
        template <typename InputType>
        void SetComputeFunctionForInputType();

        template <typename InputType>
        using ComputeFunction = std::function<void(void*, const InputType*)>;

        emitters::IRModuleEmitter& _module;
        std::string _moduleName;

        std::unique_ptr<emitters::IRExecutionEngine> _executionEngine;
        bool _verifyJittedModule = true;
        void* _context = nullptr;

        void* InternalGetContext();
        void ResolveCallbacks();

        template <typename T>
        using Vector = std::vector<std::conditional_t<std::is_same_v<bool, T>, Boolean, T>>;

        bool _computeFunctionDefined = false;
        std::variant<ComputeFunction<bool>, ComputeFunction<int>, ComputeFunction<int64_t>, ComputeFunction<float>, ComputeFunction<double>> _computeInputFunction;
        std::variant<Vector<bool>, Vector<int>, Vector<int64_t>, Vector<float>, Vector<double>> _cachedOutput;
        std::function<void(void*, void* const*, void* const*)> _computeDispatchFunction;
        std::function<void()> _resetFunction;
    };
} // namespace model
} // namespace ell

#pragma region implementation

namespace ell
{
namespace model
{
    template <typename InputType>
    void IRCompiledMap::SetComputeFunctionForInputType()
    {
        if (!_computeFunctionDefined)
        {
            _computeFunctionDefined = true;
            auto outputSize = GetOutput(0).Size();
            auto functionPointer = _executionEngine->ResolveFunctionAddress(_functionName);
            ComputeFunction<InputType> computeFunction;
            switch (GetOutput(0).GetType()) // Switch on output type
            {
            case model::Port::PortType::boolean: {
                _cachedOutput = Vector<bool>(outputSize);
                auto fn = reinterpret_cast<void (*)(void*, const InputType*, bool*)>(functionPointer);
                computeFunction = [this, fn](void* context, const InputType* input) {
                    fn(context, input, (bool*)std::get<Vector<bool>>(_cachedOutput).data());
                };
            }
            break;

            case model::Port::PortType::integer: {
                _cachedOutput = Vector<int>(outputSize);
                auto fn = reinterpret_cast<void (*)(void*, const InputType*, int*)>(functionPointer);
                computeFunction = [this, fn](void* context, const InputType* input) {
                    fn(context, input, std::get<Vector<int>>(_cachedOutput).data());
                };
            }
            break;

            case model::Port::PortType::bigInt: {
                _cachedOutput = Vector<int64_t>(outputSize);
                auto fn = reinterpret_cast<void (*)(void*, const InputType*, int64_t*)>(functionPointer);
                computeFunction = [this, fn](void* context, const InputType* input) {
                    fn(context, input, std::get<Vector<int64_t>>(_cachedOutput).data());
                };
            }
            break;

            case model::Port::PortType::smallReal: {
                _cachedOutput = Vector<float>(outputSize);
                auto fn = reinterpret_cast<void (*)(void*, const InputType*, float*)>(functionPointer);
                computeFunction = [this, fn](void* context, const InputType* input) {
                    fn(context, input, std::get<Vector<float>>(_cachedOutput).data());
                };
            }
            break;

            case model::Port::PortType::real: {
                _cachedOutput = Vector<double>(outputSize);
                auto fn = reinterpret_cast<void (*)(void*, const InputType*, double*)>(functionPointer);
                computeFunction = [this, fn](void* context, const InputType* input) {
                    fn(context, input, std::get<Vector<double>>(_cachedOutput).data());
                };
            }
            break;

            default:
                throw utilities::InputException(utilities::InputExceptionErrors::typeMismatch);
            }

            _computeInputFunction = computeFunction;
            functionPointer = _executionEngine->ResolveFunctionAddress(_functionName + "_dispatch");
            _computeDispatchFunction = reinterpret_cast<void (*)(void*, void* const*, void* const*)>(functionPointer);

            functionPointer = _executionEngine->ResolveFunctionAddress(_moduleName + "_Reset");
            _resetFunction = reinterpret_cast<void (*)()>(functionPointer);
        }
    }

    template <typename ElementType>
    ElementType* IRCompiledMap::GetGlobalValuePointer(const std::string& name)
    {
        auto& jitter = GetJitter();
        auto address = jitter.GetGlobalValueAddress(name);
        return reinterpret_cast<ElementType*>(address);
    }

} // namespace model
} // namespace ell

#pragma endregion implementation
