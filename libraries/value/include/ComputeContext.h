////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ComputeContext.h (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "EmitterContext.h"
#include "FunctionDeclaration.h"
#include "Scalar.h"

#include <atomic>
#include <forward_list>
#include <map>
#include <mutex>
#include <optional>
#include <stack>
#include <string>

namespace ell
{
namespace value
{

    /// <summary> A specialization of EmitterContext designed to execute statements directly </summary>
    class ComputeContext : public EmitterContext
    {
    public:
        /// <summary> Constructor </summary>
        /// <param name="moduleName"> The name of the module that this context represents </param>
        ComputeContext(std::string moduleName);

        const ConstantData& GetConstantData(Value value) const;

    private:
        Value AllocateImpl(ValueType type, MemoryLayout layout, size_t alignment, AllocateFlags flags) override;

        std::optional<Value> GetGlobalValue(GlobalAllocationScope scope, std::string name) override;

        Value GlobalAllocateImpl(GlobalAllocationScope scope, std::string name, ConstantData data, MemoryLayout layout, AllocateFlags flags) override;
        Value GlobalAllocateImpl(GlobalAllocationScope scope, std::string name, ValueType type, MemoryLayout layout, AllocateFlags flags) override;

        detail::ValueTypeDescription GetTypeImpl(Emittable emittable) override;

        DefinedFunction CreateFunctionImpl(FunctionDeclaration decl, DefinedFunction fn) override;
        bool IsFunctionDefinedImpl(FunctionDeclaration decl) const override;

        Value StoreConstantDataImpl(ConstantData data) override;

        void ForImpl(MemoryLayout layout, std::function<void(std::vector<Scalar>)> fn, const std::string& name) override;
        void ForImpl(Scalar start, Scalar stop, Scalar step, std::function<void(Scalar)> fn, const std::string& name) override;

        void MoveDataImpl(Value& source, Value& destination) override;

        void CopyDataImpl(const Value& source, Value& destination) override;

        Value OffsetImpl(Value begin, Value index) override;

        Value ReferenceImpl(Value source) override;

        Value DereferenceImpl(Value source) override;

        Value UnaryOperationImpl(ValueUnaryOperation op, Value destination) override;

        Value BinaryOperationImpl(ValueBinaryOperation op, Value destination, Value source) override;

        Value LogicalOperationImpl(ValueLogicalOperation op, Value source1, Value source2) override;

        Value CastImpl(Value value, ValueType type) override;

        IfContext IfImpl(Scalar test, std::function<void()> fn) override;

        void WhileImpl(Scalar test, std::function<void()> fn) override;

        std::optional<Value> CallImpl(FunctionDeclaration func, std::vector<Value> args) override;

        void PrefetchImpl(Value data, PrefetchType type, PrefetchLocality locality) override;

        void ParallelizeImpl(int numTasks, std::vector<Value> captured, std::function<void(Scalar, std::vector<Value>)> fn) override;

        void DebugBreakImpl() override;
        void DebugDumpImpl(Value value, std::string tag, std::ostream& stream) const override;
        void DebugDumpImpl(FunctionDeclaration fn, std::string tag, std::ostream& stream) const override;
        void DebugPrintImpl(std::string message) override;

        void SetNameImpl(const Value& value, const std::string& name) override;
        std::string GetNameImpl(const Value& value) const override;

        void ImportCodeFileImpl(std::string) override;

        Scalar GetFunctionAddressImpl(const FunctionDeclaration& fn) override;

        Value IntrinsicCall(FunctionDeclaration intrinsic, std::vector<Value> args);

        bool ValidateValue(Value value) const;
        bool TypeCompatible(Value value1, Value value2) const;

        ConstantData ExtractConstantData(Value value);
        bool IsGlobalValue(Value value) const;

        std::string GetScopeAdjustedName(GlobalAllocationScope scope, std::string name) const;
        std::string GetGlobalScopedName(std::string name) const;
        std::string GetCurrentFunctionScopedName(std::string name) const;

        using ConstantDataList = std::forward_list<ConstantData>;
        using Frame = std::pair<std::string, ConstantDataList>;

        Frame& GetTopFrame();
        const Frame& GetTopFrame() const;

        friend void swap(ComputeContext&, ComputeContext&) noexcept;

        class IfContextImpl;
        struct FunctionScope;

        mutable std::recursive_mutex _mutex;
        std::stack<Frame> _stack;
        std::map<std::string, std::pair<ConstantData, MemoryLayout>> _globals;
        std::unordered_map<FunctionDeclaration, DefinedFunction> _definedFunctions;
        std::unordered_map<Value, std::string> _namedValues;
        std::string _moduleName;
    };

} // namespace value
} // namespace ell
