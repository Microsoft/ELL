////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRParallelLoopEmitter.cpp (emitters)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRParallelLoopEmitter.h"
#include "IRAsyncTask.h"
#include "IRFunctionEmitter.h"
#include "IRMath.h"
#include "IRModuleEmitter.h"

#include <vector>

namespace ell
{
namespace emitters
{
    IRParallelForLoopEmitter::IRParallelForLoopEmitter(IRFunctionEmitter& functionEmitter) :
        _functionEmitter(functionEmitter)
    {}

    void IRParallelForLoopEmitter::EmitLoop(int begin, int end, int increment, const ParallelLoopOptions& options, const std::vector<LLVMValue>& capturedValues, BodyFunction body)
    {
        // numIter = ceil((end-begin)/increment)
        auto span = end - begin;
        auto numIterations = span > 0 ? (span - 1) / increment + 1 : 0;

        auto compilerSettings = _functionEmitter.GetCompilerOptions();
        ParallelLoopOptions newOptions = options;
        if (newOptions.numTasks == 0)
        {
            newOptions.numTasks = std::min(numIterations, compilerSettings.maxThreads);
        }
        EmitLoop(_functionEmitter.LocalScalar<int32_t>(begin), _functionEmitter.LocalScalar<int32_t>(end), _functionEmitter.LocalScalar<int32_t>(increment), newOptions, capturedValues, body);
    }

    void IRParallelForLoopEmitter::EmitLoop(IRLocalScalar begin, IRLocalScalar end, IRLocalScalar increment, const ParallelLoopOptions& options, const std::vector<LLVMValue>& capturedValues, BodyFunction body)
    {
        auto compilerSettings = _functionEmitter.GetCompilerOptions();
        const int numTasks = options.numTasks == 0 ? compilerSettings.maxThreads : options.numTasks;
        auto span = end - begin;
        auto numIterations = (span - 1) / increment + 1;
        // TODO: explicitly check for empty loop?

        auto taskSize = (numIterations - 1) / numTasks + 1;
        if (compilerSettings.parallelize && numTasks > 1)
        {
            auto taskFunction = GetTaskFunction(capturedValues, body);

            std::vector<std::vector<LLVMValue>> taskArgs;
            for (int taskIndex = 0; taskIndex < numTasks; ++taskIndex)
            {
                auto blockStart = begin + taskIndex * taskSize * increment;
                auto blockEnd = Min(blockStart + taskSize * increment, end);
                std::vector<LLVMValue> args{ blockStart, blockEnd, increment };
                std::copy(capturedValues.begin(), capturedValues.end(), std::back_inserter(args));
                taskArgs.push_back(args);
            }
            auto tasks = _functionEmitter.StartTasks(taskFunction, taskArgs);
            tasks.WaitAll(_functionEmitter);
        }
        else
        {
            // Emit a normal for loop if we were only going to use 1 parallel task
            _functionEmitter.For(begin, end, increment, [capturedValues, body](IRFunctionEmitter& function, LLVMValue i) {
                body(function, function.LocalScalar(i), capturedValues);
            });
        }
    }

    IRFunctionEmitter IRParallelForLoopEmitter::GetTaskFunction(const std::vector<LLVMValue>& capturedValues, BodyFunction body)
    {
        std::string name = "parForTask";

        // args = start, captured args
        auto returnType = _functionEmitter.GetModule().GetIREmitter().Type(VariableType::Void);
        auto argTypes = _functionEmitter.GetModule().GetIREmitter().GetLLVMTypes({ VariableType::Int32, VariableType::Int32, VariableType::Int32 });
        auto capturedTypes = GetLLVMTypes(capturedValues);
        std::copy(capturedTypes.begin(), capturedTypes.end(), std::back_inserter(argTypes));
        auto taskFunction = _functionEmitter.GetModule().BeginFunction(name, returnType, argTypes);

        {
            auto arguments = taskFunction.Arguments().begin();
            auto blockStart = &(*arguments++);
            auto blockEnd = &(*arguments++);
            auto increment = &(*arguments++);
            std::vector<LLVMValue> innerCapturedValues;
            int numCapturedValues = static_cast<int>(capturedValues.size());
            for (int index = 0; index < numCapturedValues; ++index)
            {
                auto capturedValue = &(*arguments++);
                capturedValue->setName("captured_" + std::to_string(index));
                innerCapturedValues.push_back(capturedValue);
            }

            taskFunction.For(blockStart, blockEnd, increment, [innerCapturedValues, body](IRFunctionEmitter& taskFunction, LLVMValue i) {
                body(taskFunction, taskFunction.LocalScalar(i), innerCapturedValues);
            });
        }
        _functionEmitter.GetModule().EndFunction();
        return taskFunction;
    }
} // namespace emitters
} // namespace ell
