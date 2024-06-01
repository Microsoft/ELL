////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRMapCompiler.cpp (model)
//  Authors:  Umesh Madan, Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRMapCompiler.h"
#include "CompilableNode.h"
#include "CompilableNodeUtilities.h"
#include "IRModelProfiler.h"
#include "Model.h"
#include "OptimizeModelTransformation.h"
#include "OutputNode.h"
#include "RefineTransformation.h"

#include <emitters/include/EmitterException.h>
#include <emitters/include/IRMetadata.h>
#include <emitters/include/LLVMUtilities.h>
#include <emitters/include/Variable.h>

#include <nodes/include/SinkNode.h>
#include <nodes/include/SourceNode.h>

#include <utilities/include/Logger.h>
#include <utilities/include/StringUtil.h>

#include <value/include/LLVMContext.h>

#include <memory>
#include <tuple>
#include <vector>

namespace ell
{
namespace model
{
    using namespace logging;
    using namespace value;

    IRMapCompiler::IRMapCompiler() :
        IRMapCompiler(MapCompilerOptions{}, ModelOptimizerOptions{})
    {
    }

    IRMapCompiler::IRMapCompiler(const MapCompilerOptions& settings, const ModelOptimizerOptions& optimizerOptions) :
        MapCompiler(settings, optimizerOptions),
        _moduleEmitter(settings.moduleName, settings.compilerSettings),
        _profiler()
    {
        Log() << "Initializing IR map compiler" << EOL;
        _nodeRegions.emplace_back();
    }

    std::string IRMapCompiler::GetNamespacePrefix() const
    {
        return GetModule().GetModuleName();
    }

    std::string IRMapCompiler::GetGlobalName(const Node& node, const std::string& baseName) const
    {
        // e.g "ELL_GRUNodeReset_1541".
        return GetNamespacePrefix() + "_" + baseName + "_" + node.GetId().ToString();
    }

    std::string IRMapCompiler::GetPredictFunctionName() const
    {
        return GetMapCompilerOptions().mapFunctionName;
    }

    IRCompiledMap IRMapCompiler::Compile(Map map)
    {
        Log() << "Compile called for map" << EOL;

        RefineAndOptimize(map);

        // Renaming callbacks based on map compiler parameters
        // Note: a more elegant solution is emit variables which get assigned to
        // function pointers at runtime (prior to computing the map).
        Log() << "Renaming callbacks..." << EOL;
        map.RenameCallbacks(GetMapCompilerOptions().sourceFunctionName, GetMapCompilerOptions().sinkFunctionName);

        // Now the model ready for compiling
        if (GetMapCompilerOptions().profile)
        {
            Log() << "Enabling profiling in emitted IR" << EOL;
            GetModule().AddPreprocessorDefinition(GetNamespacePrefix() + "_PROFILING", "1");
        }
        _profiler = { GetModule(), map.GetModel(), GetMapCompilerOptions().profile };
        _profiler.EmitInitialization();

        {
            ContextGuard<LLVMContext> guard(_moduleEmitter);

            // Now we have the refined map, compile it
            Log() << "Compiling map..." << EOL;
            CompileMap(map, GetPredictFunctionName());
        }

        // Emit runtime model APIs
        EmitModelAPIFunctions(map);

        if (GetMapCompilerOptions().compilerSettings.optimize)
        {
            // Save callback declarations in case they get optimized away
            std::vector<std::tuple<std::string, llvm::FunctionType*, std::vector<std::string>>> savedCallbacks;
            auto callbacks = emitters::GetFunctionsWithTag(_moduleEmitter, emitters::c_callbackFunctionTagName);
            for (auto& callbackInfo : callbacks)
            {
                savedCallbacks.emplace_back(callbackInfo.function->getName(), callbackInfo.function->getFunctionType(), callbackInfo.values);
            }

            emitters::IROptimizer optimizer(_moduleEmitter);
            optimizer.AddStandardPasses();
            _moduleEmitter.Optimize(optimizer);

            // Reinsert callback declarations after optimization
            for (const auto& savedCallback : savedCallbacks)
            {
                auto functionName = std::get<0>(savedCallback);
                _moduleEmitter.DeclareFunction(functionName, std::get<1>(savedCallback));
                _moduleEmitter.IncludeInCallbackInterface(functionName, std::get<2>(savedCallback)[0]);
            }
        }
        return IRCompiledMap(std::move(map), GetMapCompilerOptions().mapFunctionName, GetMapCompilerOptions(), _moduleEmitter, GetMapCompilerOptions().verifyJittedModule);
    }

    void IRMapCompiler::RefineAndOptimize(Map& map)
    {
        TransformContext context(this);
        OptimizeModelTransformation optimizer;
        map.Transform(optimizer, context);

        // Add an extra refine here, in case the optimizer doesn't do it
        RefineTransformation refiner;
        map.Transform(refiner, context);

        map.Prune();
    }

    void IRMapCompiler::EmitPredictDispatchFunction(const Map& map)
    {
        auto& emitter = _moduleEmitter.GetIREmitter();

        // Now add a dynamic predict that we can call from jitted CompiledMap that can handle multiple inputs and outputs.
        // Essentially we package up the pointers to the buffers in an array of void* pointers, into "predict_dispatch" which
        // unpacks them, casts them to the right type and calls the above predict function.
        emitters::NamedLLVMTypeList args;

        std::vector<std::string> comments;
        emitters::LLVMType returnType = emitter.Type(emitters::VariableType::Void);

        auto predictFunction = _moduleEmitter.GetFunction(GetPredictFunctionName());
        emitters::NamedLLVMTypeList predictArgs;
        for (auto arg = predictFunction->arg_begin(), end = predictFunction->arg_end(); arg != end; ++arg)
        {
            predictArgs.push_back({ arg->getName(), arg->getType() });
        }

        args.push_back(predictArgs[0]); // the context parameter

        //  we really want void** but LLVM doesn't allow that.
        emitters::LLVMType argType = llvm::PointerType::getUnqual(emitter.Type(emitters::VariableType::Char8Pointer));
        args.push_back({ "inputs", argType });
        args.push_back({ "outputs", argType });

        auto functionName = GetPredictFunctionName() + "_dispatch";
        auto function = _moduleEmitter.BeginFunction(functionName, returnType, args);

        // stops it from getting optimized away so it will always be in the JIT'd module.
        function.GetFunction()->setLinkage(llvm::GlobalValue::LinkageTypes::ExternalLinkage);

        auto arg0 = function.GetFunctionArgument("context");
        auto arg1 = function.GetFunctionArgument("inputs");
        auto arg2 = function.GetFunctionArgument("outputs");

        // unpack the array of buffers and turn them in to predict function arguments.
        emitters::IRValueList arguments;
        arguments.push_back(arg0); // pass the context through

        int predictArgIndex = 1; // skip context.
        size_t size = map.NumInputs();
        for (size_t i = 0; i < size; ++i)
        {
            auto index = function.Literal(static_cast<int>(i));
            auto offset = emitter.PointerOffset(arg1, index); // get &args[i]
            auto ptr = function.Load(offset); // char*
            auto typedPtr = function.CastPointer(ptr, predictArgs[predictArgIndex++].second); // cast it to the right type
            arguments.push_back(typedPtr);
        }

        size = map.NumOutputs();
        for (size_t i = 0; i < size; ++i)
        {
            auto index = function.Literal(static_cast<int>(i));
            auto offset = emitter.PointerOffset(arg2, index); // get &args[i]
            auto ptr = function.Load(offset); // char*
            auto typedPtr = function.CastPointer(ptr, predictArgs[predictArgIndex++].second); // cast it to the right type
            arguments.push_back(typedPtr);
        }

        function.Call(predictFunction, arguments);
        function.Return();
        _moduleEmitter.EndFunction();
    }

    void IRMapCompiler::EmitModelAPIFunctions(const Map& map)
    {
        EmitGetInputSizeFunction(map);
        EmitGetOutputSizeFunction(map);
        EmitGetSinkOutputSizeFunction(map);
        EmitGetNumNodesFunction(map);
        EmitShapeEnum();
        EmitGetInputShapeFunction(map);
        EmitGetOutputShapeFunction(map);
        EmitGetSinkOutputShapeFunction(map);
        EmitGetMetadataFunction(map);
        EmitPredictDispatchFunction(map);

        // Finish any profiling stuff we need to do and emit functions
        _profiler.EmitModelProfilerFunctions();
    }

    void IRMapCompiler::EmitGetInputSizeFunction(const Map& map)
    {
        const emitters::NamedVariableTypeList parameters = { { "index", emitters::VariableType::Int32 } };
        auto function = _moduleEmitter.BeginFunction(GetNamespacePrefix() + "_GetInputSize", emitters::VariableType::Int32, parameters);
        function.IncludeInHeader();

        std::vector<int> sizes;
        for (size_t i = 0, n = map.NumInputs(); i < n; ++i)
        {
            sizes.push_back(map.GetInputSize(i));
        }
        EmitSizeConditionals(function, sizes);
        _moduleEmitter.EndFunction();
    }

    void IRMapCompiler::EmitGetOutputSizeFunction(const Map& map)
    {
        const emitters::NamedVariableTypeList parameters = { { "index", emitters::VariableType::Int32 } };
        auto function = _moduleEmitter.BeginFunction(GetNamespacePrefix() + "_GetOutputSize", emitters::VariableType::Int32, parameters);
        function.IncludeInHeader();

        std::vector<int> sizes;
        for (size_t i = 0, n = map.NumOutputs(); i < n; ++i)
        {
            sizes.push_back(map.GetOutputSize(i));
        }
        EmitSizeConditionals(function, sizes);
        _moduleEmitter.EndFunction();
    }

    void IRMapCompiler::EmitGetSinkOutputSizeFunction(const Map& map)
    {
        auto nodes = map.GetSinkNodes();
        if (nodes.size() > 0)
        {
            const emitters::NamedVariableTypeList parameters = { { "index", emitters::VariableType::Int32 } };
            auto function = _moduleEmitter.BeginFunction(GetNamespacePrefix() + "_GetSinkOutputSize", emitters::VariableType::Int32, parameters);
            function.IncludeInHeader();

            std::vector<int> sizes;
            for (auto ptr = nodes.begin(), end = nodes.end(); ptr != end; ptr++)
            {
                const ell::model::Node* node = *ptr;
                sizes.push_back(node->GetOutputPort(0)->GetMemoryLayout().GetActiveSize().NumElements());
            }
            EmitSizeConditionals(function, sizes);
            _moduleEmitter.EndFunction();
        }
    }

    void IRMapCompiler::EmitSizeConditionals(emitters::IRFunctionEmitter& fn, std::vector<int> sizes)
    {
        // This is the type of code we are trying to generate for the GetInputSize and GetOutputSize functions:
        //
        // int model_GetInputSize(int index)
        // {
        //     if (index == 0) {
        //         return 224;
        //     }
        //     if (index == 1) {
        //         return 10;
        //     }
        //     return 0;
        // }
        //
        auto arguments = fn.Arguments().begin();
        auto indexArgument = &(*arguments++);
        auto pMainBlock = fn.GetCurrentBlock();

        std::vector<llvm::BasicBlock*> blocks;
        blocks.push_back(pMainBlock);
        // create the final block in the case the index is out of range or there are no shapes and reutrn empty TensorShape.
        auto pDoneBlock = fn.BeginBlock("NoMatchBlock");
        {
            fn.Return(fn.Literal(0));
        }

        if (!sizes.empty())
        {
            int index = 0;
            for (auto ptr = sizes.begin(), end = sizes.end(); ptr != end; ptr++, index++)
            {
                int size = *ptr;
                std::string labelSuffix = std::to_string(index);

                auto followBlock = fn.BeginBlock("FollowBlock" + labelSuffix);
                auto thenBlock = fn.BeginBlock("ThenBlock" + labelSuffix);
                {
                    fn.Return(fn.Literal(size));
                }
                auto elseBlock = fn.BeginBlock("ElseBlock" + labelSuffix);
                {
                    fn.Branch(followBlock);
                }
                auto conditionBlock = fn.BeginBlock("IfBlock" + labelSuffix);
                {
                    auto compare = new llvm::ICmpInst(*conditionBlock, llvm::ICmpInst::ICMP_EQ, indexArgument, fn.Literal(index));
                    llvm::BranchInst::Create(thenBlock, elseBlock, compare, conditionBlock);
                }
                blocks.push_back(conditionBlock);
                blocks.push_back(followBlock);
            }
            fn.SetCurrentBlock(blocks[blocks.size() - 1]);
            fn.Branch(pDoneBlock);
        }

        fn.SetCurrentBlock(pMainBlock);
        if (blocks.size() > 1)
        {
            fn.Branch(blocks[1]); // jump to first statement.
        }

        blocks.push_back(pDoneBlock);
        // ensure all blocks are properly chained with branch instructions and inserted into the function BasicBlockList.
        fn.ConcatenateBlocks(blocks);
    }

    const char* TensorShapeName = "TensorShape";

    void IRMapCompiler::EmitShapeEnum()
    {
        auto shapeType = _moduleEmitter.GetStruct(TensorShapeName);
        if (shapeType == nullptr)
        {
            auto int32Type = ell::emitters::VariableType::Int32;
            emitters::NamedVariableTypeList namedFields = { { "rows", int32Type }, { "columns", int32Type }, { "channels", int32Type } };
            auto shapeType = _moduleEmitter.GetOrCreateStruct(TensorShapeName, namedFields);
            _moduleEmitter.IncludeTypeInHeader(shapeType->getName());
        }
    }

    void IRMapCompiler::EmitShapeConditionals(emitters::IRFunctionEmitter& fn, std::vector<MemoryShape> shapes)
    {
        auto shapeType = _moduleEmitter.GetStruct(TensorShapeName);
        auto arguments = fn.Arguments().begin();
        auto indexArgument = &(*arguments++);
        auto shapeArgument = &(*arguments++);
        auto& emitter = _moduleEmitter.GetIREmitter();
        auto& irBuilder = emitter.GetIRBuilder();
        auto rowsPtr = irBuilder.CreateInBoundsGEP(shapeType, shapeArgument, { fn.Literal(0), fn.Literal(0) });
        rowsPtr->setName("rows");
        auto columnsPtr = irBuilder.CreateInBoundsGEP(shapeType, shapeArgument, { fn.Literal(0), fn.Literal(1) });
        columnsPtr->setName("columns");
        auto channelsPtr = irBuilder.CreateInBoundsGEP(shapeType, shapeArgument, { fn.Literal(0), fn.Literal(2) });
        channelsPtr->setName("channels");
        auto pMainBlock = fn.GetCurrentBlock();

        std::vector<llvm::BasicBlock*> blocks;
        blocks.push_back(pMainBlock);
        // create the final block in the case the index is out of range or there are no shapes and reutrn empty TensorShape.
        auto pDoneBlock = fn.BeginBlock("NoMatchBlock");
        {
            fn.StoreZero(rowsPtr);
            fn.StoreZero(columnsPtr);
            fn.StoreZero(channelsPtr);
            fn.Return();
        }

        if (shapes.size() > 0)
        {
            int index = 0;
            for (auto ptr = shapes.begin(), end = shapes.end(); ptr != end; ptr++, index++)
            {
                std::string labelSuffix = std::to_string(index);

                auto followBlock = fn.BeginBlock("FollowBlock" + labelSuffix);
                auto thenBlock = fn.BeginBlock("ThenBlock" + labelSuffix);
                {
                    MemoryShape shape = *ptr;
                    shape.Resize(3);

                    fn.Store(rowsPtr, fn.Literal<int>(shape[0]));
                    fn.Store(columnsPtr, fn.Literal<int>(shape[1]));
                    fn.Store(channelsPtr, fn.Literal<int>(shape[2]));
                    fn.Return();
                }
                auto elseBlock = fn.BeginBlock("ElseBlock" + labelSuffix);
                {
                    fn.Branch(followBlock);
                }
                auto conditionBlock = fn.BeginBlock("IfBlock" + labelSuffix);
                {
                    auto compare = new llvm::ICmpInst(*conditionBlock, llvm::ICmpInst::ICMP_EQ, indexArgument, fn.Literal(index));
                    llvm::BranchInst::Create(thenBlock, elseBlock, compare, conditionBlock);
                }
                blocks.push_back(conditionBlock);
                blocks.push_back(followBlock);
            }
            fn.SetCurrentBlock(blocks[blocks.size() - 1]);
            fn.Branch(pDoneBlock);
        }

        fn.SetCurrentBlock(pMainBlock);
        if (blocks.size() > 1)
        {
            fn.Branch(blocks[1]); // jump to first statement.
        }

        blocks.push_back(pDoneBlock);
        // ensure all blocks are properly chained with branch instructions and inserted into the function BasicBlockList.
        fn.ConcatenateBlocks(blocks);
    }

    // This is the type of code we are trying to generate for the GetInputShape and GetOutputShape functions:
    //
    // void foo_GetInputShape(int index, struct TensorShape* s)
    // {
    //     if (index == 0) {
    //         s->rows = 224;
    //         s->columns = 224;
    //         s->channels = 3;
    //         return;
    //     }
    //     if (index == 1) {
    //         s->rows = 10;
    //         s->columns = 10;
    //         s->channels = 20;
    //         return;
    //     }
    //     s->rows = 0;
    //     s->columns = 0;
    //     s->channels = 0;
    // }
    //

    void IRMapCompiler::EmitGetInputShapeFunction(const Map& map)
    {
        // We have to create this interface because LLVM cannot reliably return structures on the stack.
        // void darknet_GetInputShape(int index, struct TensorShape* shape)
        auto shapeType = _moduleEmitter.GetStruct(TensorShapeName);
        auto& context = _moduleEmitter.GetLLVMContext();
        auto voidType = llvm::Type::getVoidTy(context);
        auto int32Type = llvm::Type::getInt32Ty(context);

        const emitters::NamedLLVMTypeList parameters = { { "index", int32Type }, { "shape", shapeType->getPointerTo() } };
        const std::string functionName = GetNamespacePrefix() + "_GetInputShape";

        auto fn = _moduleEmitter.BeginFunction(functionName, voidType, parameters);
        fn.IncludeInHeader();

        std::vector<MemoryShape> shapes;
        for (size_t i = 0, n = map.NumInputs(); i < n; ++i)
        {
            shapes.push_back(map.GetInputShape(i));
        }

        EmitShapeConditionals(fn, shapes);
        _moduleEmitter.EndFunction();
    }

    void IRMapCompiler::EmitGetOutputShapeFunction(const Map& map)
    {
        // We have to create this interface because LLVM cannot reliably return structures on the stack.
        // void darknet_GetOutputShape(int index, struct TensorShape* shape)
        auto shapeType = _moduleEmitter.GetStruct(TensorShapeName);
        auto& context = _moduleEmitter.GetLLVMContext();
        auto voidType = llvm::Type::getVoidTy(context);
        auto int32Type = llvm::Type::getInt32Ty(context);

        const emitters::NamedLLVMTypeList parameters = { { "index", int32Type }, { "shape", shapeType->getPointerTo() } };
        const std::string functionName = GetNamespacePrefix() + "_GetOutputShape";

        auto fn = _moduleEmitter.BeginFunction(functionName, voidType, parameters);
        fn.IncludeInHeader();

        std::vector<MemoryShape> shapes;
        for (size_t i = 0, n = map.NumOutputs(); i < n; ++i)
        {
            shapes.push_back(map.GetOutputShape(i));
        }

        EmitShapeConditionals(fn, shapes);
        _moduleEmitter.EndFunction();
    }

    void IRMapCompiler::EmitGetSinkOutputShapeFunction(const Map& map)
    {
        auto nodes = map.GetSinkNodes();
        if (nodes.size() > 0)
        {
            // We have to create this interface because LLVM cannot reliably return structures on the stack.
            // void darknet_GetOutputShape(int index, struct TensorShape* shape)
            auto shapeType = _moduleEmitter.GetStruct(TensorShapeName);
            auto& context = _moduleEmitter.GetLLVMContext();
            auto voidType = llvm::Type::getVoidTy(context);
            auto int32Type = llvm::Type::getInt32Ty(context);

            const emitters::NamedLLVMTypeList parameters = { { "index", int32Type }, { "shape", shapeType->getPointerTo() } };
            const std::string functionName = GetNamespacePrefix() + "_GetSinkOutputShape";

            auto fn = _moduleEmitter.BeginFunction(functionName, voidType, parameters);
            fn.IncludeInHeader();

            std::vector<MemoryShape> shapes;
            for (auto ptr = nodes.begin(), end = nodes.end(); ptr != end; ptr++)
            {
                const ell::model::Node* node = *ptr;
                shapes.push_back(node->GetOutputPort(0)->GetMemoryLayout().GetActiveSize());
            }
            EmitShapeConditionals(fn, shapes);
            _moduleEmitter.EndFunction();
        }
    }

    void IRMapCompiler::EmitGetNumNodesFunction(const Map& map)
    {
        int numNodes = map.GetModel().Size();

        auto function = _moduleEmitter.BeginFunction(GetNamespacePrefix() + "_GetNumNodes", emitters::VariableType::Int32);
        function.IncludeInHeader();
        function.Return(function.Literal(numNodes));
        _moduleEmitter.EndFunction();
    }

    void IRMapCompiler::EmitGetMetadataFunction(const Map& map)
    {
        const emitters::NamedVariableTypeList parameters = { { "key", emitters::VariableType::Char8Pointer } };
        auto function = _moduleEmitter.BeginFunction(GetNamespacePrefix() + "_GetMetadata", emitters::VariableType::Char8Pointer, parameters);
        function.IncludeInHeader();

        std::vector<std::pair<std::string, std::string>> keyValuePairs;
        auto iter = map.GetModel().GetNodeIterator();
        while (iter.IsValid())
        {
            auto node = iter.Get();
            auto bag = node->GetMetadata();
            for (auto key : bag.Keys())
            {
                auto value = bag[key];
                auto strValue = value.ToString();
                keyValuePairs.emplace_back(key, strValue);
            }
            iter.Next();
        }
        EmitStringConditionals(function, keyValuePairs);
        _moduleEmitter.EndFunction();
    }

    void IRMapCompiler::EmitStringConditionals(emitters::IRFunctionEmitter& fn, std::vector<std::pair<std::string, std::string>> keyValuePairs)
    {
        // This is the type of code we are trying to generate for the GetInputSize and GetOutputSize functions:
        //
        // int model_GetInputSize(int index)
        // {
        //     if (name == "window_size") {
        //         return "80";
        //     }
        //     if (name == "filterbank_size") {
        //         return "40";
        //     }
        //     ...
        //     return "";
        // }
        //
        auto arguments = fn.Arguments().begin();
        auto nameArgument = &(*arguments++);
        auto pMainBlock = fn.GetCurrentBlock();

        auto strCompare = _moduleEmitter.GetRuntime().GetStringCompareFunction();

        std::vector<llvm::BasicBlock*> blocks;
        blocks.push_back(pMainBlock);
        // create the final block in the case the index is out of range or there are no shapes and reutrn empty TensorShape.
        auto pDoneBlock = fn.BeginBlock("NoMatchBlock");
        {
            fn.Return(fn.Literal(""));
        }

        if (!keyValuePairs.empty())
        {
            int index = 0;
            for (auto pair : keyValuePairs)
            {
                std::string key = pair.first;
                std::string value = pair.second;

                std::string labelSuffix = std::to_string(index);
                auto followBlock = fn.BeginBlock("FollowBlock" + labelSuffix);
                auto thenBlock = fn.BeginBlock("ThenBlock" + labelSuffix);
                {
                    fn.Return(fn.Literal(value));
                }
                auto elseBlock = fn.BeginBlock("ElseBlock" + labelSuffix);
                {
                    fn.Branch(followBlock);
                }
                auto conditionBlock = fn.BeginBlock("IfBlock" + labelSuffix);
                {
                    auto compare = fn.Call(strCompare, { nameArgument, fn.Literal(key) });
                    auto isEqual = fn.LocalScalar(compare) == 1;
                    llvm::BranchInst::Create(thenBlock, elseBlock, isEqual, conditionBlock);
                }
                blocks.push_back(conditionBlock);
                blocks.push_back(followBlock);
            }
            fn.SetCurrentBlock(blocks[blocks.size() - 1]);
            fn.Branch(pDoneBlock);
        }

        fn.SetCurrentBlock(pMainBlock);
        if (blocks.size() > 1)
        {
            fn.Branch(blocks[1]); // jump to first statement.
        }

        blocks.push_back(pDoneBlock);
        // ensure all blocks are properly chained with branch instructions and inserted into the function BasicBlockList.
        fn.ConcatenateBlocks(blocks);
    }

    //
    // Node implementor methods:
    //

    emitters::LLVMValue IRMapCompiler::EnsurePortEmitted(const InputPortBase& port)
    {
        emitters::Variable* pVar = GetVariableForPort(port.GetReferencedPort());
        if (pVar == nullptr)
        {
            const Node* node = port.GetNode();
            throw emitters::EmitterException(emitters::EmitterError::unexpected,
                                             utilities::FormatString("Error: missing port variable for '%s' port on node %s(%s)", port.GetName().c_str(), node->GetRuntimeTypeName().c_str(), node->GetId().ToString().c_str()));
        }
        return GetModule().EnsureEmitted(*pVar);
    }

    emitters::LLVMValue IRMapCompiler::EnsurePortEmitted(const OutputPortBase& port)
    {
        auto pVar = GetOrAllocatePortVariable(port);
        return GetModule().EnsureEmitted(*pVar);
    }

    void IRMapCompiler::OnBeginCompileModel(const Model& model)
    {
        auto& currentFunction = GetModule().GetCurrentFunction();
        if (currentFunction.GetCurrentRegion() == nullptr) // TODO: put this check in GetCurrentFunction()
        {
            currentFunction.AddRegion(currentFunction.GetCurrentBlock());
            Log() << "Creating a new region for " << currentFunction.GetFunctionName() << EOL;
        }

        // Tag the model function for declaration in the generated headers
        currentFunction.IncludeInHeader();
        currentFunction.IncludeInPredictInterface();

        _profiler.StartModel(currentFunction);
    }

    void IRMapCompiler::OnEndCompileModel(const Model& model)
    {
        auto& currentFunction = GetModule().GetCurrentFunction();
        _profiler.EndModel(currentFunction);
    }

    void IRMapCompiler::OnBeginCompileNode(const Node& node)
    {
        auto& currentFunction = GetModule().GetCurrentFunction();
        if (currentFunction.GetCurrentRegion() == nullptr)
        {
            Log() << "Creating new region for node in " << currentFunction.GetFunctionName() << EOL;
            currentFunction.AddRegion(currentFunction.GetCurrentBlock());
        }

        _profiler.InitNode(currentFunction, node);
        _profiler.StartNode(currentFunction, node);
    }

    void IRMapCompiler::OnEndCompileNode(const Node& node)
    {
        auto& currentFunction = GetModule().GetCurrentFunction();
        assert(currentFunction.GetCurrentRegion() != nullptr);

        _profiler.EndNode(currentFunction, node);

        auto pCurBlock = currentFunction.GetCurrentBlock();
        if (pCurBlock != currentFunction.GetCurrentRegion()->End())
        {
            Log() << "Current node block is not end of function " << currentFunction.GetFunctionName()
                  << ". Setting it so it is." << EOL;
            currentFunction.GetCurrentRegion()->SetEnd(pCurBlock);
        }

        Log() << "Finished compiling node " << DiagnosticString(node) << EOL;
    }

    void IRMapCompiler::PushScope()
    {
        MapCompiler::PushScope();
        _nodeRegions.emplace_back();
    }

    void IRMapCompiler::PopScope()
    {
        MapCompiler::PopScope();
        assert(_nodeRegions.size() > 0);
        _nodeRegions.pop_back();
    }

    NodeMap<emitters::IRBlockRegion*>& IRMapCompiler::GetCurrentNodeBlocks()
    {
        assert(_nodeRegions.size() > 0);
        return _nodeRegions.back();
    }

    const Node* IRMapCompiler::GetUniqueParent(const Node& node)
    {
        Log() << "Trying to find unique parent of node " << DiagnosticString(node) << EOL;
        auto inputs = node.GetInputPorts();

        Log() << "Node has " << inputs.size() << " input ports" << EOL;

        const Node* pParentNode = nullptr;
        emitters::IRBlockRegion* pParentRegion = nullptr;
        for (auto input : inputs)
        {
            for (auto parentNode : input->GetParentNodes())
            {
                if (!HasSingleDescendant(*parentNode))
                {
                    return nullptr;
                }
                emitters::IRBlockRegion* pNodeRegion = GetCurrentNodeBlocks().Get(*parentNode);
                if (pNodeRegion != nullptr)
                {
                    Log() << "Got block region for parent node " << DiagnosticString(*parentNode) << EOL;
                    if (pParentRegion != nullptr && pNodeRegion != pParentRegion)
                    {
                        return nullptr;
                    }
                    pParentRegion = pNodeRegion;
                    pParentNode = parentNode;
                }
                else
                {
                    Log() << "Could not get block region for parent node " << DiagnosticString(*parentNode) << EOL;
                }
            }
        }
        return pParentNode;
    }

    void IRMapCompiler::NewNodeRegion(const Node& node)
    {
        auto& currentFunction = GetModule().GetCurrentFunction();

        auto currentBlock = currentFunction.GetCurrentBlock();
        auto termInst = currentBlock->getTerminator();

        if (termInst == nullptr)
        {
            Log() << "Prev block had no terminator!" << EOL;
        }

        // Create a new block
        auto pBlock = currentFunction.BeginBlock(IdString(node), true);
        assert(pBlock != nullptr && "Got null new block");
        auto currentRegion = currentFunction.AddRegion(pBlock);

        Log() << "Created region for node: " << DiagnosticString(node) << EOL;
        GetCurrentNodeBlocks().Set(node, currentRegion);

        if (GetMapCompilerOptions().compilerSettings.includeDiagnosticInfo)
        {
            currentFunction.Print(DiagnosticString(node) + '\n');
        }
    }

    bool IRMapCompiler::TryMergeNodeRegion(const Node& node)
    {
        Log() << "Trying to merge code regions for " << DiagnosticString(node) << EOL;

        auto pRegion = GetCurrentNodeBlocks().Get(node);
        if (pRegion == nullptr)
        {
            Log() << "Could not get code region for " << DiagnosticString(node) << EOL;
            return false;
        }

        const Node* pParentNode = GetUniqueParent(node);
        if (pParentNode == nullptr)
        {
            Log() << "Could not find a unique parent for " << DiagnosticString(node) << EOL;
            return false;
        }

        return TryMergeNodeRegions(*pParentNode, node);
    }

    bool IRMapCompiler::TryMergeNodeRegions(const Node& dest, const Node& src)
    {
        Log() << "Trying to merge parent node " << DiagnosticString(dest) << " with child node " << DiagnosticString(src) << EOL;

        emitters::IRBlockRegion* pDestRegion = GetCurrentNodeBlocks().Get(dest);
        if (pDestRegion == nullptr)
        {
            Log() << "Could not get code block region for parent node " << DiagnosticString(dest) << EOL;
            return false;
        }
        return TryMergeNodeIntoRegion(pDestRegion, src);
    }

    bool IRMapCompiler::TryMergeNodeIntoRegion(emitters::IRBlockRegion* pDestRegion, const Node& src)
    {
        auto& currentFunction = GetModule().GetCurrentFunction();

        Log() << "Trying to merge emitted code for node " << DiagnosticString(src) << " with existing code region in " << currentFunction.GetFunctionName() << EOL;

        emitters::IRBlockRegion* pSrcRegion = GetCurrentNodeBlocks().Get(src);
        if (pSrcRegion == nullptr || pSrcRegion == pDestRegion)
        {
            Log() << "Code region for node " << (pSrcRegion ? " is same as region to merge" : " null") << EOL;
            return false;
        }

        Log() << "Setting end of current region to current block" << EOL;
        GetModule().GetCurrentRegion()->SetEnd(currentFunction.GetCurrentBlock());
        currentFunction.ConcatRegions(pDestRegion, pSrcRegion);
        GetCurrentNodeBlocks().Set(src, pDestRegion);
        return true;
    }

    emitters::IRBlockRegion* IRMapCompiler::GetMergeableNodeRegion(const PortElementBase& element)
    {
        const Node* pNode = nullptr;
        if (HasSingleDescendant(element))
        {
            emitters::Variable* pVar = GetVariableForPort(*element.ReferencedPort());
            if (pVar != nullptr && !pVar->IsLiteral())
            {
                pNode = element.ReferencedPort()->GetNode();
            }
        }

        return (pNode != nullptr) ? GetCurrentNodeBlocks().Get(*pNode) : nullptr;
    }

    llvm::LLVMContext& IRMapCompiler::GetLLVMContext()
    {
        return _moduleEmitter.GetLLVMContext();
    }

    //
    // Port variables
    //
    emitters::LLVMValue IRMapCompiler::LoadPortElementVariable(const PortElementBase& element)
    {
        auto& currentFunction = GetModule().GetCurrentFunction();
        emitters::Variable* pVar = GetVariableForPort(*element.ReferencedPort());
        if (pVar->IsScalar())
        {
            throw emitters::EmitterException(emitters::EmitterError::vectorVariableExpected, "Error: got a scalar port variable");
        }
        else if (element.GetIndex() >= pVar->Dimension())
        {
            throw emitters::EmitterException(emitters::EmitterError::indexOutOfRange);
        }

        emitters::LLVMValue pVal = GetModule().EnsureEmitted(*pVar);
        auto valType = pVal->getType();
        bool needsDereference = valType->isPointerTy(); // TODO: Maybe this should be `isPtrOrPtrVectorTy()` or even `isPtrOrPtrVectorTy() || isArrayTy()`
        if (needsDereference)
        {
            return currentFunction.ValueAt(pVal, currentFunction.Literal((int)element.GetIndex()));
        }
        else
        {
            return pVal;
        }
    }
} // namespace model
} // namespace ell
