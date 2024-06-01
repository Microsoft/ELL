////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRProfiler.cpp (emitters)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRProfiler.h"
#include "EmitterException.h"
#include "IRFunctionEmitter.h"
#include "IRMetadata.h"
#include "IRModuleEmitter.h"
#include "LLVMUtilities.h"

#include <utilities/include/UniqueId.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
#include <string>

namespace ell
{
namespace emitters
{
    namespace
    {
        enum class RegionInfoFields
        {
            count = 0,
            totalTime = 1,
            name = 2
        };
    }

    //
    // IRProfileRegionBlock
    //
    IRProfileRegionBlock::IRProfileRegionBlock(IRFunctionEmitter& function, const std::string& name) :
        _region(function, name)
    {
        _region.Enter();
    }

    IRProfileRegionBlock::~IRProfileRegionBlock()
    {
        _region.Exit();
    }

    //
    // IRProfileRegion
    //
    IRProfileRegion::IRProfileRegion(IRFunctionEmitter& function, const std::string& name) :
        _function(function),
        _profiler(function.GetModule().GetProfiler()),
        _index(function.LocalScalar()),
        _startTime(function.LocalScalar())
    {
        _index = _profiler.CreateRegion(_function);
        _profiler.InitRegion(*this, name);
    }

    // Private constructor for use by IRProfiler
    IRProfileRegion::IRProfileRegion(IRFunctionEmitter& function, IRProfiler& profiler, const std::string& name, IRLocalScalar index) :
        _function(function),
        _profiler(profiler),
        _index(index),
        _startTime(function.LocalScalar())
    {
    }

    void IRProfileRegion::Enter()
    {
        _profiler.EnterRegion(*this);
    }

    void IRProfileRegion::Exit()
    {
        _profiler.ExitRegion(*this);
    }

    bool IRProfileRegion::IsStartTimeValid() const
    {
        return _startTime.IsValid();
    }

    //
    // IRProfiler
    //
    IRProfiler::IRProfiler(emitters::IRModuleEmitter& module, bool enableProfiling) :
        _module(&module),
        _profilingEnabled(enableProfiling)
    {
    }

    void IRProfiler::Init() // Called by IRModuleEmitter
    {
        if (!_profilingEnabled)
            return;

        assert(_module != nullptr);

        _module->DeclarePrintf();
        CreateStructTypes();
        CreateRegionData();
        EmitProfilerFunctions();
    }

    std::string IRProfiler::GetGetNumRegionsFunctionName() const
    {
        return GetNamespacePrefix() + "_GetNumProfileRegions";
    }

    std::string IRProfiler::GetGetRegionProfilingInfoFunctionName() const
    {
        return GetNamespacePrefix() + "_GetRegionProfilingInfo";
    }

    std::string IRProfiler::GetResetRegionProfilingInfoFunctionName() const
    {
        return GetNamespacePrefix() + "_ResetRegionProfilingInfo";
    }

    std::string IRProfiler::GetNamespacePrefix() const
    {
        return _module->GetModuleName();
    }

    llvm::StructType* IRProfiler::GetRegionType() const
    {
        return _profileRegionType;
    }

    IRLocalScalar IRProfiler::GetCurrentTime(IRFunctionEmitter& function)
    {
        auto time = function.GetModule().GetRuntime().GetCurrentTime(function);
        return function.LocalScalar(time);
    }

    void IRProfiler::InitRegion(IRProfileRegion& region, const std::string& desiredName)
    {
        if (!_profilingEnabled)
            return;

        auto regionName = GetUniqueRegionName(desiredName);
        _regionNames.insert(regionName);

        auto& function = region.GetFunction();
        auto regionPtr = GetRegionPointer(function, region.GetIndex());

        // Set the name
        auto namePtr = function.GetStructFieldPointer(regionPtr, static_cast<size_t>(RegionInfoFields::name));
        function.Store(namePtr, function.Literal(regionName));
    }

    void IRProfiler::EnterRegion(IRProfileRegion& region)
    {
        if (!_profilingEnabled)
            return;

        auto& function = region.GetFunction();

        // Get the time
        auto startTime = GetCurrentTime(function);
        region.SetStartTime(startTime);

        // Increment visit count
        auto regionPtr = GetRegionPointer(function, region.GetIndex());
        auto countPtr = function.GetStructFieldPointer(regionPtr, static_cast<size_t>(RegionInfoFields::count));
        auto count = function.LocalScalar(function.Load(countPtr));
        function.Store(countPtr, count + static_cast<int64_t>(1));
    }

    void IRProfiler::ExitRegion(IRProfileRegion& region)
    {
        if (!_profilingEnabled)
            return;

        // Increment stored time
        auto& function = region.GetFunction();
        auto regionPtr = GetRegionPointer(function, region.GetIndex());
        auto timePtr = function.GetStructFieldPointer(regionPtr, static_cast<size_t>(RegionInfoFields::totalTime));
        auto startTime = region.GetStartTime();
        auto newTime = GetCurrentTime(function) - startTime;
        auto storedTime = function.LocalArray(timePtr);
        storedTime[0] = storedTime[0] + newTime;

        // reset start time to "unassigned"
        region.SetStartTime(function.LocalScalar());
    }

    void IRProfiler::ResetRegionCounts(IRFunctionEmitter& function, const IRLocalScalar& regionIndex)
    {
        if (!_profilingEnabled)
            return;

        // Reset stored time and count
        auto regionPtr = GetRegionPointer(function, regionIndex);
        auto countPtr = function.GetStructFieldPointer(regionPtr, static_cast<size_t>(RegionInfoFields::count));
        auto timePtr = function.GetStructFieldPointer(regionPtr, static_cast<size_t>(RegionInfoFields::totalTime));
        function.StoreZero(countPtr);
        function.StoreZero(timePtr);
    }

    std::string IRProfiler::GetUniqueRegionName(const std::string& desiredName) const
    {
        // Ensure name is unique
        auto regionName = desiredName;
        int index = 1;
        while (_regionNames.find(regionName) != _regionNames.end())
        {
            regionName = desiredName + "_" + std::to_string(index++);
        }
        return regionName;
    }

    IRLocalScalar IRProfiler::CreateRegion(IRFunctionEmitter& function)
    {
        if (!_profilingEnabled)
            return function.LocalScalar();

        auto index = _regionCount;
        ++_regionCount;

        // create new count and array constants
        ReallocateRegionData();
        return function.LocalScalar<int>(index);
    }

    void IRProfiler::CreateStructTypes()
    {
        assert(_profilingEnabled);
        auto& context = _module->GetLLVMContext();

        auto int64Type = llvm::Type::getInt64Ty(context);
        auto doubleType = llvm::Type::getDoubleTy(context);
        auto int8PtrType = llvm::Type::getInt8PtrTy(context);

        // ProfileRegionInfo struct fields
        emitters::NamedLLVMTypeList infoFields = { { "count", int64Type }, { "totalTime", doubleType }, { "name", int8PtrType } };
        _profileRegionType = _module->GetOrCreateStruct(GetNamespacePrefix() + "_ProfileRegionInfo", infoFields);
        _module->IncludeTypeInHeader(_profileRegionType->getName());
    }

    void IRProfiler::EmitProfilerFunctions()
    {
        if (!_profilingEnabled)
            return;

        assert(_module != nullptr);

        EmitGetRegionBufferFunction();
        EmitGetNumRegionsFunction();
        EmitGetRegionProfilingInfoFunction();
        EmitResetRegionProfilingInfoFunction();
    }

    void IRProfiler::CreateRegionData()
    {
        assert(_profileRegionsArray == nullptr);
        _profileRegionsArray = _module->GlobalArray(GetNamespacePrefix() + "_profileprofileRegionsArray_" + std::to_string(_regionCount), _profileRegionType, _regionCount);
    }

    void IRProfiler::ReallocateRegionData()
    {
        FixUpGetNumRegionsFunction();

        // reallocate the global array --- we use a new name to avoid having LLVM just give us back the existing one
        auto profileRegionsArray = _module->GlobalArray(GetNamespacePrefix() + "_profileprofileRegionsArray_" + std::to_string(_regionCount), _profileRegionType, _regionCount);
        if (_profileRegionsArray != profileRegionsArray)
        {
            if (_profileRegionsArray != nullptr)
            {
                _profileRegionsArray->removeFromParent();
            }
            _profileRegionsArray = profileRegionsArray;

            // Each time profileRegionsArray is updated, rewrite the GetRegionBuffer function to return the new profileRegionsArray`
            FixUpGetRegionBufferFunction();
        }
    }

    void IRProfiler::EmitGetNumRegionsFunction()
    {
        auto function = _module->BeginFunction(GetGetNumRegionsFunctionName(), VariableType::Int32);
        function.IncludeInHeader();
        function.Return(function.Literal(_regionCount));
        _module->EndFunction();
        _getNumRegionsFunction = function.GetFunction();
    }

    void IRProfiler::FixUpGetNumRegionsFunction()
    {
        assert(_getNumRegionsFunction != nullptr);

        auto& exitBlock = _getNumRegionsFunction->back();
        auto oldReturnInstr = exitBlock.getTerminator();
        auto& emitter = _module->GetIREmitter();

        // save old insert point
        auto oldInsertPoint = emitter.GetCurrentInsertPoint();
        emitter.SetCurrentBlock(&exitBlock);

        // add new return instruction
        emitter.Return(emitter.Literal(_regionCount));

        // remove old return instruction
        oldReturnInstr->removeFromParent();

        emitter.SetCurrentInsertPoint(oldInsertPoint);
    }

    void IRProfiler::EmitGetRegionBufferFunction()
    {
        assert(_profileRegionsArray != nullptr);
        assert(_profileRegionType != nullptr);
        auto returnType = _profileRegionType->getPointerTo();

        auto function = _module->BeginFunction("GetRegionBuffer", returnType);

        // cast _profileRegionsArray to ptr-to-region
        auto castPtr = function.CastPointer(_profileRegionsArray, returnType);
        function.Return(castPtr);
        _module->EndFunction();
        _getRegionBufferFunction = function.GetFunction();
    }

    void IRProfiler::FixUpGetRegionBufferFunction()
    {
        assert(_getRegionBufferFunction != nullptr);

        auto& exitBlock = _getRegionBufferFunction->back();
        auto oldReturnInstr = exitBlock.getTerminator();
        auto& emitter = _module->GetIREmitter();

        // save old insert point
        auto oldInsertPoint = emitter.GetCurrentInsertPoint();
        emitter.SetCurrentBlock(&exitBlock);

        // add new return instruction
        auto returnType = _profileRegionType->getPointerTo();
        auto castPtr = emitter.CastPointer(_profileRegionsArray, returnType);
        emitter.Return(castPtr);

        // remove old return instruction
        oldReturnInstr->removeFromParent();

        emitter.SetCurrentInsertPoint(oldInsertPoint);
    }

    void IRProfiler::EmitGetRegionProfilingInfoFunction()
    {
        // TODO: bounds-checking
        const emitters::NamedVariableTypeList parameters = { { "regionIndex", emitters::VariableType::Int32 } };
        auto function = _module->BeginFunction(GetGetRegionProfilingInfoFunctionName(), _profileRegionType->getPointerTo(), parameters);
        function.IncludeInHeader();

        auto regionIndex = function.GetFunctionArgument("regionIndex");
        auto regionPtr = GetRegionPointer(function, regionIndex);
        function.Return(regionPtr);
        _module->EndFunction();
    }

    void IRProfiler::EmitResetRegionProfilingInfoFunction()
    {
        auto function = _module->BeginFunction(GetResetRegionProfilingInfoFunctionName(), VariableType::Void);
        function.IncludeInHeader();
        function.IncludeInSwigInterface();

        auto numRegions = GetNumRegions(function);
        function.For(numRegions, [this](IRFunctionEmitter& function, auto regionIndex) {
            this->ResetRegionCounts(function, regionIndex);
        });

        _module->EndFunction();
    }

    LLVMValue IRProfiler::GetRegionBuffer(IRFunctionEmitter& function)
    {
        assert(_getRegionBufferFunction != nullptr);
        return function.Call(_getRegionBufferFunction, {});
    }

    LLVMValue IRProfiler::GetNumRegions(emitters::IRFunctionEmitter& function)
    {
        assert(_getNumRegionsFunction != nullptr);
        return function.Call(_getNumRegionsFunction, {});
    }

    LLVMValue IRProfiler::GetRegionPointer(emitters::IRFunctionEmitter& function, LLVMValue index)
    {
        auto regions = GetRegionBuffer(function);
        auto regionPtr = function.PointerOffset(regions, index);
        return regionPtr;
    }
} // namespace emitters
} // namespace ell
