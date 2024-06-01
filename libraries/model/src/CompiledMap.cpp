////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CompiledMap.cpp (model)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CompiledMap.h"

namespace ell
{
namespace model
{

    using namespace ell::utilities;

    CompiledMap::CompiledMap(model::Map map, std::string functionName, const MapCompilerOptions& options) :
        Map(std::move(map)),
        _functionName(functionName),
        _compilerOptions(options)
    {}

    template <>
    CallbackRegistry<float>& CompiledMap::GetCallbackRegistry() const
    {
        return const_cast<CallbackRegistry<float>&>(_floatCallbacks);
    }

    template <>
    CallbackRegistry<double>& CompiledMap::GetCallbackRegistry() const
    {
        return const_cast<CallbackRegistry<double>&>(_doubleCallbacks);
    }

    template <>
    CallbackRegistry<int>& CompiledMap::GetCallbackRegistry() const
    {
        return const_cast<CallbackRegistry<int>&>(_intCallbacks);
    }

    template <>
    CallbackRegistry<int64_t>& CompiledMap::GetCallbackRegistry() const
    {
        return const_cast<CallbackRegistry<int64_t>&>(_int64Callbacks);
    }

    template <>
    CallbackRegistry<bool>& CompiledMap::GetCallbackRegistry() const
    {
        return const_cast<CallbackRegistry<bool>&>(_boolCallbacks);
    }

    /// <summary> Returns true if the CallbackRegistry objects contain some functions. </summary>
    bool CompiledMap::HasCallbackFunctions() const
    {
        return _floatCallbacks.HasCallbackFunctions() || _doubleCallbacks.HasCallbackFunctions() || _intCallbacks.HasCallbackFunctions();
    }

} // namespace model
} // namespace ell
