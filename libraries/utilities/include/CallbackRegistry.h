////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CallbackRegistry.h (utilities)
//  Authors:  Chris Lovett
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <functional>
#include <map>
#include <vector>

namespace ell
{
namespace utilities
{
    /// <summary> Class that manages std::function callbacks </summary>
    template <typename ElementType>
    class CallbackRegistry
    {
    public:
        std::vector<std::string> GetSourceFunctionNames();
        void RegisterSourceCallback(std::string name, std::function<bool(std::vector<ElementType>&)> func);
        int GetSourceCallbackIndex(std::string name);
        std::function<bool(std::vector<ElementType>&)> GetSourceCallback(int index);

        std::vector<std::string> GetSinkFunctionNames();
        void RegisterSinkCallback(std::string name, std::function<void(const std::vector<ElementType>&)> func);
        int GetSinkCallbackIndex(std::string name);
        std::function<void(const std::vector<ElementType>&)> GetSinkCallback(int index);

        bool HasCallbackFunctions() const;

    private:
        std::map<std::string, int> _sourceCallbackMap;
        std::vector<std::function<bool(std::vector<ElementType>&)>> _sourceCallbacks;
        std::map<std::string, int> _sinkCallbackMap;
        std::vector<std::function<void(const std::vector<ElementType>&)>> _sinkCallbacks;
    };

} // namespace utilities
} // namespace ell

#pragma region implementation

namespace ell
{
namespace utilities
{

    template <typename ElementType>
    void CallbackRegistry<ElementType>::RegisterSourceCallback(std::string name, std::function<bool(std::vector<ElementType>&)> func)
    {
        _sourceCallbackMap[name] = _sourceCallbacks.size();
        _sourceCallbacks.push_back(func);
    }

    template <typename ElementType>
    int CallbackRegistry<ElementType>::GetSourceCallbackIndex(std::string name)
    {
        return _sourceCallbackMap[name];
    }

    template <typename ElementType>
    std::function<bool(std::vector<ElementType>&)> CallbackRegistry<ElementType>::GetSourceCallback(int index)
    {
        return _sourceCallbacks[index];
    }

    template <typename ElementType>
    void CallbackRegistry<ElementType>::RegisterSinkCallback(std::string name, std::function<void(const std::vector<ElementType>&)> func)
    {
        _sinkCallbackMap[name] = _sinkCallbacks.size();
        _sinkCallbacks.push_back(func);
    }

    template <typename ElementType>
    int CallbackRegistry<ElementType>::GetSinkCallbackIndex(std::string name)
    {
        return _sinkCallbackMap[name];
    }

    template <typename ElementType>
    std::function<void(const std::vector<ElementType>&)> CallbackRegistry<ElementType>::GetSinkCallback(int index)
    {
        return _sinkCallbacks[index];
    }

    template <typename ElementType>
    std::vector<std::string> CallbackRegistry<ElementType>::GetSourceFunctionNames()
    {
        std::vector<std::string> result;
        for (auto pair : _sourceCallbackMap)
        {
            result.push_back(pair.first());
        }
        return result;
    }

    template <typename ElementType>
    std::vector<std::string> CallbackRegistry<ElementType>::GetSinkFunctionNames()
    {
        std::vector<std::string> result;
        for (auto pair : _sinkCallbackMap)
        {
            result.push_back(pair.first());
        }
        return result;
    }

    template <typename ElementType>
    bool CallbackRegistry<ElementType>::HasCallbackFunctions() const
    {
        return !_sinkCallbacks.empty() || !_sourceCallbacks.empty();
    }

} // namespace utilities
} // namespace ell

#pragma endregion implementation
