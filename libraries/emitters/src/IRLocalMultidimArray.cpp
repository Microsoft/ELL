////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRLocalMultidimArray.cpp (emitter)
//  Authors:  Chuck Jacobs, Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRLocalMultidimArray.h"
#include "IRFunctionEmitter.h"

#include <llvm/IR/Value.h>

#include <functional>
#include <numeric>

namespace ell
{
namespace emitters
{
    IRLocalMultidimArray::IRLocalMultidimArray(emitters::IRFunctionEmitter& function, LLVMValue data, std::vector<int> extents) :
        IRLocalMultidimArray(function, data, extents, extents)
    {
    }

    IRLocalMultidimArray::IRLocalMultidimArray(emitters::IRFunctionEmitter& function, LLVMValue data, std::vector<int> extents, std::vector<int> memorySize) :
        function(function),
        data(data),
        extents(extents)
    {
        strides.reserve(extents.size());
        std::copy(extents.begin() + 1, extents.end(), std::back_inserter(strides));
        strides.push_back(1);
        int currentStride = 1;
        for (auto it = std::rbegin(strides); it != std::rend(strides); ++it)
        {
            *it *= currentStride;
            currentStride = *it;
        }
    }

    IRLocalMultidimArray::IRLocalArrayElement IRLocalMultidimArray::operator()(std::vector<int> indices) const
    {
        assert(indices.size() == strides.size());
        int offset = 0;
        auto stridesIt = strides.begin();
        for (auto& i : indices)
        {
            offset += i * *stridesIt;
            ++stridesIt;
        }

        assert((offset < std::accumulate(extents.begin(), extents.end(), 1, std::multiplies<int>{})));
        return IRLocalArrayElement(function, data, function.Literal(offset));
    }

    IRLocalMultidimArray::IRLocalArrayElement IRLocalMultidimArray::operator()(std::vector<IRLocalScalar> indices) const
    {
        auto offset = function.LocalScalar(0);
        auto stridesIt = strides.begin();
        for (auto& i : indices)
        {
            offset = offset + (i * *stridesIt);
            ++stridesIt;
        }
        return IRLocalArrayElement(function, data, offset);
    }

    IRLocalPointer IRLocalMultidimArray::PointerTo(std::vector<int> indices) const
    {
        assert(indices.size() == strides.size());
        int offset = 0;
        auto stridesIt = strides.begin();
        for (auto& i : indices)
        {
            offset += i * *stridesIt;
            ++stridesIt;
        }
        assert((offset < std::accumulate(extents.begin(), extents.end(), 1, std::multiplies<int>{})));
        return function.LocalPointer(function.PointerOffset(data, function.Literal(offset)));
    }

    IRLocalPointer IRLocalMultidimArray::PointerTo(std::vector<IRLocalScalar> indices) const
    {
        auto offset = function.LocalScalar(0);
        auto stridesIt = strides.begin();
        for (auto& i : indices)
        {
            offset = offset + (i * *stridesIt);
            ++stridesIt;
        }
        return function.LocalPointer(function.PointerOffset(data, offset));
    }

    IRLocalMultidimArray::IRLocalArrayElement::IRLocalArrayElement(emitters::IRFunctionEmitter& function, LLVMValue data, LLVMValue offset) :
        _function(function),
        _data(data),
        _offset(offset)
    {}

    IRLocalMultidimArray::IRLocalArrayElement& IRLocalMultidimArray::IRLocalArrayElement::operator=(const IRLocalArrayElement& value)
    {
        // cast the rhs to IRLocalScalar, which decomposes into LLVMValue thanks to the overload below
        *this = static_cast<IRLocalScalar>(value);

        return *this;
    }

    IRLocalMultidimArray::IRLocalArrayElement::operator IRLocalScalar() const
    {
        return _function.LocalScalar(_function.ValueAt(_data, _offset));
    }

    IRLocalMultidimArray::IRLocalArrayElement& IRLocalMultidimArray::IRLocalArrayElement::operator=(LLVMValue value)
    {
        _function.SetValueAt(_data, _offset, value);

        return *this;
    }
} // namespace emitters
} // namespace ell
