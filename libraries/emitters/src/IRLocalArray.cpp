////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRLocalArray.cpp (emitter)
//  Authors:  Chuck Jacobs, Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRLocalArray.h"
#include "IRFunctionEmitter.h"

#include <llvm/IR/Value.h>

namespace ell
{
namespace emitters
{
    IRLocalArray::IRLocalArrayValue IRLocalArray::operator[](int offset) const
    {
        return IRLocalArrayValue(function, value, function.Literal(offset));
    }

    IRLocalArray::IRLocalArrayValue IRLocalArray::operator[](IRLocalScalar offset) const
    {
        return IRLocalArrayValue(function, value, offset);
    }

    IRLocalArray::IRLocalArrayValue::IRLocalArrayValue(
        emitters::IRFunctionEmitter& function,
        LLVMValue value,
        LLVMValue pOffset) :
        _function(function),
        _pPointer(value),
        _pOffset(pOffset)
    {}

    IRLocalArray::IRLocalArrayValue& IRLocalArray::IRLocalArrayValue::operator=(const IRLocalArrayValue& value)
    {
        // cast the rhs to IRLocalScalar, which decomposes into LLVMValue thanks to the overload below
        *this = static_cast<IRLocalScalar>(value);

        return *this;
    }

    IRLocalArray::IRLocalArrayValue::operator IRLocalScalar() const
    {
        return _function.LocalScalar(_function.ValueAt(_pPointer, _pOffset));
    }

    IRLocalArray::IRLocalArrayValue& IRLocalArray::IRLocalArrayValue::operator=(LLVMValue value)
    {
        _function.SetValueAt(_pPointer, _pOffset, value);

        return *this;
    }
} // namespace emitters
} // namespace ell
