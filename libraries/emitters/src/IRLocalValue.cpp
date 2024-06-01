////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRLocalValue.cpp (emitter)
//  Authors:  Chuck Jacobs, Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRLocalValue.h"
#include "IRFunctionEmitter.h"
#include "IRLocalScalar.h"

#include <utilities/include/Exception.h>

#include <llvm/IR/Value.h>

#include <cassert>

namespace ell
{
namespace emitters
{
    //
    // Utility code
    //
    namespace
    {
        void VerifyFromSameFunction(const IRLocalValue& a, const IRLocalValue& b)
        {
            if (a.function.GetFunction() != b.function.GetFunction())
            {
                throw EmitterException(EmitterError::badFunctionArguments, "IRLocalValue arguments are local to different functions");
            }
        }
    } // namespace

    namespace detail
    {
        bool BothIntegral(const IRLocalValue& a, const IRLocalValue& b)
        {
            return (a.value->getType()->getScalarType()->isIntegerTy() && b.value->getType()->getScalarType()->isIntegerTy());
        }

        bool BothFloatingPoint(const IRLocalValue& a, const IRLocalValue& b)
        {
            return (a.value->getType()->getScalarType()->isFloatingPointTy() && b.value->getType()->getScalarType()->isFloatingPointTy());
        }

        void VerifyArgTypesCompatible(const IRLocalValue& a, const IRLocalValue& b)
        {
            VerifyFromSameFunction(a, b);
            if (!(BothIntegral(a, b) || BothFloatingPoint(a, b)))
            {
                throw EmitterException(EmitterError::badFunctionArguments, "IRLocalValue arguments have incompatible types");
            }
        }

        // `areCompatible` is a function that returns true if the arg types are compatible. It has the signature: bool(const IRLocalValue& a, const IRLocalValue& b)
        using CompatibleTypesPredicate = std::function<bool(const IRLocalValue&, const IRLocalValue&)>;
        void VerifyArgTypesCompatible(const IRLocalValue& a, const IRLocalValue& b, CompatibleTypesPredicate areCompatible)
        {
            VerifyFromSameFunction(a, b);
            if (!(areCompatible(a, b)))
            {
                throw EmitterException(EmitterError::badFunctionArguments, "IRLocalValue arguments have incompatible types");
            }
        }
    } // namespace detail

    using namespace detail;

    //
    // IRLocalValue
    //
    IRLocalValue::IRLocalValue(emitters::IRFunctionEmitter& function, LLVMValue value) :
        function(function),
        value(value)
    {}

    IRLocalValue& IRLocalValue::operator=(const IRLocalValue& other)
    {
        VerifyFromSameFunction(*this, other);
        this->value = other.value;
        return *this;
    }

    IRLocalValue& IRLocalValue::operator=(LLVMValue value)
    {
        this->value = value;
        return *this;
    }

    //
    // IRLocalPointer
    //
    IRLocalValue IRLocalPointer::Load() const
    {
        return { function, function.Load(value) };
    }

    IRLocalPointer IRLocalPointer::Offset(int offset) const
    {
        return { function, function.PointerOffset(value, offset) };
    }

    IRLocalPointer IRLocalPointer::Offset(LLVMValue offset) const
    {
        return { function, function.PointerOffset(value, offset) };
    }

    IRLocalPointer IRLocalPointer::Offset(const IRLocalScalar& offset) const
    {
        return { function, function.PointerOffset(value, offset) };
    }

} // namespace emitters
} // namespace ell
