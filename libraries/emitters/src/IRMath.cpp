////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRLocalScalar.cpp (emitter)
//  Authors:  Chuck Jacobs, Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRMath.h"
#include "EmitterException.h"
#include "IRModuleEmitter.h"

#include <utilities/include/Exception.h>

#include <llvm/IR/Value.h>

#include <functional>

namespace ell
{
namespace emitters
{
    //
    // Math functions
    //
    template <typename ValueType>
    IRLocalScalar Tanh(IRLocalScalar a)
    {
        auto f = a.function.GetModule().GetRuntime().GetTanhFunction<ValueType>();
        return { a.function, a.function.Call(f, { a }) };
    }

    IRLocalScalar Abs(IRLocalScalar a)
    {
        auto f = a.function.GetModule().GetRuntime().GetAbsFunction((a.value)->getType());
        return { a.function, a.function.Call(f, { a }) };
    }

    IRLocalScalar Sqrt(IRLocalScalar a)
    {
        auto f = a.function.GetModule().GetRuntime().GetSqrtFunction((a.value)->getType());
        return { a.function, a.function.Call(f, { a }) };
    }

    IRLocalScalar Exp(IRLocalScalar a)
    {
        auto f = a.function.GetModule().GetRuntime().GetExpFunction((a.value)->getType());
        return { a.function, a.function.Call(f, { a }) };
    }

    IRLocalScalar Log(IRLocalScalar a)
    {
        auto f = a.function.GetModule().GetRuntime().GetLogFunction((a.value)->getType());
        return { a.function, a.function.Call(f, { a }) };
    }

    IRLocalScalar Sin(IRLocalScalar a)
    {
        auto f = a.function.GetModule().GetRuntime().GetSinFunction((a.value)->getType());
        return { a.function, a.function.Call(f, { a }) };
    }

    IRLocalScalar Cos(IRLocalScalar a)
    {
        auto f = a.function.GetModule().GetRuntime().GetCosFunction((a.value)->getType());
        return { a.function, a.function.Call(f, { a }) };
    }

    IRLocalScalar Min(IRLocalScalar a, IRLocalScalar b)
    {
        detail::VerifyArgTypesCompatible(a, b);
        LLVMValue result = a.function.Select(a < b, a, b);
        return { a.function, result };
    }

    IRLocalScalar Max(IRLocalScalar a, IRLocalScalar b)
    {
        detail::VerifyArgTypesCompatible(a, b);
        LLVMValue result = a.function.Select(a >= b, a, b);
        return { a.function, result };
    }

    // explicit instantiations
    template IRLocalScalar Tanh<float>(IRLocalScalar a);
    template IRLocalScalar Tanh<double>(IRLocalScalar a);
    template IRLocalScalar Tanh<int>(IRLocalScalar a);

} // namespace emitters
} // namespace ell
