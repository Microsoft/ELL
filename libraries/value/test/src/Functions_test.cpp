////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Functions_test.cpp (value)
//  Authors:  Kern Handa, Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Functions_test.h"
#include "TestUtil.h"

#include <value/include/ComputeContext.h>
#include <value/include/FunctionDeclaration.h>
#include <value/include/Scalar.h>

#include <vector>

#if !defined(WIN32)
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#else
#include <windows.h>
#endif // !defined(WIN32)

using namespace ell::utilities;
using namespace ell::value;

#define PRINT_IR 0

namespace ell
{
Scalar FunctionArgType_test()
{
    auto fn = DeclareFunction("FunctionArgType_test")
                  .Parameters(
                      Value(ValueType::Float, ScalarLayout),
                      Value({ ValueType::Float, 0 }, ScalarLayout),
                      Value(ValueType::Int32, ScalarLayout),
                      Value({ ValueType::Int32, 0 }, ScalarLayout))
                  .Returns(Value(ValueType::Int32, ScalarLayout))
                  .Define([](Scalar f, Scalar f0, Scalar i, Scalar i0) {
                      auto ff = MakeScalar<float>();
                      auto ff0 = MakeScalar<float>();
                      auto ii = MakeScalar<int>();
                      auto ii0 = MakeScalar<int>();

                      ff = f;
                      ff0 = f0;
                      ii = i;
                      ii0 = i0;
                      return Scalar(0);
                  });

    auto arg1 = MakeScalar<float>();
    auto arg2 = MakeScalar<float>();
    auto arg3 = MakeScalar<int>();
    auto arg4 = MakeScalar<int>();
    return fn(arg1, arg2, arg3, arg4);
}
} // namespace ell
