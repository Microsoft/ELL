////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     TestUtil.cpp (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TestUtil.h"

#include <value/include/Array.h>
#include <value/include/ComputeContext.h>
#include <value/include/FunctionDeclaration.h>
#include <value/include/LLVMContext.h>
#include <value/include/Matrix.h>
#include <value/include/Tensor.h>
#include <value/include/Value.h>
#include <value/include/Vector.h>

#include <value/include/loopnests/Kernel.h>
#include <value/include/loopnests/LoopNest.h>

#include <math/include/Matrix.h>
#include <math/include/Tensor.h>
#include <math/include/Vector.h>

#include <utilities/include/FunctionUtils.h>

#include <testing/include/testing.h>

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <type_traits>
#include <vector>

#if !defined(WIN32)
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>
#else
#include <windows.h>
#endif // !defined(WIN32)

using namespace ell::emitters;
using namespace ell::utilities;
using namespace ell::value;
namespace math = ell::math;
using math::Dimension;
using math::MatrixLayout;

template <MatrixLayout layout>
using LayoutType = std::integral_constant<MatrixLayout, layout>;

using namespace ell::emitters;
using namespace ell::utilities;
using namespace ell::value;

namespace ell
{
void PrintMatrix(std::string indent, Matrix e)
{
    if (!e.GetValue().IsConstant())
    {
        std::cout << "cannot print non constant matrix";
        return;
    }
    if (e.Type() == ValueType::Undefined)
    {
        std::cout << "Undefined";
        return;
    }
    else if (e.Type() == ValueType::Void)
    {
        std::cout << "void";
        return;
    }
    std::cout << std::setprecision(10);
    int rows = static_cast<int>(e.Rows());
    int cols = static_cast<int>(e.Columns());
    for (int i = 0; i < rows; i++)
    {
        std::cout << indent;
        for (int j = 0; j < cols; j++)
        {
            if (j > 0)
            {
                std::cout << ", ";
            }
            Scalar s = e(i, j);
            switch (s.GetType())
            {
            case ValueType::Undefined:
                break;
            case ValueType::Void:
                break;
            case ValueType::Boolean:
                std::cout << (bool)s.Get<Boolean>();
                break;
            case ValueType::Char8:
                std::cout << s.Get<char>();
                break;
            case ValueType::Byte:
                std::cout << s.Get<uint8_t>();
                break;
            case ValueType::Int16:
                std::cout << s.Get<int16_t>();
                break;
            case ValueType::Int32:
                std::cout << s.Get<int32_t>();
                break;
            case ValueType::Int64:
                std::cout << s.Get<int64_t>();
                break;
            case ValueType::Float:
                std::cout << s.Get<float>();
                break;
            case ValueType::Double:
                std::cout << s.Get<double>();
                break;
            }
        }
        std::cout << "\n";
    }
}

void PrintLoops(const value::loopnests::LoopNest& loop, std::string tag)
{
    InvokeForContext<ComputeContext>([&](auto&) {
        std::stringstream ss;
        DebugDump(loop, tag, &ss);
        std::cout << ss.str() << std::endl;
    });
}

Scalar EqualEpsilon(Scalar x, Scalar y, double epsilon)
{
    Scalar result = Allocate<int>(ScalarLayout);
    result = 1;

    If(x == y, [&] {
#if 0 // Useful for debugging
        DebugPrint("## Scalar compare passed (exactly equal)\n");
        DebugPrint("  Expected: ");
        DebugPrintVector(AsVector(x));
        DebugPrint("\n");
        DebugPrint("  Actual:   ");
        DebugPrintVector(AsVector(y));
        DebugPrint("\n");
#endif // 0
        result = 1;
    }).Else([&] {
        if (auto type = x.GetType(); type == ValueType::Float || type == ValueType::Double)
        {
            auto tolerance = Cast<Scalar>(epsilon, type);
            If((x - y) <= tolerance, [&] {
                If((y - x) <= tolerance, [&] {
#if 0 // Useful for debugging
                    DebugPrint("## Scalar compare passed\n");
                    DebugPrint("  Expected: ");
                    DebugPrintVector(AsVector(x));
                    DebugPrint("\n");
                    DebugPrint("  Actual:   ");
                    DebugPrintVector(AsVector(y));
                    DebugPrint("\n");
#endif // 0
                    result = 1;
                });
            });
        }
        else
        {
            result = 0;
        }
    });

    return result;
}

Scalar NotEqualEpsilon(Scalar x, Scalar y, double epsilon)
{
    auto result = EqualEpsilon(x, y, epsilon);

    // TODO: overload the logical not operator
    If(result == 1, [&] { result = 0; }).Else([&] { result = 1; });
    return result;
}

Scalar VerifySame(Vector actual, Vector expected, double epsilon)
{
    Scalar ok = Allocate(ValueType::Int32, ScalarLayout);
    For(actual, [&](Scalar index) {
        Scalar x = actual(index);
        Scalar y = expected(index);

        If(ok == 0, [&] {
            ok = NotEqualEpsilon(x, y, epsilon);
        });
    });

    If(ok != 0, [&] {
        DebugPrint("## Vector compare failed\n");
        DebugPrint("  Expected: ");
        DebugPrintVector(expected);
        DebugPrint("\n");
        DebugPrint("  Actual:   ");
        DebugPrintVector(actual);
        DebugPrint("\n");
    });
    return ok;
}

Scalar VerifyDifferent(Vector actual, Vector expected, double epsilon)
{
    Scalar ok = Allocate(ValueType::Int32, ScalarLayout);
    For(actual, [&](Scalar index) {
        Scalar x = actual(index);
        Scalar y = expected(index);
        If(ok == 0, [&] {
            ok = EqualEpsilon(x, y, epsilon);
        });
    });
    If(ok != 0, [&] {
        DebugPrint("## Vector are not different\n");
        DebugPrint("  Expected: ");
        DebugPrintVector(expected);
        DebugPrint("\n");
        DebugPrint("  Actual:   ");
        DebugPrintVector(actual);
        DebugPrint("\n");
    });

    return ok;
}

Scalar VerifySame(Matrix actual, Matrix expected, double epsilon)
{
    Scalar ok = Allocate(ValueType::Int32, ScalarLayout);
    For(actual, [&](Scalar row, Scalar col) {
        Scalar x = actual(row, col);
        Scalar y = expected(row, col);
        If(ok == 0, [&] {
            ok = NotEqualEpsilon(x, y, epsilon);
        });
    });
    If(ok != 0, [&] {
        DebugPrint("## Matrices are different\n");
        InvokeForContext<ComputeContext>([&] {
            std::cout << "Expected: \n";
            PrintMatrix("   ", expected);
            std::cout << "\n";
            std::cout << "Actual:   \n";
            PrintMatrix("   ", actual);
            std::cout << "\n";
        });
    });
    return ok;
}

Scalar VerifyDifferent(Matrix actual, Matrix expected, double epsilon)
{
    Scalar ok = Allocate(ValueType::Int32, ScalarLayout);
    For(actual, [&](Scalar row, Scalar col) {
        Scalar x = actual(row, col);
        Scalar y = expected(row, col);
        If(ok == 0, [&] {
            ok = EqualEpsilon(x, y, epsilon);
        });
    });
    If(ok != 0, [&] {
        DebugPrint("## Matrices are not different\n");
        InvokeForContext<ComputeContext>([&] {
            std::cout << "Expected: \n";
            PrintMatrix("   ", expected);
            std::cout << "\n";
            std::cout << "Actual:   \n";
            PrintMatrix("   ", actual);
            std::cout << "\n";
        });
    });
    return ok;
}

Scalar VerifySame(Tensor actual, Tensor expected, double epsilon)
{
    Scalar ok = Allocate(ValueType::Int32, ScalarLayout);
    For(actual, [&](Scalar row, Scalar col, Scalar ch) {
        Scalar x = actual(row, col, ch);
        Scalar y = expected(row, col, ch);
        If(ok == 0, [&] {
            ok = NotEqualEpsilon(x, y, epsilon);
        });
    });
    return ok;
}

Scalar VerifyDifferent(Tensor actual, Tensor expected, double epsilon)
{
    Scalar fail = Cast(1, ValueType::Int32);
    Scalar ok = Allocate(ValueType::Int32, ScalarLayout);
    ok = 0;
    For(actual, [&](Scalar row, Scalar col, Scalar ch) {
        Scalar x = actual(row, col, ch);
        Scalar y = expected(row, col, ch);
        If(ok == 0, [&] {
            ok = EqualEpsilon(x, y, epsilon);
        });
    });
    return ok;
}

Scalar VerifySame(Array actual, Array expected, double epsilon)
{
    Scalar ok = Allocate(ValueType::Int32, ScalarLayout);
    For(actual, [&](const std::vector<Scalar>& indices) {
        Scalar x = actual(indices);
        Scalar y = expected(indices);
        If(ok == 0, [&] {
            ok = NotEqualEpsilon(x, y, epsilon);
        });
    });
    return ok;
}

Scalar VerifyDifferent(Array actual, Array expected, double epsilon)
{
    Scalar fail = Cast(1, ValueType::Int32);
    Scalar ok = Allocate(ValueType::Int32, ScalarLayout);
    ok = 0;
    For(actual, [&](const std::vector<Scalar>& indices) {
        Scalar x = actual(indices);
        Scalar y = expected(indices);
        If(ok == 0, [&] {
            ok = EqualEpsilon(x, y, epsilon);
        });
    });
    return ok;
}

Scalar GetTID()
{
    if (auto result = InvokeForContext<ComputeContext>(
            [](auto&) {
#if !defined(WIN32)
#if defined(__APPLE__)
#pragma message("Note: syscall() is deprecated in macOS")
    // Note: syscall() is deprecated in macOS, perhaps use pthread_self instead:
    // return static_cast<int32_t>(reinterpret_cast<uintptr_t>(pthread_self()));
#endif // defined(__APPLE__)
                return (int32_t)(pid_t)syscall(SYS_gettid);
#else
                return (int32_t)GetCurrentThreadId();
#endif // !defined(WIN32)
            });
        result)
    {
        return *result;
    }

    return Scalar(
#if !defined(WIN32)
#if defined(__APPLE__)
#pragma message("Note: syscall() is deprecated in macOS")
#endif // defined(__APPLE__)
        Cast<int>(
            *DeclareFunction("syscall")
                 .Decorated(false)
                 .Returns(Value({ ValueType::Int64, 0 }, ScalarLayout))
                 .Parameters(
                     Value({ ValueType::Int64, 0 }, ScalarLayout))
                 .Call(Scalar{ (int64_t)SYS_gettid }))
#else
        *DeclareFunction("GetCurrentThreadId")
             .Decorated(false)
             .Returns(Value({ ValueType::Int32, 0 }, ScalarLayout))
             .Call()
#endif // !defined(WIN32)
    );
}

void MultiplyMatrices(Matrix& A, Matrix& B, Matrix& C)
{
    auto M = static_cast<int>(C.Rows());
    auto N = static_cast<int>(C.Columns());
    auto K = static_cast<int>(A.Columns());

    // fill out expected with a simple for-loop gemm
    ForRange(M, [&](Scalar i) {
        ForRange(N, [&](Scalar j) {
            C(i, j) = 0;
        });
    });
    ForRange(M, [&](Scalar i) {
        ForRange(N, [&](Scalar j) {
            ForRange(K, [&](Scalar k) {
                C(i, j) += A(i, k) * B(k, j);
            });
        });
    });
}

MatMul3TestCaseParameters GetMatMul3TestCaseParameters(int M, int N, int K, int L)
{
    auto A = MakeMatrix<int>(M, K, "A");
    auto B = MakeMatrix<int>(K, N, "B");
    auto C = MakeMatrix<int>(M, N, "C");
    auto D = MakeMatrix<int>(N, L, "D");
    auto E = MakeMatrix<int>(M, L, "E");

    auto expectedC = MakeMatrix<int>(M, N, "expectedC");
    auto expectedE = MakeMatrix<int>(M, L, "expectedE");

    // initialize matrices
    ForRange(M, [&](Scalar i) {
        ForRange(K, [&](Scalar j) {
            A(i, j) = i - j;
        });
    });

    ForRange(K, [&](Scalar i) {
        ForRange(N, [&](Scalar j) {
            B(i, j) = i + 2 * j;
        });
    });

    ForRange(M, [&](Scalar i) {
        ForRange(N, [&](Scalar j) {
            C(i, j) = 0;
        });
    });

    ForRange(N, [&](Scalar i) {
        ForRange(L, [&](Scalar j) {
            D(i, j) = j - i;
        });
    });

    ForRange(M, [&](Scalar i) {
        ForRange(L, [&](Scalar j) {
            E(i, j) = 0;
        });
    });

    // fill out expected results with a simple for-loop gemm
    // C
    ForRange(M, [&](Scalar i) {
        ForRange(N, [&](Scalar j) {
            expectedC(i, j) = 0;
        });
    });
    ForRange(M, [&](Scalar i) {
        ForRange(N, [&](Scalar j) {
            ForRange(K, [&](Scalar k) {
                expectedC(i, j) += A(i, k) * B(k, j);
            });
        });
    });

    // E
    ForRange(M, [&](Scalar i) {
        ForRange(L, [&](Scalar l) {
            expectedE(i, l) = 0;
        });
    });
    ForRange(M, [&](Scalar i) {
        ForRange(N, [&](Scalar j) {
            ForRange(L, [&](Scalar l) {
                expectedE(i, l) += expectedC(i, j) * D(j, l);
            });
        });
    });

    return { M, N, K, L, A, B, C, D, E, expectedC, expectedE };
}

} // namespace ell
