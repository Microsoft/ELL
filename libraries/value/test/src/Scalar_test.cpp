////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Scalar_test.cpp (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Scalar_test.h"
#include "TestUtil.h"

#include <testing/include/testing.h>

#include <value/include/Reference.h>
#include <value/include/Scalar.h>

#include <utilities/include/MemoryLayout.h>

using namespace ell::utilities;
using namespace ell::value;

namespace ell
{
Scalar Scalar_test1()
{
    Scalar ok = Allocate(ValueType::Int32, ScalarLayout);
    ok = 0;
    Scalar s1 = 1;
    If(s1 != 1, [&] {
        ok = 1;
        DebugPrint("Scalar_test1 if 1 failed\n");
    });

    s1 += 2;

    If(s1 != 3, [&] {
        ok = 1;
        DebugPrint("Scalar_test1 if 2 failed\n");
    });

    Scalar s2 = s1 + 3;

    If(s1 != 3, [&] {
        ok = 1;
        DebugPrint("Scalar_test1 if 3 failed\n");
    });

    If(s2 != 6, [&] {
        ok = 1;
        DebugPrint("Scalar_test1 if 4 failed\n");
    });

    return ok;
}

Scalar Scalar_test2()
{
    Scalar ok = Allocate(ValueType::Int32, ScalarLayout);
    Scalar x = 2.0;
    x = x * x;
    If(x != 4.0, [&] {
        ok = 1;
        DebugPrint("Scalar_test2 failed\n");
    });
    return ok;
}

Scalar ScalarRefTest()
{
    Scalar result = Allocate<int>(ScalarLayout);
    Scalar x = Allocate<int>(ScalarLayout);
    x = 100;

    Scalar expected = Allocate<int>(ScalarLayout);
    expected = 100;

    auto scalarPtr = x.GetValue().Reference();

    Scalar scalar = scalarPtr.Dereference();

    If(scalar != expected, [&] { result = 1; });
    return result;
}

Scalar ScalarRefRefTest()
{
    Scalar result = Allocate<int>(ScalarLayout);
    Scalar x = Allocate<int>(ScalarLayout);
    x = 100;

    Scalar expected = Allocate<int>(ScalarLayout);
    expected = 100;

    auto scalarPtr = x.GetValue().Reference();
    auto scalarPtrPtr = scalarPtr.Reference();

    auto tmpValPtr = scalarPtrPtr.Dereference();
    Scalar scalar = tmpValPtr.Dereference();

    If(scalar != expected, [&] { result = 1; });
    return result;
}

Scalar ScalarRefRefRefTest()
{
    Scalar result = Allocate<int>(ScalarLayout);
    Scalar x = Allocate<int>(ScalarLayout);
    x = 100;

    Scalar expected = Allocate<int>(ScalarLayout);
    expected = 100;

    auto scalarPtr = x.GetValue().Reference();
    auto scalarPtrPtr = scalarPtr.Reference();
    auto scalarPtrPtrPtr = scalarPtrPtr.Reference();

    auto tmpValPtrPtr = scalarPtrPtrPtr.Dereference();
    auto tmpValPtr = tmpValPtrPtr.Dereference();
    Scalar scalar = tmpValPtr.Dereference();

    If(scalar != expected, [&] { result = 1; });
    return result;
}

Scalar RefScalarRefTest()
{
    Scalar result = Allocate<int>(ScalarLayout);
    Scalar x = Allocate<int>(ScalarLayout);
    x = 100;

    Scalar expected = Allocate<int>(ScalarLayout);
    expected = 100;

    Ref<Scalar> scalarPtr = x;

    Scalar scalar = *scalarPtr;

    If(scalar != expected, [&] { result = 1; });
    return result;
}

Scalar RefScalarRefCtorsTest()
{
    Scalar result = Allocate<int>(ScalarLayout);
    Scalar x = Allocate<int>(ScalarLayout);
    x = 100;

    Scalar expected = Allocate<int>(ScalarLayout);
    expected = 100;
    testing::ProcessQuietTest("Value initial pointer level", expected.GetValue().PointerLevel() == 1);
    Ref<Scalar> scalarPtr = x;
    testing::ProcessQuietTest("Ref ctor", scalarPtr.GetValue().PointerLevel() == 2);
    Ref<Scalar> scalarPtrCopy = x;
    testing::ProcessQuietTest("Ref copy semantics", scalarPtr.GetValue().PointerLevel() == scalarPtrCopy.GetValue().PointerLevel());
    Ref<Scalar> scalarPtrMove = std::move(scalarPtr);
    testing::ProcessQuietTest("Ref move semantics", !scalarPtr.GetValue().IsDefined() && scalarPtrMove.GetValue().PointerLevel() == 2);

    return result;
}

Scalar RefScalarRefRefTest()
{
    Scalar result = Allocate<int>(ScalarLayout);
    Scalar x = Allocate<int>(ScalarLayout);
    x = 100;

    Scalar expected = Allocate<int>(ScalarLayout);
    expected = 100;

    Ref<Scalar> scalarPtr = x;
    Ref<Scalar> scalarPtr2 = scalarPtr;
    Ref<Ref<Scalar>> scalarPtrPtr = scalarPtr;

    Ref<Scalar> tmpValPtr = *scalarPtrPtr;
    Scalar scalar = *tmpValPtr;

    If(scalar != expected, [&] { result = 1; });
    return result;
}

Scalar RefScalarRefRefRefTest()
{
    Scalar result = Allocate<int>(ScalarLayout);
    Scalar x = Allocate<int>(ScalarLayout);
    x = 100;

    Scalar expected = Allocate<int>(ScalarLayout);
    expected = 100;

    Ref<Scalar> scalarPtr = x;
    Ref<Ref<Scalar>> scalarPtrPtr = scalarPtr;
    Ref<Ref<Ref<Scalar>>> scalarPtrPtrPtr = scalarPtrPtr;

    Ref<Ref<Scalar>> tmpValPtrPtr = *scalarPtrPtrPtr;
    Ref<Scalar> tmpValPtr = *tmpValPtrPtr;
    Scalar scalar = *tmpValPtr;

    If(scalar != expected, [&] { result = 1; });
    return result;
}

Scalar SequenceLogicalAndTest()
{
    int fourInt = 4;
    Scalar twoScalar = 2;
    Scalar fourScalar = 4;

    Scalar fourGTTwo = fourInt > twoScalar;
    Scalar fourGTFour = fourInt > fourScalar;

    Scalar ok = Allocate<int>(ScalarLayout);
    ok = 1;
    If((fourGTTwo && fourGTFour),
       [&]() {
           DebugPrint("Error! 4 > 2 && 4 > 4\n");
       })
        .ElseIf(fourGTTwo,
                [&]() {
                    ok = 0;
                })
        .ElseIf(fourGTFour,
                [&]() {
                    DebugPrint("Error! 4 <= 2 && 4 > 4\n");
                })
        .Else(
            [&]() {
                DebugPrint("Error! 4 <= 2 && 4 <= 4\n");
            });
    return ok;
}

Scalar SequenceLogicalAndTestWithCopy()
{
    int fourInt = 4;
    Scalar twoScalar = 2;
    Scalar fourScalar = 4;

    Scalar fourGTTwo = fourInt > twoScalar;
    Scalar copyFourGTTwo = fourInt > twoScalar;
    Scalar fourGTFour = fourInt > fourScalar;
    Scalar copyFourGTFour = fourInt > fourScalar;

    Scalar ok = Allocate<int>(ScalarLayout);
    ok = 1;
    If((fourGTTwo && fourGTFour),
       [&]() {
           DebugPrint("Error! 4 > 2 && 4 > 4\n");
       })
        .ElseIf(copyFourGTTwo,
                [&]() {
                    ok = 0;
                })
        .ElseIf(copyFourGTFour,
                [&]() {
                    DebugPrint("Error! 4 <= 2 && 4 > 4\n");
                })
        .Else(
            [&]() {
                DebugPrint("Error! 4 <= 2 && 4 <= 4\n");
            });
    return ok;
}

} // namespace ell
