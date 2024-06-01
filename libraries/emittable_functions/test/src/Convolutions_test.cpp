////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Convolutions_test.cpp (emittable_functions)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Convolutions_test.h"
#include "Emittable_functions_util.h"

#include <emittable_functions/include/Convolutions.h>

#include <emitters/include/IRModuleEmitter.h>

#include <math/include/Matrix.h>
#include <math/include/Tensor.h>

#include <testing/include/testing.h>

#include <utilities/include/FunctionUtils.h>

#include <value/include/ComputeContext.h>
#include <value/include/FunctionDeclaration.h>
#include <value/include/LLVMContext.h>
#include <value/include/Matrix.h>
#include <value/include/Tensor.h>
#include <value/include/Value.h>
#include <value/include/Vector.h>

#include <iostream>
#include <memory>
#include <type_traits>
#include <vector>

using namespace ell::emitters;
using namespace ell::utilities;
using namespace ell::value;
using namespace ell::emittable_functions;

namespace ell
{

void test_simpleDepthwiseSeparableConvolve2D()
{
    auto input = std::vector<double>{ 1, 2, 3, 4, 5, 6, 1, 1, 1, 2, 3, 4, 9, 8, 7, 1, 2, 3 };
    auto filter = std::vector<double>{ 1, 2, 2, 1, 1, 1, -1, 0 };
    auto expected = std::vector<double>{ 18, 24, 17, 20, -4, -1, 16, 13 };

    Tensor inputTensor({ input, MemoryLayout({ 2, 3, 3 }, DimensionOrder(ChannelMajorTensorOrder)) });
    Tensor filterTensor({ filter, MemoryLayout({ 2, 2, 2 }, DimensionOrder(ChannelMajorTensorOrder)) });
    Tensor outputTensor(GlobalAllocate("result_test_SimpleDepthwiseSeparableConvolve2D",
                                       ValueType::Double,
                                       MemoryLayout({ 2, 2, 2 }, DimensionOrder(ChannelMajorTensorOrder))));

    math::ColumnRowChannelTensor<double> expectedTensor(2, 2, 2, expected);

    auto convolve2D = DeclareFunction("testSimpleDepthwiseSeparableConvolve2D")
                          .Parameters(inputTensor,
                                      filterTensor,
                                      Value(ValueType::Int32, ScalarLayout),
                                      Value(ValueType::Int32, ScalarLayout),
                                      outputTensor)
                          .Define(SimpleDepthwiseSeparableConvolve2D);

    InvokeForContext<ComputeContext>([&] {
        bool ok = true;
        convolve2D(inputTensor, filterTensor, 1, 1, outputTensor);
        For(outputTensor, [&](Scalar row, Scalar col, Scalar channel) {
            int rowInt = row.Get<int>(), colInt = col.Get<int>(), channelInt = channel.Get<int>();
            double expected = expectedTensor(rowInt, colInt, channelInt);
            double actual = outputTensor(row, col, channel).Get<double>();

            ok &= testing::IsEqual(expected, actual);
        });
        testing::ProcessTest("Testing DepthwiseSeparableConvolve2D", ok);
    });

    InvokeForContext<TestLLVMContext>(PrintIR);
}

} // namespace ell
