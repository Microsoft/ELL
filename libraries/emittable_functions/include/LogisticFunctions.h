////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     LogisticFunctions.h (emittable_functions)
//  Authors:  Chris Lovett
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <value/include/Vector.h>

namespace ell
{
namespace emittable_functions
{
    /// <summary> Apply the softmax function to the given input </summary>
    ///
    /// <param name="input"> The input vector remains unchanged. </param>
    /// <param name="output"> The output vector to write result to. </param>
    void Softmax(value::Vector input, value::Vector output);

    value::Scalar Sigmoid(value::Scalar s);

    value::Scalar HardSigmoid(value::Scalar s);

    value::Scalar HardTanh(value::Scalar s);

} // namespace emittable_functions
} // namespace ell
