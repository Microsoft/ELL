////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Convolutions.h (emittable_functions)
//  Authors:  Byron Changuion
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <value/include/Tensor.h>

namespace ell
{
namespace emittable_functions
{

    void SimpleConvolve1D(value::Vector signal, value::Vector filter, value::Vector output);

    void SimpleDepthwiseSeparableConvolve2D(value::Tensor signal, value::Tensor filter, value::Scalar rowStride, value::Scalar columnStride, value::Tensor output);
} // namespace emittable_functions
} // namespace ell
