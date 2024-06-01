////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     PredictorUtils.h (finetune)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <optimization/include/MatrixSolution.h>
#include <optimization/include/VectorSolution.h>

#include <vector>

namespace ell
{
template <typename ElementType>
struct WeightsAndBias
{
    ell::math::RowMatrix<ElementType> weights = { 0, 0 };
    ell::math::ColumnVector<ElementType> bias = { 0 };
};

// Optimization library stuff
using ScalarPredictor = ell::optimization::BiasedVectorSolution<float>;
using VectorPredictor = ell::optimization::BiasedMatrixSolution<float>;

template <typename ElementType>
WeightsAndBias<ElementType> GetWeightsAndBias(const ScalarPredictor& predictor);

template <typename ElementType>
WeightsAndBias<ElementType> GetWeightsAndBias(const VectorPredictor& predictor);
} // namespace ell