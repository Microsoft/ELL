////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     OptimizationUtils.cpp (utilities)
//  Authors:  Byron Changuion, Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OptimizationUtils.h"

#include <optimization/include/MultivariateLoss.h>
#include <optimization/include/SquareLoss.h>

#include <utilities/include/Exception.h>
#include <utilities/include/Logger.h>

#include <iostream>

namespace ell
{
using namespace ell::logging;

#define ADD_TO_STRING_ENTRY(NAMESPACE, OPERATOR) \
    case NAMESPACE::OPERATOR:                    \
        return #OPERATOR;
#define BEGIN_FROM_STRING if (false)
#define ADD_FROM_STRING_ENTRY(NAMESPACE, OPERATOR) else if (name == #OPERATOR) return NAMESPACE::OPERATOR

std::string ToString(LossFunction loss)
{
    switch (loss)
    {
        ADD_TO_STRING_ENTRY(LossFunction, square);
        ADD_TO_STRING_ENTRY(LossFunction, logistic);
        ADD_TO_STRING_ENTRY(LossFunction, hinge);
        ADD_TO_STRING_ENTRY(LossFunction, smoothedHinge);
        ADD_TO_STRING_ENTRY(LossFunction, huber);
    default:
        throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Unknown loss function type");
    }
}

std::string ToString(SparsifyMethod method)
{
    switch (method)
    {
        ADD_TO_STRING_ENTRY(SparsifyMethod, l1);
        ADD_TO_STRING_ENTRY(SparsifyMethod, threshold);
        ADD_TO_STRING_ENTRY(SparsifyMethod, random);
    default:
        throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Unknown sparsification method");
    }
}

ScalarOptimizerResult TrainScalarPredictor(BinaryLabelDataContainer dataset, const FineTuneOptimizationParameters& optimizerParameters)
{
    using SolutionType = ScalarPredictor;
    auto examples = std::make_shared<BinaryLabelDataContainer>(std::move(dataset));

    switch (optimizerParameters.lossFunction)
    {
    case LossFunction::hinge: {
        optimization::HingeLoss loss;
        return TrainPredictor<SolutionType>(examples, loss, optimizerParameters);
    }
    case LossFunction::huber: {
        optimization::HuberLoss loss;
        return TrainPredictor<SolutionType>(examples, loss, optimizerParameters);
    }
    case LossFunction::logistic: {
        optimization::LogisticLoss loss;
        return TrainPredictor<SolutionType>(examples, loss, optimizerParameters);
    }
    case LossFunction::smoothedHinge: {
        optimization::SmoothedHingeLoss loss;
        return TrainPredictor<SolutionType>(examples, loss, optimizerParameters);
    }
    case LossFunction::square: {
        optimization::SquareLoss loss;
        return TrainPredictor<SolutionType>(examples, loss, optimizerParameters);
    }
    default:
        throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState, "Unknown loss function type");
    }
}

// TODO: rename this function to something without "train" or "predictor" in its name
VectorOptimizerResult TrainVectorPredictor(VectorLabelDataContainer dataset,
                                           const FineTuneOptimizationParameters& optimizerParameters,
                                           bool isSpatialConvolution)
{
    using namespace logging;

    if (dataset.Size() < 1)
    {
        throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Dataset doesn't have any entries");
    }

    if (!isSpatialConvolution && !optimizerParameters.optimizeFiltersIndependently)
    {
        // Dataset: N rows from k*k*d -> d'
        return TrainVectorPredictor(dataset, optimizerParameters);
    }
    // else
    // {
    //     return TrainIndependentPredictors(dataset, optimizerParameters, isSpatialConvolution);
    // }

    // optimize spatial convolutions one-at-a-time
    FineTuneOptimizationParameters spatialParameters = optimizerParameters;

    // The input dataset consists of N rows from k*k*d -> d'.
    // We need to split it into d' datasets from k*k*d -> 1.
    auto in0 = math::RowVector<float>(dataset.Get(0).input);
    const auto& out0 = dataset.Get(0).output;
    const auto outputChannels = out0.Size(); // # output (and input) channels
    const auto filterSizeSq = in0.Size() / outputChannels;

    // In the case of spatial convolutions, for each output channel, we are just recovering
    // a k x k spatial filter.
    if (isSpatialConvolution)
    {
        in0.Resize(filterSizeSq); // # spatial elements in a filter (e.g., 9 for a 3x3 filter)
    }

    // TODO: in both the "independent channel" and "spatial filter" cases, we really are optimizing to find a scalar result.
    VectorLabelSolution resultSolution;
    resultSolution.Resize(in0, out0);

    Log() << "Optimizing " << outputChannels << " output channels independently\n";

    SolutionInfo solutionInfo;
    for (size_t i = 0; i < outputChannels; ++i)
    {
        // For each output channel, create a tiny dataset that goes from the pixels under a filter support -> output value
        VectorLabelDataContainer channelDataset;

        if (isSpatialConvolution)
        {
            channelDataset = CreateSubBlockVectorLabelDataContainer(dataset, filterSizeSq, 1, i);
        }
        else
        {
            channelDataset = CreateSingleOutputVectorLabelDataContainer(dataset, i);
        }

        auto channelResult = TrainVectorPredictor(channelDataset, spatialParameters);

        resultSolution.GetBias()[i] = channelResult.predictor.GetBias()[0];
        resultSolution.GetMatrix().GetColumn(i).CopyFrom(channelResult.predictor.GetMatrix().GetColumn(0));

        // For now, just keep the last solution info result.
        // TODO: in the "trainFiltersIndependently" case, we should
        // keep some kind of summary thing instead.
        solutionInfo = channelResult.info;
    }

    return { resultSolution, solutionInfo, {} };
}

VectorOptimizerResult TrainVectorPredictor(VectorLabelDataContainer dataset, const FineTuneOptimizationParameters& optimizerParameters)
{
    using SolutionType = VectorPredictor;
    auto examples = std::make_shared<VectorLabelDataContainer>(std::move(dataset));

    switch (optimizerParameters.lossFunction)
    {
    case LossFunction::hinge: {
        optimization::MultivariateLoss<optimization::HingeLoss> loss;
        return TrainPredictor<SolutionType>(examples, loss, optimizerParameters);
    }
    case LossFunction::huber: {
        optimization::MultivariateLoss<optimization::HuberLoss> loss;
        return TrainPredictor<SolutionType>(examples, loss, optimizerParameters);
    }
    case LossFunction::logistic: {
        optimization::MultivariateLoss<optimization::LogisticLoss> loss;
        return TrainPredictor<SolutionType>(examples, loss, optimizerParameters);
    }
    case LossFunction::smoothedHinge: {
        optimization::MultivariateLoss<optimization::SmoothedHingeLoss> loss;
        return TrainPredictor<SolutionType>(examples, loss, optimizerParameters);
    }
    case LossFunction::square: {
        optimization::MultivariateLoss<optimization::SquareLoss> loss;
        return TrainPredictor<SolutionType>(examples, loss, optimizerParameters);
    }
    default:
        throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState, "Unknown loss function type");
    }
}

VectorOptimizerResult ReoptimizeSparsePredictor(VectorOptimizerResult& sparseSolution,
                                                VectorLabelDataContainer dataset,
                                                const FineTuneOptimizationParameters& optimizerParameters,
                                                bool isSpatialConvolution)
{
    if (isSpatialConvolution || optimizerParameters.optimizeFiltersIndependently)
    {
        throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Can't reoptimize spatial or independent-channel solutions");
    }

    using SolutionType = VectorPredictor;
    auto examples = std::make_shared<VectorLabelDataContainer>(std::move(dataset));

    switch (optimizerParameters.lossFunction)
    {
    case LossFunction::hinge: {
        optimization::MultivariateLoss<optimization::HingeLoss> loss;
        return ReoptimizeSparsePredictor<SolutionType>(sparseSolution, examples, loss, optimizerParameters);
    }
    case LossFunction::huber: {
        optimization::MultivariateLoss<optimization::HuberLoss> loss;
        return ReoptimizeSparsePredictor<SolutionType>(sparseSolution, examples, loss, optimizerParameters);
    }
    case LossFunction::logistic: {
        optimization::MultivariateLoss<optimization::LogisticLoss> loss;
        return ReoptimizeSparsePredictor<SolutionType>(sparseSolution, examples, loss, optimizerParameters);
    }
    case LossFunction::smoothedHinge: {
        optimization::MultivariateLoss<optimization::SmoothedHingeLoss> loss;
        return ReoptimizeSparsePredictor<SolutionType>(sparseSolution, examples, loss, optimizerParameters);
    }
    case LossFunction::square: {
        optimization::MultivariateLoss<optimization::SquareLoss> loss;
        return ReoptimizeSparsePredictor<SolutionType>(sparseSolution, examples, loss, optimizerParameters);
    }
    default:
        throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState, "Unknown loss function type");
    }
}
} // namespace ell
