////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     DataStatistics.h (finetune)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "DataUtils.h"
#include "MultidimArray.h"

#include <model/include/Submodel.h>

#include <math/include/Vector.h>

#include <vector>

namespace ell
{
/// <summary> A struct to indicate the level of sparsity in some data (as a ratio of zeros to total elements). </summary>
struct Sparsity
{
    int64_t numValues = 0;
    int64_t numZeros = 0;
    double GetSparsity() const { return static_cast<double>(numZeros) / static_cast<double>(numValues); }
};

/// <summary>
/// A struct to hold basic statistics about a block of data, including the overall sparsity as well as
/// the mean, variance, and standard deviation.
/// The `mean`, `variance`, and `stdDev` staistics are vectors so that they can represent the statistics
/// either of the data as a whole (in which case, they have a size of 1), or along one dimension of a matrix.
/// </summary>
struct DataStatistics
{
    std::vector<Sparsity> sparsity;
    ell::math::RowVector<double> mean;
    ell::math::RowVector<double> variance;
    ell::math::RowVector<double> stdDev;
};

/// <summary> Get statistics of a data container </summary>
DataStatistics GetScalarDataStatistics(const UnlabeledDataContainer& dataset);
DataStatistics GetDataStatistics(const UnlabeledDataContainer& dataset);
DataStatistics GetDataStatistics(const UnlabeledDataContainer& dataset, const ell::utilities::MemoryLayout& layout);
DataStatistics GetDataStatistics(const UnlabeledDataContainer& dataset, const ell::utilities::MemoryLayout& layout, int dimension);

/// <summary> Get a version of a data container with its entries modified to match some given statistics. </summary>
UnlabeledDataContainer GetNormalizedData(const UnlabeledDataContainer& dataset, const DataStatistics& stats);
UnlabeledDataContainer GetNormalizedData(const UnlabeledDataContainer& dataset, const DataStatistics& stats, const ell::utilities::MemoryLayout& layout, int dimension);
UnlabeledDataContainer GetReverseNormalizedData(const UnlabeledDataContainer& dataset, const DataStatistics& stats);
UnlabeledDataContainer GetReverseNormalizedData(const UnlabeledDataContainer& dataset, const DataStatistics& stats, const ell::utilities::MemoryLayout& layout, int dimension);

/// <summary> Get the overall sparsity for the weights in all the nodes of a submodel </summary>
Sparsity GetSubmodelWeightsSparsity(const ell::model::Submodel& submodel);
} // namespace ell
