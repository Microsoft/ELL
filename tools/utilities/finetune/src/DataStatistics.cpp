////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     DataStatistics.cpp (utilities)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DataStatistics.h"
#include "ModelUtils.h"
#include "MultidimArray.h"

#include <data/include/DataVector.h>

#include <math/include/Transformations.h>
#include <math/include/Vector.h>
#include <math/include/VectorOperations.h>

#include <utilities/include/ZipIterator.h>

#include <cmath>
#include <map>
#include <numeric>

namespace ell
{

// Utilities
namespace
{
    void ThrowIfEmpty(const UnlabeledDataContainer& dataset)
    {
        if (dataset.Size() == 0)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::badData, "Empty dataset");
        }
    }

    template <typename T1, typename T2>
    void ThrowIfNotSameSize(const std::vector<T1>& a, const std::vector<T2>& b)
    {
        if (a.size() != b.size())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::badData, "Sizes don't match");
        }
    }

    template <typename T1, typename T2>
    void ThrowIfNotSameSize(const math::RowVector<T1>& a, const math::RowVector<T2>& b)
    {
        if (a.Size() != b.Size())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::badData, "Sizes don't match");
        }
    }

    template <typename T>
    math::RowVector<T> operator-(const math::RowVector<T>& a, const math::RowVector<T>& b)
    {
        ThrowIfNotSameSize(a, b);
        auto v = a;
        v -= b;
        return v;
    }

    template <typename T>
    math::RowVector<T>& operator*=(math::RowVector<T>& a, const math::RowVector<T>& b)
    {
        ThrowIfNotSameSize(a, b);
        auto size = a.Size();
        for (size_t i = 0; i < size; ++i)
        {
            a[i] *= b[i];
        }
        return a;
    }

    template <typename T>
    math::RowVector<T> operator*(const math::RowVector<T>& a, const math::RowVector<T>& b)
    {
        ThrowIfNotSameSize(a, b);
        auto v = a;
        v *= b;
        return v;
    }

    template <typename T>
    math::RowVector<T> operator/(const math::RowVector<T>& a, T denom)
    {
        auto v = a;
        v /= denom;
        return v;
    }

    template <typename T>
    math::RowVector<T>& operator/=(math::RowVector<T>& a, const math::RowVector<T>& b)
    {
        ThrowIfNotSameSize(a, b);
        auto size = a.Size();
        for (size_t i = 0; i < size; ++i)
        {
            a[i] /= b[i];
        }
        return a;
    }

    template <typename T>
    math::RowVector<T> Sqrt(const math::RowVector<T>& a)
    {
        auto v = a;
        auto xform = math::SquareRootTransformation<T>;
        v.Transform(xform);
        return v;
    }

    int GetNumRows(const UnlabeledDataContainer& dataset)
    {
        return dataset.Size();
    }

    int GetNumColumns(const UnlabeledDataContainer& dataset)
    {
        ThrowIfEmpty(dataset);
        return dataset[0].Size();
    }

    struct BasicDataStatistics
    {
        int numRows;
        std::vector<int64_t> numZeros;
        math::RowVector<double> sumElements;
        math::RowVector<double> sumSquaredElements;
    };

    BasicDataStatistics GetBasicDataStatistics(const UnlabeledDataContainer& dataset, const utilities::MemoryLayout& layout)
    {
        ThrowIfEmpty(dataset);

        BasicDataStatistics result;
        auto columns = GetNumColumns(dataset);
        auto numZeros = std::vector<int64_t>(columns);
        auto sum = math::RowVector<double>(columns);
        auto sumSquares = math::RowVector<double>(columns);
        for (const auto& row : dataset)
        {
            auto x = CastVector<double>(row);
            for (int i = 0; i < columns; ++i)
            {
                if (x[i] == 0.0)
                    ++numZeros[i];
            }
            sum += x;
            sumSquares += math::Square(x);
        }

        return {
            GetNumRows(dataset),
            numZeros,
            sum,
            sumSquares,
        };
    }

    BasicDataStatistics GetBasicDataStatistics(const UnlabeledDataContainer& dataset)
    {
        auto columns = GetNumColumns(dataset);
        utilities::MemoryLayout layout({ columns });
        return GetBasicDataStatistics(dataset, layout);
    }

    template <typename WeightsType>
    Sparsity GetWeightsSparsity(const WeightsType& weights) // TODO: add layout
    {
        auto weightsVec = weights.ToArray();
        auto numZeros = std::count_if(weightsVec.begin(), weightsVec.end(), [](auto a) { return a == 0; });
        return { static_cast<int64_t>(weightsVec.size()), numZeros };
    }

    template <typename NodeType>
    Sparsity GetNodeWeightsSparsity(const NodeType& node)
    {
        return GetWeightsSparsity(node.GetLayer().GetWeights());
    }

    Sparsity GetNodeWeightsSparsity(const model::Node& node)
    {
        if (IsConvolutionalLayerNode<float>(&node))
        {
            return GetNodeWeightsSparsity(static_cast<const nodes::ConvolutionalLayerNode<float>&>(node));
        }

        if (IsConvolutionalLayerNode<double>(&node))
        {
            return GetNodeWeightsSparsity(static_cast<const nodes::ConvolutionalLayerNode<double>&>(node));
        }

        if (IsFullyConnectedLayerNode<float>(&node))
        {
            return GetNodeWeightsSparsity(static_cast<const nodes::FullyConnectedLayerNode<float>&>(node));
        }

        if (IsFullyConnectedLayerNode<double>(&node))
        {
            return GetNodeWeightsSparsity(static_cast<const nodes::FullyConnectedLayerNode<double>&>(node));
        }
        return { 0, 0 };
    }
} // namespace

DataStatistics GetScalarDataStatistics(const UnlabeledDataContainer& dataset)
{
    auto columns = GetNumColumns(dataset);
    utilities::MemoryLayout linearLayout({ columns });

    auto basicStats = GetBasicDataStatistics(dataset, linearLayout);
    auto N = basicStats.numRows * columns;
    auto numZeros = std::accumulate(basicStats.numZeros.begin(), basicStats.numZeros.end(), 0);
    auto sum = basicStats.sumElements.Aggregate([](auto val) { return val; });
    auto sumSquares = basicStats.sumSquaredElements.Aggregate([](auto val) { return val; });
    auto mean = sum / N;
    auto variance = (sumSquares - ((sum * sum) / N)) / N; // == (sumSquares - mean*mean*N) / N
    auto stdDev = std::sqrt(variance);

    DataStatistics result;
    result.sparsity = { { static_cast<int64_t>(N), numZeros } };
    result.mean = { mean };
    result.variance = { variance };
    result.stdDev = { stdDev };

    return result;
}

DataStatistics GetDataStatistics(const UnlabeledDataContainer& dataset, const ell::utilities::MemoryLayout& layout)
{
    auto basicStats = GetBasicDataStatistics(dataset); // per-element basic statistics

    const auto N = static_cast<double>(basicStats.numRows);
    const auto& sum = basicStats.sumElements;
    const auto& sumSquares = basicStats.sumSquaredElements;
    const auto& mean = sum / N;
    const auto& variance = (sumSquares - ((sum * sum) / N)) / N; // == (sumSquares - mean*mean*N) / N
    auto stdDev = Sqrt(variance);
    DataStatistics result;

    std::transform(basicStats.numZeros.begin(), basicStats.numZeros.end(), std::back_inserter(result.sparsity), [&](int64_t nz) {
        return Sparsity{ static_cast<int64_t>(N), nz };
    });
    result.mean = mean;
    result.variance = variance;
    result.stdDev = stdDev;

    return result;
}

DataStatistics GetDataStatistics(const UnlabeledDataContainer& dataset)
{
    return GetDataStatistics(dataset, utilities::MemoryLayout({ GetNumColumns(dataset) }));
}

DataStatistics GetDataStatistics(const UnlabeledDataContainer& dataset, const ell::utilities::MemoryLayout& layout, int dimension)
{
    auto basicStats = GetBasicDataStatistics(dataset, layout); // per-element basic statistics

    const auto& sum = basicStats.sumElements;
    const auto& sumSquares = basicStats.sumSquaredElements;
    const auto& numZeros = basicStats.numZeros;

    // Squash along the given dimension
    int outputLength = static_cast<int>(layout.GetLogicalDimensionActiveSize(dimension));
    std::vector<int> dimNumZeros(outputLength);
    std::vector<double> dimSums(outputLength);
    std::vector<double> dimSumSquares(outputLength);
    const auto numElements = GetNumColumns(dataset); // the number of elements in each example
    for (int i = 0; i < numElements; ++i)
    {
        auto coords = layout.GetPhysicalCoordinatesFromOffset(i);
        auto outputIndex = coords[dimension];
        dimNumZeros[outputIndex] += numZeros[i];
        dimSums[outputIndex] += sum[i];
        dimSumSquares[outputIndex] += sumSquares[i];
    }

    int numElementsPerDim = layout.NumElements() / outputLength;
    const auto N = static_cast<double>(basicStats.numRows * numElementsPerDim);

    std::vector<Sparsity> dimSparsity(outputLength);
    ell::math::RowVector<double> dimMeans(outputLength);
    ell::math::RowVector<double> dimVariances(outputLength);
    ell::math::RowVector<double> dimStdDevs(outputLength);
    for (int i = 0; i < outputLength; ++i)
    {
        auto sumVal = dimSums[i];
        dimMeans[i] = sumVal / N;
        auto variance = (dimSumSquares[i] - ((sumVal * sumVal) / N)) / N; // == (sumSquares - mean*mean*N) / N
        dimVariances[i] = variance;
        dimStdDevs[i] = std::sqrt(variance);
        dimSparsity[i] = { numElementsPerDim / outputLength, dimNumZeros[i] };
    }

    DataStatistics result;
    result.sparsity = dimSparsity;
    result.mean = dimMeans;
    result.variance = dimVariances;
    result.stdDev = dimStdDevs;
    return result;
}

UnlabeledDataContainer GetNormalizedData(const UnlabeledDataContainer& dataset, const DataStatistics& stats)
{
    ThrowIfEmpty(dataset);

    UnlabeledDataContainer result;
    for (const auto& row : dataset)
    {
        auto newRow = CastVector<double>(row);
        newRow -= stats.mean;
        newRow /= stats.stdDev;
        result.Add(CastVector<float>(newRow));
    }
    return result;
}

UnlabeledDataContainer GetNormalizedData(const UnlabeledDataContainer& dataset, const DataStatistics& stats, const ell::utilities::MemoryLayout& layout, int dimension)
{
    ThrowIfEmpty(dataset);

    if (!layout.IsCanonicalOrder())
    {
        // Throw an exception until we're sure this code is order-independent
        throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Data not in canonical order");
    }

    UnlabeledDataContainer result;
    for (const auto& row : dataset)
    {
        // Need to broadcast stats and do the subtract / divide on a multidim array
        auto newRow = CastVector<double>(row).ToArray();
        MultidimArray newRowArray(newRow, layout);
        int size = layout.GetMemorySize();
        for (int i = 0; i < size; ++i)
        {
            auto coords = layout.GetPhysicalCoordinatesFromOffset(i);
            newRowArray[coords] -= stats.mean[coords[dimension]];
            newRowArray[coords] /= stats.stdDev[coords[dimension]];
        }

        result.Add(CastVector<float>(ell::math::RowVector<double>(newRowArray.GetData())));
    }
    return result;
}

UnlabeledDataContainer GetReverseNormalizedData(const UnlabeledDataContainer& dataset, const DataStatistics& stats)
{
    ThrowIfEmpty(dataset);

    UnlabeledDataContainer result;
    for (const auto& row : dataset)
    {
        auto newRow = CastVector<double>(row);
        newRow *= stats.stdDev;
        newRow += stats.mean;
        result.Add(CastVector<float>(newRow));
    }
    return result;
}

UnlabeledDataContainer GetReverseNormalizedData(const UnlabeledDataContainer& dataset, const DataStatistics& stats, const ell::utilities::MemoryLayout& layout, int dimension)
{
    throw 0;

    ThrowIfEmpty(dataset);

    UnlabeledDataContainer result;
    for (const auto& row : dataset)
    {
        // Need to broadcast stats and do the add / multiply on a multidim array
        auto newRow = CastVector<double>(row).ToArray();
        MultidimArray newRowArray(newRow, layout);
        int size = layout.GetMemorySize();
        for (int i = 0; i < size; ++i)
        {
            auto coords = layout.GetPhysicalCoordinatesFromOffset(i);
            newRowArray[coords] *= stats.stdDev[coords[dimension]];
            newRowArray[coords] += stats.mean[coords[dimension]];
        }

        result.Add(CastVector<float>(ell::math::RowVector<double>(newRowArray.GetData())));
    }
    return result;
}

Sparsity GetSubmodelWeightsSparsity(const model::Submodel& submodel)
{
    Sparsity sparsity;
    submodel.Visit([&](const model::Node& node) {
        auto nodeSparsity = GetNodeWeightsSparsity(node);
        sparsity.numValues += nodeSparsity.numValues;
        sparsity.numZeros += nodeSparsity.numZeros;
    });
    return sparsity;
}
} // namespace ell