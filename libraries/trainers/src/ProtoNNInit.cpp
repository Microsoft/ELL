////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ProtoNNInit.cpp (trainers)
//  Authors:  Suresh Iyengar
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ProtoNNInit.h"
#include "KMeansTrainer.h"

#include <cassert>
#include <cmath>

#define DEBUG 0
#define DUMP 1

namespace ell
{
namespace trainers
{
    ProtoNNInit::ProtoNNInit(size_t dim, size_t numLabels, size_t numPrototypesPerLabel) :
        _dim(dim),
        _numPrototypesPerLabel(numPrototypesPerLabel),
        _B(dim, numLabels * numPrototypesPerLabel),
        _Z(numLabels, numLabels * numPrototypesPerLabel)
    {}

    void ProtoNNInit::Initialize(math::ConstMatrixReference<double, math::MatrixLayout::columnMajor> WX, math::ConstMatrixReference<double, math::MatrixLayout::columnMajor> Y)
    {
        auto numLabels = Y.NumRows();
        auto numKmeansIters = 50;

        for (size_t l = 0; l < numLabels; ++l)
        {
            std::vector<size_t> indexes;

            for (size_t i = 0; i < Y.NumColumns(); ++i)
            {
                if (1 == Y(l, i))
                    indexes.push_back(i);
            }

            math::ColumnMatrix<double> wx_label(_dim, indexes.size());

            for (size_t i = 0; i < indexes.size(); ++i)
            {
                wx_label.GetColumn(i).CopyFrom(WX.GetColumn(indexes[i]));
            }

            math::ColumnVector<double> label(numLabels);
            label[l] = 1;

            KMeansTrainer kMeans(_dim, _numPrototypesPerLabel, numKmeansIters);
            kMeans.RunKMeans(wx_label);

            auto clusterMeans = kMeans.GetClusterMeans();

            for (size_t i = 0; i < _numPrototypesPerLabel; ++i)
            {
                _B.GetColumn(l * _numPrototypesPerLabel + i).CopyFrom(clusterMeans.GetColumn(i));
                _Z.GetColumn(l * _numPrototypesPerLabel + i).CopyFrom(label);
            }
        }
    }

    double ProtoNNInit::InitializeGamma(math::ConstMatrixReference<double, math::MatrixLayout::columnMajor> D, double gammaInit)
    {
        auto values = D.ToArray();
        assert(values.size() > 0);
        std::sort(values.begin(), values.end());
        auto median = values[values.size() / 2];
        auto rbfWidth = 2.5;
        return gammaInit * (rbfWidth / std::sqrt(-1.0 * std::log(median)));
    }
} // namespace trainers
} // namespace ell
