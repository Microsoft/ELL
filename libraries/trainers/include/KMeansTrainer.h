////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     KMeansTrainer.h (trainers)
//  Authors:  Suresh Iyengar
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <math/include/Matrix.h>

#include <cstddef>
#include <map>
#include <memory>

namespace ell
{
namespace trainers
{
    /// <summary> Impements KMeansTrainer++ algorithm </summary>
    ///
    class KMeansTrainer
    {
    public:
        /// <summary> Constructs an instance of KMeansTrainer trainer </summary>
        ///
        /// <param name="dimension"> The input dimension. </param>
        /// <param name="numClusters"> The number of clusters. </param>
        /// <param name="iterations"> The number of iterations. </param>
        ///
        KMeansTrainer(size_t dimension, size_t numClusters, size_t iterations);

        /// <summary> Constructs an instance of KMeansTrainer trainer </summary>
        ///
        /// <param name="numClusters"> The number of clusters. </param>
        /// <param name="iterations"> The number of iterations. </param>
        /// <param name="means"> The cluster means. </param>
        ///
        KMeansTrainer(size_t numClusters, size_t iters, math::ColumnMatrix<double> means);

        /// <summary> Runs the KMeansTrainer algorithm. </summary>
        ///
        /// <param name="X"> The input matrix. </param>
        ///
        void RunKMeans(math::ConstMatrixReference<double, math::MatrixLayout::columnMajor> X);

        /// <summary> Returns the underlying cluster means. </summary>
        ///
        /// <returns> The underlying cluster means matrix. </returns>
        const math::ColumnMatrix<double>& GetClusterMeans() const { return _means; }

        /// <summary> Returns the underlying cluster assignment. </summary>
        ///
        /// <returns> The underlying cluster assignment matrix. </returns>
        const math::ColumnVector<double>& GetClusterAssignment() const { return _clusterAssignment; }

    private:
        // Initializes the cluster means using the KMeansTrainer++ strategy.
        void initializeMeans(math::ConstMatrixReference<double, math::MatrixLayout::columnMajor> X);

        // Distance of points to all the cluster means.
        math::RowMatrix<double> pairwiseDistance(math::ConstMatrixReference<double, math::MatrixLayout::columnMajor> X, math::ConstMatrixReference<double, math::MatrixLayout::columnMajor> mu);

        // Assign each point to the closest mean.
        double assignClosestCenter(math::ConstMatrixReference<double, math::MatrixLayout::columnMajor> X, math::VectorReference<size_t, math::VectorOrientation::column> clusterAssignment);

        // Recompute the cluster means.
        void recomputeMeans(math::ConstMatrixReference<double, math::MatrixLayout::columnMajor> X, const math::ColumnVector<size_t> clusterAssignment);

        // Weighted sampling.
        size_t weightedSample(math::ColumnVector<double> weights);

        // Cluster means.
        math::ColumnMatrix<double> _means;

        // Are the means initialized?
        bool _isInitialized = false;

        // Cluster assignment for each data point.
        math::ColumnVector<double> _clusterAssignment;

        // Number of iterations of KMeansTrainer algorithm.
        size_t _iterations = 0;

        // Number of clusters.
        size_t _numClusters = 0;
    };
} // namespace trainers
} // namespace ell
