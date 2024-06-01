////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Solution_test.h (optimization_test)
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "../include/RandomDataset.h"

#include <optimization/include/IndexedContainer.h>
#include <optimization/include/L2Regularizer.h>
#include <optimization/include/MaskedMatrixSolution.h>
#include <optimization/include/MatrixSolution.h>
#include <optimization/include/MultivariateLoss.h>
#include <optimization/include/OptimizationExample.h>
#include <optimization/include/SDCAOptimizer.h>
#include <optimization/include/SGDOptimizer.h>
#include <optimization/include/SquareLoss.h>
#include <optimization/include/VectorSolution.h>

#include <testing/include/testing.h>

#include <math/include/Matrix.h>
#include <math/include/Vector.h>

#include <random>
#include <string>

using namespace ell;
using namespace ell::optimization;

/// <summary> Tests that biased and unbiased VectorSolution and biased and unbiased MatrixSolution all behave identically when given equivalent SGD optimization problems. </summary>
template <typename RealType, typename LossFunctionType, typename RegularizerType>
void TestSolutionEquivalenceSGD(double regularizationParameter);

/// <summary> Tests that biased and unbiased VectorSolution and biased and unbiased MatrixSolution all behave identically when given equivalent SDCA optimization problems. </summary>
template <typename RealType, typename LossFunctionType, typename RegularizerType>
void TestSolutionEquivalenceSDCA(double regularizationParameter);

/// <summary> Tests masked matrix solution operations </summary>
template <typename RealType, template <typename> class SolutionType>
void TestMaskedMatrixSolution();

#pragma region implementation

// Run the SGD trainer with four different solution types and confirm that the result is identical
template <typename RealType, typename LossFunctionType, typename RegularizerType>
void TestSolutionEquivalenceSGD(double regularizationParameter)
{
    std::string randomSeedString = "54321blastoff";
    std::seed_seq seed(randomSeedString.begin(), randomSeedString.end());
    std::default_random_engine randomEngine;

    const size_t numExamples = 5;
    const size_t exampleSize = 12;

    randomEngine.seed(seed);
    auto examples1 = GetRandomDataset<RealType, VectorScalarExampleType<RealType>, VectorRefScalarExampleType<RealType>>(numExamples, exampleSize, randomEngine, 1);

    randomEngine.seed(seed);
    auto examples2 = GetRandomDataset<RealType, VectorScalarExampleType<RealType>, VectorRefScalarExampleType<RealType>>(numExamples, exampleSize, randomEngine, 0);

    randomEngine.seed(seed);
    auto examples3 = GetRandomDataset<RealType, VectorVectorExampleType<RealType>, VectorRefVectorRefExampleType<RealType>>(numExamples, exampleSize, randomEngine, 1);

    randomEngine.seed(seed);
    auto examples4 = GetRandomDataset<RealType, VectorVectorExampleType<RealType>, VectorRefVectorRefExampleType<RealType>>(numExamples, exampleSize, randomEngine, 0);

    // set up four equivalent optimizers
    auto optimizer1 = MakeSGDOptimizer<VectorSolution<RealType>>(examples1, LossFunctionType{}, { regularizationParameter });
    optimizer1.Update();
    const auto& solution1 = optimizer1.GetSolution();
    const auto& vector1 = solution1.GetVector();

    auto optimizer2 = MakeSGDOptimizer<VectorSolution<RealType, true>>(examples2, LossFunctionType{}, { regularizationParameter });
    optimizer2.Update();
    const auto& solution2 = optimizer2.GetSolution();
    const auto& vector2 = solution2.GetVector();

    auto optimizer3 = MakeSGDOptimizer<MatrixSolution<RealType>>(examples3, MultivariateLoss<LossFunctionType>{}, { regularizationParameter });
    optimizer3.Update();
    const auto& solution3 = optimizer3.GetSolution();
    const auto& vector3 = solution3.GetMatrix().GetColumn(0);

    auto optimizer4 = MakeSGDOptimizer<MatrixSolution<RealType, true>>(examples4, MultivariateLoss<LossFunctionType>{}, { regularizationParameter });
    optimizer4.Update();
    const auto& solution4 = optimizer4.GetSolution();
    const auto& vector4 = solution4.GetMatrix().GetColumn(0);

    double comparisonTolerance = 1.0e-7;

    std::string realName = typeid(RealType).name();
    std::string lossName = typeid(LossFunctionType).name();
    lossName = lossName.substr(lossName.find_last_of(":") + 1);

    // test if the two solutions are identical
    testing::ProcessTest("TestSolutionEquivalenceSGD (v1 == v2) <" + realName + ", " + lossName + ">", vector1.GetSubVector(0, exampleSize).IsEqual(vector2, comparisonTolerance));
    testing::ProcessTest("TestSolutionEquivalenceSGD (v1.last == b2) <" + realName + ", " + lossName + ">", testing::IsEqual(vector1[exampleSize], solution2.GetBias(), comparisonTolerance));
    testing::ProcessTest("TestSolutionEquivalenceSGD (v1 == v3) <" + realName + ", " + lossName + ">", vector1.IsEqual(vector3, comparisonTolerance));
    testing::ProcessTest("TestSolutionEquivalenceSGD (v2 == v4) <" + realName + ", " + lossName + ">", vector2.IsEqual(vector4, comparisonTolerance));
    testing::ProcessTest("TestSolutionEquivalenceSGD (b2 == b4) <" + realName + ", " + lossName + ">", testing::IsEqual(solution4.GetBias()[0], solution2.GetBias(), comparisonTolerance));
}

// Run the SDCA trainer with four different solution types and confirm that the result is identical
template <typename RealType, typename LossFunctionType, typename RegularizerType>
void TestSolutionEquivalenceSDCA(double regularizationParameter)
{
    std::string randomSeedString = "54321blastoff";
    std::seed_seq seed(randomSeedString.begin(), randomSeedString.end());
    std::default_random_engine randomEngine;

    const size_t numExamples = 5;
    const size_t exampleSize = 7;

    randomEngine.seed(seed);
    auto examples1 = GetRandomDataset<RealType, VectorScalarExampleType<RealType>, VectorRefScalarExampleType<RealType>>(numExamples, exampleSize, randomEngine, 1);

    randomEngine.seed(seed);
    auto examples2 = GetRandomDataset<RealType, VectorScalarExampleType<RealType>, VectorRefScalarExampleType<RealType>>(numExamples, exampleSize, randomEngine, 0);

    randomEngine.seed(seed);
    auto examples3 = GetRandomDataset<RealType, VectorVectorExampleType<RealType>, VectorRefVectorRefExampleType<RealType>>(numExamples, exampleSize, randomEngine, 1);

    randomEngine.seed(seed);
    auto examples4 = GetRandomDataset<RealType, VectorVectorExampleType<RealType>, VectorRefVectorRefExampleType<RealType>>(numExamples, exampleSize, randomEngine, 0);

    randomEngine.seed(seed);
    auto examples5 = GetRandomDataset<RealType, VectorVectorExampleType<RealType>, VectorRefVectorRefExampleType<RealType>>(numExamples, exampleSize, randomEngine, 1);

    randomEngine.seed(seed);
    auto examples6 = GetRandomDataset<RealType, VectorVectorExampleType<RealType>, VectorRefVectorRefExampleType<RealType>>(numExamples, exampleSize, randomEngine, 0);

    // setup four equivalent optimizers
    auto optimizer1 = MakeSDCAOptimizer<VectorSolution<RealType>>(examples1, LossFunctionType{}, RegularizerType{}, { regularizationParameter });
    optimizer1.Update();
    const auto& solution1 = optimizer1.GetSolution();
    const auto& vector1 = solution1.GetVector();

    auto optimizer2 = MakeSDCAOptimizer<VectorSolution<RealType, true>>(examples2, LossFunctionType{}, RegularizerType{}, { regularizationParameter });
    optimizer2.Update();
    const auto& solution2 = optimizer2.GetSolution();
    const auto& vector2 = solution2.GetVector();

    auto optimizer3 = MakeSDCAOptimizer<MatrixSolution<RealType>>(examples3, MultivariateLoss<LossFunctionType>{}, RegularizerType{}, { regularizationParameter });
    optimizer3.Update();
    const auto& solution3 = optimizer3.GetSolution();
    const auto& vector3 = solution3.GetMatrix().GetColumn(0);

    auto optimizer4 = MakeSDCAOptimizer<MatrixSolution<RealType, true>>(examples4, MultivariateLoss<LossFunctionType>{}, RegularizerType{}, { regularizationParameter });
    optimizer4.Update();
    const auto& solution4 = optimizer4.GetSolution();
    const auto& vector4 = solution4.GetMatrix().GetColumn(0);

    auto optimizer5 = MakeSDCAOptimizer<MaskedMatrixSolution<MatrixSolution<RealType>>>(examples5, MultivariateLoss<LossFunctionType>{}, RegularizerType{}, { regularizationParameter });
    optimizer5.Update();
    const auto& solution5 = optimizer5.GetSolution();
    const auto& vector5 = solution5.GetMatrix().GetColumn(0);

    auto optimizer6 = MakeSDCAOptimizer<MaskedMatrixSolution<MatrixSolution<RealType, true>>>(examples6, MultivariateLoss<LossFunctionType>{}, RegularizerType{}, { regularizationParameter });
    optimizer6.Update();
    const auto& solution6 = optimizer6.GetSolution();
    const auto& vector6 = solution6.GetMatrix().GetColumn(0);

    double comparisonTolerance = 1.0e-6;

    std::string realName = typeid(RealType).name();
    std::string lossName = typeid(LossFunctionType).name();
    lossName = lossName.substr(lossName.find_last_of(":") + 1);

    // test if the two solutions are identical
    testing::ProcessTest("TestSolutionEquivalenceSDCA (v1 == v2) <" + realName + ", " + lossName + ">", vector1.GetSubVector(0, exampleSize).IsEqual(vector2, comparisonTolerance));
    testing::ProcessTest("TestSolutionEquivalenceSDCA (v1.last == b2) <" + realName + ", " + lossName + ">", testing::IsEqual(vector1[exampleSize], solution2.GetBias(), comparisonTolerance));
    testing::ProcessTest("TestSolutionEquivalenceSDCA (v1 == v3) <" + realName + ", " + lossName + ">", vector1.IsEqual(vector3, comparisonTolerance));
    testing::ProcessTest("TestSolutionEquivalenceSDCA (v1 == v5) <" + realName + ", " + lossName + ">", vector1.IsEqual(vector5, comparisonTolerance));
    testing::ProcessTest("TestSolutionEquivalenceSDCA (v2 == v4) <" + realName + ", " + lossName + ">", vector2.IsEqual(vector4, comparisonTolerance));
    testing::ProcessTest("TestSolutionEquivalenceSDCA (b2 == b4) <" + realName + ", " + lossName + ">", testing::IsEqual(solution4.GetBias()[0], solution2.GetBias(), comparisonTolerance));
    testing::ProcessTest("TestSolutionEquivalenceSDCA (v2 == v6) <" + realName + ", " + lossName + ">", vector2.IsEqual(vector6, comparisonTolerance));
    testing::ProcessTest("TestSolutionEquivalenceSDCA (b2 == b6) <" + realName + ", " + lossName + ">", testing::IsEqual(solution6.GetBias()[0], solution2.GetBias(), comparisonTolerance));
};

// Basic tests of MaskedMatrixSolution
template <typename RealType, template <typename> class SolutionType>
void TestMaskedMatrixSolution()
{
    using LossFunctionType = SquareLoss;
    using RegularizerType = L2Regularizer;
    const size_t inputSize = 6;
    const size_t outputSize = 4;
    const size_t numExamples = 10;

    {
        // A small sanity test
        math::RowVector<RealType> in(inputSize);
        math::RowVector<RealType> out(outputSize);

        SolutionType<RealType> matrixSolution;
        MaskedMatrixSolution<SolutionType<RealType>> maskedMatrixSolution;
        matrixSolution.Resize(in, out);
        maskedMatrixSolution.Resize(in, out);

        testing::ProcessTest("TestMatrixSolution matrix rows", testing::IsEqual(matrixSolution.GetMatrix().NumRows(), inputSize));
        testing::ProcessTest("TestMatrixSolution matrix cols", testing::IsEqual(matrixSolution.GetMatrix().NumColumns(), outputSize));
        testing::ProcessTest("TestMaskedMatrixSolution matrix rows", testing::IsEqual(maskedMatrixSolution.GetMatrix().NumRows(), inputSize));
        testing::ProcessTest("TestMaskedMatrixSolution matrix cols", testing::IsEqual(maskedMatrixSolution.GetMatrix().NumColumns(), outputSize));
        testing::ProcessTest("TestMaskedMatrixSolution mask rows", testing::IsEqual(maskedMatrixSolution.GetMask().NumRows(), inputSize));
        testing::ProcessTest("TestMaskedMatrixSolution mask cols", testing::IsEqual(maskedMatrixSolution.GetMask().NumColumns(), outputSize));
    }

    std::string randomSeedString = "54321blastoff";
    std::seed_seq seed(randomSeedString.begin(), randomSeedString.end());
    std::default_random_engine randomEngine;
    randomEngine.seed(seed);
    auto examples = GetRandomDataset<RealType, VectorVectorExampleType<RealType>, VectorRefVectorRefExampleType<RealType>>(numExamples, inputSize, outputSize, randomEngine, 0);

    const double regularizationParameter = 0.0001;
    auto optimizer = MakeSDCAOptimizer<SolutionType<RealType>>(examples, MultivariateLoss<LossFunctionType>{}, RegularizerType{}, { regularizationParameter });
    optimizer.Update();
    const auto& solution = optimizer.GetSolution();
    const auto& weights = solution.GetMatrix();

    typename MaskedMatrixSolution<SolutionType<RealType>>::MaskType mask(inputSize, outputSize);
    mask.Fill(0); // meaning "all elements free"
    auto maskedOptimizer = MakeSDCAOptimizer<MaskedMatrixSolution<SolutionType<RealType>>>(examples, MultivariateLoss<LossFunctionType>{}, RegularizerType{}, { regularizationParameter }, { mask });
    maskedOptimizer.Update();
    const auto& maskedSolution = maskedOptimizer.GetSolution();
    const auto& maskedWeights = maskedSolution.GetMatrix();

    testing::ProcessTest("TestMaskedMatrixSolution weights", testing::IsEqual(weights, maskedWeights));

    typename MaskedMatrixSolution<SolutionType<RealType>>::MaskType mask2(inputSize, outputSize);
    mask2(0, 0) = 1; // meaning "all elements frozen"
    typename SolutionType<RealType>::WeightsType frozenWeights(inputSize, outputSize);
    frozenWeights(0, 0) = 0.25;
    auto maskedOptimizer2 = MakeSDCAOptimizer<MaskedMatrixSolution<SolutionType<RealType>>>(examples, MultivariateLoss<LossFunctionType>{}, RegularizerType{}, { regularizationParameter }, typename MaskedMatrixSolution<SolutionType<RealType>>::ParametersType{ mask2, frozenWeights });
    maskedOptimizer2.Update();
    const auto& maskedSolution2 = maskedOptimizer2.GetSolution();
    const auto& maskedWeights2 = maskedSolution2.GetMatrix();
    const auto& frozenSolutionWeights2 = maskedSolution2.GetFrozenWeights();

    testing::ProcessTest("TestMaskedMatrixSolution2 frozen weights", testing::IsEqual(frozenSolutionWeights2(0, 0), frozenWeights(0, 0)));
    testing::ProcessTest("TestMaskedMatrixSolution2 frozen weights", testing::IsEqual(frozenSolutionWeights2(0, 0), maskedWeights2(0, 0)));

    std::cout << "Original solution weights:\n"
              << weights << std::endl;
    std::cout << "Original solution duality gap: " << optimizer.GetSolutionInfo().DualityGap() << std::endl;
    std::cout << "Masked solution weights:\n"
              << maskedWeights2 << std::endl;
    std::cout << "Masked solution duality gap: " << maskedOptimizer2.GetSolutionInfo().DualityGap() << std::endl;
};

#pragma endregion implementation
