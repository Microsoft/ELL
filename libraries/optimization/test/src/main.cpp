////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     main.cpp (optimization_test)
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "LossFunction_test.h"
#include "Optimizer_test.h"
#include "Other_test.h"
#include "Regularizer_test.h"
#include "Search_test.h"
#include "Solution_test.h"

#include <testing/include/testing.h>

// optimization

// solutions
#include <optimization/include/MatrixSolution.h>
#include <optimization/include/VectorSolution.h>

// loss functions
#include <optimization/include/AbsoluteLoss.h>
#include <optimization/include/HingeLoss.h>
#include <optimization/include/HuberLoss.h>
#include <optimization/include/LogisticLoss.h>
#include <optimization/include/MultivariateLoss.h>
#include <optimization/include/SmoothedHingeLoss.h>
#include <optimization/include/SquareLoss.h>
#include <optimization/include/SquaredHingeLoss.h>

// regularizers
#include <optimization/include/ElasticNetRegularizer.h>
#include <optimization/include/L2Regularizer.h>
#include <optimization/include/MaxRegularizer.h>

using namespace ell;
using namespace ell::optimization;

int main()
{
    // Test that Derivative() is consistent with Value()

    TestDerivative(AbsoluteLoss{}, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 });
    TestDerivative(HuberLoss{ 2.0 }, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 });
    TestDerivative(HuberLoss{ 1.0 }, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 });
    TestDerivative(HuberLoss{ 0.6 }, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 });
    TestDerivative(SquareLoss{}, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 });

    TestDerivative(HingeLoss{}, { -2.05, 0.1, 2.05 }, { -1.0, 2.0, 1.0 });
    TestDerivative(LogisticLoss{}, { -2.0, 0.1, 2.0 }, { -1.0, 2.0, 1.0 });
    TestDerivative(SmoothedHingeLoss{ 1.0 }, { -2.0, 0.1, 2.0 }, { -1.0, 2.0, 1.0 });
    TestDerivative(SmoothedHingeLoss{ 0.5 }, { -2.0, 0.1, 2.0 }, { -1.0, 2.0, 1.0 });
    TestDerivative(SmoothedHingeLoss{ 0.3 }, { -2.0, 0.1, 2.0 }, { -1.0, 2.0, 1.0 });
    TestDerivative(SquaredHingeLoss{}, { -2.0, 0.1, 2.0 }, { -1.0, 2.0, 1.0 });

    // Test that Conjugate() is consistent with Value()

    TestConjugate(AbsoluteLoss{}, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, -10, 10);
    TestConjugate(HuberLoss{ 2.0 }, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, -10, 10);
    TestConjugate(HuberLoss{ 1.0 }, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, -10, 10);
    TestConjugate(HuberLoss{ 0.6 }, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, -10, 10);
    TestConjugate(SquareLoss{}, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, -10, 10);

    TestConjugate(HingeLoss{}, { -2.0, 0.1, 2.0 }, { -1.0, 2.0, 1.0 }, -10, 10);
    TestConjugate(LogisticLoss{}, { -2.0, 0.1, 2.0 }, { -1.0, 2.0, 1.0 }, -30, 30);
    TestConjugate(SmoothedHingeLoss{ 1.0 }, { -2.0, 0.1, 2.0 }, { -1.0, 2.0, 1.0 }, -10, 10);
    TestConjugate(SmoothedHingeLoss{ 0.5 }, { -2.0, 0.1, 2.0 }, { -1.0, 2.0, 1.0 }, -10, 10);
    TestConjugate(SmoothedHingeLoss{ 0.3 }, { -2.0, 0.1, 2.0 }, { -1.0, 2.0, 1.0 }, -10, 10);
    TestConjugate(SquaredHingeLoss{}, { -2.0, 0.1, 2.0 }, { -1.0, 2.0, 1.0 }, -10, 10);

    // Test ConjugateProx() is consistent with Conjugate()

    TestConjugateProx(AbsoluteLoss{}, { 0.1, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, -1, 1);
    TestConjugateProx(HuberLoss{ 2.0 }, { 0.1, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, -1, 1);
    TestConjugateProx(HuberLoss{ 1.0 }, { 0.1, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, -1, 1);
    TestConjugateProx(HuberLoss{ 0.6 }, { 0.1, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, -1, 1);
    TestConjugateProx(SquareLoss{}, { 0.1, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, { -2.0, 0.1, 2.0 }, -10, 10);

    TestConjugateProx(HingeLoss{}, { 0.1, 0.1, 2.0 }, { -1.0, 2.0, 1.0 }, { -1.0, 2.0, 1.0 }, -1, 1);
    TestConjugateProx(LogisticLoss{}, { 0.1, 0.1, 2.0 }, { -1.0, 2.0, 1.0 }, { -1.0, 2.0, 1.0 }, -1, 1);
    TestConjugateProx(SmoothedHingeLoss{ 1.0 }, { 0.1, 0.1, 2.0 }, { -1.0, 2.0, 1.0 }, { -1.0, 2.0, 1.0 }, -1, 1);
    TestConjugateProx(SmoothedHingeLoss{ 0.5 }, { 0.1, 0.1, 2.0 }, { -1.0, 2.0, 1.0 }, { -1.0, 2.0, 1.0 }, -1, 1);
    TestConjugateProx(SmoothedHingeLoss{ 0.3 }, { 0.1, 0.1, 2.0 }, { -1.0, 2.0, 1.0 }, { -1.0, 2.0, 1.0 }, -1, 1);
    TestConjugateProx(SquaredHingeLoss{}, { 0.1, 0.1, 2.0 }, { -1.0, 2.0, 1.0 }, { -1.0, 2.0, 1.0 }, -10, 10);

    // Test convergence of SDCA on a synthetic regression problem

    TestSDCARegressionConvergence(AbsoluteLoss{}, L2Regularizer{}, { .5, true }, 1.0e-4, 1.0, 1.0, 1.0);
    TestSDCARegressionConvergence(HuberLoss{}, L2Regularizer{}, { .01, true }, 1.0e-4, 1.0, 1.0, 1.0);
    TestSDCARegressionConvergence(SquareLoss{}, L2Regularizer{}, { .1, true }, 1.0e-4, 1.0, 1.0, 1.0);

    TestSDCARegressionConvergence(AbsoluteLoss{}, ElasticNetRegularizer{ 0 }, { .5, true }, 1.0e-4, 1.0, 1.0, 1.0);
    TestSDCARegressionConvergence(AbsoluteLoss{}, ElasticNetRegularizer{ .1 }, { .5, true }, 1.0e-4, 1.0, 1.0, 1.0);
    TestSDCARegressionConvergence(HuberLoss{}, ElasticNetRegularizer{ 0 }, { .01, true }, 1.0e-4, 1.0, 1.0, 1.0);
    TestSDCARegressionConvergence(HuberLoss{}, ElasticNetRegularizer{ 5 }, { .01, true }, 1.0e-4, 1.0, 1.0, 1.0);
    TestSDCARegressionConvergence(SquareLoss{}, ElasticNetRegularizer{ 0 }, { .1, true }, 1.0e-4, 1.0, 1.0, 1.0);
    TestSDCARegressionConvergence(SquareLoss{}, ElasticNetRegularizer{ .5 }, { .1, true }, 1.0e-4, 1.0, 1.0, 1.0);

    TestSDCARegressionConvergence(AbsoluteLoss{}, MaxRegularizer{ 0 }, { .5, true }, 1.0e-4, 1.0, 1.0, 1.0);
    TestSDCARegressionConvergence(AbsoluteLoss{}, MaxRegularizer{ 2 }, { .5, true }, 1.0e-4, 1.0, 1.0, 1.0);
    TestSDCARegressionConvergence(HuberLoss{}, MaxRegularizer{ 0 }, { .01, true }, 1.0e-4, 1.0, 1.0, 1.0);
    TestSDCARegressionConvergence(HuberLoss{}, MaxRegularizer{ 20 }, { .01, true }, 1.0e-4, 1.0, 1.0, 1.0);
    TestSDCARegressionConvergence(SquareLoss{}, MaxRegularizer{ 0 }, { .1, true }, 1.0e-4, 1.0, 1.0, 1.0);
    TestSDCARegressionConvergence(SquareLoss{}, MaxRegularizer{ 1 }, { 1, true }, 1.0e-4, 1.0, 1.0, 1.0);

    // Test convergence of SDCA on a synthetic classification problem

    TestSDCAClassificationConvergence(HingeLoss{}, L2Regularizer{}, { .1, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(LogisticLoss{}, L2Regularizer{}, { .1, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(SmoothedHingeLoss{}, L2Regularizer{}, { .01, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(SquaredHingeLoss{}, L2Regularizer{}, { .1, true }, 1.0e-4, 1.0, 1.0, 3.0);

    TestSDCAClassificationConvergence(HingeLoss{}, ElasticNetRegularizer{ 0 }, { .1, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(HingeLoss{}, ElasticNetRegularizer{ .5 }, { .1, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(LogisticLoss{}, ElasticNetRegularizer{ 0 }, { .1, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(LogisticLoss{}, ElasticNetRegularizer{ .5 }, { .1, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(SmoothedHingeLoss{}, ElasticNetRegularizer{ 0 }, { .01, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(SmoothedHingeLoss{}, ElasticNetRegularizer{ 5 }, { .01, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(SquaredHingeLoss{}, ElasticNetRegularizer{ 0 }, { .1, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(SquaredHingeLoss{}, ElasticNetRegularizer{ .5 }, { .1, true }, 1.0e-4, 1.0, 1.0, 3.0);

    TestSDCAClassificationConvergence(HingeLoss{}, MaxRegularizer{ 0 }, { .5, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(HingeLoss{}, MaxRegularizer{ 1 }, { .5, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(LogisticLoss{}, MaxRegularizer{ 0 }, { .5, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(LogisticLoss{}, MaxRegularizer{ 0.2 }, { .5, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(SmoothedHingeLoss{}, MaxRegularizer{ 0 }, { .01, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(SmoothedHingeLoss{}, MaxRegularizer{ 20 }, { .01, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(SquaredHingeLoss{}, MaxRegularizer{ 0 }, { .1, true }, 1.0e-4, 1.0, 1.0, 3.0);
    TestSDCAClassificationConvergence(SquaredHingeLoss{}, MaxRegularizer{ 2 }, { .1, true }, 1.0e-4, 1.0, 1.0, 3.0);

    // SDCA Reset
    TestSDCAReset(SquaredHingeLoss{}, L2Regularizer{});
    TestGetSparseSolution(SmoothedHingeLoss{}, 0.01);

    // SGD solution equivalence tests, confirms that the four solution types behave identically when given equivalent problems

    TestSolutionEquivalenceSGD<double, AbsoluteLoss, L2Regularizer>(0.001);
    TestSolutionEquivalenceSGD<int, AbsoluteLoss, L2Regularizer>(0.001);

    TestSolutionEquivalenceSGD<double, HuberLoss, L2Regularizer>(0.001);
    TestSolutionEquivalenceSGD<int, HuberLoss, L2Regularizer>(0.001);

    TestSolutionEquivalenceSGD<double, SquareLoss, L2Regularizer>(1.0e+3);
    TestSolutionEquivalenceSGD<int, SquareLoss, L2Regularizer>(1.0e+3);

    TestSolutionEquivalenceSGD<double, HingeLoss, L2Regularizer>(0.001);
    TestSolutionEquivalenceSGD<int, HingeLoss, L2Regularizer>(0.001);

    TestSolutionEquivalenceSGD<double, LogisticLoss, L2Regularizer>(0.0001);
    TestSolutionEquivalenceSGD<float, LogisticLoss, L2Regularizer>(0.0001);
    TestSolutionEquivalenceSGD<int, LogisticLoss, L2Regularizer>(0.0001);
    TestSolutionEquivalenceSGD<double, LogisticLoss, ElasticNetRegularizer>(0.0001);
    TestSolutionEquivalenceSGD<float, LogisticLoss, ElasticNetRegularizer>(0.0001);
    TestSolutionEquivalenceSGD<int, LogisticLoss, ElasticNetRegularizer>(0.0001);
    TestSolutionEquivalenceSGD<double, LogisticLoss, MaxRegularizer>(0.0001);
    TestSolutionEquivalenceSGD<float, LogisticLoss, MaxRegularizer>(0.0001);
    TestSolutionEquivalenceSGD<int, LogisticLoss, MaxRegularizer>(0.0001);

    TestSolutionEquivalenceSGD<double, SmoothedHingeLoss, L2Regularizer>(0.001);
    TestSolutionEquivalenceSGD<int, SmoothedHingeLoss, L2Regularizer>(0.001);

    TestSolutionEquivalenceSGD<double, SquaredHingeLoss, L2Regularizer>(1.0e+3);
    TestSolutionEquivalenceSGD<int, SquaredHingeLoss, L2Regularizer>(1.0e+3);

    // SDCA solution equivalence tests, confirms that the four solution types behave identically when given equivalent problems

    TestSolutionEquivalenceSDCA<double, AbsoluteLoss, L2Regularizer>(0.001);
    TestSolutionEquivalenceSDCA<int, AbsoluteLoss, L2Regularizer>(0.001);

    TestSolutionEquivalenceSDCA<double, HuberLoss, L2Regularizer>(0.001);
    TestSolutionEquivalenceSDCA<int, HuberLoss, L2Regularizer>(0.001);

    TestSolutionEquivalenceSDCA<double, SquareLoss, L2Regularizer>(10);
    TestSolutionEquivalenceSDCA<int, SquareLoss, L2Regularizer>(10);

    TestSolutionEquivalenceSDCA<double, HingeLoss, L2Regularizer>(0.001);
    TestSolutionEquivalenceSDCA<int, HingeLoss, L2Regularizer>(0.001);

    TestSolutionEquivalenceSDCA<double, LogisticLoss, L2Regularizer>(0.0001);
    TestSolutionEquivalenceSDCA<float, LogisticLoss, L2Regularizer>(0.0001);
    TestSolutionEquivalenceSDCA<int, LogisticLoss, L2Regularizer>(0.0001);
    TestSolutionEquivalenceSDCA<double, LogisticLoss, ElasticNetRegularizer>(0.0001);
    TestSolutionEquivalenceSDCA<float, LogisticLoss, ElasticNetRegularizer>(0.0001);
    TestSolutionEquivalenceSDCA<int, LogisticLoss, ElasticNetRegularizer>(0.0001);
    TestSolutionEquivalenceSDCA<double, LogisticLoss, MaxRegularizer>(0.0001);
    TestSolutionEquivalenceSDCA<float, LogisticLoss, MaxRegularizer>(0.0001);
    TestSolutionEquivalenceSDCA<int, LogisticLoss, MaxRegularizer>(0.0001);

    TestSolutionEquivalenceSDCA<double, SmoothedHingeLoss, L2Regularizer>(0.001);
    TestSolutionEquivalenceSDCA<int, SmoothedHingeLoss, L2Regularizer>(0.001);

    TestSolutionEquivalenceSDCA<double, SquaredHingeLoss, L2Regularizer>(10);
    TestSolutionEquivalenceSDCA<int, SquaredHingeLoss, L2Regularizer>(10);

    TestMaskedMatrixSolution<int, UnbiasedMatrixSolution>();
    TestMaskedMatrixSolution<float, UnbiasedMatrixSolution>();
    TestMaskedMatrixSolution<double, UnbiasedMatrixSolution>();
    TestMaskedMatrixSolution<int, BiasedMatrixSolution>();
    TestMaskedMatrixSolution<float, BiasedMatrixSolution>();
    TestMaskedMatrixSolution<double, BiasedMatrixSolution>();

    // search techniques

    TestExponentialSearch();
    TestBinarySearch();
    TestGoldenSectionSearch();

    // Other tests
    TestRegularizerEquivalence(1.0e-1);
    TestRegularizerEquivalence(1.0e-2);
    TestRegularizerEquivalence(1.0e-3);
    TestRegularizerEquivalence(1.0e-4);
    TestL1Prox();
    TestLInfinityProx();
    TestMatrixDataset();

    return testing::DidTestFail();
}
