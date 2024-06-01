////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ProtoNNTrainer.cpp (trainers)
//  Authors:  Suresh Iyengar
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ProtoNNTrainer.h"
#include "ProtoNNInit.h"
#include "ProtoNNTrainerUtils.h"

#include <math/include/MatrixOperations.h>
#include <math/include/Vector.h>

#include <data/include/Dataset.h>

#include <utilities/include/Unused.h>

#include <cassert>
#include <cmath>
#include <ctime>
#include <iostream>

namespace ell
{
namespace trainers
{
    namespace
    {
        constexpr double ArmijoStepTolerance = 0.02;

        constexpr double DefaultStepSize = 0.2;
    } // namespace

    double safe_div(const double& num, const double& den)
    {
        assert(!std::isnan(num));
        double ret = num / den;
        assert(!std::isnan(ret));
        return ret;
    }

    ProtoNNTrainer::ProtoNNTrainer(const ProtoNNTrainerParameters& parameters) :
        _dimemsion(parameters.numFeatures),
        _parameters(parameters),
        _protoNNPredictor(parameters.numFeatures, parameters.projectedDimension, parameters.numPrototypesPerLabel * parameters.numLabels, parameters.numLabels, parameters.gamma),
        _X(0, 0),
        _Y(0, 0)
    {
    }

    void ProtoNNTrainer::SetDataset(const data::AnyDataset& anyDataset)
    {
        auto numExamples = anyDataset.NumExamples();
        _X = math::ColumnMatrix<double>(_dimemsion, numExamples);
        _Y = math::ColumnMatrix<double>(_parameters.numLabels, numExamples),
        ProtoNNTrainerUtils::GetDatasetAsMatrix(anyDataset, _X, _Y);
        _firstIteration = true;
    }

    void ProtoNNTrainer::Update()
    {
        if (_firstIteration)
        {
            Initialize();
            _firstIteration = false;
        }

        SGDWithAlternatingMinimization(_X, _Y, _parameters.gamma, _iteration++);

        _protoNNPredictor.GetProjectionMatrix() = _modelMap[ProtoNNParameterIndex::W]->GetData();
        _protoNNPredictor.GetPrototypes() = _modelMap[ProtoNNParameterIndex::B]->GetData();
        _protoNNPredictor.GetLabelEmbeddings() = _modelMap[ProtoNNParameterIndex::Z]->GetData();
        _protoNNPredictor.GetGamma() = _parameters.gamma;
    }

    void ProtoNNTrainer::Initialize()
    {
        auto numPrototypes = _parameters.numLabels * _parameters.numPrototypesPerLabel;

        size_t D = _dimemsion;
        size_t d = _parameters.projectedDimension; // projection dimension
        size_t n = _X.NumColumns();
        size_t m = numPrototypes; // number of prototypes
        size_t l = _parameters.numLabels; // number of labels

        math::ColumnMatrix<double> W(d, D);
        std::default_random_engine rng;
        std::normal_distribution<double> normal(0, 1.0);
        auto generator = [&]() { return normal(rng); };
        W.Generate(generator);

        math::ColumnMatrix<double> WX(W.NumRows(), n);
        math::MultiplyScaleAddUpdate(1.0, W, _X, 0.0, WX);

        ProtoNNInit protonnInit(d, _parameters.numLabels, _parameters.numPrototypesPerLabel);
        protonnInit.Initialize(WX, _Y);

        math::ColumnMatrix<double> B = protonnInit.GetPrototypeMatrix();
        math::ColumnMatrix<double> Z = protonnInit.GetLabelMatrix();

        _modelMap[ProtoNNParameterIndex::W] = std::make_shared<trainers::Param_W>(d, D);
        _modelMap[ProtoNNParameterIndex::Z] = std::make_shared<trainers::Param_Z>(l, m);
        _modelMap[ProtoNNParameterIndex::B] = std::make_shared<trainers::Param_B>(d, m);

        _modelMap[ProtoNNParameterIndex::W]->GetData() = W;
        _modelMap[ProtoNNParameterIndex::Z]->GetData() = Z;
        _modelMap[ProtoNNParameterIndex::B]->GetData() = B;

        // Initializing gamma
        if (-1.0 == _parameters.gamma)
        {
            auto gammaInit = 0.01;
            math::ColumnMatrix<double> WXupdate(W.NumRows(), n);
            math::MultiplyScaleAddUpdate(1.0, W, _X, 0.0, WXupdate);
            _parameters.gamma = protonnInit.InitializeGamma(SimilarityKernel(_X, WXupdate, gammaInit), gammaInit);
        }

        _stepSize[ProtoNNParameterIndex::W] = DefaultStepSize;
        _stepSize[ProtoNNParameterIndex::Z] = DefaultStepSize;
        _stepSize[ProtoNNParameterIndex::B] = DefaultStepSize;

        _sparsity[ProtoNNParameterIndex::W] = _parameters.sparsityW;
        _sparsity[ProtoNNParameterIndex::Z] = _parameters.sparsityZ;
        _sparsity[ProtoNNParameterIndex::B] = _parameters.sparsityB;

        for (auto parameterIndex : m_OptimizationOrder)
        {
            _recomputeWX[parameterIndex] = false;
        }

        // For the Projection parameter, recompoute WX is set to true
        _recomputeWX[m_projectionIndex] = true;
    }

    /// S_{ij} = exp{-gamma^2 * || B_j - W*x_i ||^2}
    /// where S_{ij} is similarity of ith input instance with the jth prototype B_j and W is the projection matrix
    /// Computed as exp(-gamma^2(||B||^2 + ||WX||^2 - 2 *  WX' * B))
    math::ColumnMatrix<double> ProtoNNTrainer::SimilarityKernel(ConstColumnMatrixReference X, math::ColumnMatrixReference<double> WX, const double gamma, const size_t begin, const size_t end, bool recomputeWX)
    {
        assert(begin < end);
        auto B = (_modelMap[ProtoNNParameterIndex::B])->GetData();
        auto W = (_modelMap[ProtoNNParameterIndex::W])->GetData();

        auto wx = WX.GetSubMatrix(0, begin, WX.NumRows(), end - begin);

        // if W has changed, recompute WX
        if (true == recomputeWX)
        {
            math::ColumnMatrix<double> wxUpdated(W.NumRows(), end - begin);
            auto x = X.GetSubMatrix(0, begin, X.NumRows(), end - begin);
            math::MultiplyScaleAddUpdate(1.0, W, x, 0.0, wxUpdated);
            wx.CopyFrom(wxUpdated);
        }

        // full(sum(B. ^ 2, 1));
        math::ColumnMatrix<double> bSquare(B.NumRows(), B.NumColumns());
        math::ElementwiseMultiplySet(B, B, bSquare);

        math::ColumnMatrix<double> bColNormSquare(1, bSquare.NumColumns());
        math::ColumnwiseSum(bSquare, bColNormSquare.GetRow(0));

        // full(sum(WX. ^ 2, 1));
        math::ColumnMatrix<double> wxSquare(wx.NumRows(), wx.NumColumns());
        math::ElementwiseMultiplySet(wx, wx, wxSquare);

        math::ColumnMatrix<double> wxColNormSquare(1, wxSquare.NumColumns());
        math::ColumnwiseSum(wxSquare, wxColNormSquare.GetRow(0));

        // wxb = (2.0 * gamma * gamma) * WX.transpose() * B;
        math::RowMatrix<double> wxb(wx.NumColumns(), B.NumColumns());
        math::MultiplyScaleAddUpdate(2 * gamma * gamma, wx.Transpose(), B, 0.0, wxb);

        // repmat of bColNormSquare and scaling by -gamma * gamma
        math::RowMatrix<double> gammaSquareCol(end - begin, 1);
        gammaSquareCol.Fill(-gamma * gamma);
        math::RowMatrix<double> bColNormSquareResized(gammaSquareCol.NumRows(), bColNormSquare.NumColumns());
        // S1 = gammaSquareCol * bColSum;
        math::MultiplyScaleAddUpdate(1.0, gammaSquareCol, bColNormSquare, 0.0, bColNormSquareResized);

        // repmat of wxColNormSquare and scaling by -gamma * gamma
        math::RowMatrix<double> gammaSquareRow(B.NumColumns(), 1);
        gammaSquareRow.Fill(-gamma * gamma);
        math::RowMatrix<double> wxColNormSquareResized(gammaSquareRow.NumRows(), wxColNormSquare.NumColumns());
        // S2 = gammaSquareRow * wxColSum;
        math::MultiplyScaleAddUpdate(1.0, gammaSquareRow, wxColNormSquare, 0.0, wxColNormSquareResized);

        // D = wxb + bColNormSquareResized
        math::RowMatrix<double> similarityMatrix(wxb.NumRows(), wxb.NumColumns());
        math::ScaleAddSet(1.0, wxb, 1.0, bColNormSquareResized, similarityMatrix);

        // D = D + wxColNormSquareResized'
        math::RowMatrix<double> distance(similarityMatrix.NumRows(), similarityMatrix.NumColumns());
        math::ScaleAddSet(1.0, similarityMatrix, 1.0, wxColNormSquareResized.Transpose(), distance);

        // similarityMatrix = exp(D)
        similarityMatrix = ProtoNNTrainerUtils::MatrixExp(distance);

        return similarityMatrix;
    }

    math::ColumnMatrix<double> ProtoNNTrainer::SimilarityKernel(ConstColumnMatrixReference X, math::ColumnMatrixReference<double> WX, const double gamma, bool recomputeWX)
    {
        return SimilarityKernel(X, WX, gamma, 0, X.NumColumns(), recomputeWX);
    }

    double ProtoNNTrainer::Loss(ConstColumnMatrixReference Y, ConstColumnMatrixReference D, const size_t begin, const size_t end)
    {
        assert(end - begin == D.NumRows());

        auto Z = (_modelMap[ProtoNNParameterIndex::Z])->GetData();

        // residual = y - ZD'
        math::ColumnMatrix<double> ZD(Z.NumRows(), D.NumRows());
        math::MultiplyScaleAddUpdate(1.0, Z, D.Transpose(), 0.0, ZD);
        auto y = Y.GetSubMatrix(0, begin, Y.NumRows(), end - begin);
        math::ColumnMatrix<double> residual(y.NumRows(), y.NumColumns());
        math::ScaleAddSet(1.0, y, -1.0, ZD, residual);

        switch (_parameters.lossFunction)
        {
        case ProtoNNLossFunction::L2:
            for (size_t j = 0; j < residual.NumColumns(); j++)
            {
                // diff .* 2
                residual.GetColumn(j).Transform([](double x) { return x * x; });
            }
            break;
        case ProtoNNLossFunction::L4:
            for (size_t j = 0; j < residual.NumColumns(); j++)
            {
                // diff .* 4
                residual.GetColumn(j).Transform([](double x) { return x * x * x * x; });
            }
            break;
        }

        auto loss = 0.0;

        // Aggregate all the loss values
        for (size_t j = 0; j < residual.NumColumns(); j++)
        {
            loss += residual.GetColumn(j).Aggregate([](double x) { return x; });
        }

        return loss;
    }

    double ProtoNNTrainer::Loss(ConstColumnMatrixReference Y, ConstColumnMatrixReference D)
    {
        return Loss(Y, D, 0, Y.NumColumns());
    }

    double ProtoNNTrainer::ComputeObjective(ConstColumnMatrixReference X, ConstColumnMatrixReference Y, math::ColumnMatrixReference<double> WX, double gamma, bool recomputeWX)
    {
        double objective = 0.0;

        size_t n = X.NumColumns();
        size_t maxBatchSize = (size_t)std::ceil(std::sqrt(n));

        if (maxBatchSize > n) maxBatchSize = n;

        size_t batchSize = maxBatchSize;
        size_t numBatches = (n + batchSize - 1) / batchSize;

        // Aggregate loss over the batches
        for (size_t i = 0; i < numBatches; ++i)
        {
            size_t idx1 = (i * batchSize) % n;
            size_t idx2 = ((i + 1) * (batchSize) % n);
            if (idx2 <= idx1) idx2 = n;

            assert(idx1 < idx2);
            assert(idx2 - idx1 <= maxBatchSize);

            auto D = SimilarityKernel(X, WX, gamma, idx1, idx2, recomputeWX);
            auto y = Y.GetSubMatrix(0, idx1, Y.NumRows(), idx2 - idx1);

            objective += Loss(y, D);
        }

        return objective;
    }

    //See https://blogs.princeton.edu/imabandit/2013/04/01/acceleratedgradientdescent/ for the accelerated gradient_paramS descent version we use
    //We use stochastic version of the above algorithm
    //paramQ_new[t+1]=paramS[t]-stepSize*gradient_paramS(paramS[t]) //gradient_paramS descent update
    //paramS[t+1]=(1-alpha)paramQ_new[t+1]+alpha*paramQ_new[t] //paramS_new update
    //paramAvg=running average of all but first burn_period paramS.
    //stepSize is decaying at approximately initialStepSize/(1+t) rate (if decayRate=-1) or initialStepSize/sqrt(1+t) (if decayRate=0)
    void ProtoNNTrainer::AcceleratedProximalGradient(
        ProtoNNParameterIndex parameterIndex,
        std::function<math::ColumnMatrix<double>(const ConstColumnMatrixReference, const size_t, const size_t)> gradf,
        std::function<void(math::MatrixReference<double, math::MatrixLayout::columnMajor>)> prox,
        math::MatrixReference<double, math::MatrixLayout::columnMajor> param,
        const size_t& epochs,
        const size_t& n,
        const size_t& batchSize,
        const double& eta,
        const int& decayRate)
    {
        math::ColumnMatrix<double> paramt(param.NumRows(), param.NumColumns());
        math::ColumnMatrix<double> paramAvg(param.NumRows(), param.NumColumns());
        math::ColumnMatrix<double> gradient_paramS(param.NumRows(), param.NumColumns());
        math::ColumnMatrix<double> paramQ(param.NumRows(), param.NumColumns());

        math::ColumnMatrix<double> paramS(param.NumRows(), param.NumColumns());

        paramQ.CopyFrom(param);
        paramS.CopyFrom(param);

        int burn_period = 50;
        double lambda = 1;
        double stepSize, lambda_new, alpha;

        assert(batchSize <= n);

        size_t iters = ((uint64_t)n * (uint64_t)epochs) / (uint64_t)batchSize;
        assert(iters < 0x7fffffff);

        for (size_t i = 0; i < iters; ++i)
        {
            int t = static_cast<int>(i); // this algorithm requires signed integer arithmetic
            size_t idx1 = (t * batchSize) % n;
            size_t idx2 = ((t + 1) * batchSize) % n;
            if (idx2 <= idx1) idx2 = n;

            switch (decayRate)
            {
            case -1:
                stepSize = safe_div(eta, (1 + (double)0.2 * ((double)t + 1.0))); //stepSize=eta/(1+.2*(t+1)), eta is initial stepsize
                break;
            case 0:
                stepSize = safe_div(eta, pow((double)t + 1.0, (double)0.5)); //stepSize=eta/sqrt(1+t), eta is initial stepsize
                break;
            default:
                stepSize = safe_div(eta, pow((double)t + 1.0, (double)0.5)); //stepSize=eta/sqrt(1+t), eta is initial stepsize
                break;
            }

            lambda_new = 0.5 + 0.5 * pow(1 + 4 * lambda * lambda, 0.5);
            alpha = safe_div((1 - lambda), lambda_new); // alpha: weight for paramS_new term

            _modelMap[parameterIndex]->GetData() = paramS; //paramS

            gradient_paramS = gradf(paramS, idx1, idx2);

            math::ColumnMatrix<double> paramQ_new(paramS.NumRows(), paramS.NumColumns());
            math::ScaleAddSet(1.0, paramS, -stepSize, gradient_paramS, paramQ_new); //paramQ_new=paramS-stepSize*grad(paramS)

            prox(paramQ_new); //paramQ_new = HardThresholding(paramQ_new)

            math::ColumnMatrix<double> paramS_new(paramQ_new.NumRows(), paramQ_new.NumColumns());
            math::ScaleAddSet(1 - alpha, paramQ_new, alpha, paramQ, paramS_new);

            // paramS_new = (1-alpha)*paramQ_new+alpha*paramQ; paramS=paramS_new
            paramS.CopyFrom(paramS_new);

            double runningAvgWeight = ((t - burn_period) > 1) ? (t - burn_period) : 1.0; //runningAvgWeight
            assert(runningAvgWeight >= 0.999999);

            //Running average of all but first burn_period paramS's; paramAvg_new=(1-1/runningAvgWeight)*paramAvg+ 1/runningAvgWeight*paramS_new
            math::ColumnMatrix<double> paramAvg_new(paramS_new.NumRows(), paramS_new.NumColumns());
            math::ScaleAddSet(safe_div(1.0, runningAvgWeight), paramS_new, safe_div(runningAvgWeight - 1.0, runningAvgWeight), paramAvg, paramAvg_new);

            //Initializing parameters for next iteration
            lambda = lambda_new;
            paramQ = paramQ_new;

            paramS.CopyFrom(paramS_new);
            paramAvg.CopyFrom(paramAvg_new);
        }

        param.CopyFrom(paramAvg);
        paramS.CopyFrom(paramAvg);

        _modelMap[parameterIndex]->GetData() = paramS;
    }

    //minimize f(W, B, Z) = \sum_{i = 1} ^ numTrainData Loss(Y[i], Z* D[i]) where D[i][j] = exp(-gamma^2 || B[j]-WX[i] || ^ 2) where j = 1:numPrototypes
    void ProtoNNTrainer::SGDWithAlternatingMinimization(ConstColumnMatrixReference X, ConstColumnMatrixReference Y, double gamma, size_t iter)
    {
        // Start Initializations
        size_t n = X.NumColumns(); //numTrainPoints
        size_t epochs = _parameters.numInnerIterations; // number of SGD iterations(epochs) over each of the parameters

        size_t sgdBatchSize = 1 << 8 < n ? 1 << 8 : n; //expose to user parameter structure, default is sqrt(n)

        double armijoStepTolerance = ArmijoStepTolerance;

        //knob for step-size decay, decayRate = 0 sets the decay to be 1/sqrt(t), decayRate = 1 sets it to be 1/t
        int eta_update = 0;

        const double smallPerturbation = 0.001; // used to calculate the step size while approximating Hessian

        double fOld, fCur, paramStepSize;

        //Projection onto low-d space
        auto projectionMatrix = _modelMap[m_projectionIndex]->GetData();
        math::ColumnMatrix<double> WX(projectionMatrix.NumRows(), n);
        math::MultiplyScaleAddUpdate(1.0, projectionMatrix, X, 0.0, WX);

        fCur = ComputeObjective(X, Y, WX, gamma, false);

        // End Initializations

        // SGD iterations
        for (auto parameterIndex : m_OptimizationOrder)
        {
            math::RowVector<double> eta(10);

            auto parameter = _modelMap[parameterIndex];
            auto parameterMatrix = parameter->GetData();
            math::ColumnMatrix<double> currentGradient(parameterMatrix.NumRows(), parameterMatrix.NumColumns());

            if (_parameters.verbose)
                std::cout << "Iteration " << iter << std::endl;

            // Step-size estimation: We try out 10 random batches of data, estimate Hessian (H) of the function using each batch.
            // Select median of 1/H as the stepsize.
            for (size_t j = 0; j < eta.Size(); ++j)
            {
                size_t idx1 = (j * sgdBatchSize) % n;
                size_t idx2 = ((j + 1) * sgdBatchSize) % n;
                if (idx2 <= idx1) idx2 = n;

                // gradient_paramS at current parameter
                currentGradient = _modelMap[parameterIndex]->gradient(_modelMap, X, Y, WX, SimilarityKernel(X, WX, gamma, idx1, idx2, _recomputeWX[parameterIndex]), gamma, idx1, idx2, _parameters.lossFunction);

                math::ColumnMatrix<double> thresholdedGradient(parameterMatrix.NumRows(), parameterMatrix.NumColumns());

                thresholdedGradient.CopyFrom(currentGradient);

                ProtoNNTrainerUtils::HardThresholding(thresholdedGradient, _sparsity[parameterIndex]);

                auto coeff = smallPerturbation * safe_div(ProtoNNTrainerUtils::MaxAbsoluteElement(parameterMatrix), ProtoNNTrainerUtils::MaxAbsoluteElement(currentGradient));

                // perturb the parameter
                math::ColumnMatrix<double> perturbedParameter(parameterMatrix.NumRows(), parameterMatrix.NumColumns());
                math::ScaleAddSet(1.0, parameterMatrix, -1.0 * coeff, thresholdedGradient, perturbedParameter);

                auto WX_old = WX;
                _modelMap[parameterIndex]->GetData() = perturbedParameter;

                // Compute gradient_paramS with updated parameter
                math::MultiplyScaleAddUpdate(1.0, _modelMap[m_projectionIndex]->GetData(), X, 0.0, WX);

                math::ColumnMatrix<double> gradientEstimate(parameterMatrix.NumRows(), parameterMatrix.NumColumns());
                auto grad = _modelMap[parameterIndex]->gradient(_modelMap, X, Y, WX, SimilarityKernel(X, WX, gamma, idx1, idx2, _recomputeWX[parameterIndex]), gamma, idx1, idx2, _parameters.lossFunction);
                math::ScaleAddSet(1.0, currentGradient, -1.0, grad, gradientEstimate);

                currentGradient = gradientEstimate;

                // revert the old parameter value and projected input
                _modelMap[parameterIndex]->GetData() = parameterMatrix;
                WX = WX_old;

                if (ProtoNNTrainerUtils::MatrixNorm(currentGradient) <= 1e-20L)
                {
                    if (_parameters.verbose)
                        std::cerr << "Difference between consecutive gradients has become really low..\n";
                    eta[j] = 1;
                }
                else
                {
                    math::ColumnMatrix<double> deltaParameter(parameterMatrix.NumRows(), parameterMatrix.NumColumns());
                    math::ScaleAddSet(1.0, perturbedParameter, -1.0, parameterMatrix, deltaParameter);
                    eta[j] = safe_div(ProtoNNTrainerUtils::MatrixNorm(deltaParameter), ProtoNNTrainerUtils::MatrixNorm(currentGradient));
                }
            }

            std::vector<double> etaVector(eta.GetDataPointer(), eta.GetDataPointer() + eta.Size());
            std::sort(etaVector.begin(), etaVector.end());

            paramStepSize = _stepSize[parameterIndex] * etaVector[4];

            // Call the accelerated proximal gradient_paramS method for optimizing this parameter
            AcceleratedProximalGradient(
                parameterIndex, [&](ConstColumnMatrixReference /*W*/, const size_t begin, const size_t end) -> math::ColumnMatrix<double> { return _modelMap[parameterIndex]->gradient(_modelMap, X, Y, WX, SimilarityKernel(X, WX, gamma, begin, end, _recomputeWX[parameterIndex]), gamma, begin, end, _parameters.lossFunction); }, [&](auto arg) { ProtoNNTrainerUtils::HardThresholding(arg, _sparsity[parameterIndex]); }, parameterMatrix, epochs, n, sgdBatchSize, paramStepSize, eta_update);

            math::MultiplyScaleAddUpdate(1.0, _modelMap[m_projectionIndex]->GetData(), X, 0.0, WX);
            fOld = fCur;
            fCur = ComputeObjective(X, Y, WX, gamma, _recomputeWX[parameterIndex]);

            // Armijo step
            // If function value has increased, decrease the step size else increase
            if (fCur >= fOld * (1 + safe_div(armijoStepTolerance * log(3), log(2 + iter))))
                _stepSize[parameterIndex] *= 0.7;
            else if (fCur <= fOld * (1 - safe_div(3 * armijoStepTolerance * log(3), log(2 + iter))))
                _stepSize[parameterIndex] *= 1.1;
        }
    }

    ProtoNNModelParameter::ProtoNNModelParameter() :
        _data(0, 0)
    {
    }

    ProtoNNModelParameter::ProtoNNModelParameter(size_t dim1, size_t dim2) :
        _data(dim1, dim2)
    {
    }

    ProtoNNModelParameter::~ProtoNNModelParameter() = default;

    Param_W::Param_W(size_t dim1, size_t dim2) :
        ProtoNNModelParameter(dim1, dim2)
    {
    }

    math::ColumnMatrix<double> Param_W::gradient(ProtoNNModelMap& modelMap, ConstColumnMatrixReference X, ConstColumnMatrixReference Y, ConstColumnMatrixReference WX, ConstColumnMatrixReference D, double gamma, size_t begin, size_t end, ProtoNNLossFunction lossType)
    {
        UNUSED(WX);
        assert(end - begin == D.NumRows());

        auto W = modelMap[ProtoNNParameterIndex::W]->GetData();
        auto B = modelMap[ProtoNNParameterIndex::B]->GetData();
        auto Z = modelMap[ProtoNNParameterIndex::Z]->GetData();

        auto y = Y.GetSubMatrix(0, begin, Y.NumRows(), end - begin).Transpose();

        math::ColumnMatrix<double> DZ(D.NumRows(), Z.NumRows());
        math::MultiplyScaleAddUpdate(1.0, D, Z.Transpose(), 0.0, DZ);

        //diff = Y - D*Z';
        math::RowMatrix<double> residual(y.NumRows(), y.NumColumns());
        math::ScaleAddSet(1.0, y, -1.0, DZ, residual);

        switch (lossType)
        {
        case ProtoNNLossFunction::L2:
            for (size_t j = 0; j < residual.NumColumns(); j++)
            {
                // 4 * gamma * gamma * diff
                double multiplication_scalar = 4.0 * gamma * gamma;
                residual.GetColumn(j).Transform([multiplication_scalar](double x) { return multiplication_scalar * x; });
            }
            break;
        case ProtoNNLossFunction::L4:
            for (size_t j = 0; j < residual.NumColumns(); j++)
            {
                // 8 * gamma * gamma * diff
                double multiplication_scalar = 8.0 * gamma * gamma;
                residual.GetColumn(j).Transform([multiplication_scalar](double x) { return multiplication_scalar * x * x * x; });
            }
            break;
        }

        // T = diff * Z
        math::RowMatrix<double> T(D.NumRows(), D.NumColumns());
        math::MultiplyScaleAddUpdate(1.0, residual, Z, 0.0, T);

        // T = D .* T
        math::ElementwiseMultiplySet(T, D, T);

        math::ColumnMatrix<double> colMult(1, T.NumRows());
        math::ColumnwiseSum(T.Transpose(), colMult.GetRow(0));

        auto xSub = X.GetSubMatrix(0, begin, X.NumRows(), end - begin);
        math::ColumnMatrix<double> wxScaled(W.NumRows(), end - begin);
        math::MultiplyScaleAddUpdate(1.0, W, xSub, 0.0, wxScaled);

        for (size_t j = 0; j < wxScaled.NumColumns(); j++)
        {
            auto t = colMult(0, j);
            wxScaled.GetColumn(j).Transform([t](double x) { return t * x; });
        }

        //wx_scaled = wx_scaled - B*T
        math::MultiplyScaleAddUpdate(-1.0, B, T.Transpose(), 1.0, wxScaled);

        // gradient_paramS -= wx_scaled * x_submat'
        math::ColumnMatrix<double> gradient(W.NumRows(), W.NumColumns());
        math::MultiplyScaleAddUpdate(1.0, wxScaled, xSub.Transpose(), 0.0, gradient);

        return gradient;
    }

    math::ColumnMatrix<double> Param_W::gradient(ProtoNNModelMap& modelMap, ConstColumnMatrixReference X, ConstColumnMatrixReference Y, ConstColumnMatrixReference WX, ConstColumnMatrixReference D, double gamma, ProtoNNLossFunction lossType)
    {
        return gradient(modelMap, X, Y, WX, D, gamma, 0, Y.NumColumns(), lossType);
    }

    Param_Z::Param_Z(size_t dim1, size_t dim2) :
        ProtoNNModelParameter(dim1, dim2)
    {
    }

    math::ColumnMatrix<double> Param_Z::gradient(ProtoNNModelMap& modelMap, ConstColumnMatrixReference X, ConstColumnMatrixReference Y, ConstColumnMatrixReference WX, ConstColumnMatrixReference Similarity, double gamma, size_t begin, size_t end, ProtoNNLossFunction lossType)
    {
        UNUSED(X, WX, gamma);

        assert(end - begin == Similarity.NumRows());

        auto Z = modelMap[ProtoNNParameterIndex::Z]->GetData();

        auto y = Y.GetSubMatrix(0, begin, Y.NumRows(), end - begin);

        // ZD = Z * D'
        math::ColumnMatrix<double> ZD(Z.NumRows(), Similarity.NumRows());
        math::MultiplyScaleAddUpdate(1.0, Z, Similarity.Transpose(), 0.0, ZD);

        // yMinusZD = y - ZD'
        math::ColumnMatrix<double> residual(y.NumRows(), y.NumColumns());
        math::ScaleAddSet(1.0, y, -1.0, ZD, residual);

        math::ColumnMatrix<double> gradient(residual.NumRows(), Similarity.NumColumns());
        switch (lossType)
        {
        case ProtoNNLossFunction::L2:
            // gradient_paramS = -2 * yMinusZD * D
            math::MultiplyScaleAddUpdate(-2.0, residual, Similarity, 0.0, gradient);
            break;

        case ProtoNNLossFunction::L4:
            for (size_t j = 0; j < residual.NumColumns(); j++)
            {
                residual.GetColumn(j).Transform([](double x) { return x * x * x; });
            }
            // gradient_paramS = -4 * (yMinusZD .^ 3) * D
            math::MultiplyScaleAddUpdate(-4.0, residual, Similarity, 0.0, gradient);
            break;
        }

        return gradient;
    }

    math::ColumnMatrix<double> Param_Z::gradient(ProtoNNModelMap& modelMap, ConstColumnMatrixReference X, ConstColumnMatrixReference Y, ConstColumnMatrixReference WX, ConstColumnMatrixReference D, double gamma, ProtoNNLossFunction lossType)
    {
        return gradient(modelMap, X, Y, WX, D, gamma, 0, Y.NumColumns(), lossType);
    }

    Param_B::Param_B(size_t dim1, size_t dim2) :
        ProtoNNModelParameter(dim1, dim2)
    {
    }

    math::ColumnMatrix<double> Param_B::gradient(ProtoNNModelMap& modelMap, ConstColumnMatrixReference X, ConstColumnMatrixReference Y, ConstColumnMatrixReference WX, ConstColumnMatrixReference Similarity, double gamma, size_t begin, size_t end, ProtoNNLossFunction lossType)
    {
        UNUSED(X, WX);
        assert(end - begin == Similarity.NumRows());

        auto B = modelMap[ProtoNNParameterIndex::B]->GetData();
        auto Z = modelMap[ProtoNNParameterIndex::Z]->GetData();

        auto y = Y.GetSubMatrix(0, begin, Y.NumRows(), end - begin).Transpose();
        auto wx = WX.GetSubMatrix(0, begin, WX.NumRows(), end - begin);
        math::ColumnMatrix<double> DZ(Similarity.NumRows(), Z.NumRows());
        math::MultiplyScaleAddUpdate(1.0, Similarity, Z.Transpose(), 0.0, DZ);

        // residual = y - D*Z';
        math::RowMatrix<double> residual(y.NumRows(), y.NumColumns());
        math::ScaleAddSet(1.0, y, -1.0, DZ, residual);

        switch (lossType)
        {
        case ProtoNNLossFunction::L2:
            for (size_t j = 0; j < residual.NumColumns(); j++)
            {
                // 4 * gamma * gamma * diff
                double multScalar = 4.0 * gamma * gamma;
                residual.GetColumn(j).Transform([multScalar](double x) { return multScalar * x; });
            }
            break;
        case ProtoNNLossFunction::L4:
            for (size_t j = 0; j < residual.NumColumns(); j++)
            {
                // 8 * gamma * gamma * diff
                double multScalar = 8.0 * gamma * gamma;
                residual.GetColumn(j).Transform([multScalar](double x) { return multScalar * x * x * x; });
            }

            break;
        }

        // T = residual * Z
        math::RowMatrix<double> T(Similarity.NumRows(), Similarity.NumColumns());
        math::MultiplyScaleAddUpdate(1.0, residual, Z, 0.0, T);

        // T = D .* T (final output T = residual * Z * Similarity)
        math::ElementwiseMultiplySet(T, Similarity, T);

        // gradient_paramS = B (initialized gradient_paramS to prototypes)
        math::ColumnMatrix<double> gradient(B.NumRows(), B.NumColumns());

        gradient.CopyFrom(B);

        // sum(T, 1)
        math::ColumnMatrix<double> colMult(1, T.NumColumns());
        math::ColumnwiseSum(T, colMult.GetRow(0));

        // gradient_paramS = gradient_paramS .* sum(T, 1)
        for (size_t j = 0; j < gradient.NumColumns(); j++)
        {
            auto t = colMult(0, j);
            gradient.GetColumn(j).Transform([t](double x) { return t * x; });
        }

        // gradient_paramS = gradient_paramS - wx * T
        math::MultiplyScaleAddUpdate(-1.0, wx, T, 1.0, gradient);

        return gradient;
    }

    math::ColumnMatrix<double> Param_B::gradient(ProtoNNModelMap& modelMap, ConstColumnMatrixReference X, ConstColumnMatrixReference Y, ConstColumnMatrixReference WX, ConstColumnMatrixReference D, double gamma, ProtoNNLossFunction lossType)
    {
        return gradient(modelMap, X, Y, WX, D, gamma, 0, Y.NumColumns(), lossType);
    }

    std::unique_ptr<trainers::ProtoNNTrainer> MakeProtoNNTrainer(const trainers::ProtoNNTrainerParameters& parameters)
    {
        return std::make_unique<trainers::ProtoNNTrainer>(parameters);
    }
} // namespace trainers
} // namespace ell
