////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     main.cpp (utilities)
//  Authors:  Byron Changuion
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "RetargetArguments.h"

#include <utilities/include/CommandLineParser.h>
#include <utilities/include/Exception.h>
#include <utilities/include/MillisecondTimer.h>

#include <common/include/DataLoaders.h>
#include <common/include/LoadModel.h>
#include <common/include/MakeEvaluator.h>
#include <common/include/MakeTrainer.h>

#include <data/include/Dataset.h>

#include <evaluators/include/Evaluator.h>

#include <functions/include/HingeLoss.h>
#include <functions/include/L2Regularizer.h>
#include <functions/include/LogLoss.h>
#include <functions/include/SmoothHingeLoss.h>
#include <functions/include/SquaredLoss.h>

#include <model/include/InputNode.h>
#include <model/include/Map.h>
#include <model/include/OutputNode.h>

#include <nodes/include/ActivationFunctions.h>
#include <nodes/include/BinaryOperationNode.h>
#include <nodes/include/BroadcastFunctionNode.h>
#include <nodes/include/ConstantNode.h>
#include <nodes/include/LinearPredictorNode.h>
#include <nodes/include/MatrixVectorProductNode.h>
#include <nodes/include/NeuralNetworkPredictorNode.h>
#include <nodes/include/SinkNode.h>

#include <predictors/include/Normalizer.h>

#include <algorithm>
#include <iostream>

using namespace ell;

// predictor type
using PredictorType = predictors::LinearPredictor<double>;

template <typename ElementType>
model::Map AppendTrainedLinearPredictorToMap(const PredictorType& trainedPredictor, model::Map& map, size_t dimension)
{
    predictors::LinearPredictor<ElementType> predictor(trainedPredictor);
    predictor.Resize(dimension);

    const auto& mapOutput = map.GetOutput<ElementType>(0);
    const auto& predictorOutput = nodes::LinearPredictor(mapOutput, predictor);
    const auto& sink = nodes::Sink(predictorOutput);
    const auto& output = model::Output(sink);
    auto outputMap = model::Map(map.GetModel(), { { "input", map.GetInput() } }, { { "output", output } });
    return outputMap;
}

template <typename ElementType>
bool RedirectNeuralNetworkOutputByLayer(model::Map& map, size_t numLayersFromEnd)
{
    bool found = false;
    auto nodes = (map.GetModel()).GetNodesByType<nodes::NeuralNetworkPredictorNode<ElementType>>();
    if (nodes.size() > 0)
    {
        auto predictor = nodes[0]->GetPredictor();
        predictor.RemoveLastLayers(numLayersFromEnd);
        model::Model model;
        model::MemoryShape inputShape(predictor.GetInputShape());
        auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputShape);
        const auto& neuralNetwork = nodes::NeuralNetwork(inputNode->output, predictor);

        map = model::Map(model, { { "input", inputNode } }, { { "output", neuralNetwork } });
        found = true;
    }

    return found;
}

void PrintModel(model::Map& map, size_t refineIterations)
{
    map.Refine(refineIterations);
    map.GetModel().Print(std::cout);
}

bool RedirectModelOutputByPortElements(model::Map& map, const std::string& targetPortElements, size_t refineIterations)
{
    // Refine the model
    map.Refine(refineIterations);

    try
    {
        // Create a port elements from the target port output
        auto elementsProxy = model::ParsePortElementsProxy(targetPortElements);
        auto originalPortElements = model::ProxyToPortElements(map.GetModel(), elementsProxy);

        // Create a copy of the refined model, setting the
        // input to be the original input node and the output to be from the target
        // port elements.
        model::TransformContext context;
        model::ModelTransformer transformer;
        auto model = map.GetModel().DeepCopy();
        auto input = transformer.GetCorrespondingInputNode(map.GetInput());
        const auto& output = transformer.GetCorrespondingOutputs(*originalPortElements.GetRanges()[0].ReferencedPort());

        map = model::Map(model, { { "input", input } }, { { "output", output } });
    }
    catch (const utilities::Exception& exception)
    {
        std::cerr << "Couldn't redirect model output from " << targetPortElements << ", error :" << exception.GetMessage() << std::endl;
        return false;
    }

    return true;
}

void PrintSDCAPredictorInfoHeader(std::ostream& os)
{
    os << "\tPrimal Objective";
    os << "\tDual Objective";
    os << "\tDuality gap";
    os << std::endl;
}

void PrintSDCAPredictorInfoValues(const trainers::SDCAPredictorInfo& info, std::ostream& os)
{
    auto originalPrecision = os.precision(6);
    auto originalFlags = os.setf(std::ios::fixed);

    os << "\t" << info.primalObjective;
    os << "\t\t" << info.dualObjective;
    os << "\t" << std::abs(info.primalObjective - info.dualObjective) << std::endl;

    os.setf(originalFlags);
    os.precision(originalPrecision);
}

void PrintEvaluation(double dualityGap, double desiredPrecision, evaluators::IEvaluator<PredictorType>* evaluator, std::ostream& os)
{
    // Print evaluation of training
    os << "Final duality Gap: " << dualityGap << std::endl
       << std::endl;
    evaluator->Print(os);
    os << std::endl
       << std::endl;
    if (dualityGap < desiredPrecision)
    {
        os << "Training completed successfully." << std::endl;
    }
    else
    {
        os << "Warning: After maxEpochs, training has not converged to desired duality gap." << std::endl;
    }
}

template <typename LossFunctionType>
PredictorType RetargetModelUsingLinearPredictor(ParsedRetargetArguments& retargetArguments, data::AutoSupervisedDataset& dataset)
{
    trainers::SDCATrainerParameters trainerParameters{ retargetArguments.regularization, retargetArguments.desiredPrecision, retargetArguments.maxEpochs, retargetArguments.permute, retargetArguments.randomSeedString };

    auto trainer = trainers::SDCATrainer<LossFunctionType, functions::L2Regularizer>(LossFunctionType(), functions::L2Regularizer(), trainerParameters);
    if (retargetArguments.verbose) std::cout << "Created linear trainer ..." << std::endl;

    // create an evaluator
    evaluators::EvaluatorParameters evaluatorParameters{ 1, true };
    auto evaluator = common::MakeEvaluator<PredictorType>(dataset.GetAnyDataset(), evaluatorParameters, retargetArguments.lossFunctionArguments);

    // Train the predictor
    std::cout << "Training ..." << std::endl;
    trainer.SetDataset(dataset.GetAnyDataset());
    size_t epoch = 0;
    double dualityGap = std::numeric_limits<double>::max();

    if (retargetArguments.verbose) PrintSDCAPredictorInfoHeader(std::cout);
    while ((++epoch <= retargetArguments.maxEpochs) && (dualityGap > retargetArguments.desiredPrecision))
    {
        trainer.Update();
        auto info = trainer.GetPredictorInfo();
        dualityGap = std::abs(info.primalObjective - info.dualObjective);
        if (retargetArguments.verbose) PrintSDCAPredictorInfoValues(trainer.GetPredictorInfo(), std::cout);
    }

    // Print evaluation of training
    evaluator->Evaluate(trainer.GetPredictor());
    PrintEvaluation(dualityGap, retargetArguments.desiredPrecision, evaluator.get(), std::cout);

    return PredictorType(trainer.GetPredictor());
}

PredictorType RetargetModelUsingLinearPredictor(ParsedRetargetArguments& retargetArguments, data::AutoSupervisedDataset& dataset)
{
    using LossFunctionEnum = common::LossFunctionArguments::LossFunction;
    PredictorType trainedPredictor;
    switch (retargetArguments.lossFunctionArguments.lossFunction)
    {
    case LossFunctionEnum::squared:
        trainedPredictor = RetargetModelUsingLinearPredictor<functions::SquaredLoss>(retargetArguments, dataset);
        break;

    case LossFunctionEnum::log:
        trainedPredictor = RetargetModelUsingLinearPredictor<functions::LogLoss>(retargetArguments, dataset);
        break;

    case LossFunctionEnum::smoothHinge:
        trainedPredictor = RetargetModelUsingLinearPredictor<functions::SmoothHingeLoss>(retargetArguments, dataset);
        break;

    default:
        throw utilities::CommandLineParserErrorException("chosen loss function is not supported by this trainer");
    }
    return trainedPredictor;
}

std::vector<data::AutoSupervisedDataset> CreateDatasetsForOneVersusRest(data::AutoSupervisedMultiClassDataset& multiclassDataset)
{
    std::vector<data::AutoSupervisedDataset> datasets;

    // Get statistics for the multi-class dataset
    std::map<size_t, size_t> classCounts;
    for (size_t i = 0; i < multiclassDataset.NumExamples(); ++i)
    {
        const auto& example = multiclassDataset.GetExample(i);
        size_t classIndex = example.GetMetadata().classIndex;

        classCounts[classIndex] += 1;
    }

    // Create binary classification datasets for each class
    size_t numClasses = classCounts.size();
    size_t totalCount = multiclassDataset.NumExamples();
    datasets.resize(numClasses);
    for (size_t i = 0; i < numClasses; ++i)
    {
        // For any class x, create a binary classification dataset where Example is:
        //  weight = 1 / number of examples for x, or 1 / number of examples for all classes not x
        //  label = 1.0 for examples in x, or -1.0 for all examples not in x
        //  data = shared_ptr to existing data
        size_t positiveCount = classCounts[i];
        size_t negativeCount = (totalCount - classCounts[i]);
        double weightPositiveCase = 1.0 / (positiveCount ? positiveCount : 1.0);
        double weightNegativeCase = 1.0 / (negativeCount ? negativeCount : 1.0);
        datasets[i] = multiclassDataset.Transform<data::AutoSupervisedExample>([i, weightPositiveCase, weightNegativeCase](const auto& example) {
            if (example.GetMetadata().classIndex == i)
            {
                // Positive case
                return data::AutoSupervisedExample(example.GetSharedDataVector(), data::WeightLabel{ weightPositiveCase, 1.0 });
            }
            else
            {
                // Negative case
                return data::AutoSupervisedExample(example.GetSharedDataVector(), data::WeightLabel{ weightNegativeCase, -1.0 });
            }
        });
    }

    return datasets;
}

template <typename ElementType>
model::Map GetMultiClassMapFromBinaryPredictors(std::vector<PredictorType>& binaryPredictors, model::Map& map)
{
    if (binaryPredictors.empty())
    {
        throw std::invalid_argument("binaryPredictors vector has no elements");
    }
    auto inputDimension = map.GetOutput(0).Size();
    for (auto& binaryPredictor : binaryPredictors)
    {
        binaryPredictor.Resize(inputDimension);
    }

    math::RowMatrix<ElementType> weights(binaryPredictors.size(), inputDimension);
    math::ColumnVector<ElementType> bias(binaryPredictors.size());

    // Set the weights matrix and bias from the predictors.
    // Each row in the weights is the learned weights from that predictor.
    // Each element in the bias is the learned bias from that predictor.
    for (size_t i = 0; i < binaryPredictors.size(); ++i)
    {
        auto predictorWeights = binaryPredictors[i].GetWeights();
        for (size_t j = 0; j < inputDimension; ++j)
        {
            weights(i, j) = predictorWeights[j];
        }
        bias[i] = binaryPredictors[i].GetBias();
    }

    model::Model& model = map.GetModel();
    const auto& mapOutput = map.GetOutput<ElementType>(0);
    const auto& predictorOutput = nodes::MatrixVectorProduct(mapOutput, weights);
    const auto& biasValues = nodes::Constant(model, bias.ToArray());
    const auto& biasedOutput = nodes::Add(predictorOutput, biasValues);

    // Apply a sigmoid function so that output can be treated as a probability or
    // confidence score.
    const auto& sigmoid = nodes::BroadcastUnaryFunction<nodes::SigmoidActivationFunction<ElementType>>(biasedOutput);
    const auto& sink = nodes::Sink(sigmoid);
    auto& output = model::Output(sink);
    auto outputMap = model::Map(model, { { "input", map.GetInput() } }, { { "output", output } });

    return outputMap;
}

model::Map GetRetargetedModel(std::vector<PredictorType>& binaryPredictors, model::Map& map)
{
    model::Map result;
    // Create a new map with the output of the combined linear predictors appended.
    switch (map.GetOutputType())
    {
    case model::Port::PortType::smallReal: {
        result = GetMultiClassMapFromBinaryPredictors<float>(binaryPredictors, map);
        break;
    }
    case model::Port::PortType::real: {
        result = GetMultiClassMapFromBinaryPredictors<double>(binaryPredictors, map);
        break;
    }
    default:
        throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Unexpected output type for model. Should be double or float.");
        break;
    };
    return result;
}

model::Map GetRetargetedModel(const PredictorType& trainedPredictor, model::Map& map)
{
    model::Map result;
    auto mappedDatasetDimension = map.GetOutput(0).Size();
    // Create a new map with the output of the linear predictor appended.
    switch (map.GetOutputType())
    {
    case model::Port::PortType::smallReal: {
        result = AppendTrainedLinearPredictorToMap<float>(trainedPredictor, map, mappedDatasetDimension);
        break;
    }
    case model::Port::PortType::real: {
        result = AppendTrainedLinearPredictorToMap<double>(trainedPredictor, map, mappedDatasetDimension);
        break;
    }
    default:
        throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Unexpected output type for model. Should be double or float.");
        break;
    };

    return model::Map();
}

int main(int argc, char* argv[])
{
    utilities::MillisecondTimer _overallTimer;
    utilities::MillisecondTimer _timer;
    try
    {
        _overallTimer.Start();
        // create a command line parser
        utilities::CommandLineParser commandLineParser(argc, argv);

        // add arguments to the command line parser
        ParsedRetargetArguments retargetArguments;
        commandLineParser.AddOptionSet(retargetArguments);

        // parse command line
        commandLineParser.Parse();
        if (retargetArguments.verbose) std::cout << commandLineParser.GetCurrentValuesString() << std::endl;

        _timer.Start();
        // load map
        if (retargetArguments.verbose) std::cout << "Loading model from " << retargetArguments.inputModelFilename;
        auto map = common::LoadMap(retargetArguments.inputModelFilename);
        if (retargetArguments.verbose) std::cout << "(" << _timer.Elapsed() << " ms)" << std::endl;

        if (retargetArguments.print)
        {
            PrintModel(map, retargetArguments.refineIterations);
            return 0;
        }

        // Create a map by redirecting a layer or node to be output
        bool redirected = false;
        if (retargetArguments.removeLastLayers > 0)
        {
            if (map.GetOutputType() == model::Port::PortType::smallReal)
            {
                redirected = RedirectNeuralNetworkOutputByLayer<float>(map, retargetArguments.removeLastLayers);
            }
            else
            {
                redirected = RedirectNeuralNetworkOutputByLayer<double>(map, retargetArguments.removeLastLayers);
            }
            std::cout << "Removed last " << retargetArguments.removeLastLayers << " layers from neural network" << std::endl;
        }
        else if (retargetArguments.targetPortElements.length() > 0)
        {
            redirected = RedirectModelOutputByPortElements(map, retargetArguments.targetPortElements, retargetArguments.refineIterations);
            std::cout << "Redirected output for port elements " << retargetArguments.targetPortElements << " from model" << std::endl;
        }
        else
        {
            std::cerr << "Error: Expected valid arguments for either --removeLastLayers or --targetPortElements" << std::endl;
            return 1;
        }

        if (!redirected)
        {
            std::cerr << "Could not splice model, exiting" << std::endl;
            return 1;
        }

        auto node = map.GetOutput(0).GetNode();
        std::cout << "Using output from node of type " << node->GetRuntimeTypeName() << std::endl;

        // load dataset and map the output
        if (retargetArguments.verbose) std::cout << "Loading data ...";
        model::Map result;
        if (retargetArguments.multiClass)
        {
            // This is a multi-class dataset
            _timer.Start();
            auto stream = utilities::OpenIfstream(retargetArguments.inputDataFilename);
            auto multiclassDataset = common::GetMultiClassDataset(stream);
            if (retargetArguments.verbose) std::cout << "(" << _timer.Elapsed() << " ms)" << std::endl;

            // Obtain a new training dataset for the set of Linear Predictors by running the
            // multiclassDataset through the modified model
            if (retargetArguments.verbose) std::cout << std::endl
                                                     << "Transforming dataset with compiled model...";
            _timer.Start();

            auto dataset = common::TransformDatasetWithCompiledMap(multiclassDataset, map, retargetArguments.useBlas);
            if (retargetArguments.verbose) std::cout << "(" << _timer.Elapsed() << " ms)" << std::endl;

            // Create binary classification datasets for each one versus rest (OVR) case
            if (retargetArguments.verbose) std::cout << std::endl
                                                     << "Creating datasets for One vs Rest...";
            _timer.Start();
            auto datasets = CreateDatasetsForOneVersusRest(dataset);
            if (retargetArguments.verbose) std::cout << "(" << _timer.Elapsed() << " ms)" << std::endl;

            // Next, train a binary classifier for each case and combine into a
            // single model.
            _timer.Start();
            std::vector<PredictorType> predictors(datasets.size());
            for (size_t i = 0; i < datasets.size(); ++i)
            {
                std::cout << std::endl
                          << "=== Training binary classifier for class " << i << " vs Rest ===" << std::endl;

                predictors[i] = RetargetModelUsingLinearPredictor(retargetArguments, datasets[i]);
            }
            if (retargetArguments.verbose) std::cout << "Training completed ...(" << _timer.Elapsed() << " ms)" << std::endl;

            // Save the newly spliced model
            result = GetRetargetedModel(predictors, map);
        }
        else
        {
            // This is a binary classification dataset
            _timer.Start();
            auto stream = utilities::OpenIfstream(retargetArguments.inputDataFilename);
            auto binaryDataset = common::GetDataset(stream);
            if (retargetArguments.verbose) std::cout << "Loading dataset took :" << _timer.Elapsed() << " ms" << std::endl;
            // Obtain a new training dataset for the Linear Predictor by running the
            // binaryDataset through the modified model
            if (retargetArguments.verbose) std::cout << std::endl
                                                     << "Transforming dataset with compiled model...";
            _timer.Start();

            auto dataset = common::TransformDatasetWithCompiledMap(binaryDataset, map);
            if (retargetArguments.verbose) std::cout << "(" << _timer.Elapsed() << " ms)" << std::endl;

            // Train a linear predictor whose input comes from the previous model
            _timer.Start();
            auto predictor = RetargetModelUsingLinearPredictor(retargetArguments, dataset);
            if (retargetArguments.verbose) std::cout << "Training completed... (" << _timer.Elapsed() << " ms)" << std::endl;

            // Save the newly spliced model
            result = GetRetargetedModel(predictor, map);
        }
        common::SaveMap(result, retargetArguments.outputModelFilename);
        if (retargetArguments.verbose) std::cout << std::endl
                                                 << "RetargetTrainer completed... (" << _overallTimer.Elapsed() << " ms)" << std::endl;
        std::cout << std::endl
                  << "New model saved as " << retargetArguments.outputModelFilename << std::endl;
    }
    catch (const utilities::CommandLineParserPrintHelpException& exception)
    {
        std::cout << exception.GetHelpText() << std::endl;
        return 0;
    }
    catch (const utilities::CommandLineParserErrorException& exception)
    {
        std::cerr << "Command line parse error:" << std::endl;
        for (const auto& error : exception.GetParseErrors())
        {
            std::cerr << error.GetMessage() << std::endl;
        }
        return 1;
    }
    catch (const utilities::Exception& exception)
    {
        std::cerr << "exception: " << exception.GetMessage() << std::endl;
        return 1;
    }
    return 0;
}
