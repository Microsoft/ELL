////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     main.cpp (linearTrainer)
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "LinearTrainerArguments.h"

#include <utilities/include/CommandLineParser.h>
#include <utilities/include/Exception.h>
#include <utilities/include/Files.h>
#include <utilities/include/OutputStreamImpostor.h>

#include <data/include/Dataset.h>

#include <common/include/DataLoadArguments.h>
#include <common/include/DataLoaders.h>
#include <common/include/EvaluatorArguments.h>
#include <common/include/LoadModel.h>
#include <common/include/MakeEvaluator.h>
#include <common/include/MakeTrainer.h>
#include <common/include/MapLoadArguments.h>
#include <common/include/ModelSaveArguments.h>
#include <common/include/TrainerArguments.h>

#include <model/include/Map.h>
#include <model/include/Model.h>
#include <model/include/OutputNode.h>

#include <nodes/include/LinearPredictorNode.h>

#include <trainers/include/MeanCalculator.h>

#include <evaluators/include/Evaluator.h>

#include <predictors/include/Normalizer.h>

#include <iostream>
#include <memory>

using namespace ell;

template <typename ElementType>
model::Map AppendTrainedLinearPredictorToMap(const predictors::LinearPredictor<ElementType>& trainedPredictor, model::Map& map, size_t dimension)
{
    predictors::LinearPredictor<ElementType> predictor(trainedPredictor);
    predictor.Resize(dimension);

    const auto& mapOutput = map.GetOutput<ElementType>(0);
    const auto& predictorOutput = nodes::LinearPredictor(mapOutput, predictor);
    const auto& output = model::Output(predictorOutput);
    auto outputMap = model::Map(map.GetModel(), { { "input", map.GetInput() } }, { { "output", output } });

    return outputMap;
}

int main(int argc, char* argv[])
{
    try
    {
        // create a command line parser
        utilities::CommandLineParser commandLineParser(argc, argv);

        // add arguments to the command line parser
        ParsedLinearTrainerArguments linearTrainerArguments;
        common::ParsedDataLoadArguments dataLoadArguments;
        common::ParsedMapLoadArguments mapLoadArguments;
        common::ParsedModelSaveArguments modelSaveArguments;
        common::ParsedTrainerArguments trainerArguments;
        common::ParsedEvaluatorArguments evaluatorArguments;

        commandLineParser.AddOptionSet(linearTrainerArguments);
        commandLineParser.AddOptionSet(dataLoadArguments);
        commandLineParser.AddOptionSet(mapLoadArguments);
        commandLineParser.AddOptionSet(modelSaveArguments);
        commandLineParser.AddOptionSet(trainerArguments);
        commandLineParser.AddOptionSet(evaluatorArguments);

        // parse command line
        commandLineParser.Parse();

        if (trainerArguments.verbose)
        {
            std::cout << "Linear Trainer" << std::endl;
            std::cout << commandLineParser.GetCurrentValuesString() << std::endl;
        }

        if (dataLoadArguments.inputDataFilename.empty())
        {
            throw utilities::CommandLineParserPrintHelpException(commandLineParser.GetHelpString());
        }

        // load map
        mapLoadArguments.defaultInputSize = dataLoadArguments.parsedDataDimension;
        model::Map map;

        if (mapLoadArguments.HasInputFilename())
        {
            map = common::LoadMap(mapLoadArguments);
        }
        else
        {
            model::Model model;
            auto input = model.AddNode<model::InputNode<float>>(dataLoadArguments.parsedDataDimension);
            const auto& output = model::Output(input->output);
            map = model::Map(model, { { "input", input } }, { { "output", output } });
        }

        // load dataset
        if (trainerArguments.verbose) std::cout << "Loading data ..." << std::endl;
        auto stream = utilities::OpenIfstream(dataLoadArguments.inputDataFilename);
        auto parsedDataset = common::GetDataset(stream);
        auto mappedDataset = common::TransformDataset(parsedDataset, map);
        auto mappedDatasetDimension = map.GetOutput(0).Size();

        // normalize data
        if (linearTrainerArguments.normalize)
        {
            if (trainerArguments.verbose) std::cout << "Sparisty-preserving data normalization ..." << std::endl;

            // find inverse absolute mean
            auto scaleVector = trainers::CalculateSparseTransformedMean(mappedDataset.GetAnyDataset(), [](data::IndexValue x) { return std::abs(x.value); });
            scaleVector.Transform([](double x) { return x > 0.0 ? 1.0 / x : 0.0; });

            // create normalizer
            auto coordinateTransformation = [&](data::IndexValue x) { return x.value * scaleVector[x.index]; };
            auto normalizer = predictors::MakeTransformationNormalizer<data::IterationPolicy::skipZeros>(coordinateTransformation);

            // apply normalizer to data
            auto normalizedDataset = common::TransformDataset(mappedDataset, normalizer);

            mappedDataset.Swap(normalizedDataset);
        }

        // predictor type
        using PredictorType = predictors::LinearPredictor<double>;

        // create linear trainer
        std::unique_ptr<trainers::ITrainer<PredictorType>> trainer;
        switch (linearTrainerArguments.algorithm)
        {
        case LinearTrainerArguments::Algorithm::SGD:
            trainer = common::MakeSGDTrainer(trainerArguments.lossFunctionArguments, { linearTrainerArguments.regularization, linearTrainerArguments.randomSeedString });
            break;
        case LinearTrainerArguments::Algorithm::SparseDataSGD:
            trainer = common::MakeSparseDataSGDTrainer(trainerArguments.lossFunctionArguments, { linearTrainerArguments.regularization, linearTrainerArguments.randomSeedString });
            break;
        case LinearTrainerArguments::Algorithm::SparseDataCenteredSGD: {
            auto mean = trainers::CalculateMean(mappedDataset.GetAnyDataset());
            trainer = common::MakeSparseDataCenteredSGDTrainer(trainerArguments.lossFunctionArguments, mean, { linearTrainerArguments.regularization, linearTrainerArguments.randomSeedString });
            break;
        }
        case LinearTrainerArguments::Algorithm::SDCA: {
            trainer = common::MakeSDCATrainer(trainerArguments.lossFunctionArguments, { linearTrainerArguments.regularization, linearTrainerArguments.desiredPrecision, linearTrainerArguments.maxEpochs, linearTrainerArguments.permute, linearTrainerArguments.randomSeedString });
            break;
        }
        default:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "unrecognized algorithm type");
        }

        // create an evaluator
        auto evaluator = common::MakeEvaluator<PredictorType>(mappedDataset.GetAnyDataset(), evaluatorArguments, trainerArguments.lossFunctionArguments);

        // Train the predictor
        if (trainerArguments.verbose) std::cout << "Training ..." << std::endl;
        trainer->SetDataset(mappedDataset.GetAnyDataset());

        for (size_t epoch = 0; epoch < trainerArguments.numEpochs; ++epoch)
        {
            trainer->Update();
            evaluator->Evaluate(trainer->GetPredictor());
        }

        // Print loss and errors
        if (trainerArguments.verbose)
        {
            std::cout << "Finished training.\n";

            // print evaluation
            std::cout << "Training error\n";
            evaluator->Print(std::cout);
            std::cout << std::endl;
        }

        // Save predictor model
        if (modelSaveArguments.outputModelFilename != "")
        {
            // Create a new map with the linear predictor appended.
            switch (map.GetOutputType())
            {
            case model::Port::PortType::smallReal: {
                auto outputMap = AppendTrainedLinearPredictorToMap<float>(trainer->GetPredictor(), map, mappedDatasetDimension);
                common::SaveMap(outputMap, modelSaveArguments.outputModelFilename);
            }
            break;
            case model::Port::PortType::real: {
                auto outputMap = AppendTrainedLinearPredictorToMap<double>(trainer->GetPredictor(), map, mappedDatasetDimension);
                common::SaveMap(outputMap, modelSaveArguments.outputModelFilename);
            }
            break;
            default:
                std::cerr << "Unexpected output type for model. Should be double or float." << std::endl;
                break;
            };
        }
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
