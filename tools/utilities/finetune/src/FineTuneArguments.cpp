////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     FineTuneArguments.cpp (finetune)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FineTuneArguments.h"
#include "ModelUtils.h"

#include <common/include/LoadModel.h>

#include <utilities/include/Exception.h>

#include <iostream>

namespace ell
{

TargetNodeFlags operator|(TargetNodeType t1, TargetNodeType t2)
{
    return { static_cast<unsigned int>(t1) | static_cast<unsigned int>(t2) };
}

TargetNodeFlags operator|(TargetNodeFlags t1, TargetNodeType t2)
{
    return { t1.flags | static_cast<unsigned int>(t2) };
}

TargetNodeFlags operator|(TargetNodeType t1, TargetNodeFlags t2)
{
    return { static_cast<unsigned int>(t1) | t2.flags };
}

TargetNodeFlags operator|(TargetNodeFlags t1, TargetNodeFlags t2)
{
    return { t1.flags | t2.flags };
}

bool operator&(TargetNodeType t1, TargetNodeType t2)
{
    return t1 == t2;
}

bool operator&(TargetNodeFlags t1, TargetNodeType t2)
{
    return t1.flags & static_cast<unsigned int>(t2);
}

bool operator&(TargetNodeType t1, TargetNodeFlags t2)
{
    return static_cast<unsigned int>(t1) & t2.flags;
}

TargetNodeFlags operator&(TargetNodeFlags t1, TargetNodeFlags t2)
{
    return { t1.flags & t2.flags };
}

FineTuneOptimizationParameters FineTuneArguments::GetFineTuneParameters() const
{
    optimization::SDCAOptimizerParameters sdcaParams;
    sdcaParams.regularizationParameter = l2Regularization;
    sdcaParams.permuteData = permute;

    FineTuneOptimizationParameters params;
    params.optimizerParameters = sdcaParams;
    params.lossFunction = lossFunction;
    params.l1Regularization.reset();
    params.maxEpochs = maxEpochs;
    params.desiredPrecision = desiredPrecision;
    params.requiredPrecision = requiredPrecision;
    params.optimizeFiltersIndependently = optimizeFiltersIndependently;
    params.randomSeed = randomSeed;
    return params;
}

FineTuneOptimizationParameters FineTuneArguments::GetSparsifyParameters() const
{
    auto params = GetFineTuneParameters();
    params.l1Regularization = l1Regularization;
    params.sparsifyMethod = sparsifyMethod;
    params.sparsityTarget = sparsityTarget;
    params.sparsityTargetEpsilon = sparsityTargetEpsilon;
    params.sparsifyMethod = sparsifyMethod;
    params.reoptimizeSparseWeights = reoptimizeSparseWeights;
    return params;
}

FineTuneOptimizationParameters FineTuneArguments::GetReoptimizeParameters() const
{
    auto params = GetFineTuneParameters();
    params.l1Regularization = 0;
    params.reoptimizeSparseWeights = reoptimizeSparseWeights;
    return params;
}

FineTuneProblemParameters FineTuneArguments::GetFineTuneProblemParameters() const
{
    FineTuneProblemParameters params;
    params.normalizeInputs = normalizeInputs;
    params.normalizeOutputs = normalizeOutputs;
    params.fineTuneParameters = GetFineTuneParameters();
    params.sparsifyParameters = GetSparsifyParameters();
    params.reoptimizeParameters = GetReoptimizeParameters();
    return params;
};

const ell::model::OutputPortBase& FineTuneArguments::GetInputModelTargetOutput() const
{
    auto model = LoadInputModel();
    const auto& output = GetSpecifiedOutput(model);
    const auto& newOutput = RemoveSourceAndSinkNodes(output);

    ComputeValidNodes(model, newOutput);
    return newOutput;
}

void FineTuneArguments::ComputeValidNodes(ell::model::Model& model, const ell::model::OutputPortBase& output) const
{
    // get list of all nodes
    std::vector<std::string> allValidNodes;
    model.VisitSubmodel(&output, [&](const ell::model::Node& node) {
        if (((fineTuneFullyConnectedNodes && IsFullyConnectedLayerNode(&node)) ||
             (fineTuneConvolutionalNodes && IsConvolutionalLayerNode(&node))))
        {
            allValidNodes.push_back(node.GetId().ToString());
        }
    });

    // skip from beginning
    allValidNodes.erase(allValidNodes.begin(), allValidNodes.begin() + numPrefixNodesToSkip);

    // skip from end
    allValidNodes.resize(std::max(size_t(0), allValidNodes.size() - numSuffixNodesToSkip));

    // skip specific nodes
    for (auto id : specificNodesToSkip)
    {
        auto it = std::find(allValidNodes.begin(), allValidNodes.end(), id);
        if (it != allValidNodes.end())
        {
            allValidNodes.erase(it);
        }
    }
    validNodes = allValidNodes;
}

bool FineTuneArguments::SkipNode(std::string id) const
{
    // return "true" (i.e., skip this node) if it's not in the list of valid nodes
    return (std::find(validNodes.begin(), validNodes.end(), id) == validNodes.end());
}

const ell::model::OutputPortBase& FineTuneArguments::GetSpecifiedOutput(ell::model::Model& model) const
{
    if (targetPortElements.empty())
    {
        auto outputNode = GetOutputNode(model);
        return outputNode->GetOutputPort();
    }

    auto targetElementsProxy = model::ParsePortElementsProxy(targetPortElements);
    auto targetElements = model::ProxyToPortElements(model, targetElementsProxy);
    if (targetElements.IsFullPortOutput())
    {
        return *(targetElements.GetRanges()[0].ReferencedPort());
    }
    else
    {
        throw utilities::InputException(utilities::InputExceptionErrors::badData, "Only full port outputs supported");
    }
}

ell::model::Model FineTuneArguments::LoadInputModel() const
{
    auto map = common::LoadMap(mapLoadArguments);
    return map.GetModel().ShallowCopy();
}

utilities::OutputStreamImpostor FineTuneArguments::GetReportStream() const
{
    if (reportFilename.empty())
    {
        return utilities::OutputStreamImpostor(std::cout);
    }
    else
    {
        return utilities::OutputStreamImpostor(reportFilename);
    }
}

void InitArguments(utilities::CommandLineParser& parser, FineTuneArguments& args)
{
    using namespace common;

    parser.AddDocumentationString("Model input / output");
    parser.AddOptionSet(args.mapLoadArguments);
    parser.AddOption(args.targetPortElements,
                     "targetPortElements",
                     "tpe",
                     "The port elements of the pre-trained model to use as input to the subsequent linear predictor "
                     "e.g. \"1115.output\" to use the full output from Node 1115",
                     "");
    parser.AddOptionSet(args.mapSaveArguments);

    parser.AddOption(args.printModel, "print", "", "Print the model", false);

    parser.AddDocumentationString("");
    parser.AddDocumentationString("Train / test datasets");
    parser.AddOptionSet(args.trainDataArguments);
    parser.AddOptionSet(args.testDataArguments);

    parser.AddOption(args.maxTrainingRows,
                     "maxTrainingRows",
                     "",
                     "Maximum number of rows from dataset to use for training (0 = 'all')",
                     0);

    parser.AddOption(args.maxTestingRows,
                     "maxTestingRows",
                     "",
                     "Maximum number of rows from dataset to use for testing (0 = 'all')",
                     0);

    parser.AddOption(args.multiClass,
                     "multiClass",
                     "mc",
                     "Indicates whether the input dataset is multi-class or binary.",
                     true);

    parser.AddOption(args.dataFormat, "format", "", "Dataset format (GSDF, CIFAR, MNIST; default: guess)", "");

    parser.AddDocumentationString("");
    parser.AddDocumentationString("Node selection");
    parser.AddOption(args.numPrefixNodesToSkip, "skipStart", "", "Number of nodes in the beginning of the model to skip", 0);

    parser.AddOption(args.numSuffixNodesToSkip, "skipEnd", "", "Number of nodes at the end of the model to skip", 0);

    parser.AddOption(args.specificNodesToSkip, "skipNode", "", "A specific node to skip (can be specified multiple times)", std::vector<std::string>{});

    parser.AddOption(args.fineTuneFullyConnectedNodes, "dense", "", "Fine-tune dense (fully-connected) layers", true);

    parser.AddOption(args.fineTuneConvolutionalNodes, "conv", "", "Fine-tune convolutional layers", true);

    parser.AddDocumentationString("");
    parser.AddDocumentationString("Optimization parameters");
    parser.AddOption(args.l2Regularization, "l2regularization", "l2", "The L2 regularization parameter", 0.005);

    parser.AddOption(args.l1Regularization, "l1regularization", "l1", "The L1 regularization parameter", 0);

    parser.AddOption(args.desiredPrecision,
                     "desiredPrecision",
                     "",
                     "The desired duality gap at which to stop optimizing",
                     1.0e-4);

    parser.AddOption(args.requiredPrecision,
                     "requiredPrecision",
                     "",
                     "The duality gap at which we give up and use the unoptimized result (0 means \"don't give up\")",
                     0);

    parser.AddOption(args.maxEpochs, "maxEpochs", "e", "The maximum number of optimization epochs to run", 25);

    parser.AddOption(
        args.lossFunction,
        "lossFunction",
        "lf",
        "Choice of loss function",
        { { "squared", LossFunction::square }, { "hinge", LossFunction::hinge }, { "log", LossFunction::logistic }, { "smoothHinge", LossFunction::smoothedHinge }, { "huber", LossFunction::huber } },
        "squared");

    parser.AddOption(args.normalizeInputs, "normalizeInputs", "", "Normalize inputs", false);

    parser.AddOption(args.normalizeOutputs, "normalizeOutputs", "", "Normalize outputs", false);

    parser.AddOption(args.optimizeFiltersIndependently, "optimizePerFilter", "", "Re-optimize filters independently", false);

    parser.AddOption(args.reoptimizeSparseWeights, "reoptimizeSparseWeights", "", "Re-optimize sparsified weights", false);

    parser.AddOption(args.permute,
                     "permute",
                     "",
                     "Whether or not to randomly permute the training data before each epoch",
                     true);

    parser.AddDocumentationString("");
    parser.AddDocumentationString("Sparsification parameters");
    parser.AddOption(args.fineTuneTargets,
                     "finetune",
                     "",
                     "The types of nodes to fine-tune.",
                     { { "none", TargetNodeType::none },
                       { "pointwise", TargetNodeType::pointwiseConvolution },
                       { "depthwise", TargetNodeType::spatialConvolution },
                       { "full", TargetNodeType::fullConvolution },
                       { "dense", TargetNodeType::fullyConnected },
                       { "pointwise+depthwise", TargetNodeType::pointwiseConvolution | TargetNodeType::spatialConvolution },
                       { "depthwise+full", TargetNodeType::spatialConvolution | TargetNodeType::fullConvolution },
                       { "pointwise+full", TargetNodeType::pointwiseConvolution | TargetNodeType::fullConvolution },
                       { "pointwise+depthwise+full", TargetNodeType::pointwiseConvolution | TargetNodeType::spatialConvolution | TargetNodeType::fullConvolution },
                       { "pointwise+full+dense", TargetNodeType::pointwiseConvolution | TargetNodeType::fullConvolution | TargetNodeType::fullyConnected },
                       { "all", TargetNodeType::fullConvolution | TargetNodeType::spatialConvolution | TargetNodeType::pointwiseConvolution | TargetNodeType::fullyConnected } },
                     "pointwise+full+dense");

    parser.AddOption(args.sparsifyTargets,
                     "sparsify",
                     "",
                     "The types of nodes to sparsify via L1 regularization.",
                     { { "none", TargetNodeType::none },
                       { "pointwise", TargetNodeType::pointwiseConvolution },
                       { "depthwise", TargetNodeType::spatialConvolution },
                       { "full", TargetNodeType::fullConvolution },
                       { "pointwise+depthwise", TargetNodeType::pointwiseConvolution | TargetNodeType::spatialConvolution },
                       { "depthwise+full", TargetNodeType::spatialConvolution | TargetNodeType::fullConvolution },
                       { "pointwise+full", TargetNodeType::pointwiseConvolution | TargetNodeType::fullConvolution },
                       { "pointwise+depthwise+full", TargetNodeType::pointwiseConvolution | TargetNodeType::spatialConvolution | TargetNodeType::fullConvolution },
                       { "dense", TargetNodeType::fullyConnected },
                       { "pointwise+full+dense", TargetNodeType::pointwiseConvolution | TargetNodeType::fullConvolution | TargetNodeType::fullyConnected },
                       { "all", TargetNodeType::fullConvolution | TargetNodeType::spatialConvolution | TargetNodeType::pointwiseConvolution | TargetNodeType::fullyConnected } },
                     "pointwise");

    parser.AddOption(args.sparsityTarget, "sparsity", "", "The target sparsity level to aim for when sparisfying layers via L1 regularization. When this value is zero, the l1Regularization parameter is used, when this value is nonzero, the l1Regularization parameter is ignored.", 0);

    parser.AddOption(args.sparsityTargetEpsilon, "sparsityPrecision", "", "The amount by which the output sparsity level is allowed to deviate from the desired sparsity level.", 0.01);

    parser.AddOption(args.sparsifyMethod, "sparsifyMethod", "", "The method to use for sparsifying weights", { { "l1", SparsifyMethod::l1 }, { "threshold", SparsifyMethod::threshold }, { "random", SparsifyMethod::random } }, "l1");

    parser.AddDocumentationString("");
    parser.AddDocumentationString("Misc parameters");
    parser.AddOption(args.randomSeed, "randomSeed", "seed", "The random seed string", "ABCDEFG");

    parser.AddOption(args.reportFilename,
                     "reportFilename",
                     "",
                     "Output filename for report (empty for standard output)",
                     "");

    parser.AddOption(args.writeOutputStats, "writeOutputStats", "", "Write output stats", false);

    parser.AddOption(args.testOnly, "testOnly", "", "Report accuracy of model and exit", false);

    parser.AddOption(args.compile, "compile", "", "Compile the model when evaluating", true);

    parser.AddOption(args.verbose, "verbose", "v", "Turn on verbose mode", false);
}

void ParsedFineTuneArguments::AddArgs(utilities::CommandLineParser& parser)
{
    InitArguments(parser, *this);
}

ParsedFineTuneArguments ParsedFineTuneArguments::ParseCommandLine(int argc, char* argv[])
{
    try
    {
        utilities::CommandLineParser commandLineParser(argc, argv);
        ParsedFineTuneArguments args;
        commandLineParser.AddOptionSet(args);
        commandLineParser.Parse();

        // if no input specified, print help and exit
        if (!args.mapLoadArguments.HasInputFilename())
        {
            if (commandLineParser.GetPositionalArgs().size() == 1)
            {
                args.mapLoadArguments.inputMapFilename = commandLineParser.GetPositionalArgs()[0];
            }
            else
            {
                throw utilities::CommandLineParserPrintHelpException(commandLineParser.GetHelpString());
            }
        }

        if (args.verbose)
        {
            std::cout << commandLineParser.GetCurrentValuesString() << std::endl;
        }

        return args;
    }
    catch (const utilities::CommandLineParserPrintHelpException& exception)
    {
        std::cout << exception.GetHelpText() << std::endl;
        throw;
    }
    catch (const utilities::CommandLineParserErrorException& exception)
    {
        std::cerr << "Command line parse error:" << std::endl;
        for (const auto& error : exception.GetParseErrors())
        {
            std::cerr << error.GetMessage() << std::endl;
        }
        throw;
    }
}

} // namespace ell