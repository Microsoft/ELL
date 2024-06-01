////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     MapCompilerArguments.cpp (common)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MapCompilerArguments.h"

#include <utilities/include/Archiver.h>
#include <utilities/include/Files.h>
#include <utilities/include/JsonArchiver.h>
#include <utilities/include/StringUtil.h>

namespace ell
{
namespace common
{
    void ParsedMapCompilerArguments::AddArgs(utilities::CommandLineParser& parser)
    {
        parser.AddDocumentationString("Compiler options");
        parser.AddOption(
            compiledModuleName,
            "compiledModuleName",
            "cmn",
            "Name for compiled module",
            "ELL");

        parser.AddOption(
            compiledFunctionName,
            "compiledFunctionName",
            "cfn",
            "Name for compiled function (if none specified, use <moduleName>_Predict",
            "");

        parser.AddOption(
            profile,
            "profile",
            "p",
            "Emit profiling code",
            false);

        parser.AddOption(
            optimize,
            "optimize",
            "opt",
            "Optimize output code",
            true);

        parser.AddOption(
            useBlas,
            "blas",
            "",
            "Emit code that calls BLAS",
            true);

        parser.AddOption(
            fuseLinearOperations,
            "fuseLinearOps",
            "",
            "Fuse sequences of linear operations with constant coefficients into a single operation",
            true);

        parser.AddOption(
            optimizeReorderDataNodes,
            "optimizeReorderDataNodes",
            "",
            "Optimize sequences of reordering nodes",
            true);

        parser.AddOption(
            convolutionMethod,
            "convolutionMethod",
            "",
            "Set the preferred convolution method",
            { { "unrolled", PreferredConvolutionMethod::unrolled },
              { "simple", PreferredConvolutionMethod::simple },
              { "diagonal", PreferredConvolutionMethod::diagonal },
              { "winograd", PreferredConvolutionMethod::winograd },
              { "auto", PreferredConvolutionMethod::automatic } },
            "auto");

        parser.AddOption(
            modelOptions,
            "modelOption",
            "",
            "Add a model-specific option",
            std::vector<std::string>{});

        parser.AddOption(
            nodeOptions,
            "nodeOption",
            "",
            "Add a node-specific option",
            std::vector<std::string>{});

        parser.AddOption(
            enableVectorization,
            "vectorize",
            "vec",
            "Enable ELL's vectorization",
            false);

        parser.AddOption(
            vectorWidth,
            "vectorWidth",
            "vw",
            "Size of vector units",
            4);

        parser.AddOption(
            parallelize,
            "parallelize",
            "par",
            "Enable ELL's parallelization",
            false);

        parser.AddOption(
            useThreadPool,
            "threadPool",
            "tp",
            "Use thread pool for parallelization (if parallelization enabled)",
            true);

        parser.AddOption(
            maxThreads,
            "threads",
            "th",
            "Maximum num of parallel threads",
            4);

        parser.AddOption(
            debug,
            "debug",
            "dbg",
            "Emit debug code",
            false);

        parser.AddDocumentationString("");
        parser.AddDocumentationString("Target device options");
        parser.AddOption(
            target,
            "target",
            "t",
            "Target name",
            { { "host" }, { "pi0" }, { "pi3" }, { "orangepi0" }, { "pi3_64" }, { "mac" }, { "linux" }, { "windows" }, { "ios" }, { "aarch64" }, { "custom" } },
            "host");

        parser.AddOption(
            numBits,
            "numBits",
            "b",
            "Number of bits for target [0 == auto]",
            0);

        parser.AddOption(
            cpu,
            "cpu",
            "cpu",
            "The CPU target for generating code",
            "");

        parser.AddOption(
            targetTriple,
            "triple",
            "",
            "The triple describing the target architecture",
            "");

        parser.AddOption(
            targetDataLayout,
            "datalayout",
            "",
            "The string describing the target data layout",
            "");

        parser.AddOption(
            targetFeatures,
            "features",
            "",
            "A string describing target-specific features to enable or disable (these are LLVM attributes, in the format the llc -mattr option uses)",
            "");

        parser.AddOption(
            positionIndependentCode,
            "positionIndependentCode",
            "pic",
            "Generate position independent code (equivalent to -fPIC)",
            { { "auto", utilities::Optional<bool>() },
              { "true", utilities::Optional<bool>(true) },
              { "false", utilities::Optional<bool>(false) } },
            "auto");

        parser.AddOption(
            globalValueAlignment,
            "globalValueAlignment",
            "gva",
            "The number of bytes to align global buffers to",
            32);

        parser.AddOption(
            skip_ellcode,
            "skip_ellcode",
            "skip_ellcode",
            "To skip ELLCode",
            false);
    }

    model::MapCompilerOptions MapCompilerArguments::GetMapCompilerOptions(const std::string& modelName) const
    {
        model::MapCompilerOptions settings;
        bool namespaceSpecified = true;
        auto namespacePrefix = compiledModuleName;
        if (namespacePrefix == "")
        {
            namespacePrefix = "ELL";
            namespaceSpecified = false;
        }

        std::string functionName;
        if (compiledFunctionName == "")
        {
            if (namespaceSpecified)
            {
                functionName = namespacePrefix + "_Predict";
            }
            else
            {
                functionName = modelName;
            }
        }
        else
        {
            functionName = namespacePrefix + "_" + compiledFunctionName;
        }

        settings.moduleName = namespacePrefix;
        settings.mapFunctionName = functionName;
        settings.compilerSettings.optimize = optimize;
        settings.compilerSettings.useBlas = useBlas;
        settings.compilerSettings.allowVectorInstructions = enableVectorization;
        settings.compilerSettings.parallelize = parallelize;
        settings.compilerSettings.vectorWidth = vectorWidth;
        settings.profile = profile;
        settings.compilerSettings.profile = profile;
        settings.compilerSettings.positionIndependentCode = positionIndependentCode;
        settings.compilerSettings.globalValueAlignment = globalValueAlignment;
        settings.compilerSettings.skip_ellcode = skip_ellcode;

        if (target != "")
        {
            settings.compilerSettings.targetDevice.deviceName = target;
        }

        if (cpu != "")
        {
            settings.compilerSettings.targetDevice.cpu = cpu;
        }

        if (targetTriple != "")
        {
            settings.compilerSettings.targetDevice.triple = targetTriple;
        }

        if (targetDataLayout != "")
        {
            settings.compilerSettings.targetDevice.dataLayout = targetDataLayout;
        }

        if (targetFeatures != "")
        {
            settings.compilerSettings.targetDevice.features = targetFeatures;
        }

        if (numBits != 0)
        {
            settings.compilerSettings.targetDevice.numBits = numBits;
        }

        // Now add any settings specified in the --modelOptions metadata
        auto metadata = GetOptionsMetadata();
        if (metadata.HasEntry("model"))
        {
            settings = settings.AppendOptions(metadata);
        }

        return settings;
    }

    model::ModelOptimizerOptions MapCompilerArguments::GetModelOptimizerOptions() const
    {
        model::ModelOptimizerOptions options;
        options["fuseLinearFunctionNodes"] = fuseLinearOperations;
        options["optimizeReorderDataNodes"] = optimizeReorderDataNodes;
        options["preferredConvolutionMethod"] = convolutionMethod;

        auto metadata = GetOptionsMetadata();
        if (metadata.HasEntry("model"))
        {
            AppendMetadataToOptions(metadata.GetEntry<utilities::PropertyBag>("model"), options);
        }

        return options;
    }

    bool MapCompilerArguments::HasOptionsMetadata() const
    {
        return !nodeOptions.empty() || !modelOptions.empty();
    }

    utilities::PropertyBag MapCompilerArguments::GetOptionsMetadata() const
    {
        utilities::PropertyBag result;
        utilities::PropertyBag modelMetadata = GetModelOptionsMetadata();
        if (!modelMetadata.IsEmpty())
        {
            result["model"] = modelMetadata;
        }

        utilities::PropertyBag nodesMetadata = GetNodeOptionsMetadata();
        if (!nodesMetadata.IsEmpty())
        {
            result["nodes"] = nodesMetadata;
        }

        return result;
    }

    utilities::PropertyBag MapCompilerArguments::GetModelOptionsMetadata() const
    {
        utilities::PropertyBag modelMetadata;
        for (const auto& entry : modelOptions)
        {
            if (!entry.empty())
            {
                auto parts = utilities::Split(entry, ',');
                if (parts.size() == 2u) // key, value
                {
                    // key, value
                    modelMetadata[parts[0]] = parts[1];
                }
                else
                {
                    auto msg = "Model options must be in the format \"<option_name>,<option_value>\", got: " + entry;
                    throw utilities::CommandLineParserInvalidOptionsException(msg.c_str());
                }
            }
        }
        return modelMetadata;
    }

    utilities::PropertyBag MapCompilerArguments::GetNodeOptionsMetadata() const
    {
        utilities::PropertyBag nodesMetadata;
        for (const auto& entry : nodeOptions)
        {
            if (!entry.empty())
            {
                auto parts = utilities::Split(entry, ',');
                if (parts.size() == 3u) // node id, key, value
                {
                    // node Id, key, value
                    auto nodeMetadata = nodesMetadata.GetEntry(parts[0], utilities::PropertyBag{});
                    nodeMetadata[parts[1]] = parts[2];
                    nodesMetadata[parts[0]] = nodeMetadata;
                }
                else
                {
                    auto msg = "Node options must be in the format \"<node_id>,<option_name>,<option_value>\", got: " + entry;
                    throw utilities::CommandLineParserInvalidOptionsException(msg.c_str());
                }
            }
        }
        return nodesMetadata;
    }

} // namespace common
} // namespace ell
