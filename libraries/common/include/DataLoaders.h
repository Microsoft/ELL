////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     DataLoaders.h (common)
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "DataLoadArguments.h"

#include <data/include/Dataset.h>
#include <data/include/ExampleIterator.h>

#include <model/include/Map.h>

#include <utilities/include/StringUtil.h>

#include <istream>
#include <string>

namespace ell
{
namespace common
{
    /// <summary> Gets an ExampleIterator from an input stream. </summary>
    ///
    /// <typeparam name="TextLineIteratorType"> Line iterator type. </typeparam>
    /// <typeparam name="MetadataParserType"> Metadata parser type. </typeparam>
    /// <typeparam name="DataVectorParserType"> DataVector parser type. </typeparam>
    /// <param name="stream"> Input stream to load data from. </param>
    ///
    /// <returns> The data iterator. </returns>
    template <typename TextLineIteratorType, typename MetadataParserType, typename DataVectorParserType>
    auto GetExampleIterator(std::istream& stream);

    /// <summary> Gets an AutoSupervisedExampleIterator iterator from an input stream. </summary>
    ///
    /// <param name="stream"> Input stream to load data from. </param>
    ///
    /// <returns> The data iterator. </returns>
    data::AutoSupervisedExampleIterator GetAutoSupervisedExampleIterator(std::istream& stream);

    /// <summary> Gets an AutoSupervisedMultiClassExampleIterator iterator from an input stream. </summary>
    ///
    /// <param name="stream"> Input stream to load data from. </param>
    ///
    /// <returns> The data iterator. </returns>
    data::AutoSupervisedMultiClassExampleIterator GetAutoSupervisedMultiClassExampleIterator(std::istream& stream);

    /// <summary> Gets an AutoSupervisedDataset dataset from data load arguments. </summary>
    ///
    /// <param name="stream"> Input stream to load data from. </param>
    ///
    /// <returns> The dataset. </returns>
    data::AutoSupervisedDataset GetDataset(std::istream& stream);

    /// <summary> Gets a dataset from data load arguments. </summary>
    ///
    /// <param name="stream"> Input stream to load data from. </param>
    ///
    /// <returns> The dataset. </returns>
    data::AutoSupervisedMultiClassDataset GetMultiClassDataset(std::istream& stream);

    /// <summary>
    /// Gets a new dataset by running an existing dataset through a map.
    /// </summary>
    ///
    /// <typeparam name="ExampleType"> Example type. </typeparam>
    /// <typeparam name="MapType"> Map type. </typeparam>
    /// <param name="input"> Input dataset. </param>
    /// <param name="map"> Map to run input dataset on. </param>
    ///
    /// <returns> The transformed dataset. </returns>
    template <typename ExampleType, typename MapType>
    auto TransformDataset(data::Dataset<ExampleType>& input, const MapType& map);

    /// <summary>
    /// The map is first compiled, then a new dataset is returned
    /// by running an existing dataset through the compiled map.
    /// </summary>
    ///
    /// <typeparam name="ExampleType"> Example type. </typeparam>
    /// <typeparam name="MapType"> Map type. </typeparam>
    /// <param name="input"> Input dataset. </param>
    /// <param name="map"> Map to run input dataset on. </param>
    /// <param name="useBlas"> Use BLAS in the emitted code to speed up linear algerbra operations. </param>
    ///
    /// <returns> The transformed dataset. </returns>
    template <typename ExampleType, typename MapType>
    auto TransformDatasetWithCompiledMap(data::Dataset<ExampleType>& input, const MapType& map, bool useBlas = true);
} // namespace common
} // namespace ell

#pragma region implementation

#include <data/include/SingleLineParsingExampleIterator.h>

#include <model/include/IRCompiledMap.h>
#include <model/include/IRMapCompiler.h>

#include <nodes/include/ClockNode.h> // for nodes::TimeTickType

namespace ell
{
namespace common
{
    template <typename TextLineIteratorType, typename MetadataParserType, typename DataVectorParserType>
    auto GetExampleIterator(std::istream& stream)
    {
        TextLineIteratorType textLineIterator(stream);

        MetadataParserType metadataParser;

        DataVectorParserType dataVectorParser;

        return data::MakeSingleLineParsingExampleIterator(std::move(textLineIterator), std::move(metadataParser), std::move(dataVectorParser));
    }

    template <typename ExampleType, typename MapType>
    auto TransformDataset(data::Dataset<ExampleType>& input, MapType& map)
    {
        return input.template Transform<ExampleType>([&map](const ExampleType& example) {
            auto transformedDataVector = map.template Compute<data::DoubleDataVector>(example.GetDataVector());
            return ExampleType(std::move(transformedDataVector), example.GetMetadata());
        });
    }

    namespace detail
    {
        // Context used by callback functions
        struct CallbackContext
        {
            std::vector<double> inputValues;
        };
    } // namespace detail

    // C functions called by compiled maps
    extern "C" {
    inline bool InputCallback_Double(void* context, double* input)
    {
        auto dataContext = static_cast<detail::CallbackContext*>(context);
        std::copy(dataContext->inputValues.begin(), dataContext->inputValues.end(), input);
        return true;
    }

    inline bool InputCallback_Float(void* context, float* input)
    {
        auto dataContext = static_cast<detail::CallbackContext*>(context);
        std::transform(dataContext->inputValues.begin(), dataContext->inputValues.end(), input, [](double val) { return static_cast<float>(val); });
        return true;
    }
    }

    namespace detail
    {
        // Sets up the function address that the LLVM jit will call for the source function callback
        // Note that this only supports a single source node, but can be extended in the future
        // to support multiple source nodes (e.g. by switching the function on node id).
        template <typename MapType>
        void ResolveInputCallback(const MapType& map, llvm::Module* module, ell::emitters::IRExecutionEngine& jitter)
        {
            const std::string defaultCallbackName("ELL_InputCallback");
            auto callback = module->getFunction(defaultCallbackName);

            ptrdiff_t callbackAddress = 0;
            switch (map.GetInputType())
            {
            case model::Port::PortType::smallReal: {
                callbackAddress = reinterpret_cast<ptrdiff_t>(&InputCallback_Float);
                break;
            }
            case model::Port::PortType::real: {
                callbackAddress = reinterpret_cast<ptrdiff_t>(&InputCallback_Double);
                break;
            }
            default:
                throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Unexpected source input type for model. Should be double or float.");
            }

            jitter.DefineFunction(callback, callbackAddress);
        }
    } // namespace detail

    template <typename ExampleType, typename MapType>
    auto TransformDatasetWithCompiledMap(data::Dataset<ExampleType>& input, const MapType& map, bool useBlas)
    {
        ell::model::MapCompilerOptions settings;
        settings.compilerSettings.useBlas = useBlas;
        ell::model::ModelOptimizerOptions optimizerOptions;

        detail::CallbackContext dataContext;
        model::IRMapCompiler compiler(settings, optimizerOptions);

        auto module = compiler.GetModule().GetLLVMModule();
        auto compiledMap = compiler.Compile(map);
        compiledMap.SetContext(&dataContext);

        // Unlike reference maps, compiled maps receive the current time as the parameter input and
        // values through the input callback.
        if (map.GetSourceNodes().size() > 0)
        {
            detail::ResolveInputCallback(map, module, compiledMap.GetJitter());
            return input.template Transform<ExampleType>([&compiledMap, &dataContext](const ExampleType& example) {
                dataContext.inputValues = example.GetDataVector().ToArray();
                compiledMap.SetInputValue(0, std::vector<nodes::TimeTickType>({ 0 /*currentTime*/ }));
                auto transformedDataVector = compiledMap.template ComputeOutput<typename ExampleType::DataVectorType>(0);
                return ExampleType(std::move(transformedDataVector), example.GetMetadata());
            });
        }
        else
        {
            auto type = map.GetInputType();
            switch (type)
            {
            case model::Port::PortType::smallReal: {
                return input.template Transform<ExampleType>([&compiledMap](const ExampleType& example) {
                    auto data = example.GetDataVector().ToArray();
                    std::vector<float> smallData(data.size());
                    std::transform(data.begin(), data.end(), smallData.begin(), [](double val) { return static_cast<float>(val); });
                    compiledMap.SetInputValue(0, smallData);
                    auto transformedDataVector = compiledMap.template ComputeOutput<typename ExampleType::DataVectorType>(0);
                    return ExampleType(std::move(transformedDataVector), example.GetMetadata());
                });
            }
            case model::Port::PortType::real: {
                return input.template Transform<ExampleType>([&compiledMap](const ExampleType& example) {
                    compiledMap.SetInputValue(0, example.GetDataVector().ToArray());
                    auto transformedDataVector = compiledMap.template ComputeOutput<typename ExampleType::DataVectorType>(0);
                    return ExampleType(std::move(transformedDataVector), example.GetMetadata());
                });
            }
            default:
                throw utilities::InputException(utilities::InputExceptionErrors::typeMismatch,
                                                utilities::FormatString("Unexpected input type %d, expecting float or double", type));
            }
        }
    }
} // namespace common
} // namespace ell

#pragma endregion implementation
