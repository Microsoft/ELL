////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     DSPNodesTest.cpp (nodes_test)
//  Authors:  Chuck Jacobs, Byron Changuion, Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DSPNodesTests.h"
#include "DTWPrototype.h"
#include "NodesTestData.h"
#include "NodesTestUtilities.h"

#include <model_testing/include/ModelTestUtilities.h>

#include <common/include/LoadModel.h>

#include <dsp/include/Convolution.h>

#include <math/include/MathConstants.h>
#include <math/include/Tensor.h>
#include <math/include/TensorOperations.h>
#include <math/include/Vector.h>

#include <model/include/InputNode.h>
#include <model/include/Model.h>
#include <model/include/Node.h>

#include <nodes/include/BufferNode.h>
#include <nodes/include/ConstantNode.h>
#include <nodes/include/DTWDistanceNode.h>
#include <nodes/include/DelayNode.h>
#include <nodes/include/DiagonalConvolutionNode.h>
#include <nodes/include/FFTNode.h>
#include <nodes/include/FilterBankNode.h>
#include <nodes/include/GRUNode.h>
#include <nodes/include/IIRFilterNode.h>
#include <nodes/include/LSTMNode.h>
#include <nodes/include/RNNNode.h>
#include <nodes/include/ReorderDataCodeNode.h>
#include <nodes/include/SimpleConvolutionNode.h>
#include <nodes/include/UnrolledConvolutionNode.h>
#include <nodes/include/WinogradConvolutionNode.h>

#include <predictors/include/NeuralNetworkPredictor.h>

#include <predictors/neural/include/ConvolutionalLayer.h>
#include <predictors/neural/include/SigmoidActivation.h>
#include <predictors/neural/include/TanhActivation.h>

#include <testing/include/testing.h>

#include <utilities/include/Exception.h>
#include <utilities/include/Files.h>
#include <utilities/include/RandomEngines.h>
#include <utilities/include/StringUtil.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>

using namespace ell;
using namespace nodes;
using namespace data;
using namespace std::string_literals;

//
// Helpers
//
namespace
{

std::string GetConvAlgName(dsp::ConvolutionMethodOption alg)
{
    switch (alg)
    {
    case dsp::ConvolutionMethodOption::automatic:
        return "automatic";
    case dsp::ConvolutionMethodOption::simple:
        return "simple";
    case dsp::ConvolutionMethodOption::unrolled:
        return "unrolled";
    case dsp::ConvolutionMethodOption::diagonal:
        return "diagonal";
    case dsp::ConvolutionMethodOption::winograd:
        return "winograd";
    }
    return "";
}

model::PortMemoryLayout CalculateMemoryLayout(int numRows, int numColumns, int numChannels, int padding)
{
    // Calculate dimension parameters
    model::MemoryShape size{ numRows, numColumns, numChannels };
    model::MemoryShape offset{ padding, padding, 0 };
    model::MemoryShape stride{ numRows + 2 * padding, numColumns + 2 * padding, numChannels };

    return { size, stride, offset };
}
} // namespace

//
// Test compute functions
//

static void TestDelayNodeCompute()
{
    const int delay = 4;

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(1);
    auto outputNode = model.AddNode<nodes::DelayNode<double>>(inputNode->output, delay);

    std::vector<std::vector<double>> data = { { 1 }, { 2 }, { 3 }, { 4 }, { 5 }, { 6 }, { 7 }, { 8 }, { 9 }, { 10 } };

    std::vector<double> outputVec;

    for (size_t index = 0; index < data.size(); ++index)
    {
        auto inputValue = data[index];
        inputNode->SetInput(inputValue);
        outputVec = model.ComputeOutput(outputNode->output);
        if (index >= delay)
        {
            testing::ProcessTest("Testing DelayNode compute", testing::IsEqual(outputVec, data[index - delay]));
        }
    }
}

static void TestFFTNodeCompute(size_t N, size_t nfft)
{
    using ValueType = double;
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ValueType>>(N);
    nodes::FFTNode<ValueType>* fftNode = nullptr;
    if (nfft == 0)
    {
        nfft = pow(2, ceil(log2(N)));
        fftNode = model.AddNode<nodes::FFTNode<ValueType>>(inputNode->output);
    }
    else
    {
        fftNode = model.AddNode<nodes::FFTNode<ValueType>>(inputNode->output, nfft);
    }

    // FFT of constant value
    std::vector<ValueType> signal(N, 1.0);
    inputNode->SetInput(signal);
    auto computeOutput = model.ComputeOutput(fftNode->output);
    for (size_t index = 0; index < computeOutput.size(); ++index)
    {
        auto x = computeOutput[index];
        if (N >= nfft || index == 0)
        {
            auto expected = (N > nfft) ? nfft : N;
            testing::ProcessTest("Testing real-valued FFT of DC signal", testing::IsEqual(x, static_cast<ValueType>(index == 0 ? expected : 0)));
        }
        else
        {
            // in the zero-padded case the ramp down from N is a bit smoother, it is not just a step function.
            testing::ProcessTest("Testing real-valued FFT of DC signal", testing::IsTrue(x < N));
        }
    }

    // FFT of impulse signal
    signal.assign(N, 0);
    signal[0] = 1.0;
    inputNode->SetInput(signal);
    computeOutput = model.ComputeOutput(fftNode->output);
    for (size_t index = 0; index < computeOutput.size(); ++index)
    {
        auto x = computeOutput[index];
        testing::ProcessTest("Testing real-valued FFT of impulse signal", testing::IsEqual(x, static_cast<ValueType>(1)));
    }

    // FFT of some arbitrary sine waves
    for (size_t freq : { 1, 3, 6, 11 })
    {
        for (size_t index = 0; index < N; ++index)
        {
            signal[index] = std::sin(2 * math::Constants<ValueType>::pi * index * freq / N);
        }
        inputNode->SetInput(signal);
        computeOutput = model.ComputeOutput(fftNode->output);
        if (N == nfft)
        {
            for (size_t index = 0; index < computeOutput.size(); ++index)
            {
                auto x = computeOutput[index];
                bool isPeak = (index == freq) || (index == (nfft - freq));
                testing::ProcessTest("Testing real-valued FFT of sine wave", testing::IsEqual(x, static_cast<ValueType>(isPeak ? N / 2 : 0)));
            }
        }
        else
        {
            // with zero-padding the output is smoother, but the argmax should still be at the given frequency band.
            // but that frequence band is shifted because of the larger nfft value.
            int expectedPeak = static_cast<int>(std::round(static_cast<ValueType>(freq) * static_cast<ValueType>(nfft) / static_cast<ValueType>(N)));
            auto result = std::max_element(computeOutput.begin(), computeOutput.end());
            int actualPeak = static_cast<int>(result - computeOutput.begin());
            testing::ProcessTest("Testing real-valued FFT of sine wave", testing::IsTrue(actualPeak == expectedPeak || actualPeak == expectedPeak - 1));
        }
    }
}

static void TestDTWDistanceNodeCompute()
{
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<double>>(3);
    auto prototype = GetNextSlidePrototype();
    auto dtwNode = model.AddNode<nodes::DTWDistanceNode<double>>(inputNode->output, prototype);

    //
    auto prototypeLength = prototype.size();
    size_t numSamples = 200;
    size_t increment = 3;
    for (size_t index = 0; index < numSamples; ++index)
    {
        auto sampleIndex = (index * increment) % prototypeLength;
        auto inputValue = prototype[sampleIndex];
        inputNode->SetInput(inputValue);
        std::vector<double> outputVec = model.ComputeOutput(dtwNode->output);
    }
}

//
// Combined tests
//

template <typename ValueType>
static void TestIIRFilterNode1()
{
    const ValueType epsilon = static_cast<ValueType>(1e-6);

    std::vector<std::vector<ValueType>> data = { { 1 }, { 0 }, { 0 }, { 0 } };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ValueType>>(data[0].size());
    auto outputNode = model.AddNode<nodes::IIRFilterNode<ValueType>>(inputNode->output, std::vector<ValueType>{ static_cast<ValueType>(1.0) }, std::vector<ValueType>{ static_cast<ValueType>(-0.95) });

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });
    model::MapCompilerOptions settings;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    std::vector<std::vector<ValueType>> expectedOutput = { { static_cast<ValueType>(1.0) }, { static_cast<ValueType>(0.95) }, { static_cast<ValueType>(0.95 * 0.95) }, { static_cast<ValueType>(0.95 * 0.95 * 0.95) } };
    for (size_t index = 0; index < data.size(); ++index)
    {
        auto input = data[index];

        map.SetInputValue(0, input);
        auto computedResult = map.ComputeOutput<ValueType>(0);

        compiledMap.SetInputValue(0, input);
        auto compiledResult = compiledMap.ComputeOutput<ValueType>(0);

        testing::ProcessTest("Testing IIRFilterNode compute", testing::IsEqual(computedResult, expectedOutput[index], epsilon));
        testing::ProcessTest("Testing IIRFilterNode compile", testing::IsEqual(compiledResult, expectedOutput[index], epsilon));
    }
}

template <typename ValueType>
static void TestIIRFilterNode2()
{
    const ValueType epsilon = static_cast<ValueType>(1e-6);

    std::vector<std::vector<ValueType>> data = { { 1, 0, 0, 0 } };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ValueType>>(data[0].size());
    auto outputNode = model.AddNode<nodes::IIRFilterNode<ValueType>>(inputNode->output, std::vector<ValueType>{ static_cast<ValueType>(1.0) }, std::vector<ValueType>{ static_cast<ValueType>(-0.95) });

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });
    model::MapCompilerOptions settings;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    std::vector<std::vector<ValueType>> expectedOutput = { { static_cast<ValueType>(1.0), static_cast<ValueType>(0.95), static_cast<ValueType>(0.95 * 0.95), static_cast<ValueType>(0.95 * 0.95 * 0.95) } };
    for (size_t index = 0; index < data.size(); ++index)
    {
        auto input = data[index];

        map.SetInputValue(0, input);
        auto computedResult = map.ComputeOutput<ValueType>(0);

        compiledMap.SetInputValue(0, input);
        auto compiledResult = compiledMap.ComputeOutput<ValueType>(0);

        testing::ProcessTest("Testing IIRFilterNode compute 2", testing::IsEqual(computedResult, expectedOutput[index], epsilon));
        testing::ProcessTest("Testing IIRFilterNode compile 2", testing::IsEqual(compiledResult, expectedOutput[index], epsilon));
    }
}

template <typename ValueType>
static void TestIIRFilterNode3()
{
    const ValueType epsilon = static_cast<ValueType>(1e-6);

    std::vector<ValueType> datapoint(128);
    datapoint[0] = 1.0;
    std::vector<std::vector<ValueType>> data = { datapoint };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ValueType>>(data[0].size());
    std::vector<ValueType> aCoeffs = { static_cast<ValueType>(0.0125), static_cast<ValueType>(-0.0125) };
    std::vector<ValueType> bCoeffs = { static_cast<ValueType>(1.0), static_cast<ValueType>(0.25), static_cast<ValueType>(-0.125) };
    auto outputNode = model.AddNode<nodes::IIRFilterNode<ValueType>>(inputNode->output, bCoeffs, aCoeffs);

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });
    model::MapCompilerOptions settings;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    for (size_t index = 0; index < data.size(); ++index)
    {
        auto input = data[index];

        map.SetInputValue(0, input);
        auto computedResult = map.ComputeOutput<ValueType>(0);

        compiledMap.SetInputValue(0, input);
        auto compiledResult = compiledMap.ComputeOutput<ValueType>(0);

        testing::ProcessTest("Testing IIRFilterNode compile 3", testing::IsEqual(compiledResult, computedResult, epsilon));
    }
}

template <typename ValueType>
static void TestIIRFilterNode4()
{
    const ValueType epsilon = static_cast<ValueType>(1e-6);

    std::vector<std::vector<ValueType>> data = { { 1, 0, 0, 0, 0, 0, 0 } };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ValueType>>(data[0].size());
    std::vector<ValueType> aCoeffs = { 0 };
    std::vector<ValueType> bCoeffs = { static_cast<ValueType>(1.0), static_cast<ValueType>(0.25), static_cast<ValueType>(-0.125) };
    auto outputNode = model.AddNode<nodes::IIRFilterNode<ValueType>>(inputNode->output, bCoeffs, aCoeffs);

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });
    model::MapCompilerOptions settings;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    std::vector<std::vector<ValueType>> expectedOutput = { bCoeffs };
    expectedOutput.resize(data.size());
    for (size_t index = 0; index < data.size(); ++index)
    {
        auto input = data[index];

        map.SetInputValue(0, input);
        auto computedResult = map.ComputeOutput<ValueType>(0);

        compiledMap.SetInputValue(0, input);
        auto compiledResult = compiledMap.ComputeOutput<ValueType>(0);

        testing::ProcessTest("Testing IIRFilterNode compute 4", testing::IsEqual(computedResult, expectedOutput[index], epsilon));
        testing::ProcessTest("Testing IIRFilterNode compile 4", testing::IsEqual(compiledResult, expectedOutput[index], epsilon));
    }
}

template <typename ValueType>
static void TestMelFilterBankNode()
{
    const ValueType epsilon = static_cast<ValueType>(1e-5);
    const size_t numFilters = 13;
    const size_t windowSize = 512;
    const size_t fftSize = 512;
    const double sampleRate = 16000;

    std::vector<ValueType> signal(windowSize);
    FillRandomVector(signal);
    std::vector<std::vector<ValueType>> data = { signal };

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ValueType>>(windowSize);
    auto filters = dsp::MelFilterBank(windowSize, sampleRate, fftSize, numFilters);
    auto outputNode = model.AddNode<nodes::MelFilterBankNode<ValueType>>(inputNode->output, filters);

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });
    model::MapCompilerOptions settings;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    for (size_t index = 0; index < data.size(); ++index)
    {
        auto input = data[index];

        map.SetInputValue(0, input);
        auto computedResult = map.ComputeOutput<ValueType>(0);

        compiledMap.SetInputValue(0, input);
        auto compiledResult = compiledMap.ComputeOutput<ValueType>(0);

        testing::ProcessTest("Testing MelFilterBankNode compile", testing::IsEqual(compiledResult, computedResult, epsilon));
    }
}

template <typename ValueType>
static void TestBufferNode()
{
    const size_t inputSize = 16;
    const size_t windowSize = 40;

    std::vector<std::vector<ValueType>> data;
    const int numEntries = 8; // 8 input buffers of consecutive numbers
    for (int index = 0; index < numEntries; ++index)
    {
        std::vector<ValueType> item(inputSize);
        std::iota(item.begin(), item.end(), inputSize * index);
        data.push_back(item);
    }

    std::vector<ValueType> window(windowSize);

    std::vector<std::vector<ValueType>> expected;
    // compute the expected sliding buffer output by actually sliding the
    // input through the window.
    for (auto input : data)
    {
        auto shifted = windowSize - inputSize;
        std::copy_n(window.begin() + inputSize, shifted, window.begin());
        std::copy(input.begin(), input.end(), window.begin() + shifted);
        expected.push_back(window);
    }

    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ValueType>>(inputSize);
    auto outputNode = model.AddNode<nodes::BufferNode<ValueType>>(inputNode->output, windowSize);

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", outputNode->output } });

    TestWithSerialization(map, "TestBufferNode", [&](model::Map& map, int iteration) {
        model::MapCompilerOptions settings;
        settings.compilerSettings.optimize = false;
        model::ModelOptimizerOptions optimizerOptions;
        model::IRMapCompiler compiler(settings, optimizerOptions);
        auto compiledMap = compiler.Compile(map);

        auto message = utilities::FormatString("Testing BufferNode compile iteration %d", iteration);
        VerifyCompiledOutputAndResult<ValueType, ValueType>(map, compiledMap, data, expected, message);
    });
}

template <typename ValueType>
static void TestConvolutionNodeCompile(dsp::ConvolutionMethodOption convolutionMethod)
{
    using Tensor = math::ChannelColumnRowTensor<ValueType>;

    const ValueType epsilon = static_cast<ValueType>(1e-7);
    const int inputRows = 10;
    const int inputColumns = 10;
    const int outputRows = 10;
    const int outputColumns = 10;
    const int numChannels = 1;
    const int filterSize = 3;
    const int numFilters = 1;
    const int inputPadding = 1;
    const int outputPadding = 0;
    const int stride = 1;

    const int winogradTileSize = 2;
    const auto winogradFilterOrder = nodes::WinogradConvolutionNode<ValueType>::FilterOrder::tilesFirst;

    auto data = GetConvolutionTestData<ValueType>();
    auto filter = GetConvolutionTestFilter<ValueType>();
    auto reference = GetCorrelationTestResultSame<ValueType>();

    auto inputMemoryLayout = CalculateMemoryLayout(inputRows, inputColumns, numChannels, inputPadding);
    auto outputMemoryLayout = CalculateMemoryLayout(outputRows, outputColumns, numFilters, outputPadding);
    auto filterWeights = Tensor(numFilters * filterSize, filterSize, numChannels, filter);

    // auto inputSize = data.size();
    auto inputSize = inputMemoryLayout.GetMemorySize();
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ValueType>>(inputSize);

    model::Node* outputNode = nullptr;
    switch (convolutionMethod)
    {
    case dsp::ConvolutionMethodOption::automatic:
        std::cout << "Testing 'automatic' method --- using 'simple' instead" << std::endl;
    // fallthrough
    case dsp::ConvolutionMethodOption::simple:
        outputNode = model.AddNode<nodes::SimpleConvolutionNode<ValueType>>(inputNode->output, inputMemoryLayout, outputMemoryLayout, filterWeights, stride);
        break;
    case dsp::ConvolutionMethodOption::diagonal:
        outputNode = model.AddNode<nodes::DiagonalConvolutionNode<ValueType>>(inputNode->output, inputMemoryLayout, outputMemoryLayout, filterWeights, stride);
        break;
    case dsp::ConvolutionMethodOption::unrolled:
        outputNode = model.AddNode<nodes::UnrolledConvolutionNode<ValueType>>(inputNode->output, inputMemoryLayout, outputMemoryLayout, filterWeights, stride);
        break;
    case dsp::ConvolutionMethodOption::winograd:
        outputNode = model.AddNode<nodes::WinogradConvolutionNode<ValueType>>(inputNode->output, inputMemoryLayout, outputMemoryLayout, filterWeights, stride, winogradTileSize, winogradFilterOrder);
        break;
    }

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", *(outputNode->GetOutputPort(0)) } });

    auto rawDataTensor = Tensor(inputRows, inputColumns, numChannels, data);
    auto paddedDataTensor = Tensor(inputRows + 2, inputColumns + 2, numChannels);
    paddedDataTensor.Fill(0);
    auto dataTensorReference = paddedDataTensor.GetSubTensor(inputPadding, inputPadding, 0, inputRows, inputColumns, numChannels);
    dataTensorReference.CopyFrom(rawDataTensor);
    auto paddedDataArray = paddedDataTensor.ToArray();

    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = false;
    settings.compilerSettings.useBlas = true;
    settings.verifyJittedModule = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);
    auto compiledMap = compiler.Compile(map);

    // Check for errors in module
    auto& module = compiledMap.GetModule();
    auto hasErrors = module.CheckForErrors(std::cerr);
    testing::ProcessTest("Testing compiled "s + GetConvAlgName(convolutionMethod) + " convolution node model for errors", !hasErrors);

    compiledMap.SetInputValue(0, paddedDataArray);
    auto compiledResult = compiledMap.ComputeOutput<ValueType>(0);
    auto ok = testing::IsEqual(reference, compiledResult, epsilon);

#if 0
    // Helpful debugging output
    if (!ok)
    {
        std::cout << "Compiled result: " << compiledResult << std::endl;
        std::cout << "Reference result: " << reference << std::endl;
    }
#endif
    testing::ProcessTest("Testing compiled "s + GetConvAlgName(convolutionMethod) + " convolution node", ok);
}

template <typename ValueType>
static void TestConvolutionNodeCompileVsReference(ImageShape inputShape, FiltersShape filterShape, int stride, dsp::ConvolutionMethodOption convolutionMethod, ConvolutionOptions options = {})
{
    int inputRows = inputShape.numRows;
    int inputColumns = inputShape.numColumns;
    int numChannels = inputShape.numChannels;

    int numFilters = filterShape.numFilters;
    int filterSize = filterShape.numRows;
    assert(filterShape.numColumns == filterSize);
    int numFilterChannels = filterShape.numChannels == 0 ? numChannels : filterShape.numChannels; // "0" means "number of input channels"
    auto isDepthwiseSeparable = (numFilterChannels == 1) && (numChannels > 1);

    using Tensor = math::ChannelColumnRowTensor<ValueType>;

    const ValueType epsilon = static_cast<ValueType>(1e-4);
    const int outputRows = inputRows / stride;
    const int outputColumns = inputColumns / stride;
    const int inputPadding = (filterSize - 1) / 2;
    const int outputPadding = 0;

    auto dataSize = inputRows * inputColumns * numChannels;
    auto data = std::vector<ValueType>(dataSize);
    FillRandomVector(data);

    auto filterWeightsSize = numFilters * filterSize * filterSize * numChannels;
    auto filter = std::vector<ValueType>(filterWeightsSize);
    FillRandomVector(filter);

    auto inputMemoryLayout = CalculateMemoryLayout(inputRows, inputColumns, numChannels, inputPadding);
    auto outputMemoryLayout = CalculateMemoryLayout(outputRows, outputColumns, numFilters, outputPadding);
    auto filterWeights = Tensor(numFilters * filterSize, filterSize, numFilterChannels, filter);

    auto inputSize = inputMemoryLayout.GetMemorySize();

    // Create compiler for models
    model::MapCompilerOptions settings;
    settings.compilerSettings.optimize = true;
    settings.compilerSettings.useBlas = true;
    settings.verifyJittedModule = true;
    model::ModelOptimizerOptions optimizerOptions;
    model::IRMapCompiler compiler(settings, optimizerOptions);

    // Create "test" model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ValueType>>(inputSize);

    auto shouldReorderToChannelMajor = isDepthwiseSeparable && (convolutionMethod == dsp::ConvolutionMethodOption::simple);

    auto convInputLayout = inputMemoryLayout.ReorderedCopy({ shouldReorderToChannelMajor ? utilities::ChannelMajorTensorOrder : utilities::RowMajorTensorOrder });
    auto convOutputLayout = outputMemoryLayout.ReorderedCopy({ shouldReorderToChannelMajor ? utilities::ChannelMajorTensorOrder : utilities::RowMajorTensorOrder });

    auto preConvReorderNode = model.AddNode<nodes::ReorderDataCodeNode<ValueType>>(inputNode->output, inputMemoryLayout, convInputLayout);
    const auto* newInput = &preConvReorderNode->output;

    model::PortElements<ValueType> convOutput;

    switch (convolutionMethod)
    {
    case dsp::ConvolutionMethodOption::automatic:
        std::cout << "Testing 'automatic' method --- using 'simple' instead" << std::endl;
    // fallthrough
    case dsp::ConvolutionMethodOption::simple: {
        auto convNode = model.AddNode<nodes::SimpleConvolutionNode<ValueType>>(*newInput, convInputLayout, convOutputLayout, filterWeights, stride);
        convOutput = convNode->output;
        break;
    }
    case dsp::ConvolutionMethodOption::diagonal: {
        auto convNode = model.AddNode<nodes::DiagonalConvolutionNode<ValueType>>(*newInput, convInputLayout, convOutputLayout, filterWeights, stride);
        convOutput = convNode->output;
        break;
    }
    case dsp::ConvolutionMethodOption::unrolled: {
        auto convNode = model.AddNode<nodes::UnrolledConvolutionNode<ValueType>>(*newInput, convInputLayout, convOutputLayout, filterWeights, stride);
        convOutput = convNode->output;
        break;
    }
    case dsp::ConvolutionMethodOption::winograd: {
        auto convNode = model.AddNode<nodes::WinogradConvolutionNode<ValueType>>(*newInput, convInputLayout, convOutputLayout, filterWeights, stride, options.winogradOptions.tileSize, options.winogradOptions.filterOrder);
        convOutput = convNode->output;
        break;
    }
    }

    auto postConvReorderNode = model.AddNode<nodes::ReorderDataCodeNode<ValueType>>(convOutput, convOutputLayout, outputMemoryLayout);

    auto map = model::Map(model, { { "input", inputNode } }, { { "output", postConvReorderNode->output } });

    auto rawDataTensor = Tensor(inputRows, inputColumns, numChannels, data);
    auto paddedDataTensor = Tensor(inputRows + 2 * inputPadding, inputColumns + 2 * inputPadding, numChannels);
    paddedDataTensor.Fill(0);
    auto dataTensorReference = paddedDataTensor.GetSubTensor(inputPadding, inputPadding, 0, inputRows, inputColumns, numChannels);
    dataTensorReference.CopyFrom(rawDataTensor);
    auto paddedDataArray = paddedDataTensor.ToArray();
    auto compiledMap = compiler.Compile(map);

    // Get reference value from dsp library
    std::vector<ValueType> reference;
    if (isDepthwiseSeparable)
    {
        reference = dsp::Convolve2DDepthwiseSeparable(paddedDataTensor, filterWeights, numFilters).ToArray();
    }
    else
    {
        reference = dsp::Convolve2D(paddedDataTensor, filterWeights, numFilters, stride).ToArray();
    }

    compiledMap.SetInputValue(0, paddedDataArray);
    auto compiledResult = compiledMap.ComputeOutput<ValueType>(0);

    auto ok = testing::IsEqual(reference, compiledResult, epsilon);
    testing::ProcessTest("Testing compiled "s + GetConvAlgName(convolutionMethod) + " convolution node vs reference for  " + std::to_string(inputRows) + " x " + std::to_string(inputColumns) + " x " + std::to_string(numChannels) + " image and " + std::to_string(numFilters) + " " + std::to_string(filterSize) + " x " + std::to_string(filterSize) + " x " + std::to_string(numFilterChannels) + " filters, stride " + std::to_string(stride), ok);

    // Helpful debugging output
    if (!ok)
    {
        std::vector<ValueType> diff(reference.size());
        for (size_t index = 0; index < reference.size(); ++index)
        {
            diff[index] = reference[index] - compiledResult[index];
        }
        auto minmax = std::minmax_element(diff.begin(), diff.end());

        std::cout << "Error: min diff: " << *minmax.first << " max diff: " << *minmax.second << "\n";
#if 0
        // Helpful debugging output
        if (compiledResult.size() < 500)
        {
            std::cout << "Compiled result:\n"
                      << compiledResult << "\n\n";
            std::cout << "Reference result:\n"
                      << reference << "\n\n";
        }
#endif
    }
}

//
// Recurrent layer nodes (Recurrent, GRU, LSTM)
//
void TestRNNNode()
{
    using ElementType = double;
    using namespace ell::predictors;
    using namespace ell::predictors::neural;
    using ConstVectorReference = math::ConstColumnVectorReference<ElementType>;

    double epsilon = 1e-5;
    size_t hiddenSize = 3;

    // Precomputed weights created by GenerateGRUTest.py
    const double x_t[] = { 0.11864984035491943, -1.497725486755371, 0.3899663984775543, -0.742249608039856, 0.38884925842285156, -0.8346691131591797, 0.9489753246307373, 1.0470960140228271, -1.3924566507339478, -0.6278074979782104 };
    const double w_i[] = { -0.3215062916278839, -0.8250587582588196, -0.07175730913877487, 0.7295218110084534, 0.29332873225212097, 0.8628071546554565, 0.7429170608520508, 1.1023180484771729, -0.8306611776351929, 0.29230356216430664, 1.1557507514953613, -0.6685269474983215, 0.5184255242347717, -0.45642054080963135, -0.7934108376502991, -0.9269002676010132, 1.1365916728973389, -1.291425108909607, -0.24487516283988953, -0.07714151591062546, 1.0650510787963867, -1.2603979110717773, 0.5447753071784973, 0.6149663925170898, 1.0477608442306519, -1.3045274019241333, -2.4990055561065674, 0.04219631850719452, -0.3878266215324402, 0.047706957906484604 };
    const double w_h[] = { 0.8688994646072388, -1.0314407348632812, 0.8147369623184204, 1.8119542598724365, -0.3671615421772003, -0.05160994082689285, -0.6416834592819214, -1.1712406873703003, 0.510130763053894 };
    const double b_i[] = { 0.18072627484798431, 0.028830422088503838, 1.657913088798523 };
    const double b_h[] = { 0.607840895652771, 0.32387951016426086, 0.3496452271938324 };
    // Expected output
    const double h_1[] = { 0.998630702495575, 0.9893065094947815, 0.9980131387710571 };
    const double h_2[] = { 0.999634325504303, 0.9993395209312439, 0.9740726351737976 };
    const double h_3[] = { 0.9996125102043152, 0.999338686466217, 0.9727824926376343 };
    const double* h_t[] = { h_1, h_2, h_3 };

    ConstVectorReference input(x_t, sizeof(x_t) / sizeof(double));
    size_t inputSize = input.Size();
    ConstVectorReference inputWeights(w_i, sizeof(w_i) / sizeof(double));
    ConstVectorReference hiddenWeights(w_h, sizeof(w_h) / sizeof(double));
    ConstVectorReference inputBias(b_i, sizeof(b_i) / sizeof(double));
    ConstVectorReference hiddenBias(b_h, sizeof(b_h) / sizeof(double));

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputSize);
    auto resetTriggerNode = model.AddNode<nodes::ConstantNode<int>>(0);
    auto inputWeightsNode = model.AddNode<nodes::ConstantNode<ElementType>>(inputWeights.ToArray());
    auto hiddenWeightsNode = model.AddNode<nodes::ConstantNode<ElementType>>(hiddenWeights.ToArray());
    auto inputBiasNode = model.AddNode<nodes::ConstantNode<ElementType>>(inputBias.ToArray());
    auto hiddenBiasNode = model.AddNode<nodes::ConstantNode<ElementType>>(hiddenBias.ToArray());
    auto activation = ell::predictors::neural::Activation<ElementType>(new ell::predictors::neural::TanhActivation<ElementType>());

    auto rnnNode = model.AddNode<nodes::RNNNode<ElementType>>(inputNode->output, resetTriggerNode->output, hiddenSize, inputWeightsNode->output, hiddenWeightsNode->output, inputBiasNode->output, hiddenBiasNode->output, activation);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", rnnNode->output } });

    TestWithSerialization(map, "TestRNNNode", [&](model::Map& map, int iteration) {
        // Compile model
        model::MapCompilerOptions settings;
        settings.compilerSettings.useBlas = true;
        model::ModelOptimizerOptions optimizerOptions;
        model::IRMapCompiler compiler(settings, optimizerOptions);
        auto compiledMap = compiler.Compile(map);
        auto name = rnnNode->GetRuntimeTypeName();

        std::vector<std::vector<ElementType>> signal = { input.ToArray() };
        map.SetInputValue(0, signal[0]);
        std::vector<ElementType> computedResult = map.ComputeOutput<ElementType>(0);
        if (IsEqual(computedResult, std::vector<ElementType>(computedResult.size()), static_cast<double>(epsilon)))
        {
            std::cout << "#############################################################################################\n";
            std::cout << "### bugbug: weird case where first compute randomly fails ... see work item 1918 \n";
            std::cout << "#############################################################################################\n";
        }
        else
        {
            map.Reset(); // test that model reset works.
        }

        // test statefulness of the GRU node
        for (size_t i = 0; i < 3; i++)
        {
            ConstVectorReference expectedOutput(h_t[i], sizeof(h_1) / sizeof(double));

            std::string message = name + utilities::FormatString(" iteration %d row %d", iteration, i);

            // compare computed vs. compiled output
            VerifyCompiledOutputAndResult<ElementType, ElementType>(map, compiledMap, signal, { expectedOutput.ToArray() }, message);
        }
    });
}
void TestGRUNode()
{
    using ElementType = double;
    using namespace ell::predictors;
    using namespace ell::predictors::neural;

    using ConstVectorReference = math::ConstColumnVectorReference<ElementType>;

    // Precomputed weights created by GenerateGRUTest.py
    const double x_t[] = { -1.866538166999817, 2.6257550716400146, 0.06279680877923965, 0.42149317264556885 };
    const double w_i[] = { -1.6192892789840698, 0.36051392555236816, 2.053159475326538, -1.6098523139953613, 0.4089716374874115, -2.4427011013031006, 0.3615369200706482, 0.35893726348876953, 0.6950929164886475, -0.15558069944381714, -0.6605814099311829, -0.17566926777362823, -0.5587524771690369, 1.0622758865356445, -0.7328291535377502, -0.46356719732284546, -0.3172260820865631, 1.1607599258422852, -1.774704933166504, 2.080854654312134, 1.4166220426559448, 0.08074117451906204, 0.6439551711082458, 0.6682876348495483, 0.0031769759953022003, -1.2152390480041504, 0.15362346172332764, 0.22646228969097137, -0.2625025808811188, -0.020441772416234016, -1.1608366966247559, -0.9417181611061096, 0.3165226876735687, 0.36076638102531433, 0.17325237393379211, 1.2059754133224487 };
    const double w_h[] = { -0.6580407619476318, 1.8833234310150146, -1.8858188390731812, -0.21750696003437042, -0.7450492978096008, -0.8853184580802917, -0.3482202887535095, 0.07999972254037857, -1.643333077430725, -0.7072165012359619, -1.0991154909133911, 0.6662507653236389, -0.05446276068687439, -0.6393508911132812, -0.8782468438148499, -0.1724170297384262, -0.8084756135940552, 0.07150451093912125, 1.2091819047927856, 0.04685645550489426, 0.7719306349754333, 0.35313302278518677, 0.23132576048374176, 0.27549979090690613, 1.466962218284607, 1.2321659326553345, -3.2861289978027344 };
    const double b_i[] = { 1.8049829006195068, -0.045422252267599106, 0.13240598142147064, 1.2670079469680786, 1.5549393892288208, -0.5999399423599243, -0.13771949708461761, -1.3751298189163208, -1.840890884399414 };
    const double b_h[] = { 0.7539071440696716, -0.3806458115577698, -0.14616608619689941, -0.42660772800445557, -0.7125017046928406, -0.8799905180931091, 2.8359553813934326, 1.2625319957733154, 0.4183560013771057 };
    // Expected output
    const double h_1[] = { -0.001017451286315918, -0.15146352350711823, -0.6395260691642761 };
    const double h_2[] = { -0.001752614974975586, -0.34548428654670715, -0.6916687488555908 };
    const double h_3[] = { -0.0027370452880859375, -0.3964291512966156, -0.7047065496444702 };
    const double* h_t[] = { h_1, h_2, h_3 };

    size_t hiddenSize = 3;
    double epsilon = 1e-5;

    ConstVectorReference input(x_t, sizeof(x_t) / sizeof(double));
    size_t inputSize = input.Size();
    ConstVectorReference inputWeights(w_i, sizeof(w_i) / sizeof(double));
    ConstVectorReference hiddenWeights(w_h, sizeof(w_h) / sizeof(double));
    ConstVectorReference inputBias(b_i, sizeof(b_i) / sizeof(double));
    ConstVectorReference hiddenBias(b_h, sizeof(b_h) / sizeof(double));

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputSize);
    auto resetTriggerNode = model.AddNode<nodes::ConstantNode<int>>(0);
    auto inputWeightsNode = model.AddNode<nodes::ConstantNode<ElementType>>(inputWeights.ToArray());
    auto hiddenWeightsNode = model.AddNode<nodes::ConstantNode<ElementType>>(hiddenWeights.ToArray());
    auto inputBiasNode = model.AddNode<nodes::ConstantNode<ElementType>>(inputBias.ToArray());
    auto hiddenBiasNode = model.AddNode<nodes::ConstantNode<ElementType>>(hiddenBias.ToArray());
    auto activation = ell::predictors::neural::Activation<ElementType>(new ell::predictors::neural::TanhActivation<ElementType>());
    auto recurrentActivation = ell::predictors::neural::Activation<ElementType>(new ell::predictors::neural::SigmoidActivation<ElementType>());

    auto gruNode = model.AddNode<nodes::GRUNode<ElementType>>(inputNode->output, resetTriggerNode->output, hiddenSize, inputWeightsNode->output, hiddenWeightsNode->output, inputBiasNode->output, hiddenBiasNode->output, activation, recurrentActivation);
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", gruNode->output } });

    TestWithSerialization(map, "TestGRUNode", [&](model::Map& map, int iteration) {
        // Compile model
        model::MapCompilerOptions settings;
        settings.compilerSettings.useBlas = true;
        model::ModelOptimizerOptions optimizerOptions;
        model::IRMapCompiler compiler(settings, optimizerOptions);
        auto compiledMap = compiler.Compile(map);
        auto name = gruNode->GetRuntimeTypeName();

        std::vector<std::vector<ElementType>> signal = { input.ToArray() };
        map.SetInputValue(0, signal[0]);
        std::vector<ElementType> computedResult = map.ComputeOutput<ElementType>(0);
        if (IsEqual(computedResult, std::vector<ElementType>(computedResult.size()), static_cast<double>(epsilon)))
        {
            std::cout << "#############################################################################################\n";
            std::cout << "### bugbug: weird case where first compute randomly fails ... see work item 1918 \n";
            std::cout << "#############################################################################################\n";
        }
        else
        {
            map.Reset(); // test that model reset works.
        }

        // test statefulness of the GRU node
        for (size_t i = 0; i < 3; i++)
        {
            ConstVectorReference expectedOutput(h_t[i], sizeof(h_1) / sizeof(double));

            std::string message = name + utilities::FormatString(" iteration %d row %d", iteration, i);

            // compare computed vs. compiled output
            VerifyCompiledOutputAndResult<ElementType, ElementType>(map, compiledMap, signal, { expectedOutput.ToArray() }, message);
        }
    });
}

void TestLSTMNode()
{
    using ElementType = double;
    using namespace ell::predictors;
    using namespace ell::predictors::neural;

    using ConstVectorReference = math::ConstColumnVectorReference<ElementType>;

    // Precomputed weights created by GenerateLSGMTest.py
    const double x_t[] = { -0.43309685587882996, -1.130162000656128, -0.0909687802195549, -0.4426236152648926, -0.11253798007965088, 1.380946159362793, 0.8205883502960205, -0.05229336395859718, -2.0741474628448486, 0.3278883993625641 };
    const double w_i[] = { 0.6511253118515015, 2.0175061225891113, 0.19230778515338898, 0.1317894458770752, 0.1585828810930252, -0.07296566665172577, -1.2601498365402222, 0.2603142559528351, -1.2849689722061157, -0.1481228917837143, -0.45136135816574097, 1.3159385919570923, 0.8316872119903564, -0.1754256933927536, 1.6749238967895508, 0.3764311969280243, -0.4301822781562805, -0.4683826267719269, -1.028944730758667, 1.1821211576461792, -0.86643385887146, 0.27833691239356995, 1.5544127225875854, -0.14951558411121368, -0.5922301411628723, -1.1712491512298584, 0.8437608480453491, -2.096470355987549, -2.362649917602539, 1.0517401695251465, 0.9789304137229919, 1.1866481304168701, 2.19455885887146, -1.5573855638504028, -1.4857137203216553, -0.7441024780273438, 1.209663987159729, -0.6159052848815918, -0.7103704810142517, -1.1879568099975586, -0.10851240158081055, 0.02951694466173649, -1.0808030366897583, 0.47296342253685, 0.8303858637809753, 1.3805022239685059, 0.8054493069648743, 0.0033771514426916838, 0.24905401468276978, 1.06377112865448, 0.6644095182418823, -0.3728560507297516, 1.8050099611282349, -1.4735825061798096, -0.37865114212036133, 0.7331714034080505, -0.5518970489501953, 0.7309926748275757, 1.2089462280273438, 0.7720244526863098, -0.5284073948860168, -0.060446847230196, -0.7342194318771362, -0.8396680355072021, -0.8295539021492004, 0.46687257289886475, 1.3178989887237549, -0.4714529514312744, -0.36685019731521606, -0.3204497694969177, 0.5232697129249573, 1.0299279689788818, 0.1968117356300354, -0.6680045127868652, -0.6849696636199951, 0.25415563583374023, 1.4508898258209229, -1.0641292333602905, 0.02521374635398388, -0.828795850276947, -0.1840386986732483, -0.0035157897509634495, -0.5138935446739197, 1.7330914735794067, 0.563833475112915, -0.3131422698497772, -1.2656675577163696, 0.3594684898853302, 0.46927720308303833, 0.4585743844509125, 1.3152965307235718, -0.5655897259712219, 0.32646334171295166, -0.8774658441543579, -0.9450925588607788, -0.9361690282821655, 0.31880906224250793, -0.08327312022447586, -0.71200031042099, 0.41611140966415405, 0.5446979999542236, 1.659063458442688, -0.8110974431037903, 0.4108612537384033, 0.6116517186164856, 2.3877499103546143, -0.646440863609314, 0.7542923092842102, -1.3344032764434814, -0.6392279267311096, -2.4335315227508545, 1.1413220167160034, 0.26730069518089294, -1.2412208318710327, 0.6426215767860413, 1.486965537071228, 0.8783390522003174, -1.8383007049560547, -0.8426035642623901, 0.8142386078834534, 0.14272619783878326, 0.814831554889679, -1.1428523063659668, 0.381057471036911, 0.9837798476219177, -0.580453097820282, 0.6704853177070618, 0.4560665190219879, 0.7043464183807373, 0.02900850772857666, -0.7768147587776184, 0.723414421081543, 0.6402895450592041, -1.935476541519165, 0.5388496518135071, 0.8460615873336792, -0.3884362578392029, -0.28912603855133057, -1.328397512435913, -0.00650961697101593, -1.1806731224060059, -1.1872210502624512, -0.11407879739999771, -1.6614559888839722, -0.13863477110862732, 0.15498283505439758, -3.231074810028076, -2.042465925216675, -1.1988264322280884, 0.5179974436759949, 1.838789701461792, -0.4401094317436218, 0.8016218543052673, -0.9823786020278931, -0.14128735661506653, 0.9628021121025085, 0.5577496290206909, -0.41265949606895447, -0.21254649758338928, -0.6768006682395935 };
    const double w_h[] = { -1.554459571838379, -1.1459623575210571, -0.4674321413040161, -1.57246994972229, 1.5664836168289185, 0.07712340354919434, -0.9089983701705933, 0.49248918890953064, 0.022358577698469162, -0.6295645236968994, 0.21086902916431427, -0.3291659355163574, 0.3653954863548279, 1.7737113237380981, -0.8169741630554199, -0.3816412389278412, -0.041717443615198135, 1.1712921857833862, -0.802385151386261, 0.7553011178970337, 1.6707643270492554, 1.0745068788528442, -0.4649391770362854, 0.5538312792778015, -0.6652563810348511, 0.003600509138777852, 0.17086449265480042, -0.9902800917625427, -0.2772578299045563, -0.009141645394265652, 0.5040040016174316, -1.7911834716796875, -1.5239406824111938, 1.4718868732452393, 0.45045509934425354, 0.06189517304301262, -1.6212736368179321, -0.536480724811554, -1.7301037311553955, -1.7541351318359375, 0.36702921986579895, -0.1488613337278366, 0.4395482838153839, -0.32868295907974243, 2.0901262760162354, 0.8486853837966919, 0.698390543460846, -1.2518192529678345, -1.113065242767334, -0.9914770722389221, -0.1959114521741867, -0.2975504398345947, 1.6278289556503296, 0.18092399835586548, -0.8431065678596497, 1.0035691261291504, 0.822422981262207, -1.7101740837097168, 1.2035664319992065, 0.9784857034683228, -0.7354406118392944, 1.3279316425323486, 0.0412302128970623, 0.028288977220654488 };
    const double b_i[] = { -0.3474038541316986, -2.0073912143707275, -1.3713533878326416, -0.7574060559272766, -1.8490254878997803, -0.20404112339019775, 0.44390836358070374, -0.4153674244880676, 0.8775933980941772, 0.3997775614261627, -0.2047869712114334, 0.22270329296588898, -0.2717428207397461, 1.1116859912872314, -0.12611441314220428, -0.05235645920038223 };
    const double b_h[] = { -0.6316860318183899, 0.771199107170105, 0.30606597661972046, 0.597480058670044, -0.7204434275627136, -0.5604809522628784, 0.9371622800827026, -0.1999434381723404, 2.272083044052124, 0.6642170548439026, -0.780116856098175, 0.8419560194015503, -0.9387738108634949, -0.7556354403495789, 0.05611182004213333, 1.7009730339050293 };
    // Expected output
    const double h_1[] = { 0.00022874458227306604, 0.445499062538147, 0.6932798027992249, 0.41498056054115295 };
    const double h_2[] = { 0.00022241008991841227, 0.09827625751495361, 0.9197579026222229, 0.5392394661903381 };
    const double h_3[] = { 0.00028709517209790647, -0.12299935519695282, 0.9793951511383057, 0.4173615574836731 };
    const double* h_t[] = { h_1, h_2, h_3 };

    size_t hiddenSize = 4;
    double epsilon = 1e-5;

    ConstVectorReference input(x_t, sizeof(x_t) / sizeof(double));
    size_t inputSize = input.Size();
    ConstVectorReference inputWeights(w_i, sizeof(w_i) / sizeof(double));
    ConstVectorReference hiddenWeights(w_h, sizeof(w_h) / sizeof(double));
    ConstVectorReference inputBias(b_i, sizeof(b_i) / sizeof(double));
    ConstVectorReference hiddenBias(b_h, sizeof(b_h) / sizeof(double));

    // Create model
    model::Model model;
    auto inputNode = model.AddNode<model::InputNode<ElementType>>(inputSize);
    auto resetTriggerNode = model.AddNode<nodes::ConstantNode<int>>(0);
    auto inputWeightsNode = model.AddNode<nodes::ConstantNode<ElementType>>(inputWeights.ToArray());
    auto hiddenWeightsNode = model.AddNode<nodes::ConstantNode<ElementType>>(hiddenWeights.ToArray());
    auto inputBiasNode = model.AddNode<nodes::ConstantNode<ElementType>>(inputBias.ToArray());
    auto hiddenBiasNode = model.AddNode<nodes::ConstantNode<ElementType>>(hiddenBias.ToArray());
    auto lstmNode = model.AddNode<nodes::LSTMNode<ElementType>>(inputNode->output, resetTriggerNode->output, hiddenSize, inputWeightsNode->output, hiddenWeightsNode->output, inputBiasNode->output, hiddenBiasNode->output, ell::predictors::neural::Activation<ElementType>(new ell::predictors::neural::TanhActivation<ElementType>()), ell::predictors::neural::Activation<ElementType>(new ell::predictors::neural::SigmoidActivation<ElementType>()));
    auto map = model::Map(model, { { "input", inputNode } }, { { "output", lstmNode->output } });

    TestWithSerialization(map, "TestLSTMNode", [&](model::Map& map, int iteration) {
        // Compile model
        model::MapCompilerOptions settings;
        settings.compilerSettings.useBlas = true;
        model::ModelOptimizerOptions optimizerOptions;
        model::IRMapCompiler compiler(settings, optimizerOptions);
        auto compiledMap = compiler.Compile(map);
        auto name = lstmNode->GetRuntimeTypeName();

        std::vector<std::vector<ElementType>> signal = { input.ToArray() };
        map.SetInputValue(0, signal[0]);
        std::vector<ElementType> computedResult = map.ComputeOutput<ElementType>(0);
        if (IsEqual(computedResult, std::vector<ElementType>(computedResult.size()), static_cast<double>(epsilon)))
        {
            std::cout << "#############################################################################################\n";
            std::cout << "### bugbug: weird case where first compute randomly fails ... see work item 1918 \n";
            std::cout << "#############################################################################################\n";
        }
        else
        {
            map.Reset(); // test that model reset works.
        }

        // test statefulness of the LSTM node
        for (size_t i = 0; i < 3; i++)
        {
            ConstVectorReference expectedOutput(h_t[i], sizeof(h_1) / sizeof(double));

            std::string message = name + utilities::FormatString(" iteration %d row %d", iteration, i);

            // compare computed vs. compiled output
            VerifyCompiledOutputAndResult<ElementType, ElementType>(map, compiledMap, signal, { expectedOutput.ToArray() }, message);
        }
    });
}

//
// Main driver function to call all the tests
//
void TestDSPNodes(const std::string& path)
{
    TestRNNNode();
    TestGRUNode();
    TestLSTMNode();

    //
    // Compute tests
    //
    TestDelayNodeCompute();
    TestDTWDistanceNodeCompute();
    TestFFTNodeCompute(32, 32);
    TestFFTNodeCompute(40, 64);
    TestFFTNodeCompute(40, 32);
    TestFFTNodeCompute(100, 0); // test default constructor with no fftSize parameter

    //
    // Combined tests
    //
    TestIIRFilterNode1<float>();
    TestIIRFilterNode2<float>();
    TestIIRFilterNode3<float>();
    TestIIRFilterNode4<float>();

    TestMelFilterBankNode<float>();
    TestMelFilterBankNode<double>();

    TestBufferNode<float>();

    TestConvolutionNodeCompile<float>(dsp::ConvolutionMethodOption::simple);
    // TestConvolutionNodeCompile<float>(dsp::ConvolutionMethodOption::diagonal); // ERROR: diagonal test currently broken
    TestConvolutionNodeCompile<float>(dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompile<float>(dsp::ConvolutionMethodOption::winograd);

    // Test simple convolution
    TestConvolutionNodeCompileVsReference<float>({ 2, 2, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 2, 3, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 3, 2, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 3, 3, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 2, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 2 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 4, 5, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 5, 4, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 2 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 2, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 5, 15, 4 }, { 7, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 8, 8, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 32, 32, 8 }, { 8, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 64, 64, 8 }, { 8, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 120, 80, 8 }, { 16, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 120, 80, 8 }, { 16, 3, 3, 0 }, 2, dsp::ConvolutionMethodOption::simple);

    // Test unrolled convolution
    TestConvolutionNodeCompileVsReference<float>({ 2, 2, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 2, 3, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 3, 2, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 3, 3, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 2, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 2 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 4, 5, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 5, 4, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 2 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 2, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 5, 15, 4 }, { 7, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 8, 8, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 32, 32, 8 }, { 8, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 64, 64, 8 }, { 8, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 120, 80, 8 }, { 16, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::unrolled);
    TestConvolutionNodeCompileVsReference<float>({ 120, 80, 8 }, { 16, 3, 3, 0 }, 2, dsp::ConvolutionMethodOption::unrolled);

    // Test Winograd convolution with tile size 2
    TestConvolutionNodeCompileVsReference<float>({ 2, 2, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 2, 3, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 3, 2, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 3, 3, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 2, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 2 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 5, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 4, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 2 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 2, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 15, 4 }, { 7, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 8, 8, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 32, 32, 8 }, { 8, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 64, 64, 8 }, { 8, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 120, 80, 8 }, { 16, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::tilesFirst });
    // TestConvolutionNodeCompileVsReference<float>({120, 80, 8}, {16, 3, 3, 0}, 2, dsp::ConvolutionMethodOption::winograd, {2, dsp::WinogradFilterOrder::tilesFirst}); // Commented-out because Winograd doesn't support non-1 stride

    TestConvolutionNodeCompileVsReference<float>({ 2, 2, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 2, 3, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 3, 2, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 3, 3, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 2, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 2 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 5, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 4, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 2 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 2, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 15, 4 }, { 7, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 8, 8, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 32, 32, 8 }, { 8, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 64, 64, 8 }, { 8, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 120, 80, 8 }, { 16, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });

    // Test Winograd convolution with tile size 4
    TestConvolutionNodeCompileVsReference<float>({ 2, 2, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 2, 3, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 3, 2, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 3, 3, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 2, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 2 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 5, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 4, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 2 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 2, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 15, 4 }, { 7, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 8, 8, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 32, 32, 8 }, { 8, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 64, 64, 8 }, { 8, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    TestConvolutionNodeCompileVsReference<float>({ 120, 80, 8 }, { 16, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::tilesFirst });
    // TestConvolutionNodeCompileVsReference<float>({120, 80, 8}, {16, 3, 3, 0}, 2, dsp::ConvolutionMethodOption::winograd, {4, dsp::WinogradFilterOrder::tilesFirst}); // Commented-out because Winograd doesn't support non-1 stride

    TestConvolutionNodeCompileVsReference<float>({ 2, 2, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 2, 3, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 3, 2, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 3, 3, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 2, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 2 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 5, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 4, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 2 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 2, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 15, 4 }, { 7, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 8, 8, 1 }, { 1, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 32, 32, 8 }, { 8, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 64, 64, 8 }, { 8, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 120, 80, 8 }, { 16, 3, 3, 0 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });

    //
    // Depthwise-separable convolution tests
    //
    TestConvolutionNodeCompileVsReference<float>({ 2, 2, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 3, 2, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 2, 3, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 3, 3, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 2 }, { 2, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 4, 5, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 5, 4, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 2 }, { 2, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    // Non-square inputs not supported in simple depthwise separable yet
    //TestConvolutionNodeCompileVsReference<float>({ 5, 15, 4 }, { 4, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 8, 8, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 32, 32, 8 }, { 8, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    TestConvolutionNodeCompileVsReference<float>({ 64, 64, 8 }, { 8, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);
    // Non-square inputs not supported in simple depthwise separable yet
    //TestConvolutionNodeCompileVsReference<float>({ 120, 80, 8 }, { 8, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::simple);

    TestConvolutionNodeCompileVsReference<float>({ 2, 2, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 3, 2, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 2, 3, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 3, 3, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 2 }, { 2, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 5, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 4, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 2 }, { 2, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 15, 4 }, { 4, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 8, 8, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 32, 32, 8 }, { 8, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 64, 64, 8 }, { 8, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 120, 80, 8 }, { 8, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 2, dsp::WinogradFilterOrder::filtersFirst });

    TestConvolutionNodeCompileVsReference<float>({ 2, 2, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 2, 3, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 3, 2, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 3, 3, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 4, 2 }, { 2, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 4, 5, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 4, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 2 }, { 2, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 5, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 5, 15, 4 }, { 4, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 8, 8, 1 }, { 1, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 32, 32, 8 }, { 8, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 64, 64, 8 }, { 8, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
    TestConvolutionNodeCompileVsReference<float>({ 120, 80, 8 }, { 8, 3, 3, 1 }, 1, dsp::ConvolutionMethodOption::winograd, { 4, dsp::WinogradFilterOrder::filtersFirst });
}
