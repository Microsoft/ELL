////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ConvolutionalLayer.h (neural)
//  Authors:  Byron Changuion
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once
#include "Layer.h"

#include <dsp/include/Convolution.h>

#include <math/include/Matrix.h>

namespace ell
{
namespace predictors
{
    namespace neural
    {
        /// <summary> The method for performing convolutions. </summary>
        enum class ConvolutionMethod : int
        {
            /// <summary> Allow the function to choose the algorithm to use. </summary>
            automatic = 0,
            /// <summary> A different method of doing convolution which avoids reshaping the input, and uses gemm on smaller matrices with diagonal sums to create output. </summary>
            diagonal,
            /// <summary> A simple, straightforward nested-loop implementation. </summary>
            simple,
            /// <summary> An implementation that performs convolution with fewer arithmetic operations. </summary>
            winograd,
            /// <summary> Normal method of doing convolution via reshaping input into columns and performing a gemm operation. </summary>
            unrolled
        };

        /// <summary> Specifies the hyper parameters of the convolutional layer. </summary>
        struct ConvolutionalParameters
        {
            /// <summary> Width and height of the receptive field that is slid over the input. </summary>
            size_t receptiveField;

            /// <summary> Number of elements to move/jump when sliding over the input. Typically this is 1 to 3. </summary>
            size_t stride;

            /// <summary> Method for doing convolution. </summary>
            ConvolutionMethod method;

            /// <summary> Number of filters to batch at a time when using the Diagonal method. </summary>
            size_t numFiltersAtATime;
        };

        /// <summary> A layer in a neural network that implements a fully connected layer, meaning all nodes in this layer are connected to all
        /// outputs of the previous layer (which are the inputs of this layer). </summary>
        template <typename ElementType>
        class ConvolutionalLayer : public Layer<ElementType>
        {
        public:
            using LayerParameters = typename Layer<ElementType>::LayerParameters;
            using MatrixType = typename Layer<ElementType>::MatrixType;
            using TensorType = typename Layer<ElementType>::TensorType;
            using ConstTensorReferenceType = typename Layer<ElementType>::ConstTensorReferenceType;
            using Layer<ElementType>::GetOutputMinusPadding;
            using Layer<ElementType>::NumOutputRowsMinusPadding;
            using Layer<ElementType>::NumOutputColumnsMinusPadding;
            using Layer<ElementType>::NumOutputChannels;

            /// <summary> Instantiates an instance of a convolutional layer. </summary>
            ///
            /// <param name="layerParameters"> The parameters common to every layer. </param>
            /// <param name="convolutionalParameters"> The hyperparameters for this convolutional layer. </param>
            /// <param name="weights"> The set of weights to apply. </param>
            ConvolutionalLayer(const LayerParameters& layerParameters, const ConvolutionalParameters& convolutionalParameters, TensorType weights);

            /// <summary> Instantiates a blank instance. Used for unarchiving purposes only. </summary>
            ConvolutionalLayer() :
                _weights(math::IntegerTriplet{ 0, 0, 0 }),
                _outputMatrix(0, 0) {}

            /// <summary> Feeds the input forward through the layer and returns a reference to the output. </summary>
            void Compute() override;

            /// <summary> Indicates the kind of layer. </summary>
            ///
            /// <returns> An enum indicating the layer type. </returns>
            LayerType GetLayerType() const override { return LayerType::convolution; }

            /// <summary> Get the parameters used to control convolution. </summary>
            ///
            /// <returns> A ConvolutionalParameters struct. </returns>
            const ConvolutionalParameters& GetConvolutionalParameters() const { return _convolutionalParameters; }

            /// <summary> Get the weights for the convolution filters. </summary>
            ///
            /// <returns> The weights, packed into a Tensor. </returns>
            const TensorType& GetWeights() const { return _weights; }

            /// <summary> Gets the name of this type (for serialization). </summary>
            ///
            /// <returns> The name of this type. </returns>
            static std::string GetTypeName() { return utilities::GetCompositeTypeName<ElementType>("ConvolutionalLayer"); }

            /// <summary> Gets the name of this type (for serialization). </summary>
            ///
            /// <returns> The name of this type. </returns>
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }

            /// <summary> Checks if this layer represents a depthwise-separable convolution. </summary>
            ///
            /// <returns> `true` if this layer represents a depthwise-separable convolution. </returns>
            bool IsDepthwiseSeparable() const;

        protected:
            void WriteToArchive(utilities::Archiver& archiver) const override;
            void ReadFromArchive(utilities::Unarchiver& archiver) override;

        private:
            // Fills a matrix (backed by the array outputMatrix) where the columns the set of input values corresponding to a filter, stretched into a vector.
            // The number of columns is equal to the number of locations that a filter is slide over the input tensor.
            void ReceptiveFieldToColumns(ConstTensorReferenceType input, MatrixType& shapedInput);
            void ComputeWeightsMatrix();
            void InitializeIOMatrices();
            void Validate() const;
            void CalculateConvolutionMethod();
            void ComputeSimpleMethod();
            void ComputeUnrolledMethod();
            void ComputeWinogradMethod();
            void ComputeDiagonalMethod();
            void ComputeDepthwiseSeparable();

            using Layer<ElementType>::_layerParameters;
            using Layer<ElementType>::_output;

            ConvolutionalParameters _convolutionalParameters;
            TensorType _weights;

            MatrixType _outputMatrix;

            ConvolutionMethod _originalConvolutionMethod;
        };

    } // namespace neural
} // namespace predictors
} // namespace ell

#pragma region implementation

#include <dsp/include/SimpleConvolution.h>
#include <dsp/include/UnrolledConvolution.h>
#include <dsp/include/WinogradConvolution.h>

namespace ell
{
namespace predictors
{
    namespace neural
    {
        template <typename ElementType>
        ConvolutionalLayer<ElementType>::ConvolutionalLayer(const LayerParameters& layerParameters, const ConvolutionalParameters& convolutionalParameters, TensorType weights) :
            Layer<ElementType>(layerParameters),
            _convolutionalParameters(convolutionalParameters),
            _weights(std::move(weights)),
            _outputMatrix{ NumOutputChannels(), NumOutputRowsMinusPadding() * NumOutputColumnsMinusPadding() },
            _originalConvolutionMethod(convolutionalParameters.method)
        {
            Validate();
            CalculateConvolutionMethod();
        }

        template <typename ElementType>
        void ConvolutionalLayer<ElementType>::Compute()
        {
            if (!IsDepthwiseSeparable())
            {
                switch (_convolutionalParameters.method)
                {
                case ConvolutionMethod::simple:
                    ComputeSimpleMethod();
                    break;

                case ConvolutionMethod::unrolled:
                    ComputeUnrolledMethod();
                    break;

                case ConvolutionMethod::winograd:
                    ComputeWinogradMethod();
                    break;

                case ConvolutionMethod::diagonal:
                    ComputeDiagonalMethod();
                    break;

                default:
                    throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, "Convolution method not supported");
                }
            }
            else
            {
                ComputeDepthwiseSeparable();
            }
        }

        template <typename ElementType>
        void ConvolutionalLayer<ElementType>::ComputeSimpleMethod()
        {
            auto output = GetOutputMinusPadding();
            auto& input = _layerParameters.input;
            auto stride = static_cast<int>(_convolutionalParameters.stride);
            const int numFilters = static_cast<int>(output.NumChannels());
            dsp::Convolve2DSimple(input, _weights, numFilters, stride, output);
        }

        template <typename ElementType>
        void ConvolutionalLayer<ElementType>::ComputeUnrolledMethod()
        {
            auto output = GetOutputMinusPadding();
            auto& input = _layerParameters.input;
            auto stride = static_cast<int>(_convolutionalParameters.stride);
            const int numFilters = static_cast<int>(output.NumChannels());
            auto result = dsp::Convolve2DUnrolled(input, _weights, numFilters, stride);
            output.CopyFrom(result);
        }

        template <typename ElementType>
        void ConvolutionalLayer<ElementType>::ComputeWinogradMethod()
        {
            auto output = GetOutputMinusPadding();
            auto& input = _layerParameters.input;
            const int numFilters = static_cast<int>(output.NumChannels());
            auto result = dsp::Convolve2DWinograd(input, _weights, numFilters);
            output.CopyFrom(result);
        }

        template <typename ElementType>
        void ConvolutionalLayer<ElementType>::ComputeDiagonalMethod()
        {
            auto output = GetOutputMinusPadding();
            auto& input = _layerParameters.input;

            // Flatten the input
            auto inputMatrix = input.ReferenceAsMatrix();

            const size_t depth = input.NumChannels();
            const size_t kt = _convolutionalParameters.receptiveField * depth;
            const size_t paddingSize = _layerParameters.inputPaddingParameters.paddingSize;
            const size_t numConvolutions = (inputMatrix.NumColumns() - kt) / depth + 1;
            const size_t numFiltersAtAtime = _convolutionalParameters.numFiltersAtATime;
            const size_t numFilters = _layerParameters.outputShape.NumChannels();
            auto weightsMatrix = _weights.ReferenceAsMatrix().Transpose();

            for (size_t j = 0; j < numConvolutions; j++)
            {
                // Get the sub matrix for Vj
                auto Vj = inputMatrix.GetSubMatrix(0, j * depth, inputMatrix.NumRows(), kt);

                for (size_t filterStart = 0; filterStart < numFilters; filterStart += numFiltersAtAtime)
                {
                    size_t numFiltersToUse = std::min(numFiltersAtAtime, numFilters - filterStart);

                    auto Wl = weightsMatrix.GetSubMatrix(0, filterStart * _convolutionalParameters.receptiveField, weightsMatrix.NumRows(), numFiltersToUse * _convolutionalParameters.receptiveField);

                    MatrixType A(Vj.NumRows(), _convolutionalParameters.receptiveField * numFiltersToUse);

                    math::MultiplyScaleAddUpdate(static_cast<ElementType>(1.0), Vj, Wl, static_cast<ElementType>(0.0), A);

                    for (size_t l = 0; l < numFiltersToUse; l++)
                    {
                        for (size_t row = 0; row < (A.NumRows() - 2 * paddingSize); row++)
                        {
                            ElementType sum = 0.0;
                            for (size_t diagonal = 0; diagonal < _convolutionalParameters.receptiveField; diagonal++)
                            {
                                sum += A(row + diagonal, l * _convolutionalParameters.receptiveField + diagonal);
                            }
                            output(row, j, filterStart + l) = sum;
                        }
                    }
                }
            }
        }

        template <typename ElementType>
        void ConvolutionalLayer<ElementType>::ComputeDepthwiseSeparable()
        {
            auto output = GetOutputMinusPadding();
            auto& input = _layerParameters.input;
            auto stride = static_cast<int>(_convolutionalParameters.stride);
            const int numFilters = 1;
            const size_t numInputRows = input.NumRows();
            const size_t numInputColumns = input.NumColumns();
            const size_t numOutputRows = output.NumRows();
            const size_t numOutputColumns = output.NumColumns();
            const size_t filterRows = _convolutionalParameters.receptiveField;

            for (size_t channel = 0; channel < output.NumChannels(); ++channel)
            {
                using TensorType = typename Layer<ElementType>::TensorType;
                using TensorReferenceType = typename Layer<ElementType>::TensorReferenceType;

                TensorType weights(_weights.GetSubTensor(filterRows * channel, 0, 0, filterRows, filterRows, 1));
                const auto& inputChannelTensor = input.GetSubTensor(0, 0, channel, numInputRows, numInputColumns, 1);
                TensorReferenceType outputChannelTensor = output.GetSubTensor(0, 0, channel, numOutputRows, numOutputColumns, 1);

                switch (_convolutionalParameters.method)
                {
                case ConvolutionMethod::simple: {
                    auto result = dsp::Convolve2DSimpleDepthwiseSeparable(inputChannelTensor, weights, numFilters, stride);
                    outputChannelTensor.CopyFrom(result);
                }
                break;
                case ConvolutionMethod::unrolled: {
                    auto result = dsp::Convolve2DUnrolled(inputChannelTensor, weights, numFilters, stride);
                    outputChannelTensor.CopyFrom(result);
                }
                break;
                case ConvolutionMethod::winograd: {
                    auto result = dsp::Convolve2DWinogradDepthwiseSeparable(inputChannelTensor, weights, numFilters); // Stide of 1 is assumed
                    outputChannelTensor.CopyFrom(result);
                }
                break;
                default:
                    throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, "Convolution method not supported for depthwise separable convolution");
                }
            }
        }

        template <typename ElementType>
        void ConvolutionalLayer<ElementType>::WriteToArchive(utilities::Archiver& archiver) const
        {
            Layer<ElementType>::WriteToArchive(archiver);

            archiver["receptiveField"] << _convolutionalParameters.receptiveField;
            archiver["stride"] << _convolutionalParameters.stride;
            archiver["method"] << static_cast<int>(_originalConvolutionMethod);
            archiver["numFiltersAtATime"] << static_cast<int>(_convolutionalParameters.numFiltersAtATime);

            math::TensorArchiver::Write(_weights, "weights", archiver);
        }

        template <typename ElementType>
        void ConvolutionalLayer<ElementType>::ReadFromArchive(utilities::Unarchiver& archiver)
        {
            Layer<ElementType>::ReadFromArchive(archiver);

            archiver["receptiveField"] >> _convolutionalParameters.receptiveField;
            archiver["stride"] >> _convolutionalParameters.stride;
            int method;
            archiver["method"] >> method;
            _originalConvolutionMethod = static_cast<ConvolutionMethod>(method);
            int numFilters;
            archiver["numFiltersAtATime"] >> numFilters;
            _convolutionalParameters.numFiltersAtATime = static_cast<size_t>(numFilters);

            math::TensorArchiver::Read(_weights, "weights", archiver);
            CalculateConvolutionMethod();
            InitializeIOMatrices();
        }

        template <typename ElementType>
        void ConvolutionalLayer<ElementType>::InitializeIOMatrices()
        {
            _outputMatrix = { NumOutputChannels(), NumOutputRowsMinusPadding() * NumOutputColumnsMinusPadding() };
        }

        template <typename ElementType>
        void ConvolutionalLayer<ElementType>::Validate() const
        {
            if (_weights.GetConstDataPointer() == nullptr)
            {
                throw utilities::InputException(utilities::InputExceptionErrors::nullReference, "weights tensor has null data field");
            }

            if (IsDepthwiseSeparable() && (_output.NumChannels() != _layerParameters.input.NumChannels()))
            {
                throw utilities::InputException(utilities::InputExceptionErrors::sizeMismatch, "Input and output channel sizes must match for a depthwise-separable convolutional layer");
            }

            if ((_weights.NumRows() != _output.NumChannels() * _convolutionalParameters.receptiveField) || (_weights.NumColumns() != _convolutionalParameters.receptiveField) || (!IsDepthwiseSeparable() && (_weights.NumChannels() != _layerParameters.input.NumChannels())))
            {
                throw utilities::InputException(utilities::InputExceptionErrors::sizeMismatch, "Weights tensor size mismatch");
            }
        }

        template <typename ElementType>
        void ConvolutionalLayer<ElementType>::CalculateConvolutionMethod()
        {
            _convolutionalParameters.method = _originalConvolutionMethod;
            switch (_convolutionalParameters.method)
            {
            case ConvolutionMethod::automatic:
                _convolutionalParameters.method = IsDepthwiseSeparable() ? ConvolutionMethod::simple : ConvolutionMethod::unrolled;
                break;
            case ConvolutionMethod::simple:
            case ConvolutionMethod::unrolled: // fallthrough
                // do nothing
                break;
            case ConvolutionMethod::diagonal:
                // Verify that we meet the criteria for doing Diagonal method. If not,
                // choose the normal method.
                if ((_convolutionalParameters.receptiveField % 2 == 0) || _convolutionalParameters.stride != 1)
                {
                    _convolutionalParameters.method = IsDepthwiseSeparable() ? ConvolutionMethod::simple : ConvolutionMethod::unrolled;
                }
                break;
            case ConvolutionMethod::winograd:
                // Verify that we meet the criteria for doing Winograd method. If not,
                // choose the normal method.
                if (_convolutionalParameters.stride != 1 || _convolutionalParameters.receptiveField != 3)
                {
                    _convolutionalParameters.method = IsDepthwiseSeparable() ? ConvolutionMethod::simple : ConvolutionMethod::unrolled;
                }
                break;
            }
            if (IsDepthwiseSeparable())
            {
                // Verify we can use a workable method for depthwise separable convolutions.
                if ((_convolutionalParameters.method != ConvolutionMethod::unrolled) && (_convolutionalParameters.method != ConvolutionMethod::simple) && (_convolutionalParameters.method != ConvolutionMethod::winograd))
                {
                    _convolutionalParameters.method = ConvolutionMethod::simple;
                }
            }
        }

        template <typename ElementType>
        bool ConvolutionalLayer<ElementType>::IsDepthwiseSeparable() const
        {
            return (_weights.NumChannels() == 1) && (_layerParameters.input.NumChannels() > 1);
        }
    } // namespace neural
} // namespace predictors
} // namespace ell

#pragma endregion implementation
