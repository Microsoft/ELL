////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     NeuralNetworkPredictorInterface.cpp (interfaces)
//  Authors:  Chris Lovett
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "NeuralNetworkPredictorInterface.h"
#include "MathInterface.h"
#include "ModelBuilderInterface.h"
#include "NeuralLayersInterface.h"

#include <predictors/include/NeuralNetworkPredictor.h>

#include <predictors/neural/include/HardSigmoidActivation.h>
#include <predictors/neural/include/HardTanhActivation.h>
#include <predictors/neural/include/LeakyReLUActivation.h>
#include <predictors/neural/include/MaxPoolingFunction.h>
#include <predictors/neural/include/MeanPoolingFunction.h>
#include <predictors/neural/include/ParametricReLUActivation.h>
#include <predictors/neural/include/ReLUActivation.h>
#include <predictors/neural/include/SigmoidActivation.h>

#include <utilities/include/Exception.h>
#include <utilities/include/Files.h>
#include <utilities/include/JsonArchiver.h>
#include <utilities/include/StringUtil.h>

#include <string>
#include <vector>

using Layer = ell::api::predictors::neural::Layer;
using LayerShape = ell::api::math::TensorShape;
namespace underlying = ell::predictors::neural;
using namespace ELL_API;
using namespace ell::utilities;

namespace ell
{
namespace api
{
    namespace predictors
    {

        class NeuralNetworkPredictorImpl
        {
        public:
            virtual ~NeuralNetworkPredictorImpl() = default;
            virtual std::vector<double> PredictDouble(const std::vector<double>& input) = 0;
            virtual std::vector<float> PredictFloat(const std::vector<float>& input) = 0;
            virtual LayerShape GetInputShape() const = 0;
            virtual LayerShape GetOutputShape() const = 0;
            virtual void RemoveLastLayers(size_t numLayersToRemove) = 0;
        };

        template <typename ElementType>
        class TypedNeuralNetworkPredictorImpl : public NeuralNetworkPredictorImpl
        {
        public:
            using UnderlyingPredictor = ell::predictors::NeuralNetworkPredictor<ElementType>;
            using UnderlyingLayer = typename ell::predictors::neural::Layer<ElementType>;
            using UnderlyingLayers = typename ell::predictors::NeuralNetworkPredictor<ElementType>::Layers;
            using UnderlyingInputParameters = typename ell::predictors::neural::InputLayer<ElementType>::InputParameters;
            using UnderlyingInputLayer = typename ell::predictors::neural::InputLayer<ElementType>;
            using UnderlyingLayerParameters = typename ell::predictors::neural::Layer<ElementType>::LayerParameters;
            using TensorType = typename ell::predictors::neural::Layer<ElementType>::TensorType;

            TypedNeuralNetworkPredictorImpl(const std::vector<Layer*>& layers, ElementType inputScaleFactor = 1.0f);

            std::vector<double> PredictDouble(const std::vector<double>& input) override;
            std::vector<float> PredictFloat(const std::vector<float>& input) override;
            LayerShape GetInputShape() const override;
            LayerShape GetOutputShape() const override;
            void RemoveLastLayers(size_t numLayersToRemove) override;

            // helpers
            template <typename DataType>
            std::vector<DataType> Predict(const std::vector<DataType>& input);

            static void AddLayer(Layer* layer, const std::unique_ptr<UnderlyingInputLayer>& underlyingInputLayer, UnderlyingLayers& underlyingLayers);

            // Specific layer factory functions
            static std::unique_ptr<UnderlyingLayer> CreateActivationLayer(ell::api::predictors::neural::ActivationLayer& layer, const UnderlyingLayerParameters& parameters);

            UnderlyingPredictor& GetUnderlyingPredictor() { return _predictor; }

        private:
            template <typename DerivedLayer>
            static bool LayerIs(Layer* layer);

            template <typename DerivedLayer>
            static auto& LayerAs(Layer* layer);

            UnderlyingPredictor _predictor;
        };

        template <typename ElementType>
        template <typename DerivedLayer>
        bool TypedNeuralNetworkPredictorImpl<ElementType>::LayerIs(Layer* layer)
        {
            return layer->template Is<DerivedLayer>();
        }

        template <typename ElementType>
        template <typename DerivedLayer>
        auto& TypedNeuralNetworkPredictorImpl<ElementType>::LayerAs(Layer* layer)
        {
            if (layer == nullptr)
            {
                throw InputException(InputExceptionErrors::invalidArgument, "Trying to cast null layer");
            }
            return layer->template As<DerivedLayer>();
        }
        //
        // API classes for the neural predictor
        //
        template <typename ElementType>
        TypedNeuralNetworkPredictorImpl<ElementType>::TypedNeuralNetworkPredictorImpl(const std::vector<Layer*>& layers, ElementType inputScaleFactor)
        {
            if (layers.size() > 0)
            {

                auto& parameters = layers.front()->parameters;
                // Construct the input layer
                UnderlyingInputParameters inputParameters = {
                    { static_cast<size_t>(parameters.inputShape.rows - (2 * parameters.inputPaddingParameters.paddingSize)), static_cast<size_t>(parameters.inputShape.columns - (2 * parameters.inputPaddingParameters.paddingSize)), static_cast<size_t>(parameters.inputShape.channels) },
                    underlying::NoPadding(),
                    { static_cast<size_t>(parameters.inputShape.rows), static_cast<size_t>(parameters.inputShape.columns), static_cast<size_t>(parameters.inputShape.channels) },
                    parameters.inputPaddingParameters,
                    inputScaleFactor
                };
                auto inputLayer = std::make_unique<underlying::InputLayer<ElementType>>(inputParameters);

                UnderlyingLayers underlyingLayers;

                // Construct up the layers
                for (size_t i = 0; i < layers.size(); i++)
                {
                    AddLayer(layers[i], inputLayer, underlyingLayers);
                }

                // Create the predictor
                _predictor = UnderlyingPredictor(std::move(inputLayer), std::move(underlyingLayers));
            }
            else
            {
                throw InputException(InputExceptionErrors::invalidArgument, "'layers' parameters does not contain any neural network layers");
            }
        }

        template <typename ElementType>
        ell::predictors::neural::Activation<ElementType> ell::api::predictors::neural::ActivationLayer::CreateActivation(ell::api::predictors::neural::ActivationType type)
        {
            std::unique_ptr<ell::predictors::neural::ActivationImpl<ElementType>> activation;
            switch (type)
            {
            case ell::api::predictors::neural::ActivationType::relu:
                activation = std::make_unique<underlying::ReLUActivation<ElementType>>();
                break;
            case ell::api::predictors::neural::ActivationType::leaky:
                activation = std::make_unique<underlying::LeakyReLUActivation<ElementType>>();
                break;
            case ell::api::predictors::neural::ActivationType::hardSigmoid:
                activation = std::make_unique<underlying::HardSigmoidActivation<ElementType>>();
                break;
            case ell::api::predictors::neural::ActivationType::hardTanh:
                activation = std::make_unique<underlying::HardTanhActivation<ElementType>>();
                break;
            case ell::api::predictors::neural::ActivationType::sigmoid:
                activation = std::make_unique<underlying::SigmoidActivation<ElementType>>();
                break;
            case ell::api::predictors::neural::ActivationType::tanh:
                activation = std::make_unique<underlying::TanhActivation<ElementType>>();
                break;
            default:
                throw InputException(InputExceptionErrors::invalidArgument,
                                     FormatString("Encountered unsupported activation type in neural network predictor: %d", static_cast<int>(type)));
            }
            return ell::predictors::neural::Activation<ElementType>(activation);
        }

        //
        // CreateActivationLayer
        //
        template <typename ElementType>
        std::unique_ptr<underlying::Layer<ElementType>> TypedNeuralNetworkPredictorImpl<ElementType>::CreateActivationLayer(neural::ActivationLayer& layer, const UnderlyingLayerParameters& parameters)
        {
            using TensorType = typename underlying::Layer<ElementType>::TensorType;
            using ActivationImplType = ell::predictors::neural::ActivationImpl<ElementType>;

            ell::predictors::neural::Activation<ElementType> activation;
            switch (layer.activation)
            {
            case neural::ActivationType::relu:
            case neural::ActivationType::hardSigmoid:
            case neural::ActivationType::hardTanh:
            case neural::ActivationType::sigmoid:
            case neural::ActivationType::tanh:
                activation = neural::ActivationLayer::CreateActivation<ElementType>(layer.activation);
                break;
            case neural::ActivationType::leaky: {
                ActivationImplType* implementation = nullptr;
                if (LayerIs<neural::LeakyReLUActivationLayer>(&layer))
                {
                    auto& leakyReluApiLayer = LayerAs<neural::LeakyReLUActivationLayer>(&layer);
                    implementation = new underlying::LeakyReLUActivation<ElementType>(static_cast<ElementType>(leakyReluApiLayer._alpha));
                }
                else
                {
                    implementation = new underlying::LeakyReLUActivation<ElementType>();
                }
                activation = ell::predictors::neural::Activation<ElementType>(implementation);
                break;
            }
            case neural::ActivationType::prelu: {
                auto& preluApiLayer = LayerAs<neural::PReLUActivationLayer>(&layer);
                TensorType alpha(preluApiLayer.alpha.shape.rows, preluApiLayer.alpha.shape.columns, preluApiLayer.alpha.shape.channels, CastVector<ElementType>(preluApiLayer.alpha.data));
                activation = ell::predictors::neural::Activation<ElementType>(new underlying::ParametricReLUActivation<ElementType>(alpha));
                break;
            }
            default:
                throw InputException(InputExceptionErrors::invalidArgument,
                                     FormatString("Encountered unsupported activation type in neural network predictor: %d", static_cast<int>(layer.activation)));
            }

            return std::make_unique<underlying::ActivationLayer<ElementType>>(parameters, activation);
        }

        template <typename ElementType>
        std::vector<double> TypedNeuralNetworkPredictorImpl<ElementType>::PredictDouble(const std::vector<double>& input)
        {
            if (typeid(ElementType) == typeid(double))
            {
                // types match, yay
                return Predict<double>(input);
            }
            else
            {
                // need to cast.
                return Predict<double>(CastVector<double>(input));
            }
        }

        template <typename ElementType>
        std::vector<float> TypedNeuralNetworkPredictorImpl<ElementType>::PredictFloat(const std::vector<float>& input)
        {
            if (typeid(ElementType) == typeid(float))
            {
                // types match, yay
                return Predict<float>(input);
            }
            else
            {
                // need to cast.
                return Predict<float>(CastVector<float>(input));
            }
        }

        template <typename ElementType>
        template <typename DataType>
        std::vector<DataType> TypedNeuralNetworkPredictorImpl<ElementType>::Predict(const std::vector<DataType>& input)
        {
            std::vector<DataType> result;
            // Call the underlying predictor with the specified input.
            // Note that we return a copy and not a reference, since we cannot control what the
            // api caller is going to do with it.
            if (typeid(ElementType) == typeid(DataType))
            {
                result = input;
            }
            else
            {
                result = CastVector<DataType>(_predictor.Predict(CastVector<ElementType>(input)));
            }
            return result;
        }

        template <typename ElementType>
        void TypedNeuralNetworkPredictorImpl<ElementType>::RemoveLastLayers(size_t numLayersToRemove)
        {
            _predictor.RemoveLastLayers(numLayersToRemove);
        }

        template <typename ElementType>
        LayerShape TypedNeuralNetworkPredictorImpl<ElementType>::GetInputShape() const
        {
            auto shape = _predictor.GetInputShape();
            return ell::api::math::TensorShape::FromMathTensorShape(shape);
        }

        template <typename ElementType>
        LayerShape TypedNeuralNetworkPredictorImpl<ElementType>::GetOutputShape() const
        {
            auto shape = _predictor.GetOutputShape();
            return ell::api::math::TensorShape::FromMathTensorShape(shape);
        }

        template <typename ElementType>
        void TypedNeuralNetworkPredictorImpl<ElementType>::AddLayer(Layer* layer, const std::unique_ptr<UnderlyingInputLayer>& underlyingInputLayer, UnderlyingLayers& underlyingLayers)
        {
            using UnderlyingLayerParameters = typename ell::predictors::neural::Layer<ElementType>::LayerParameters;
            using TensorType = typename underlying::Layer<ElementType>::TensorType;
            if (layer != nullptr)
            {
                // Set the layer parameters. Note that if this is the first layer, we set the input reference to the output of the InputLayer.
                // Otherwise, we set it to the output of the last layer.
                UnderlyingLayerParameters parameters = {
                    ((underlyingLayers.size() > 0) ? underlyingLayers.back()->GetOutput() : underlyingInputLayer->GetOutput()),
                    layer->parameters.inputPaddingParameters,
                    { static_cast<size_t>(layer->parameters.outputShape.rows), static_cast<size_t>(layer->parameters.outputShape.columns), static_cast<size_t>(layer->parameters.outputShape.channels) },
                    layer->parameters.outputPaddingParameters,
                };

                // Instantiate the specific layer type
                underlying::LayerType layerType = layer->GetLayerType();
                switch (layerType)
                {
                case (underlying::LayerType::activation): {
                    auto& apiLayer = LayerAs<neural::ActivationLayer>(layer);
                    underlyingLayers.push_back(CreateActivationLayer(apiLayer, parameters));
                }
                break;
                case (underlying::LayerType::batchNormalization): {
                    auto& apiLayer = LayerAs<neural::BatchNormalizationLayer>(layer);
                    auto epsilonSummand = (apiLayer.epsilonSummand == neural::EpsilonSummand::variance) ? underlying::EpsilonSummand::Variance : underlying::EpsilonSummand::SqrtVariance;
                    underlyingLayers.push_back(std::make_unique<underlying::BatchNormalizationLayer<ElementType>>(parameters,
                                                                                                                  CastVector<ElementType>(apiLayer.mean),
                                                                                                                  CastVector<ElementType>(apiLayer.variance),
                                                                                                                  static_cast<ElementType>(apiLayer.epsilon),
                                                                                                                  epsilonSummand));
                }
                break;
                case (underlying::LayerType::bias): {
                    auto& apiLayer = LayerAs<neural::BiasLayer>(layer);
                    underlyingLayers.push_back(std::make_unique<underlying::BiasLayer<ElementType>>(parameters, CastVector<ElementType>(apiLayer.bias)));
                }
                break;
                case (underlying::LayerType::binaryConvolution): {
                    auto& apiLayer = LayerAs<neural::BinaryConvolutionalLayer>(layer);
                    TensorType weights(apiLayer.weights.shape.rows, apiLayer.weights.shape.columns, apiLayer.weights.shape.channels, CastVector<ElementType>(apiLayer.weights.data));
                    underlyingLayers.push_back(std::make_unique<underlying::BinaryConvolutionalLayer<ElementType>>(parameters, apiLayer.convolutionalParameters, weights));
                }
                break;
                case (underlying::LayerType::convolution): {
                    auto& apiLayer = LayerAs<neural::ConvolutionalLayer>(layer);
                    TensorType weights(apiLayer.weights.shape.rows, apiLayer.weights.shape.columns, apiLayer.weights.shape.channels, CastVector<ElementType>(apiLayer.weights.data));
                    underlyingLayers.push_back(std::make_unique<underlying::ConvolutionalLayer<ElementType>>(parameters, apiLayer.convolutionalParameters, weights));
                }
                break;
                case (underlying::LayerType::fullyConnected): {
                    auto& apiLayer = LayerAs<neural::FullyConnectedLayer>(layer);

                    TensorType weights(apiLayer.weights.shape.rows, apiLayer.weights.shape.columns, apiLayer.weights.shape.channels, CastVector<ElementType>(apiLayer.weights.data));
                    underlyingLayers.push_back(std::make_unique<underlying::FullyConnectedLayer<ElementType>>(parameters, weights));
                }
                break;
                case (underlying::LayerType::pooling): {
                    auto& apiLayer = LayerAs<neural::PoolingLayer>(layer);
                    if (apiLayer.poolingType == neural::PoolingType::max)
                    {
                        underlyingLayers.push_back(std::make_unique<underlying::PoolingLayer<ElementType, underlying::MaxPoolingFunction>>(parameters, apiLayer.poolingParameters));
                    }
                    else
                    {
                        underlyingLayers.push_back(std::make_unique<underlying::PoolingLayer<ElementType, underlying::MeanPoolingFunction>>(parameters, apiLayer.poolingParameters));
                    }
                }
                break;
                case (underlying::LayerType::region): {
                    auto& apiLayer = LayerAs<neural::RegionDetectionLayer>(layer);
                    underlyingLayers.push_back(std::make_unique<underlying::RegionDetectionLayer<ElementType>>(parameters, apiLayer.detectionParameters));
                }
                break;
                case (underlying::LayerType::scaling): {
                    auto& apiLayer = LayerAs<neural::ScalingLayer>(layer);
                    underlyingLayers.push_back(std::make_unique<underlying::ScalingLayer<ElementType>>(parameters, CastVector<ElementType>(apiLayer.scales)));
                }
                break;
                case (underlying::LayerType::softmax): {
                    underlyingLayers.push_back(std::make_unique<underlying::SoftmaxLayer<ElementType>>(parameters));
                }
                break;
                default:
                    throw InputException(InputExceptionErrors::invalidArgument, "Encountered unknown layer type in neural network predictor");
                    break;
                }
            }
            else
            {
                throw InputException(InputExceptionErrors::invalidArgument, "Encountered null layer in neural network predictor");
            }
        }

        ELL_API::PortType GetBiggestDataType(const std::vector<Layer*>& layers)
        {
            auto result = ELL_API::PortType::none;

            for (size_t i = 0; i < layers.size(); i++)
            {
                Layer* layer = layers[i];
                auto dt = layer->parameters.dataType;
                switch (dt)
                {
                case PortType::real:
                    result = dt;
                    break;
                case PortType::smallReal:
                    if (result != PortType::real)
                    {
                        result = dt;
                    }
                    break;
                default:
                    throw DataFormatException(DataFormatErrors::badFormat, FormatString("Unsupported DataType %d for Layer %s", static_cast<int>(dt), typeid(*layer).name()));
                }
            }
            return result;
        }

        NeuralNetworkPredictor::NeuralNetworkPredictor(const std::vector<ell::api::predictors::neural::Layer*>& layers, double inputScaleFactor)
        {
            _dataType = GetBiggestDataType(layers);
            switch (_dataType)
            {
            case PortType::real:
                _predictor = std::make_unique<TypedNeuralNetworkPredictorImpl<double>>(layers, static_cast<double>(inputScaleFactor));
                break;
            case PortType::smallReal:
                _predictor = std::make_unique<TypedNeuralNetworkPredictorImpl<float>>(layers, static_cast<float>(inputScaleFactor));
                break;
                //case PortType::bigInt:
                //    _predictor = std::make_unique<TypedNeuralNetworkPredictorImpl<int64_t>>(layers, static_cast<int64_t>(inputScaleFactor));
                //    break;
                //case PortType::integer:
                //    _predictor = std::make_unique<TypedNeuralNetworkPredictorImpl<int>>(layers, static_cast<int>(inputScaleFactor));
                //    break;
                //case PortType::boolean:
                //    _predictor = std::make_unique<TypedNeuralNetworkPredictorImpl<bool>>(layers, static_cast<bool>(inputScaleFactor));
                //    break;
            default:
                throw DataFormatException(DataFormatErrors::badFormat, FormatString("Unsupported DataType %d for NeuralNetworkPredictor", static_cast<int>(_dataType)));
            }
        }

        std::vector<double> NeuralNetworkPredictor::PredictDouble(const std::vector<double>& input)
        {
            return _predictor->PredictDouble(input);
        }

        std::vector<float> NeuralNetworkPredictor::PredictFloat(const std::vector<float>& input)
        {
            return _predictor->PredictFloat(input);
        }

        LayerShape NeuralNetworkPredictor::GetInputShape() const
        {
            return _predictor->GetInputShape();
        }

        LayerShape NeuralNetworkPredictor::GetOutputShape() const
        {
            return _predictor->GetOutputShape();
        }

        void NeuralNetworkPredictor::RemoveLastLayers(size_t numLayersToRemove)
        {
            _predictor->RemoveLastLayers(numLayersToRemove);
        }

        ELL_API::PortType NeuralNetworkPredictor::GetDataType() const
        {
            return _dataType;
        }

        template <typename ElementType>
        ell::predictors::NeuralNetworkPredictor<ElementType>& NeuralNetworkPredictor::GetUnderlyingPredictor()
        {
            if (typeid(ElementType) == typeid(double) && _dataType == ELL_API::PortType::real)
            {
                TypedNeuralNetworkPredictorImpl<ElementType>* ptr = static_cast<TypedNeuralNetworkPredictorImpl<ElementType>*>(_predictor.get());
                return ptr->GetUnderlyingPredictor();
            }
            else if (typeid(ElementType) == typeid(float) && _dataType == ELL_API::PortType::smallReal)
            {
                TypedNeuralNetworkPredictorImpl<ElementType>* ptr = static_cast<TypedNeuralNetworkPredictorImpl<ElementType>*>(_predictor.get());
                return ptr->GetUnderlyingPredictor();
            }
            throw InputException(InputExceptionErrors::invalidArgument, "Trying to cast underlying predictor to the wrong type");
        }

        // explicit instantiations

        template ell::predictors::NeuralNetworkPredictor<float>& NeuralNetworkPredictor::GetUnderlyingPredictor();
        template ell::predictors::NeuralNetworkPredictor<double>& NeuralNetworkPredictor::GetUnderlyingPredictor();

    } // namespace predictors
} // namespace api
} // namespace ell
