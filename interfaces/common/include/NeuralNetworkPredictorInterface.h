////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     NeuralNetworkPredictorInterface.h (interfaces)
//  Authors:  Byron Changuion
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "MathInterface.h"
#include "NeuralLayersInterface.h"
#include "Ports.h"

#ifndef SWIG
#include <predictors/include/NeuralNetworkPredictor.h>

#include <memory>
#include <string>
#include <vector>
#endif

namespace ell
{
namespace api
{
    namespace predictors
    {
#ifndef SWIG
        class NeuralNetworkPredictorImpl;
#endif
        //
        // API classes for the neural predictor
        //
        class NeuralNetworkPredictor
        {
        public:
            using LayerShape = ell::api::math::TensorShape;
            using DataType = ELL_API::PortType;

            NeuralNetworkPredictor(const std::vector<ell::api::predictors::neural::Layer*>& layers, double inputScaleFactor = 1.0f);

            std::vector<double> PredictDouble(const std::vector<double>& input);
            std::vector<float> PredictFloat(const std::vector<float>& input);

            void RemoveLastLayers(size_t numberToRemove = 1);

            LayerShape GetInputShape() const;
            LayerShape GetOutputShape() const;

            DataType GetDataType() const;

#ifndef SWIG
            template <typename ElementType>
            ell::predictors::NeuralNetworkPredictor<ElementType>& GetUnderlyingPredictor();

        private:
            std::shared_ptr<NeuralNetworkPredictorImpl> _predictor;
            DataType _dataType;
#endif
        };
    } // namespace predictors
} // namespace api
} // namespace ell
