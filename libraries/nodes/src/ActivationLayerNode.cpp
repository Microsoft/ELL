////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ActivationLayerNode.cpp (nodes)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ActivationLayerNode.h"
#include "ActivationFunctions.h"
#include "BinaryFunctionNode.h"
#include "BroadcastFunctionNode.h"
#include "ConstantNode.h"

#include <predictors/neural/include/HardSigmoidActivation.h>
#include <predictors/neural/include/HardTanhActivation.h>
#include <predictors/neural/include/LeakyReLUActivation.h>
#include <predictors/neural/include/ParametricReLUActivation.h>
#include <predictors/neural/include/ReLUActivation.h>
#include <predictors/neural/include/SigmoidActivation.h>
#include <predictors/neural/include/TanhActivation.h>

namespace ell
{
namespace nodes
{
    //
    // ActivationLayerNode
    //
    template <typename ValueType>
    ActivationLayerNode<ValueType>::ActivationLayerNode(const model::OutputPort<ValueType>& input, const predictors::neural::ActivationLayer<ValueType>& layer) :
        BaseType(input, layer)
    {
        auto&& inputLayout = this->GetInputMemoryLayout();
        auto&& outputLayout = this->GetOutputMemoryLayout();
        if (inputLayout.GetActiveSize() != outputLayout.GetActiveSize())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "ActivationLayerNode: Input and output active area sizes don't match");
        }
    }

    template <typename ValueType>
    bool ActivationLayerNode<ValueType>::Refine(model::ModelTransformer& transformer) const
    {
        const auto& newInput = transformer.GetCorrespondingInputs(this->input);

        predictors::neural::ActivationImpl<ValueType>* ptr = this->_layer.GetActivationFunction().GetImpl();

        // hmmm, now we have the opposite problem, unless we push the same idea down into BroadcastUnaryFunctionNode...
        auto hardSigmoid = dynamic_cast<predictors::neural::HardSigmoidActivation<ValueType>*>(ptr);
        auto hardTanh = dynamic_cast<predictors::neural::HardTanhActivation<ValueType>*>(ptr);
        auto leakyReLU = dynamic_cast<predictors::neural::LeakyReLUActivation<ValueType>*>(ptr);
        auto sigmoid = dynamic_cast<predictors::neural::SigmoidActivation<ValueType>*>(ptr);
        auto relu = dynamic_cast<predictors::neural::ReLUActivation<ValueType>*>(ptr);
        auto tanh = dynamic_cast<predictors::neural::TanhActivation<ValueType>*>(ptr);
        auto prelu = dynamic_cast<predictors::neural::ParametricReLUActivation<ValueType>*>(ptr);

        auto inputLayout = this->GetInputMemoryLayout();
        auto outputLayout = this->GetOutputMemoryLayout();

        ell::model::Node* computeNode = nullptr;
        if (hardSigmoid)
        {
            computeNode = transformer.AddNode<BroadcastUnaryFunctionNode<ValueType, HardSigmoidActivationFunction<ValueType>>>(newInput, inputLayout, outputLayout, HardSigmoidActivationFunction<ValueType>{});
        }
        else if (hardTanh)
        {
            computeNode = transformer.AddNode<BroadcastUnaryFunctionNode<ValueType, HardTanhActivationFunction<ValueType>>>(newInput, inputLayout, outputLayout, HardTanhActivationFunction<ValueType>{});
        }
        else if (leakyReLU)
        {
            computeNode = transformer.AddNode<BroadcastUnaryFunctionNode<ValueType, LeakyReLUActivationFunction<ValueType>>>(newInput, inputLayout, outputLayout, LeakyReLUActivationFunction<ValueType>(leakyReLU->GetLeakyFactor()));
        }
        else if (sigmoid)
        {
            computeNode = transformer.AddNode<BroadcastUnaryFunctionNode<ValueType, SigmoidActivationFunction<ValueType>>>(newInput, inputLayout, outputLayout, SigmoidActivationFunction<ValueType>{});
        }
        else if (relu)
        {
            computeNode = transformer.AddNode<BroadcastUnaryFunctionNode<ValueType, ReLUActivationFunction<ValueType>>>(newInput, inputLayout, outputLayout, ReLUActivationFunction<ValueType>{});
        }
        else if (tanh)
        {
            computeNode = transformer.AddNode<BroadcastUnaryFunctionNode<ValueType, TanhActivationFunction<ValueType>>>(newInput, inputLayout, outputLayout, TanhActivationFunction<ValueType>{});
        }
        else if (prelu)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "ActivationLayerNode cannot be used on ParametricReLUActivations");
        }
        else
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "ActivationLayerNode given a new Activation type it doesn't recognize");
        }

        transformer.MapNodeOutput(this->output, *(computeNode->GetOutputPort(0)));
        return true;
    }

    template <typename ValueType>
    void ActivationLayerNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInput = transformer.GetCorrespondingInputs(this->_input);
        auto newNode = transformer.AddNode<ActivationLayerNode<ValueType>>(newInput, this->_layer);
        transformer.MapNodeOutput(this->_output, newNode->output);
    }

    //
    // ParametricReLUActivationLayerNode
    //

    template <typename ValueType>
    ParametricReLUActivationLayerNode<ValueType>::ParametricReLUActivationLayerNode(const model::OutputPort<ValueType>& input, const LayerType& layer) :
        BaseType(input, layer)
    {
        auto&& inputLayout = this->GetInputMemoryLayout();
        auto&& outputLayout = this->GetOutputMemoryLayout();
        if (inputLayout.GetActiveSize() != outputLayout.GetActiveSize())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "ActivationLayerNode: Input and output active area sizes don't match");
        }
        auto paf = dynamic_cast<const predictors::neural::ParametricReLUActivation<ValueType>*>(layer.GetActivationFunction().GetImpl());
        if (!paf)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "The layer activation function must be 'ParametricReLUActivation'");
        }
    }

    template <typename ValueType>
    bool ParametricReLUActivationLayerNode<ValueType>::Refine(model::ModelTransformer& transformer) const
    {
        using ActivationFunction = ParametricReLUActivationFunction<ValueType>;
        const auto& newInput = transformer.GetCorrespondingInputs(this->input);

        auto paf = dynamic_cast<const predictors::neural::ParametricReLUActivation<ValueType>*>(this->_layer.GetActivationFunction().GetImpl());
        auto alphaValues = paf->GetAlpha().ToArray();
        const auto& alphaValuesOut = Constant(transformer, alphaValues);

        // PReLU is a coordinate-wise operation
        auto computeNode = transformer.AddNode<BinaryFunctionNode<ValueType, ActivationFunction>>(
            newInput,
            alphaValuesOut,
            this->GetInputMemoryLayout(),
            this->GetOutputMemoryLayout(),
            ActivationFunction{});

        transformer.MapNodeOutput(this->output, computeNode->output);
        return true;
    }

    template <typename ValueType>
    void ParametricReLUActivationLayerNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInputs = transformer.GetCorrespondingInputs(this->_input);
        auto newNode = transformer.AddNode<ParametricReLUActivationLayerNode<ValueType>>(newInputs, this->_layer);
        transformer.MapNodeOutput(this->_output, newNode->output);
    }

    // Explicit specialization
    template class ActivationLayerNode<float>;
    template class ActivationLayerNode<double>;
    template class ParametricReLUActivationLayerNode<float>;
    template class ParametricReLUActivationLayerNode<double>;

} // namespace nodes
} // namespace ell
