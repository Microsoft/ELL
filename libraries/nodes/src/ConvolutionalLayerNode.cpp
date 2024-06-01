////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ConvolutionalLayerNode.cpp (nodes)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ConvolutionalLayerNode.h"
#include "DiagonalConvolutionNode.h"
#include "ReorderDataCodeNode.h"
#include "SimpleConvolutionNode.h"
#include "SpatialConvolutionNode.h"
#include "UnrolledConvolutionNode.h"
#include "WinogradConvolutionNode.h"

using namespace ell::math;
using namespace ell::math::Blas;

namespace ell
{
namespace nodes
{
    template <typename ValueType>
    ConvolutionalLayerNode<ValueType>::ConvolutionalLayerNode(const model::OutputPort<ValueType>& input, const predictors::neural::ConvolutionalLayer<ValueType>& layer) :
        NeuralNetworkLayerNode<ConvolutionalLayerNode<ValueType>, predictors::neural::ConvolutionalLayer<ValueType>, ValueType>(input, layer)
    {
    }

    template <typename ValueType>
    bool ConvolutionalLayerNode<ValueType>::Refine(model::ModelTransformer& transformer) const
    {
        using predictors::neural::ConvolutionMethod;
        auto originalInputLayout = this->GetInputMemoryLayout();
        const auto originalOutputLayout = this->GetOutputMemoryLayout();
        const auto convParams = this->GetLayer().GetConvolutionalParameters();

        const auto* newInput = &transformer.GetCorrespondingInputs(this->input);

        // Terminology:
        // fw: filter width
        // d: # input channels
        // f: # filters (== output channels)
        const auto& weights = this->GetLayer().GetWeights();

        auto isDepthwiseSeparable = (weights.NumChannels() == 1);
        auto shouldReorderToChannelMajor = isDepthwiseSeparable && (convParams.method == ConvolutionMethod::simple || convParams.method == ConvolutionMethod::winograd);

        auto convInputLayout = originalInputLayout.ReorderedCopy({ shouldReorderToChannelMajor ? utilities::ChannelMajorTensorOrder : utilities::RowMajorTensorOrder });
        auto convOutputLayout = originalOutputLayout.ReorderedCopy({ shouldReorderToChannelMajor ? utilities::ChannelMajorTensorOrder : utilities::RowMajorTensorOrder });

        const auto& preConvReorder = ReorderDataWithCodeNode(*newInput, originalInputLayout, convInputLayout);
        newInput = &preConvReorder;

        const model::OutputPort<ValueType>* convOutput;

        switch (convParams.method)
        {
        case ConvolutionMethod::simple: {
            if (isDepthwiseSeparable)
            {
                auto convNode = transformer.AddNode<SpatialConvolutionNode<ValueType>>(*newInput, this->GetLayer(), convOutputLayout);
                convOutput = &convNode->output;
            }
            else
            {
                auto convNode = transformer.AddNode<SimpleConvolutionNode<ValueType>>(*newInput, convInputLayout, convOutputLayout, weights, convParams.stride);
                convOutput = &convNode->output;
            }
        }
        break;
        case ConvolutionMethod::unrolled: {
            auto convNode = transformer.AddNode<UnrolledConvolutionNode<ValueType>>(*newInput, convInputLayout, convOutputLayout, weights, convParams.stride);
            convOutput = &convNode->output;
        }
        break;
        case ConvolutionMethod::diagonal: {
            auto convNode = transformer.AddNode<DiagonalConvolutionNode<ValueType>>(*newInput, convInputLayout, convOutputLayout, weights, convParams.stride);
            convOutput = &convNode->output;
        }
        break;
        case ConvolutionMethod::winograd: {
            auto convNode = transformer.AddNode<WinogradConvolutionNode<ValueType>>(*newInput, convInputLayout, convOutputLayout, weights, convParams.stride);
            convOutput = &convNode->output;
        }
        break;
        default:
            throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
        }

        // Copy metadata
        const_cast<model::Node*>(convOutput->GetNode())->GetMetadata() = this->GetMetadata();

        const auto& postConvReorder = ReorderDataWithCodeNode(*convOutput, originalOutputLayout);
        transformer.MapNodeOutput(this->output, postConvReorder);

        return true;
    }

    template <typename ValueType>
    void ConvolutionalLayerNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInputs = transformer.GetCorrespondingInputs(this->_input);
        auto newNode = transformer.AddNode<ConvolutionalLayerNode<ValueType>>(newInputs, this->_layer);
        transformer.MapNodeOutput(this->_output, newNode->output);
    }

    // Explicit specializations
    template class ConvolutionalLayerNode<float>;
    template class ConvolutionalLayerNode<double>;
} // namespace nodes
} // namespace ell
