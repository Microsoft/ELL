////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     DetectLowPrecisionConvolutionTransformation.h (passes)
//  Authors:  Byron Changuion
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <model/include/Transformation.h>

namespace ell
{
namespace passes
{
    /// <summary> A transformation that detects when a ConvolutionalLayerNode node can be replaced with a
    /// reduced bit version such as BinaryConvolutionalLayerNode. </summary>
    class DetectLowPrecisionConvolutionTransformation : public model::Transformation
    {
    public:
        /// <summary> Change `ConvolutionalLayerNode` to a reduced bit version if possible. </summary>
        model::Submodel Transform(const model::Submodel& submodel, model::ModelTransformer& transformer, const ell::model::TransformContext& context) const override;

        /// <summary> Returns the ID for this transformation </summary>
        std::string GetRuntimeTypeName() const override { return { "DetectLowPrecisionConvolutionTransformation" }; };
    };
} // namespace passes
} // namespace ell
