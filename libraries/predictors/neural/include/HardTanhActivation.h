////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     HardTanhActivation.h (neural)
//  Authors:  Yash Gaurkar
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Activation.h"

#include <math/include/Tensor.h>
#include <math/include/Vector.h>

#include <value/include/EmitterContext.h>

#include <cmath>

namespace ell
{
namespace predictors
{
    namespace neural
    {
        /// <summary> Implements the hyperbolic tangent function: hardtanh(x) = clip(x, -1, 1) </summary>
        template <typename ElementType>
        class HardTanhActivation : public ActivationImpl<ElementType>
        {
        public:
            /// <summary> Returns the output as a function of the input. </summary>
            ///
            /// <param name="input"> The input value. </param>
            ElementType Apply(const ElementType input) const override;

            /// <summary> Returns the output as a function of the input. </summary>
            ///
            /// <param name="input"> The input value as a value library Scalar. </param>
            ///
            /// <returns> The computed output. </param>
            value::Scalar Apply(value::Scalar input) const override;

            /// <summary> Gets the name of this type. </summary>
            ///
            /// <returns> The name of this type. </returns>
            static std::string GetTypeName() { return utilities::GetCompositeTypeName<ElementType>("HardTanhActivation"); }

            /// <summary> Gets the name of this type (for serialization). </summary>
            ///
            /// <returns> The name of this type. </returns>
            virtual std::string GetRuntimeTypeName() const override { return GetTypeName(); }

            /// <summary> Make a copy of this activation. </summary>
            ///
            /// <returns> The copy in a unique pointer. </param>
            std::unique_ptr<ActivationImpl<ElementType>> Copy() const override;
        };
    } // namespace neural
} // namespace predictors
} // namespace ell

#pragma region implementation

namespace ell
{
namespace predictors
{
    namespace neural
    {
        template <typename ElementType>
        ElementType HardTanhActivation<ElementType>::Apply(const ElementType input) const
        {
            return ((input < -1) ? -1 : (input > 1) ? 1
                                                    : input);
        }

        template <typename ElementType>
        value::Scalar HardTanhActivation<ElementType>::Apply(value::Scalar input) const
        {
            value::Scalar result;
            If(input < -1, [&] {
                result = value::Cast<ElementType>(-1);
            }).ElseIf(input > 1, [&] {
                  result = value::Cast<ElementType>(1);
              }).Else([&] {
                result = input;
            });
            return result;
        }

        template <typename ElementType>
        std::unique_ptr<ActivationImpl<ElementType>> HardTanhActivation<ElementType>::Copy() const
        {
            return std::make_unique<HardTanhActivation<ElementType>>();
        }
    } // namespace neural
} // namespace predictors
} // namespace ell

#pragma endregion implementation
