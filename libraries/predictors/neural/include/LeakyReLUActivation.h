////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     LeakyReLUActivation.h (neural)
//  Authors:  Byron Changuion
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Activation.h"

#include <math/include/Tensor.h>

#include <utilities/include/Exception.h>

namespace ell
{
namespace predictors
{
    namespace neural
    {
        /// <summary> Implements the leaky ReLU function: activation(x) = (x > 0) ? x : leakyFactor * x  </summary>
        template <typename ElementType>
        class LeakyReLUActivation : public ActivationImpl<ElementType>
        {
        public:
            /// <summary> Instantiates an instance of this class with appropriate leaky factor.
            /// Typical values are 0.1 and 0.01.
            /// </summary>
            ///
            /// <param name="leakyFactor"> The value to multiply the input by if it is less than zero. </param>
            LeakyReLUActivation(ElementType leakyFactor = static_cast<ElementType>(0.1)) :
                _leakyFactor(leakyFactor) {}

            /// <summary> Returns the output as a function of the input. </summary>
            ///
            /// <param name="input"> The input value. </param>
            ///
            /// <returns> The computed output. </param>
            ElementType Apply(const ElementType input) const override;

            /// <summary> Returns the output as a function of the input. </summary>
            ///
            /// <param name="input"> The input value as a value library Scalar. </param>
            ///
            /// <returns> The computed output. </param>
            value::Scalar Apply(value::Scalar input) const override;

            /// <summary> Gets the leaky factor parameter. </summary>
            ///
            /// <returns> The leaky factor parameter. </returns>
            ElementType GetLeakyFactor() const { return _leakyFactor; }

            /// <summary> Make a copy of this activation. </summary>
            ///
            /// <returns> The copy in a unique pointer. </param>
            std::unique_ptr<ActivationImpl<ElementType>> Copy() const override;

            /// <summary> Gets the name of this type. </summary>
            ///
            /// <returns> The name of this type. </returns>
            static std::string GetTypeName() { return utilities::GetCompositeTypeName<ElementType>("LeakyReLUActivation"); }

            /// <summary> Gets the name of this type (for serialization). </summary>
            ///
            /// <returns> The name of this type. </returns>
            std::string GetRuntimeTypeName() const override { return GetTypeName(); }

            /// <summary> Archives this object. </summary>
            ///
            /// <param name="archiver"> The archiver. </param>
            void WriteToArchive(utilities::Archiver& /*archiver*/) const override;

            /// <summary> Unarchives this object. </summary>
            ///
            /// <param name="archiver"> The unarchiver. </param>
            void ReadFromArchive(utilities::Unarchiver& /*archiver*/) override;

        private:
            ElementType _leakyFactor;
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
        ElementType LeakyReLUActivation<ElementType>::Apply(const ElementType input) const
        {
            return ((input > 0) ? input : _leakyFactor * input);
        }

        template <typename ElementType>
        value::Scalar LeakyReLUActivation<ElementType>::Apply(value::Scalar input) const
        {
            value::Scalar result;
            If(input > 0, [&] {
                result = input;
            }).Else([&] {
                value::Scalar s = value::Cast<ElementType>(_leakyFactor);
                result = input * s;
            });
            return result;
        }

        template <typename ElementType>
        void LeakyReLUActivation<ElementType>::WriteToArchive(utilities::Archiver& archiver) const
        {
            archiver["leakyFactor"] << _leakyFactor;
        }

        template <typename ElementType>
        void LeakyReLUActivation<ElementType>::ReadFromArchive(utilities::Unarchiver& archiver)
        {
            archiver["leakyFactor"] >> _leakyFactor;
        }

        template <typename ElementType>
        std::unique_ptr<ActivationImpl<ElementType>> LeakyReLUActivation<ElementType>::Copy() const
        {
            return std::make_unique<LeakyReLUActivation<ElementType>>(_leakyFactor);
        }
    } // namespace neural
} // namespace predictors
} // namespace ell

#pragma endregion implementation
