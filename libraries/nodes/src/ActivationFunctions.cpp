////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ActivationFunctions.cpp (nodes)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ActivationFunctions.h"
#include "UnaryOperationNode.h"

#include <predictors/neural/include/Activation.h>

#include <emitters/include/IRLocalValue.h>
#include <emitters/include/IRMath.h>

#include <value/include/Emittable.h>

namespace ell
{
namespace nodes
{

    //
    // Hard sigmoid activation function
    //
    template <typename ValueType>
    ValueType HardSigmoidActivationFunction<ValueType>::Compute(ValueType x) const
    {
        // y = clip (scale*x + bias) to [0,1]
        //   = scale * (clip x to [a, b]) + bias, where scale*a+bias = 0, scale*b+bias = 1; so, a = -bias/scale, b = (1-bias)/scale
        constexpr auto scale = static_cast<ValueType>(0.2);
        constexpr auto bias = static_cast<ValueType>(0.5);
        auto lowBound = -bias / scale;
        auto highBound = (1 - bias) / scale;
        return x < lowBound ? 0 : x > highBound ? 1
                                                : (scale * x) + bias;
    }

    template <typename ValueType>
    emitters::LLVMValue HardSigmoidActivationFunction<ValueType>::Compile(emitters::IRFunctionEmitter& function, emitters::LLVMValue xValue) const
    {
        emitters::IRLocalScalar x{ function, xValue };
        return Compile(x).value;
    }

    template <typename ValueType>
    emitters::IRLocalScalar HardSigmoidActivationFunction<ValueType>::Compile(emitters::IRLocalScalar x) const
    {
        // y = clip (scale*x + bias) to [0,1]
        //   = scale * (clip x to [a, b]) + bias, where scale*a+bias = 0, scale*b+bias = 1; so, a = -bias/scale, b = (1-bias)/scale
        auto function = x.function;
        const auto zero = function.Literal(ValueType{ 0 });
        const auto one = emitters::IRLocalScalar(function, function.Literal(static_cast<ValueType>(1)));
        constexpr auto scale = static_cast<ValueType>(0.2);
        constexpr auto bias = static_cast<ValueType>(0.5);
        auto lowBound = -bias / scale;
        auto highBound = (one - bias) / scale;
        auto result = function.Select(x <= lowBound, zero, function.Select(x >= highBound, one, (scale * x) + bias));
        return { function, result };
    }

    //
    // Hard Tanh activation function
    //
    template <typename ValueType>
    ValueType HardTanhActivationFunction<ValueType>::Compute(ValueType x) const
    {
        return x <= -1 ? -1 : x >= 1 ? 1
                                     : x;
    }

    template <typename ValueType>
    emitters::LLVMValue HardTanhActivationFunction<ValueType>::Compile(emitters::IRFunctionEmitter& function, emitters::LLVMValue xValue) const
    {
        auto x = function.LocalScalar(xValue);
        auto lowBound = function.LocalScalar<ValueType>(-1);
        auto highBound = function.LocalScalar<ValueType>(1);
        auto result = function.Select(x <= lowBound, lowBound, function.Select(x >= highBound, highBound, x));
        return result;
    }

    //
    // ReLU activation function
    //
    template <typename ValueType>
    ValueType ReLUActivationFunction<ValueType>::Compute(ValueType x) const
    {
        return x >= 0 ? x : 0;
    }

    template <typename ValueType>
    emitters::LLVMValue ReLUActivationFunction<ValueType>::Compile(emitters::IRFunctionEmitter& function, emitters::LLVMValue xValue) const
    {
        auto x = function.LocalScalar(xValue);
        auto zero = function.LocalScalar<ValueType>(0);
        auto result = function.Select(x >= zero, x, zero);
        return result;
    }

    //
    // Leaky ReLU activation function
    //
    template <typename ValueType>
    ValueType LeakyReLUActivationFunction<ValueType>::Compute(ValueType x) const
    {
        return x >= 0 ? x : x * GetLeakyFactor();
    }

    template <typename ValueType>
    emitters::LLVMValue LeakyReLUActivationFunction<ValueType>::Compile(emitters::IRFunctionEmitter& function, emitters::LLVMValue xValue) const
    {
        auto x = function.LocalScalar(xValue);
        auto zero = function.LocalScalar<ValueType>(0.0);
        auto result = function.Select(x >= zero, x, x * function.LocalScalar(GetLeakyFactor()));
        return result;
    }

    //
    // Sigmoid activation function
    //
    template <typename ValueType>
    ValueType SigmoidActivationFunction<ValueType>::Compute(ValueType x) const
    {
        if (x > 0)
        {
            auto result = static_cast<ValueType>(1) / (std::exp(-x) + static_cast<ValueType>(1));
            return result;
        }
        else
        {
            auto expInput = std::exp(x);
            return expInput / (expInput + static_cast<ValueType>(1));
        }
    }

    template <typename ValueType>
    emitters::LLVMValue SigmoidActivationFunction<ValueType>::Compile(emitters::IRFunctionEmitter& function, emitters::LLVMValue xValue) const
    {
        emitters::IRLocalScalar x{ function, xValue };
        return Compile(x).value;
    }

    template <typename ValueType>
    emitters::IRLocalScalar SigmoidActivationFunction<ValueType>::Compile(emitters::IRLocalScalar x) const
    {
        auto function = x.function;
        const auto zero = emitters::IRLocalScalar(function, function.Literal(ValueType{ 0.0 }));
        const auto one = emitters::IRLocalScalar(function, function.Literal(static_cast<ValueType>(1.0)));
        auto a = one / (emitters::Exp(x * -one) + one);
        auto b = emitters::Exp(x);
        auto c = b / (b + one);
        auto result = function.Select(x >= zero, a, c);
        return { function, result };
    }

    //
    // Tanh activation function
    //
    template <typename ValueType>
    ValueType TanhActivationFunction<ValueType>::Compute(ValueType x) const
    {
        return std::tanh(x);
    }

    template <typename ValueType>
    emitters::LLVMValue TanhActivationFunction<ValueType>::Compile(emitters::IRFunctionEmitter& function, emitters::LLVMValue xValue) const
    {
        auto x = function.LocalScalar(xValue);
        return emitters::Tanh<ValueType>(x);
    }

    //
    // Parametric ReLU activation function
    //
    template <typename ValueType>
    ValueType ParametricReLUActivationFunction<ValueType>::Compute(ValueType x, ValueType a) const
    {
        return ((x > 0) ? x : a * x);
    }

    template <typename ValueType>
    emitters::LLVMValue ParametricReLUActivationFunction<ValueType>::Compile(emitters::IRFunctionEmitter& function, emitters::LLVMValue xValue, emitters::LLVMValue aValue) const
    {
        auto x = function.LocalScalar(xValue);
        auto a = function.LocalScalar(aValue);
        return function.Select(x > static_cast<ValueType>(0), x, x * a);
    }

    // Explicit instantiation
    template class HardSigmoidActivationFunction<float>;
    template class HardSigmoidActivationFunction<double>;
    template class HardTanhActivationFunction<float>;
    template class HardTanhActivationFunction<double>;
    template class LeakyReLUActivationFunction<float>;
    template class LeakyReLUActivationFunction<double>;
    template class ParametricReLUActivationFunction<float>;
    template class ParametricReLUActivationFunction<double>;
    template class ReLUActivationFunction<float>;
    template class ReLUActivationFunction<double>;
    template class SigmoidActivationFunction<float>;
    template class SigmoidActivationFunction<double>;
    template class TanhActivationFunction<float>;
    template class TanhActivationFunction<double>;

} // namespace nodes
} // namespace ell
