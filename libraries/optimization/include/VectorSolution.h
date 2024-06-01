////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     VectorSolution.h (optimization)
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "Expression.h"
#include "IndexedContainer.h"
#include "OptimizationExample.h"

#include <math/include/Vector.h>
#include <math/include/VectorOperations.h>

#include <type_traits>

namespace ell
{
namespace optimization
{
    /// <summary> An vector solution that applies to vector inputs and scalar outputs. </summary>
    template <typename IOElementType, bool isBiased = false>
    class VectorSolution : public Scalable
    {
    public:
        using InputType = math::ConstRowVectorReference<IOElementType>;
        using OutputType = IOElementType;
        using AuxiliaryDoubleType = double;
        using ExampleType = Example<InputType, OutputType>;
        using DatasetType = IndexedContainer<ExampleType>;

        /// <summary> Solutions are expected to have a ParameterType. Empty here because this solution type doesn't need any parameters. </summary>
        struct ParametersType
        {};

        /// <summary> Default constructor. </summary>
        VectorSolution() = default;

        /// <summary> Constructs a solution of a given size. </summary>
        VectorSolution(size_t size) :
            _weights(size) {}

        /// <summary> Resize the solution to match the sizes of an input and an output. </summary>
        void Resize(const InputType& inputExample, OutputType);

        /// <summary> Resets the solution to zero. </summary>
        void Reset();

        /// <summary> Sets the solution parameters. This solution type doesn't have any parameters, so it does nothing. </summary>
        void SetParameters(const ParametersType&) {}

        /// <summary> Returns a reference to the vector. </summary>
        math::ColumnVectorReference<double> GetVector() { return _weights; }

        /// <summary> Returns a const reference to the vector. </summary>
        math::ConstColumnVectorReference<double> GetVector() const { return _weights; }

        /// <summary> Returns the bias. </summary>
        template <bool B = isBiased, typename Concept = std::enable_if_t<B>>
        double& GetBias()
        {
            return _bias;
        }

        /// <summary> Returns the bias. </summary>
        template <bool B = isBiased, typename Concept = std::enable_if_t<B>>
        double GetBias() const
        {
            return _bias;
        }

        /// <summary> Assignment operator. </summary>
        void operator=(const VectorSolution<IOElementType, isBiased>& other);

        /// <summary> Adds another scaled solution to a scaled version of this solution. </summary>
        void operator=(SumExpression<ScaledExpression<VectorSolution<IOElementType, isBiased>>, ScaledExpression<VectorSolution<IOElementType, isBiased>>> expression);

        /// <summary> Adds a scaled column vector to a scaled version of this solution. </summary>
        void operator=(SumExpression<ScaledExpression<VectorSolution<IOElementType, isBiased>>, ScaledColumnVectorExpression<IOElementType>> expression);

        /// <summary> Subtracts another solution from this one. </summary>
        void operator-=(const VectorSolution<IOElementType, isBiased>& other);

        /// <summary> Adds a scaled column vector to this solution. </summary>
        void operator+=(ScaledColumnVectorExpression<IOElementType> expression);

        /// <summary> Computes input * weights, or input * weights + bias (if a bias exists). </summary>
        double Multiply(const InputType& input) const;

        /// <summary> Returns the squared 2-norm of a given input. </summary>
        static double GetNorm2SquaredOf(const InputType& input);

        /// <summary> Initializes an auxiliary double variable. </summary>
        void InitializeAuxiliaryVariable(AuxiliaryDoubleType& aux) { aux = 0; }

    private:
        math::ColumnVector<double> _weights;

        struct Nothing
        {};

        // if the solution is biased, allocate a bias term
        std::conditional_t<isBiased, double, Nothing> _bias = {};

        // if the IO element type is not double, allocate a double row vector
        static constexpr bool isDouble = std::is_same_v<IOElementType, double>;
        mutable std::conditional_t<isDouble, Nothing, math::RowVector<double>> _doubleInput;
    };

    /// <summary> Returns the squared 2-norm of a VectorSolutionBase. </summary>
    template <typename IOElementType, bool isBiased>
    double Norm2Squared(const VectorSolution<IOElementType, isBiased>& solution);

    /// <summary> vector-solution product. </summary>
    template <typename IOElementType, bool isBiased>
    double operator*(math::ConstRowVectorReference<IOElementType> input, const VectorSolution<IOElementType, isBiased>& solution);

    /// <summary> An unbiased vector solution that applies to vector inputs and scalar outputs. </summary>
    template <typename IOElementType>
    using UnbiasedVectorSolution = VectorSolution<IOElementType, false>;

    /// <summary> A biased vector solution that applies to vector inputs and scalar outputs. </summary>
    template <typename IOElementType>
    using BiasedVectorSolution = VectorSolution<IOElementType, true>;
} // namespace optimization
} // namespace ell

#pragma region implementation

#include "Common.h"

#include <math/include/VectorOperations.h>

namespace ell
{
namespace optimization
{
    template <typename IOElementType, bool isBiased>
    void VectorSolution<IOElementType, isBiased>::Resize(const InputType& inputExample, OutputType)
    {
        _weights.Resize(inputExample.Size());

        if constexpr (!isDouble)
        {
            _doubleInput.Resize(inputExample.Size());
        }
    }

    template <typename IOElementType, bool isBiased>
    void VectorSolution<IOElementType, isBiased>::Reset()
    {
        _weights.Reset();

        if constexpr (isBiased)
        {
            _bias = 0;
        }
    }

    template <typename IOElementType, bool isBiased>
    void VectorSolution<IOElementType, isBiased>::operator=(const VectorSolution<IOElementType, isBiased>& other)
    {
        _weights.CopyFrom(other._weights);

        if constexpr (isBiased)
        {
            _bias = other._bias;
        }
    }

    template <typename IOElementType, bool isBiased>
    void VectorSolution<IOElementType, isBiased>::operator=(SumExpression<ScaledExpression<VectorSolution<IOElementType, isBiased>>, ScaledExpression<VectorSolution<IOElementType, isBiased>>> expression)
    {
        const auto& thisTerm = expression.lhs;
        const auto& otherTerm = expression.rhs;

        if (&(thisTerm.lhs.get()) != this)
        {
            throw OptimizationException("First term should be a scaled version of this solution");
        }

        double thisScale = thisTerm.rhs;
        const auto& otherSolution = otherTerm.lhs.get();
        double otherScale = otherTerm.rhs;
        math::ScaleAddUpdate(otherScale, otherSolution.GetVector(), thisScale, _weights);

        if constexpr (isBiased)
        {
            _bias = thisScale * _bias + otherScale * otherSolution.GetBias();
        }
    }

    template <typename IOElementType, bool isBiased>
    void VectorSolution<IOElementType, isBiased>::operator=(SumExpression<ScaledExpression<VectorSolution<IOElementType, isBiased>>, ScaledColumnVectorExpression<IOElementType>> expression)
    {
        const auto& thisTerm = expression.lhs;
        const auto& updateTerm = expression.rhs;

        if (&(thisTerm.lhs.get()) != this)
        {
            throw OptimizationException("One of the terms should be a scaled version of this solution");
        }

        double thisScale = thisTerm.rhs;
        auto updateVector = updateTerm.lhs;
        double updateScale = updateTerm.rhs;

        if constexpr (isDouble)
        {
            math::ScaleAddUpdate(updateScale, updateVector, thisScale, _weights);
        }
        else
        {
            auto doubleColumnVector = _doubleInput.Transpose();
            doubleColumnVector.CopyFrom(updateVector);
            math::ScaleAddUpdate(updateScale, doubleColumnVector, thisScale, _weights);
        }

        if constexpr (isBiased)
        {
            _bias = thisScale * _bias + updateScale;
        }
    }

    template <typename IOElementType, bool isBiased>
    void VectorSolution<IOElementType, isBiased>::operator-=(const VectorSolution<IOElementType, isBiased>& other)
    {
        _weights -= other._weights;
        if constexpr (isBiased)
        {
            _bias -= other._bias;
        }
    }

    template <typename IOElementType, bool isBiased>
    void VectorSolution<IOElementType, isBiased>::operator+=(ScaledColumnVectorExpression<IOElementType> expression)
    {
        const auto& updateVector = expression.lhs;
        double updateScale = expression.rhs;

        if constexpr (isDouble)
        {
            math::ScaleAddUpdate(updateScale, updateVector, 1.0, _weights);
        }
        else
        {
            auto doubleColumnVector = _doubleInput.Transpose();
            doubleColumnVector.CopyFrom(updateVector);
            math::ScaleAddUpdate(updateScale, doubleColumnVector, 1.0, _weights);
        }

        if constexpr (isBiased)
        {
            _bias += updateScale;
        }
    }

    template <typename IOElementType, bool isBiased>
    double VectorSolution<IOElementType, isBiased>::Multiply(const InputType& input) const
    {
        double result;

        if constexpr (isDouble)
        {
            result = math::Dot(input, _weights);
        }
        else
        {
            _doubleInput.CopyFrom(input);
            result = math::Dot(_doubleInput, _weights);
        }

        if constexpr (isBiased)
        {
            result += _bias;
        }

        return result;
    }

    template <typename IOElementType, bool isBiased>
    double VectorSolution<IOElementType, isBiased>::GetNorm2SquaredOf(const InputType& input)
    {
        double result = input.Norm2Squared();

        if constexpr (isBiased)
        {
            result += 1.0;
        }

        return result;
    }

    template <typename IOElementType, bool isBiased>
    double Norm2Squared(const VectorSolution<IOElementType, isBiased>& solution)
    {
        double result = solution.GetVector().Norm2Squared();

        if constexpr (isBiased)
        {
            result += solution.GetBias() * solution.GetBias();
        }

        return result;
    }

    template <typename IOElementType, bool isBiased>
    double operator*(math::ConstRowVectorReference<IOElementType> input, const VectorSolution<IOElementType, isBiased>& solution)
    {
        return solution.Multiply(input);
    }
} // namespace optimization
} // namespace ell

#pragma endregion implementation
