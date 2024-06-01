////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Tensor.cpp (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Tensor.h"
#include "EmitterContext.h"

#include <utilities/include/Exception.h>

#include <cassert>
#include <functional>
#include <numeric>

namespace ell
{
using namespace utilities;

namespace value
{
    Tensor::Tensor() = default;

    Tensor::Tensor(Value value, const std::string& name) :
        _value(value)
    {
        if (!_value.IsDefined() || !_value.IsConstrained() || _value.GetLayout().NumDimensions() != 3)
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Value passed in must be three-dimensional");
        }
        if (!name.empty())
        {
            SetName(name);
        }
    }

    Tensor::~Tensor() = default;
    Tensor::Tensor(const Tensor&) = default;
    Tensor::Tensor(Tensor&&) noexcept = default;

    Tensor& Tensor::operator=(const Tensor& other)
    {
        if (this != &other)
        {
            _value = other._value;
        }
        return *this;
    }

    Tensor& Tensor::operator=(Tensor&& other)
    {
        if (this != &other)
        {
            _value = std::move(other._value);
            other._value = Value();
        }
        return *this;
    }

    Scalar Tensor::operator()(Scalar rowIndex, Scalar columnIndex, Scalar channelIndex)
    {
        Value indexedValue = GetContext().Offset(_value, { rowIndex, columnIndex, channelIndex });
        indexedValue.SetLayout(ScalarLayout);

        return indexedValue;
    }

    Scalar Tensor::operator()(Scalar rowIndex, Scalar columnIndex, Scalar channelIndex) const
    {
        Value indexedValue = GetContext().Offset(_value, { rowIndex, columnIndex, channelIndex });
        indexedValue.SetLayout(ScalarLayout);

        return Scalar(indexedValue).Copy();
    }

    Value Tensor::GetValue() const
    {
        return _value;
    }

    Tensor Tensor::SubTensor(Scalar row, Scalar column, Scalar channel, int numRows, int numColumns, int numChannels) const
    {
        const MemoryLayout& currentLayout = _value.GetLayout();

        if (numRows > currentLayout.GetLogicalDimensionActiveSize(0) ||
            numColumns > currentLayout.GetLogicalDimensionActiveSize(1) ||
            numChannels > currentLayout.GetLogicalDimensionActiveSize(2))
        {
            throw InputException(InputExceptionErrors::indexOutOfRange);
        }

        Value indexedValue = GetContext().Offset(_value, { row, column, channel });
        const auto& dimensionOrder = currentLayout.GetLogicalDimensionOrder();
        const int logicalDimemnsions[] = { numRows, numColumns, numChannels };
        std::vector<int> physicalDimensions(3);

        assert(dimensionOrder.NumDimensions() == 3);
        for (unsigned index = 0; index < 3; ++index)
        {
            physicalDimensions[index] = logicalDimemnsions[dimensionOrder[index]];
        }

        MemoryLayout newLayout(physicalDimensions, currentLayout.GetExtent(), { 0, 0, 0 }, dimensionOrder);
        indexedValue.SetLayout(newLayout);
        return indexedValue;
    }

    Tensor Tensor::Copy() const
    {
        auto newValue = Allocate(_value.GetBaseType(), _value.GetLayout());
        newValue = _value;
        return newValue;
    }

    size_t Tensor::Size() const
    {
        return _value.GetLayout().NumElements();
    }

    Matrix Tensor::Slice(Scalar row, value::Slice mode1, value::Slice mode2) const
    {
        Value indexedValue = GetContext().Offset(_value, { row, 0, 0 });
        const auto& currentLayout = _value.GetLayout();

        indexedValue.SetLayout(currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(0)));

        return indexedValue;
    }

    Matrix Tensor::Slice(value::Slice mode1, Scalar column, value::Slice mode2) const
    {
        Value indexedValue = GetContext().Offset(_value, { 0, column, 0 });
        const auto& currentLayout = _value.GetLayout();

        indexedValue.SetLayout(currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(1)));

        return indexedValue;
    }

    Matrix Tensor::Slice(value::Slice mode1, value::Slice mode2, Scalar channel) const
    {
        Value indexedValue = GetContext().Offset(_value, { 0, 0, channel });
        const auto& currentLayout = _value.GetLayout();

        indexedValue.SetLayout(currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(2)));

        return indexedValue;
    }

    Vector Tensor::Slice(Scalar row, Scalar column, value::Slice /*mode*/) const
    {
        Value indexedValue = GetContext().Offset(_value, { row, column, 0 });
        const auto& currentLayout = _value.GetLayout();

        auto newLayout = currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(0));
        newLayout = newLayout.GetSliceLayout(newLayout.GetPhysicalDimension(0));

        indexedValue.SetLayout(newLayout);
        return indexedValue;
    }

    Vector Tensor::Slice(Scalar row, value::Slice /*mode*/, Scalar channel) const
    {
        Value indexedValue = GetContext().Offset(_value, { row, 0, channel });
        const auto& currentLayout = _value.GetLayout();

        auto newLayout = currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(0));
        newLayout = newLayout.GetSliceLayout(newLayout.GetPhysicalDimension(1));

        indexedValue.SetLayout(newLayout);
        return indexedValue;
    }

    Vector Tensor::Slice(value::Slice /*mode*/, Scalar column, Scalar channel) const
    {
        Value indexedValue = GetContext().Offset(_value, { 0, column, channel });
        const auto& currentLayout = _value.GetLayout();

        auto newLayout = currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(1));
        newLayout = newLayout.GetSliceLayout(newLayout.GetPhysicalDimension(1));

        indexedValue.SetLayout(newLayout);
        return indexedValue;
    }

    size_t Tensor::Rows() const
    {
        return static_cast<size_t>(_value.GetLayout().GetLogicalDimensionActiveSize(0));
    }

    size_t Tensor::Columns() const
    {
        return static_cast<size_t>(_value.GetLayout().GetLogicalDimensionActiveSize(1));
    }

    size_t Tensor::Channels() const
    {
        return static_cast<size_t>(_value.GetLayout().GetLogicalDimensionActiveSize(2));
    }

    ValueType Tensor::Type() const
    {
        return _value.GetBaseType();
    }

    void Tensor::SetName(const std::string& name)
    {
        _value.SetName(name);
    }

    std::string Tensor::GetName() const
    {
        return _value.GetName();
    }

    Tensor& Tensor::operator+=(Scalar s)
    {
        if (s.GetType() != Type())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [&, this](Scalar row, Scalar column, Scalar channel) {
            (*this)(row, column, channel) += s;
        });

        return *this;
    }

    Tensor& Tensor::operator-=(Scalar s)
    {
        if (s.GetType() != Type())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [&, this](Scalar row, Scalar column, Scalar channel) {
            (*this)(row, column, channel) -= s;
        });

        return *this;
    }

    Tensor& Tensor::operator*=(Scalar s)
    {
        if (s.GetType() != Type())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [&, this](Scalar row, Scalar column, Scalar channel) {
            (*this)(row, column, channel) *= s;
        });

        return *this;
    }

    Tensor& Tensor::operator/=(Scalar s)
    {
        if (s.GetType() != Type())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [&, this](Scalar row, Scalar column, Scalar channel) {
            (*this)(row, column, channel) /= s;
        });

        return *this;
    }

    // Free function operator overloads
    Tensor operator+(Tensor t, Scalar s)
    {
        Tensor copy = t.Copy();
        return copy += s;
    }

    Tensor operator-(Tensor t, Scalar s)
    {
        Tensor copy = t.Copy();
        return copy -= s;
    }

    Tensor operator*(Tensor t, Scalar s)
    {
        Tensor copy = t.Copy();
        return copy *= s;
    }

    Tensor operator/(Tensor t, Scalar s)
    {
        Tensor copy = t.Copy();
        return copy /= s;
    }

} // namespace value
} // namespace ell
