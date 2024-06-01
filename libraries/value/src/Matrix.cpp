////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Matrix.cpp (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Matrix.h"
#include "EmitterContext.h"

#include <utilities/include/Exception.h>

#include <cassert>
#include <functional>

namespace ell
{
using namespace utilities;

namespace value
{

    Matrix::Matrix() = default;

    Matrix::Matrix(Value value, const std::string& name) :
        _value(value)
    {
        if (!_value.IsDefined() || !_value.IsConstrained() || _value.GetLayout().NumDimensions() != 2)
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Value passed in must be two-dimensional");
        }

        if (!name.empty())
        {
            SetName(name);
        }
    }

    Matrix::~Matrix() = default;
    Matrix::Matrix(const Matrix&) = default;
    Matrix::Matrix(Matrix&&) noexcept = default;

    Matrix& Matrix::operator=(const Matrix& other)
    {
        if (this != &other)
        {
            _value = other._value;
        }
        return *this;
    }

    Matrix& Matrix::operator=(Matrix&& other)
    {
        if (this != &other)
        {
            _value = std::move(other._value);
            other._value = Value();
        }
        return *this;
    }

    Scalar Matrix::operator()(Scalar rowIndex, Scalar columnIndex)
    {
        Value indexedValue = GetContext().Offset(_value, { rowIndex, columnIndex });
        indexedValue.SetLayout(ScalarLayout);

        return indexedValue;
    }

    Scalar Matrix::operator()(Scalar rowIndex, Scalar columnIndex) const
    {
        Value indexedValue = GetContext().Offset(_value, { rowIndex, columnIndex });
        indexedValue.SetLayout(ScalarLayout);

        return Scalar(indexedValue).Copy();
    }

    Value Matrix::GetValue() const
    {
        return _value;
    }

    Matrix Matrix::SubMatrix(Scalar row, Scalar column, int numRows, int numColumns) const
    {
        const MemoryLayout& currentLayout = _value.GetLayout();

        if (numRows > currentLayout.GetLogicalDimensionActiveSize(0) ||
            numColumns > currentLayout.GetLogicalDimensionActiveSize(1))
        {
            throw InputException(InputExceptionErrors::indexOutOfRange);
        }

        Value indexedValue = GetContext().Offset(_value, { row, column });
        const auto& dimensionOrder = currentLayout.GetLogicalDimensionOrder();
        const int logicalDimemnsions[] = { numRows, numColumns };
        std::vector<int> physicalDimensions(2);

        assert(dimensionOrder.NumDimensions() == 2);
        for (unsigned index = 0; index < 2; ++index)
        {
            physicalDimensions[index] = logicalDimemnsions[dimensionOrder[index]];
        }

        MemoryLayout newLayout(physicalDimensions, currentLayout.GetExtent(), { 0, 0 }, dimensionOrder);
        indexedValue.SetLayout(newLayout);
        return indexedValue;
    }

    Matrix Matrix::Copy() const
    {
        auto newValue = Allocate(_value.GetBaseType(), _value.GetLayout());
        newValue = _value;
        return newValue;
    }

    size_t Matrix::Size() const
    {
        return _value.GetLayout().NumElements();
    }

    Vector Matrix::Row(Scalar index) const
    {
        Value indexedValue = GetContext().Offset(_value, { index, 0 });
        const MemoryLayout& currentLayout = _value.GetLayout();

        indexedValue.SetLayout(currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(0)));
        return indexedValue;
    }

    Vector Matrix::Column(Scalar index) const
    {
        Value indexedValue = GetContext().Offset(_value, { 0, index });
        const MemoryLayout& currentLayout = _value.GetLayout();

        indexedValue.SetLayout(currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(1)));
        return indexedValue;
    }

    size_t Matrix::Rows() const
    {
        return static_cast<size_t>(_value.GetLayout().GetLogicalDimensionActiveSize(0));
    }

    size_t Matrix::Columns() const
    {
        return static_cast<size_t>(_value.GetLayout().GetLogicalDimensionActiveSize(1));
    }

    Matrix::MatrixLayout Matrix::GetMatrixLayout() const
    {
        return _value.GetLayout().IsCanonicalOrder() ? MatrixLayout::rowMajor : MatrixLayout::columnMajor;
    }

    ValueType Matrix::Type() const
    {
        return _value.GetBaseType();
    }

    void Matrix::SetName(const std::string& name)
    {
        _value.SetName(name);
    }

    std::string Matrix::GetName() const
    {
        return _value.GetName();
    }

    Matrix& Matrix::operator+=(Matrix m)
    {
        if (m.Rows() != Rows() && m.Columns() != Columns())
        {
            throw InputException(InputExceptionErrors::sizeMismatch);
        }

        if (m.Type() != Type())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(m, [this, &m](Scalar row, Scalar column) {
            (*this)(row, column) += m(row, column);
        });

        return *this;
    }

    Matrix& Matrix::operator-=(Matrix m)
    {
        if (m.Rows() != Rows() && m.Columns() != Columns())
        {
            throw InputException(InputExceptionErrors::sizeMismatch);
        }

        if (m.Type() != Type())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(m, [this, &m](Scalar row, Scalar column) {
            (*this)(row, column) -= m(row, column);
        });

        return *this;
    }

    Matrix& Matrix::operator+=(Scalar s)
    {
        if (s.GetType() != Type())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](Scalar row, Scalar column) {
            (*this)(row, column) += s;
        });

        return *this;
    }

    Matrix& Matrix::operator-=(Scalar s)
    {
        if (s.GetType() != Type())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](Scalar row, Scalar column) {
            (*this)(row, column) -= s;
        });

        return *this;
    }

    Matrix& Matrix::operator*=(Scalar s)
    {
        if (s.GetType() != Type())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](Scalar row, Scalar column) {
            (*this)(row, column) *= s;
        });

        return *this;
    }

    Matrix& Matrix::operator/=(Scalar s)
    {
        if (s.GetType() != Type())
        {
            throw InputException(InputExceptionErrors::typeMismatch);
        }

        For(*this, [this, &s](Scalar row, Scalar column) {
            (*this)(row, column) /= s;
        });

        return *this;
    }

    // Free function operator overloads
    Matrix operator+(Matrix m1, Matrix m2)
    {
        Matrix copy = m1.Copy();
        return copy += m2;
    }

    Matrix operator+(Matrix m, Scalar s)
    {
        Matrix copy = m.Copy();
        return copy += s;
    }

    Matrix operator-(Matrix m1, Matrix m2)
    {
        Matrix copy = m1.Copy();
        return copy -= m2;
    }
    Matrix operator-(Matrix m, Scalar s)
    {
        Matrix copy = m.Copy();
        return copy -= s;
    }

    Matrix operator*(Matrix m, Scalar s)
    {
        Matrix copy = m.Copy();
        return copy *= s;
    }

    Matrix operator/(Matrix m, Scalar s)
    {
        Matrix copy = m.Copy();
        return copy /= s;
    }

} // namespace value
} // namespace ell
