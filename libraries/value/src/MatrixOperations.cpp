////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     MatrixOperations.h (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MatrixOperations.h"
#include "EmitterContext.h"
#include "Matrix.h"
#include "Scalar.h"

#include <utilities/include/StringUtil.h>

namespace ell
{
using namespace utilities;

namespace value
{

    Matrix ToMatrix(Value data, int numRows, int numCols)
    {
        Value matrix = data;
        auto size = data.GetLayout().GetActiveSize().NumElements();
        if (size != numRows * numCols || !data.GetLayout().IsContiguous())
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 ell::utilities::FormatString("data must be contiguous and have size %zu = %d * %d", size, numRows, numCols));
        }
        matrix.SetLayout(utilities::MemoryLayout{ { numRows, numCols } });
        return matrix;
    }

    Scalar Sum(Matrix matrix)
    {
        Scalar result = Allocate(matrix.Type(), ScalarLayout);

        For(matrix, [&](auto row, auto column) {
            result += matrix(row, column);
        });

        return result;
    }

    void For(Matrix matrix, std::function<void(Scalar, Scalar)> fn)
    {
        For(std::string{}, matrix, fn);
    }

    void For(const std::string& name, Matrix matrix, std::function<void(Scalar, Scalar)> fn)
    {
        auto layout = matrix.GetValue().GetLayout();
        if (layout.NumDimensions() != 2)
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Layout being looped over must be two-dimensional");
        }

        GetContext().For(
            layout,
            [fn = std::move(fn)](std::vector<Scalar> coordinates) {
                fn(coordinates[0], coordinates[1]);
            },
            name);
    }

    Matrix GEMM(Matrix m1, Matrix m2)
    {
        throw LogicException(LogicExceptionErrors::notImplemented);
    }

    Vector GEMV(Matrix m, Vector v)
    {
        Vector result = Allocate(v.GetType(), m.Rows());
        Scalar first = Allocate(ValueType::Int32, ScalarLayout);
        if (m.Columns() != v.Size())
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 ell::utilities::FormatString("Vector size %d must match number of columns in the matrix %d", v.Size(), m.Columns()));
        }
        first = 1;
        For(m, [&](Scalar row, Scalar col) {
            If(first == 1, [&] {
                result[row] = m(row, col) * v(col);
                first = 0;
            }).Else([&] {
                result[row] += m(row, col) * v(col);
            });
        });
        return result;
    }

} // namespace value
} // namespace ell
