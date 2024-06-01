////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     WinogradConvolution.cpp (dsp)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WinogradConvolution.h"
#include "SimpleConvolution.h"

#include <math/include/BlasWrapper.h>
#include <math/include/MatrixOperations.h>
#include <math/include/TensorOperations.h>

#include <utilities/include/Debug.h>
#include <utilities/include/Exception.h>
#include <utilities/include/Unused.h>

#include <array>
#include <cassert>
#include <initializer_list>

//
// Table of contents for this file:
//
// * Utility code
// * Winograd matrix functions
// * 1D convolution implementation
// * 2D convolution implementation
//   - Filter-transforming functions
//   - ComputeOutput ("convolve" transformed input with transformed output)
//   - Templated classes implementing transforms for different tile,filter sizes
//   - Templated class implementing convolution
//   - Wrapper functions to call templates
// * API wrapper functions
//   - 1D
//   - 2D
// * Explicit instantiations

//
// Notation used throughout:
//
// Input image: r x c x d
// filters: nf x fr x fc x d
// windows: wr x wc
// output tiles: tr x tc
//
// r: input rows
// c: input columns
// d: input channels
// fr: filter rows
// fc: filter columns
// nf: num filters

namespace ell
{
namespace dsp
{
    namespace
    {
        //
        // Helper function to avoid annoying double-to-float errors when creating matrices from literals
        //
        template <typename ValueType, typename ValueType2>
        math::RowMatrix<ValueType> MakeMatrix(std::initializer_list<std::initializer_list<ValueType2>> list)
        {
            auto numRows = list.size();
            auto numColumns = list.begin()->size();
            std::vector<ValueType> data;
            data.reserve(numRows * numColumns);
            for (const auto& row : list)
            {
                DEBUG_THROW(row.size() != numColumns, utilities::InputException(utilities::InputExceptionErrors::sizeMismatch, "incorrect number of elements in initializer list"));
                std::transform(row.begin(), row.end(), std::back_inserter(data), [](ValueType2 x) { return static_cast<ValueType>(x); });
            }
            return math::RowMatrix<ValueType>(numRows, numColumns, data);
        }

        template <typename ValueType>
        void ElementwiseMultiply(const ValueType* A, const ValueType* B, int size, ValueType* C)
        {
            auto end = A + size;
            while (A < end)
            {
                *(C++) = (*(A++)) * (*(B++));
            }
        }

        //
        // Helper function for basic matrix multiplication
        //
        template <typename ValueType, math::MatrixLayout layout1, math::MatrixLayout layout2, math::MatrixLayout layout3>
        void Multiply(const math::ConstMatrixReference<ValueType, layout1>& A, const math::ConstMatrixReference<ValueType, layout2>& B, math::MatrixReference<ValueType, layout3>& C)
        {
            math::MultiplyScaleAddUpdate(static_cast<ValueType>(1.0), A, B, static_cast<ValueType>(0.0), C);
        }

        //
        // Tensor utility functions
        //

        // Extract a non-contiguous slice from a tensor by copying
        template <typename ValueType>
        void GetChannelSlice(const math::ConstChannelColumnRowTensorReference<ValueType>& tensor, int channelIndex, math::RowMatrix<ValueType>& slice)
        {
            const auto numRows = static_cast<int>(tensor.NumRows());
            const auto numColumns = static_cast<int>(tensor.NumColumns());
            assert(numRows == static_cast<int>(slice.NumRows()));
            assert(numColumns == static_cast<int>(slice.NumColumns()));
            for (int rowIndex = 0; rowIndex < numRows; ++rowIndex)
            {
                for (int columnIndex = 0; columnIndex < numColumns; ++columnIndex)
                {
                    slice(rowIndex, columnIndex) = tensor(rowIndex, columnIndex, channelIndex);
                }
            }
        }

        // Convert incoming f x r x c tensor into a (f*r) x c x 1 one
        template <typename ValueType>
        math::ConstChannelColumnRowTensorReference<ValueType> ReorderSeparableFiltersTensor(const math::ConstChannelColumnRowTensorReference<ValueType>& filters)
        {
            return { filters.GetConstDataPointer(), { filters.NumRows() * filters.NumColumns(), filters.NumChannels(), 1 } };
        }

        //
        // 2D array class for fixed-size arrays
        //
        template <typename ValueType, int rows, int columns>
        class Fixed2DArray
        {
        public:
            constexpr Fixed2DArray() :
                _data({ 0 }) {}

            void CopyFrom(const math::ConstChannelColumnRowTensorReference<ValueType>& data)
            {
                const auto dataPtr = data.GetConstDataPointer();
                if (data.IsContiguous())
                {
                    std::copy(dataPtr, dataPtr + (rows * columns), _data.data());
                }
                else if (data.GetIncrement1() == 1)
                {
                    auto stride = data.GetIncrement2();
                    for (int rowIndex = 0; rowIndex < rows; ++rowIndex)
                    {
                        std::copy(dataPtr + rowIndex * stride, dataPtr + rowIndex * stride + columns, _data.data() + rowIndex * columns);
                    }
                }
                else
                {
                    for (int rowIndex = 0; rowIndex < rows; ++rowIndex)
                    {
                        for (int columnIndex = 0; columnIndex < columns; ++columnIndex)
                        {
                            (*this)(rowIndex, columnIndex) = data(rowIndex, columnIndex, 0);
                        }
                    }
                }
            }

            void CopyFrom(const ValueType* dataPtr, int startRow, int startColumn, int channelIndex, int increment1, int increment2)
            {
                CopyFrom(dataPtr, startRow, startColumn, channelIndex, rows, columns, increment1, increment2);
            }

            void CopyFrom(const ValueType* dataPtr, int startRow, int startColumn, int channelIndex, int numRows, int numColumns, int increment1, int increment2)
            {
                UNUSED(numColumns, numRows);
                for (int rowIndex = 0; rowIndex < rows; ++rowIndex)
                {
                    for (int columnIndex = 0; columnIndex < columns; ++columnIndex)
                    {
                        _data[rowIndex * columns + columnIndex] = dataPtr[(rowIndex + startRow) * increment2 + (columnIndex + startColumn) * increment1 + channelIndex];
                    }
                }
            }

            ValueType operator()(int row, int column) const
            {
                return _data[row * columns + column];
            }

            ValueType& operator()(int row, int column)
            {
                return _data[row * columns + column];
            }

            ValueType* GetDataPointer() { return _data.data(); }

            const ValueType* GetDataPointer() const { return _data.data(); }

        private:
            std::array<ValueType, rows * columns> _data;
        };

        //
        // 3D array class for fixed-size arrays
        //
        template <typename ValueType, int M, int N, int K>
        class Fixed3DArray
        {
        public:
            static constexpr int rows = M;
            static constexpr int columns = N;
            static constexpr int channels = K;

            constexpr Fixed3DArray() :
                _data({ 0 }) {}

            ValueType operator()(int row, int column, int channel) const
            {
                return _data[(row * columns * channels) + (column * channels) + channel];
            }

            ValueType& operator()(int row, int column, int channel)
            {
                return _data[(row * columns * channels) + (column * channels) + channel];
            }

            inline int Offset(int row, int column, int channel)
            {
                return (row * columns * channels) + (column * channels) + channel;
            }

            void CopyFrom(const math::ConstChannelColumnRowTensorReference<ValueType>& tensor, int startRow, int startColumn, int startChannel, int numRows, int numColumns, int numChannels, int rowStride, int columnStride)
            {
                const auto tensorPtr = tensor.GetConstDataPointer() + startRow * rowStride + startColumn * columnStride + startChannel;
                auto dataPtr = _data.data();
                for (int rowIndex = 0; rowIndex < numRows; ++rowIndex)
                {
                    for (int columnIndex = 0; columnIndex < numColumns; ++columnIndex)
                    {
                        auto src = tensorPtr + rowIndex * rowStride + columnIndex * columnStride;
                        auto dst = dataPtr + rowIndex * columns * channels + columnIndex * channels;
                        std::copy(src, src + numChannels, dst);
                    }
                }
            }

            ValueType* GetDataPointer() { return _data.data(); }

            const ValueType* GetDataPointer() const { return _data.data(); }

        private:
            std::array<ValueType, rows * columns * channels> _data;
        };

        template <typename ElementType, int rows, int columns>
        std::ostream& operator<<(std::ostream& stream, const Fixed2DArray<ElementType, rows, columns>& matrix)
        {
            stream << "{ ";
            for (size_t i = 0; i < rows; ++i)
            {
                auto firstRow = (i == 0);
                if (!firstRow)
                    stream << "  ";
                stream << "{ ";
                for (size_t j = 0; j < columns; ++j)
                {
                    stream << matrix(i, j);
                    auto lastColumn = (j == columns - 1);
                    if (!lastColumn)
                        stream << ", ";
                }
                stream << " }";
                auto lastRow = (i == rows - 1);
                if (!lastRow)
                    stream << ",\n";
            }
            stream << " }" << std::endl;

            return stream;
        }

        template <typename ElementType, int rows, int columns, int channels>
        std::ostream& operator<<(std::ostream& stream, const Fixed3DArray<ElementType, rows, columns, channels>& array)
        {
            stream << "{ ";
            for (size_t i = 0; i < rows; ++i)
            {
                auto firstRow = (i == 0);
                if (!firstRow)
                    stream << "  ";
                stream << "{ ";
                for (size_t j = 0; j < columns; ++j)
                {
                    stream << "{ ";
                    for (size_t k = 0; k < channels; ++k)
                    {
                        stream << array(i, j, k);
                        auto lastChannel = (k == channels - 1);
                        if (!lastChannel)
                            stream << ", ";
                    }
                    stream << " }";
                    auto lastColumn = (j == columns - 1);
                    if (!lastColumn)
                        stream << ", ";
                }
                stream << " }";
                auto lastRow = (i == rows - 1);
                if (!lastRow)
                    stream << ",\n";
            }
            stream << " }" << std::endl;

            return stream;
        }

    } // End anonymous namespace

    //
    // Winograd matrix functions
    //

    // Winograd convolution works by tiling the input into a set of 'input tiles' or 'windows' and
    // performing a low-operation-count convolution of that window with the filter, producing an output
    // tile. The input tiles must be larger than the output tiles, and overlap. The input tile size is
    // determined by the output tile size and the filter size, and Winograd kernels are referred to using
    // those parameters. For instance, in 1D, F(2,3) is the filtering algorithm for a size 2 output tile and a
    // size 3 filter. The 2D equivalent is F(2x2, 3x3), which produces a 2x2 output tile using a 3x3 filter.
    // The relationship between output tile size, filter size, and input tile (window) size is:
    //
    //     window_size = tile_size + filter_size - 1
    //
    // Therefore, the F(2,3) algorithm requires (2+3-1), or 4, inputs to produce its 2-output tile.
    //
    // The algorithm uses three matrices to perform an algebraic transformation from the input into
    // a representation that enables performing the convolution with fewer operations. These matrices
    // are:
    //
    // B': the input transformation matrix. A square matrix of dimension window_size x window_size that
    //     is multiplied with the input tile.
    //
    // G:  the filter transformation matrix. A rectangular matrix of dimension window_size x filter_size
    //     that is multiplied with the filter, transforming it into something of dimension window_size
    //
    // A': the output transformation matrix. A rectangular matrix of dimension tile_size x window_size that
    //     transforms intermediate results into output tiles
    //
    // Usage:
    //
    // To convolve an input tile with the filter, we first use the B' and G matrices to transform the filter
    // and the input into window_size pieces. Then perform elementwise multiplication between the two, and
    // finally transform that result into an output tile with the A' matrix:
    //
    //     Y = d (*) g             (the input signal 'd' convolved with the filter 'g')
    //       = A' * (Gg .* B'd)    (where '.*' denotes elementwise multiplication)
    //
    //     d = input signal (e.g., [d0 d1 d2 d3]')
    //     g = filter (e.g., [g0 g1 g2]')
    //
    // To perform 2D convolution, we start with 2D input tile d and 2D input filter g,
    // and transform the data by multiplying on the left by the transform matrix and on
    // the right by its transpose:
    //
    //     Y = d (*) g
    //       = A' * (GgG' .* B'dB) * A
    //
    //
    // I don't know why the input and output transformation matrices are called B' and A' instead
    // of just B and A.
    //
    //
    // The matrices:
    //
    // For F(2,3)
    //
    //       1   0  -1   0
    // B' =  0   1   1   0
    //       0  -1   1   0
    //       0   1   0  -1
    //
    //        1     0     0
    // G =  1/2   1/2   1/2
    //      1/2  -1/2   1/2
    //        0     0     1
    //
    // A' =  1   1   1   0
    //       0   1  -1  -1
    //
    //
    // For F(4,3)
    //
    //      4   0  -5   0   1   0
    //      0  -4  -4   1   1   0
    // B' = 0   4  -4  -1   1   0
    //      0  -2  -1   2   1   0
    //      0   2  -1  -2   1   0
    //      0   4   0  -5   0   1
    //
    //
    //       1/4      0      0
    //      -1/6   -1/6   -1/6
    // G =  -1/6    1/6   -1/6
    //      1/24   1/12    1/6
    //      1/24  -1/12    1/6
    //         0      0      1
    //
    //
    //       1   1   1   1   1   0
    // A' =  0   1  -1   2  -2   0
    //       0   1   1   4   4   0
    //       0   1  -1   8  -8   1
    //

    /// <summary> Gets the data-transforming matrix for Winograd convolution (commonly notated as B') </summary>
    template <typename ValueType>
    math::RowMatrix<ValueType> GetLeftDataTransformMatrix(int tileSize, int filterSize)
    {
        if (tileSize == 2 && filterSize == 3)
        {
            // clang-format off
            return MakeMatrix<ValueType>({ { 1,  0, -1,  0 },
                                           { 0,  1,  1,  0 },
                                           { 0, -1,  1,  0 },
                                           { 0,  1,  0, -1 } });
            // clang-format on
        }
        if (tileSize == 4 && filterSize == 3)
        {
            // clang-format off
            return MakeMatrix<ValueType>({ { 4,  0, -5,  0,  1,  0 },
                                           { 0, -4, -4,  1,  1,  0 },
                                           { 0,  4, -4, -1,  1,  0 },
                                           { 0, -2, -1,  2,  1,  0 },
                                           { 0,  2, -1, -2,  1,  0 },
                                           { 0,  4,  0, -5,  0,  1 } });
            // clang-format on
        }
        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
    }

    template <typename ValueType>
    math::RowMatrix<ValueType> GetRightDataTransformMatrix(int tileSize, int filterSize)
    {
        return { GetLeftDataTransformMatrix<ValueType>(tileSize, filterSize).Transpose() };
    }

    /// <summary> Gets the filter-transforming matrix for Winograd convolution (commonly notated as G) </summary>
    template <typename ValueType>
    math::RowMatrix<ValueType> GetLeftFilterTransformMatrix(int tileSize, int filterSize)
    {
        if (tileSize == 2 && filterSize == 3)
        {
            // clang-format off
            return MakeMatrix<ValueType>({ {     1.0,      0.0,     0.0 },
                                           { 1.0 / 2,  1.0 / 2, 1.0 / 2 },
                                           { 1.0 / 2, -1.0 / 2, 1.0 / 2 },
                                           {     0.0,      0.0,     1.0 } });
            // clang-format on
        }
        if (tileSize == 4 && filterSize == 3)
        {
            // clang-format off
            return MakeMatrix<ValueType>({ {  1.0 / 4,       0.0,       0.0 },
                                           { -1.0 / 6,  -1.0 / 6,  -1.0 / 6 },
                                           { -1.0 / 6,   1.0 / 6,  -1.0 / 6 },
                                           {  1.0 / 24,  1.0 / 12,  1.0 / 6 },
                                           {  1.0 / 24, -1.0 / 12,  1.0 / 6 },
                                           {       0.0,       0.0,      1.0 } });
            // clang-format on
        }
        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
    }

    template <typename ValueType>
    math::RowMatrix<ValueType> GetRightFilterTransformMatrix(int tileSize, int filterSize)
    {
        return { GetLeftFilterTransformMatrix<ValueType>(tileSize, filterSize).Transpose() };
    }

    /// <summary> Gets the result-transforming matrix for Winograd convolution (commonly notated as A') </summary>
    template <typename ValueType>
    math::RowMatrix<ValueType> GetLeftResultTransformMatrix(int tileSize, int filterSize)
    {
        if (tileSize == 2 && filterSize == 3)
        {
            // clang-format off
            return MakeMatrix<ValueType>({ { 1,  1,  1,  0 },
                                           { 0,  1, -1, -1 } });
            // clang-format on
        }
        if (tileSize == 4 && filterSize == 3)
        {
            // clang-format off
            return MakeMatrix<ValueType>({ { 1,  1,  1,  1,  1,  0 },
                                           { 0,  1, -1,  2, -2,  0 },
                                           { 0,  1,  1,  4,  4,  0 },
                                           { 0,  1, -1,  8, -8,  1 } });
            // clang-format on
        }
        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
    }

    template <typename ValueType>
    math::RowMatrix<ValueType> GetRightResultTransformMatrix(int tileSize, int filterSize)
    {
        return { GetLeftResultTransformMatrix<ValueType>(tileSize, filterSize).Transpose() };
    }

    //
    // 1D Winograd convolution implementation
    //

    //
    // Templated class used for implementing 1D Winograd convolution for sizes known at compile time
    //

    template <typename ValueType, int tileSize, int filterSize>
    struct FixedWinograd1D;

    template <typename ValueType>
    struct FixedWinograd1D<ValueType, 2, 3>
    {
        static constexpr int tileSize = 2;
        static constexpr int filterSize = 3;

        static void Convolve(const math::RowVector<ValueType>& input, const math::RowVector<ValueType>& filter, math::RowVector<ValueType>& output)
        {
            assert(filter.Size() == filterSize);
            const int outputSize = static_cast<int>(output.Size());

            // Y = A' * (Gg .* B'd)
            //   See matrix section above for definition of G, B', and A' matrices
            //
            // g = filter ([g0 g1 g2]')
            // d = input ([d0 d1 d2 d3]')
            //
            // Precompute Gg:
            //
            //            g0
            // Gg = (g0 + g1 + g2) / 2
            //      (g0 - g1 + g2) / 2
            //            g2

            auto Gg0 = filter[0];
            auto Gg1 = (filter[0] + filter[1] + filter[2]) / static_cast<ValueType>(2);
            auto Gg2 = (filter[0] - filter[1] + filter[2]) / static_cast<ValueType>(2);
            auto Gg3 = filter[2];

            // prefetch 2 values so we don't have to get them again (check to see if this really matters)
            auto d = input.GetConstDataPointer();
            ValueType d0 = d[0];
            ValueType d1 = d[1];
            int numFullTiles = outputSize / tileSize;
            for (int tileIndex = 0; tileIndex < numFullTiles; ++tileIndex)
            {
                const int index = tileIndex * tileSize;
                ValueType d2 = d[index + 2];
                ValueType d3 = d[index + 3];

                // elementwise vector multiply
                const auto m1 = (d0 - d2) * Gg0;
                const auto m2 = (d1 + d2) * Gg1;
                const auto m3 = (d2 - d1) * Gg2;
                const auto m4 = (d1 - d3) * Gg3;
                output[index] = m1 + m2 + m3;
                output[index + 1] = m2 - m3 - m4;

                // shift d0, d1
                d0 = d2;
                d1 = d3;
            }

            // If the last tile is only partially full, compute it here
            auto remainder = outputSize % tileSize;
            if (remainder > 0)
            {
                assert(remainder == 1); // with tileSize == 2, the only possible remainder is '1'
                const int tileIndex = numFullTiles; // the last tile
                const int index = tileIndex * tileSize;
                ValueType d2 = d[index + 2];

                // elementwise vector multiply
                auto m1 = (d0 - d2) * Gg0;
                auto m2 = (d1 + d2) * Gg1;
                auto m3 = (d2 - d1) * Gg2;
                output[index] = m1 + m2 + m3;
            }
        }
    };

    //
    // 2D convolution implementation
    //

    template <typename ValueType>
    void TransformFilters(const math::ConstChannelColumnRowTensorReference<ValueType>& filters, int numFilters, int tileSize, WinogradFilterOrder order, math::ChannelColumnRowTensorReference<ValueType>& transformedFilters)
    {
        using Matrix = math::RowMatrix<ValueType>;
        assert(numFilters > 0);
        assert(order == WinogradFilterOrder::filtersFirst || order == WinogradFilterOrder::tilesFirst);

        // filters is a (numFilters * filterRows) x (filterColumns) x (numChannels) tensor
        const auto filterSize = static_cast<int>(filters.NumRows()) / numFilters;
        const auto windowSize = tileSize + filterSize - 1;
        const auto numChannels = static_cast<int>(filters.NumChannels());
        assert(static_cast<int>(filters.NumColumns()) == filterSize);

        if (order == WinogradFilterOrder::tilesFirst)
        {
            assert(static_cast<int>(transformedFilters.NumRows()) == windowSize * windowSize);
            assert(static_cast<int>(transformedFilters.NumColumns()) == numFilters);
            assert(static_cast<int>(transformedFilters.NumChannels()) == numChannels);
        }

        // Get the strides for the various dimensions (which depend on the order parameter)
        // 'filtersFirst': (numFilters) x (numChannels) x (windowRows * windowColumns)
        // 'tilesFirst': (windowRows * windowColumns) x (numFilters) x (numChannels)
        const int windowEntryStride = order == WinogradFilterOrder::filtersFirst ? 1 : numFilters * numChannels;
        const int filterStride = order == WinogradFilterOrder::filtersFirst ? windowSize * windowSize * numChannels : numChannels;
        const int channelStride = order == WinogradFilterOrder::filtersFirst ? windowSize * windowSize : 1;

        // Precompute GgG', the transformed filter:
        Matrix G = GetLeftFilterTransformMatrix<ValueType>(tileSize, filterSize);
        Matrix Gt = GetRightFilterTransformMatrix<ValueType>(tileSize, filterSize);

        // Temporaries
        Matrix Gg(windowSize, filterSize); // temporary to store the product G * g
        Matrix filterSlice(filterSize, filterSize); // temporary for g
        Matrix transformedFilterSlice(windowSize, windowSize); // temporary for result G * g * Gt
        auto transformedFiltersPtr = transformedFilters.GetDataPointer();

        // for each "pixel" in a tile, we want to generate a (numFilters) x (numChannels) matrix
        // we gather these matrices into a (windowSize*windowSize) x (numFilters) x (numChannels) tensor
        for (int filterIndex = 0; filterIndex < numFilters; ++filterIndex)
        {
            auto filter = filters.GetSubTensor(filterIndex * filterSize, 0, 0, filterSize, filterSize, numChannels);
            for (int channelIndex = 0; channelIndex < numChannels; ++channelIndex)
            {
                // u = Gg_(k,c)Gt   (a windowSize x windowSize matrix)
                GetChannelSlice(filter, channelIndex, filterSlice);

                Multiply(G, filterSlice, Gg);
                Multiply(Gg, Gt, transformedFilterSlice);

                auto transformedFilterSlicePtr = transformedFilterSlice.GetDataPointer();

                // Splat this tile into the appropriately-shaped tensor
                for (int i = 0; i < windowSize * windowSize; ++i)
                {
                    transformedFiltersPtr[filterIndex * filterStride + channelIndex * channelStride + i * windowEntryStride] = transformedFilterSlicePtr[i];
                }
            }
        }
    }

    template <typename ValueType>
    math::ChannelColumnRowTensor<ValueType> GetTransformedFilters(const math::ConstChannelColumnRowTensorReference<ValueType>& filters, int numFilters, int tileSize, WinogradFilterOrder order)
    {
        using Tensor = math::ChannelColumnRowTensor<ValueType>;

        // Input filters tensor is (numFilters*filterRows) x filterColumns x numChannels
        const auto filterSize = static_cast<int>(filters.NumColumns());
        assert(filterSize == static_cast<int>(filters.NumRows()) / numFilters);
        const auto windowSize = tileSize + filterSize - 1;
        const auto numChannels = static_cast<int>(filters.NumChannels());

        // The two algorithm versions use different orderings of the transformed filter tensor, but they're the same size
        Tensor transformedFilters(0, 0, 0);
        if (order == WinogradFilterOrder::filtersFirst)
        {
            transformedFilters = Tensor(numFilters, numChannels, windowSize * windowSize);
        }
        else if (order == WinogradFilterOrder::tilesFirst)
        {
            transformedFilters = Tensor(windowSize * windowSize, numFilters, numChannels);
        }
        else
        {
            throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
        }

        TransformFilters(filters, numFilters, tileSize, order, transformedFilters);
        return transformedFilters;
    }

    template <typename ValueType>
    void ComputeTransformedOutput(const math::ConstChannelColumnRowTensorReference<ValueType>& transformedInput,
                                  const math::ConstChannelColumnRowTensorReference<ValueType>& transformedFilters,
                                  int numOutputRows,
                                  int numOutputColumns,
                                  int numChannels,
                                  int numFilters,
                                  int tileSize,
                                  int filterSize,
                                  math::ChannelColumnRowTensor<ValueType>& transformedOutput)
    {
        // Do a matrix multiply to reduce many entries in parallel
        //
        // transformedInput is a (windowRows*windowColumns) x (tr * tc) x (numChannels) tensor containing the entire transformed input signal
        // transformedFilters is a (windowRows*windowColumns) x (numFilters) x (numChannels) tensor
        // transformedOutput is a (windowRows*windowColumns) x (tr * tc) x (numFilters) tensor containing the entire transformed output signal

        const auto windowSize = filterSize + tileSize - 1;
        const auto numTileRows = ((numOutputRows - 1) / tileSize) + 1;
        const auto numTileColumns = ((numOutputColumns - 1) / tileSize) + 1;

        // Pointers to the beginning of memory for the 3 tensors we're using
        const auto signalBasePointer = transformedInput.GetConstDataPointer();
        const auto filterBasePointer = transformedFilters.GetConstDataPointer();
        auto outputBasePointer = transformedOutput.GetDataPointer();

        // Strides between adjacent pixels of a window, for the 3 tensors we're using
        const int signalWindowStride = numTileRows * numTileColumns * numChannels;
        const int filterWindowStride = numFilters * numChannels;
        const int outputWindowStride = numTileRows * numTileColumns * numFilters;

        auto A = signalBasePointer;
        auto B = filterBasePointer;
        auto C = outputBasePointer;
        const int m = numTileRows * numTileColumns;
        const int n = numFilters;
        const int k = numChannels;
        for (int windowPosition = 0; windowPosition < windowSize * windowSize; ++windowPosition)
        {
            // Multiply directly into transformedOutput
            math::ConstRowMatrixReference<ValueType> matrixA(A, m, k);
            math::ConstColumnMatrixReference<ValueType> matrixB(B, k, n);
            math::RowMatrixReference<ValueType> matrixC(C, m, n);
            math::MultiplyScaleAddUpdate(static_cast<ValueType>(1.0), matrixA, matrixB, static_cast<ValueType>(0.0), matrixC);

            A += signalWindowStride;
            B += filterWindowStride;
            C += outputWindowStride;
        }
    }

    //
    // Helper class encapsulating Winograd transform matrix application
    //
    // The code inside these classes was generated by the `winograd.py` script.
    //
    template <typename ValueType, int tileSize, int filterSize>
    struct FixedWinogradTransform2D;

    // F(2,3)
    template <typename ValueType>
    struct FixedWinogradTransform2D<ValueType, 2, 3>
    {
        static constexpr int tileSize = 2;
        static constexpr int filterSize = 3;
        static constexpr auto windowSize = filterSize + tileSize - 1;

        template <typename MatrixType1, typename MatrixType2>
        static void TransformInputWindow(const MatrixType1& d, MatrixType2& X)
        {
            // Compute B'dB
            X(0, 0) = ((d(0, 0) - d(2, 0)) - (d(0, 2) - d(2, 2)));
            X(0, 1) = ((d(0, 1) - d(2, 1)) + (d(0, 2) - d(2, 2)));
            X(0, 2) = ((d(0, 2) - d(2, 2)) - (d(0, 1) - d(2, 1)));
            X(0, 3) = ((d(0, 1) - d(2, 1)) - (d(0, 3) - d(2, 3)));
            X(1, 0) = ((d(1, 0) + d(2, 0)) - (d(1, 2) + d(2, 2)));
            X(1, 1) = ((d(1, 1) + d(2, 1)) + (d(1, 2) + d(2, 2)));
            X(1, 2) = ((d(1, 2) + d(2, 2)) - (d(1, 1) + d(2, 1)));
            X(1, 3) = ((d(1, 1) + d(2, 1)) - (d(1, 3) + d(2, 3)));
            X(2, 0) = ((d(2, 0) - d(1, 0)) - (d(2, 2) - d(1, 2)));
            X(2, 1) = ((d(2, 1) - d(1, 1)) + (d(2, 2) - d(1, 2)));
            X(2, 2) = ((d(2, 2) - d(1, 2)) - (d(2, 1) - d(1, 1)));
            X(2, 3) = ((d(2, 1) - d(1, 1)) - (d(2, 3) - d(1, 3)));
            X(3, 0) = ((d(1, 0) - d(3, 0)) - (d(1, 2) - d(3, 2)));
            X(3, 1) = ((d(1, 1) - d(3, 1)) + (d(1, 2) - d(3, 2)));
            X(3, 2) = ((d(1, 2) - d(3, 2)) - (d(1, 1) - d(3, 1)));
            X(3, 3) = ((d(1, 1) - d(3, 1)) - (d(1, 3) - d(3, 3)));
        }

        template <typename BlockType1, typename BlockType2>
        static inline void TransformInputBlock(const BlockType1& d, int blockSize, BlockType2& X)
        {
            // Compute B'dB
            for (int index = 0; index < blockSize; ++index)
            {
                X(0, 0, index) = ((d(0, 0, index) - d(2, 0, index)) - (d(0, 2, index) - d(2, 2, index)));
                X(0, 1, index) = ((d(0, 1, index) - d(2, 1, index)) + (d(0, 2, index) - d(2, 2, index)));
                X(0, 2, index) = ((d(0, 2, index) - d(2, 2, index)) - (d(0, 1, index) - d(2, 1, index)));
                X(0, 3, index) = ((d(0, 1, index) - d(2, 1, index)) - (d(0, 3, index) - d(2, 3, index)));
                X(1, 0, index) = ((d(1, 0, index) + d(2, 0, index)) - (d(1, 2, index) + d(2, 2, index)));
                X(1, 1, index) = ((d(1, 1, index) + d(2, 1, index)) + (d(1, 2, index) + d(2, 2, index)));
                X(1, 2, index) = ((d(1, 2, index) + d(2, 2, index)) - (d(1, 1, index) + d(2, 1, index)));
                X(1, 3, index) = ((d(1, 1, index) + d(2, 1, index)) - (d(1, 3, index) + d(2, 3, index)));
                X(2, 0, index) = ((d(2, 0, index) - d(1, 0, index)) - (d(2, 2, index) - d(1, 2, index)));
                X(2, 1, index) = ((d(2, 1, index) - d(1, 1, index)) + (d(2, 2, index) - d(1, 2, index)));
                X(2, 2, index) = ((d(2, 2, index) - d(1, 2, index)) - (d(2, 1, index) - d(1, 1, index)));
                X(2, 3, index) = ((d(2, 1, index) - d(1, 1, index)) - (d(2, 3, index) - d(1, 3, index)));
                X(3, 0, index) = ((d(1, 0, index) - d(3, 0, index)) - (d(1, 2, index) - d(3, 2, index)));
                X(3, 1, index) = ((d(1, 1, index) - d(3, 1, index)) + (d(1, 2, index) - d(3, 2, index)));
                X(3, 2, index) = ((d(1, 2, index) - d(3, 2, index)) - (d(1, 1, index) - d(3, 1, index)));
                X(3, 3, index) = ((d(1, 1, index) - d(3, 1, index)) - (d(1, 3, index) - d(3, 3, index)));
            }
        }

        template <typename MatrixType1, typename MatrixType2>
        static void TransformOutputTile(const MatrixType1& X, MatrixType2& result)
        {
            result(0, 0) = ((((X(0, 0) + X(1, 0)) + X(2, 0)) + ((X(0, 1) + X(1, 1)) + X(2, 1))) + ((X(0, 2) + X(1, 2)) + X(2, 2)));
            result(0, 1) = ((((X(0, 1) + X(1, 1)) + X(2, 1)) - ((X(0, 2) + X(1, 2)) + X(2, 2))) - ((X(0, 3) + X(1, 3)) + X(2, 3)));
            result(1, 0) = ((((X(1, 0) - X(2, 0)) - X(3, 0)) + ((X(1, 1) - X(2, 1)) - X(3, 1))) + ((X(1, 2) - X(2, 2)) - X(3, 2)));
            result(1, 1) = ((((X(1, 1) - X(2, 1)) - X(3, 1)) - ((X(1, 2) - X(2, 2)) - X(3, 2))) - ((X(1, 3) - X(2, 3)) - X(3, 3)));
        }

        template <typename BlockType1, typename BlockType2>
        static inline void TransformOutputBlock(const BlockType1& X, int blockSize, BlockType2& result)
        {
            for (int index = 0; index < blockSize; ++index)
            {
                result(0, 0, index) = ((((X(0, 0, index) + X(1, 0, index)) + X(2, 0, index)) + ((X(0, 1, index) + X(1, 1, index)) + X(2, 1, index))) + ((X(0, 2, index) + X(1, 2, index)) + X(2, 2, index)));
                result(0, 1, index) = ((((X(0, 1, index) + X(1, 1, index)) + X(2, 1, index)) - ((X(0, 2, index) + X(1, 2, index)) + X(2, 2, index))) - ((X(0, 3, index) + X(1, 3, index)) + X(2, 3, index)));
                result(1, 0, index) = ((((X(1, 0, index) - X(2, 0, index)) - X(3, 0, index)) + ((X(1, 1, index) - X(2, 1, index)) - X(3, 1, index))) + ((X(1, 2, index) - X(2, 2, index)) - X(3, 2, index)));
                result(1, 1, index) = ((((X(1, 1, index) - X(2, 1, index)) - X(3, 1, index)) - ((X(1, 2, index) - X(2, 2, index)) - X(3, 2, index))) - ((X(1, 3, index) - X(2, 3, index)) - X(3, 3, index)));
            }
        }
    };

    // F(4,3)
    template <typename ValueType>
    struct FixedWinogradTransform2D<ValueType, 4, 3>
    {
        static constexpr int tileSize = 4;
        static constexpr int filterSize = 3;
        static constexpr auto windowSize = filterSize + tileSize - 1;

        using TileArray = Fixed2DArray<ValueType, tileSize, tileSize>;
        using WindowArray = Fixed2DArray<ValueType, windowSize, windowSize>;

        template <typename MatrixType1, typename MatrixType2>
        static void TransformInputWindow(const MatrixType1& d, MatrixType2& X)
        {
            // Compute B'dB
            X(0, 0) = ((((((4 * d(0, 0)) + (-5 * d(2, 0))) + d(4, 0)) * 4) + ((((4 * d(0, 2)) + (-5 * d(2, 2))) + d(4, 2)) * -5)) + (((4 * d(0, 4)) + (-5 * d(2, 4))) + d(4, 4)));
            X(0, 1) = (((((((4 * d(0, 1)) + (-5 * d(2, 1))) + d(4, 1)) * -4) + ((((4 * d(0, 2)) + (-5 * d(2, 2))) + d(4, 2)) * -4)) + (((4 * d(0, 3)) + (-5 * d(2, 3))) + d(4, 3))) + (((4 * d(0, 4)) + (-5 * d(2, 4))) + d(4, 4)));
            X(0, 2) = (((((((4 * d(0, 1)) + (-5 * d(2, 1))) + d(4, 1)) * 4) + ((((4 * d(0, 2)) + (-5 * d(2, 2))) + d(4, 2)) * -4)) - (((4 * d(0, 3)) + (-5 * d(2, 3))) + d(4, 3))) + (((4 * d(0, 4)) + (-5 * d(2, 4))) + d(4, 4)));
            X(0, 3) = (((((((4 * d(0, 1)) + (-5 * d(2, 1))) + d(4, 1)) * -2) - (((4 * d(0, 2)) + (-5 * d(2, 2))) + d(4, 2))) + ((((4 * d(0, 3)) + (-5 * d(2, 3))) + d(4, 3)) * 2)) + (((4 * d(0, 4)) + (-5 * d(2, 4))) + d(4, 4)));
            X(0, 4) = (((((((4 * d(0, 1)) + (-5 * d(2, 1))) + d(4, 1)) * 2) - (((4 * d(0, 2)) + (-5 * d(2, 2))) + d(4, 2))) + ((((4 * d(0, 3)) + (-5 * d(2, 3))) + d(4, 3)) * -2)) + (((4 * d(0, 4)) + (-5 * d(2, 4))) + d(4, 4)));
            X(0, 5) = ((((((4 * d(0, 1)) + (-5 * d(2, 1))) + d(4, 1)) * 4) + ((((4 * d(0, 3)) + (-5 * d(2, 3))) + d(4, 3)) * -5)) + (((4 * d(0, 5)) + (-5 * d(2, 5))) + d(4, 5)));
            X(1, 0) = (((((((-4 * d(1, 0)) + (-4 * d(2, 0))) + d(3, 0)) + d(4, 0)) * 4) + (((((-4 * d(1, 2)) + (-4 * d(2, 2))) + d(3, 2)) + d(4, 2)) * -5)) + ((((-4 * d(1, 4)) + (-4 * d(2, 4))) + d(3, 4)) + d(4, 4)));
            X(1, 1) = ((((((((-4 * d(1, 1)) + (-4 * d(2, 1))) + d(3, 1)) + d(4, 1)) * -4) + (((((-4 * d(1, 2)) + (-4 * d(2, 2))) + d(3, 2)) + d(4, 2)) * -4)) + ((((-4 * d(1, 3)) + (-4 * d(2, 3))) + d(3, 3)) + d(4, 3))) + ((((-4 * d(1, 4)) + (-4 * d(2, 4))) + d(3, 4)) + d(4, 4)));
            X(1, 2) = ((((((((-4 * d(1, 1)) + (-4 * d(2, 1))) + d(3, 1)) + d(4, 1)) * 4) + (((((-4 * d(1, 2)) + (-4 * d(2, 2))) + d(3, 2)) + d(4, 2)) * -4)) - ((((-4 * d(1, 3)) + (-4 * d(2, 3))) + d(3, 3)) + d(4, 3))) + ((((-4 * d(1, 4)) + (-4 * d(2, 4))) + d(3, 4)) + d(4, 4)));
            X(1, 3) = ((((((((-4 * d(1, 1)) + (-4 * d(2, 1))) + d(3, 1)) + d(4, 1)) * -2) - ((((-4 * d(1, 2)) + (-4 * d(2, 2))) + d(3, 2)) + d(4, 2))) + (((((-4 * d(1, 3)) + (-4 * d(2, 3))) + d(3, 3)) + d(4, 3)) * 2)) + ((((-4 * d(1, 4)) + (-4 * d(2, 4))) + d(3, 4)) + d(4, 4)));
            X(1, 4) = ((((((((-4 * d(1, 1)) + (-4 * d(2, 1))) + d(3, 1)) + d(4, 1)) * 2) - ((((-4 * d(1, 2)) + (-4 * d(2, 2))) + d(3, 2)) + d(4, 2))) + (((((-4 * d(1, 3)) + (-4 * d(2, 3))) + d(3, 3)) + d(4, 3)) * -2)) + ((((-4 * d(1, 4)) + (-4 * d(2, 4))) + d(3, 4)) + d(4, 4)));
            X(1, 5) = (((((((-4 * d(1, 1)) + (-4 * d(2, 1))) + d(3, 1)) + d(4, 1)) * 4) + (((((-4 * d(1, 3)) + (-4 * d(2, 3))) + d(3, 3)) + d(4, 3)) * -5)) + ((((-4 * d(1, 5)) + (-4 * d(2, 5))) + d(3, 5)) + d(4, 5)));
            X(2, 0) = (((((((4 * d(1, 0)) + (-4 * d(2, 0))) - d(3, 0)) + d(4, 0)) * 4) + (((((4 * d(1, 2)) + (-4 * d(2, 2))) - d(3, 2)) + d(4, 2)) * -5)) + ((((4 * d(1, 4)) + (-4 * d(2, 4))) - d(3, 4)) + d(4, 4)));
            X(2, 1) = ((((((((4 * d(1, 1)) + (-4 * d(2, 1))) - d(3, 1)) + d(4, 1)) * -4) + (((((4 * d(1, 2)) + (-4 * d(2, 2))) - d(3, 2)) + d(4, 2)) * -4)) + ((((4 * d(1, 3)) + (-4 * d(2, 3))) - d(3, 3)) + d(4, 3))) + ((((4 * d(1, 4)) + (-4 * d(2, 4))) - d(3, 4)) + d(4, 4)));
            X(2, 2) = ((((((((4 * d(1, 1)) + (-4 * d(2, 1))) - d(3, 1)) + d(4, 1)) * 4) + (((((4 * d(1, 2)) + (-4 * d(2, 2))) - d(3, 2)) + d(4, 2)) * -4)) - ((((4 * d(1, 3)) + (-4 * d(2, 3))) - d(3, 3)) + d(4, 3))) + ((((4 * d(1, 4)) + (-4 * d(2, 4))) - d(3, 4)) + d(4, 4)));
            X(2, 3) = ((((((((4 * d(1, 1)) + (-4 * d(2, 1))) - d(3, 1)) + d(4, 1)) * -2) - ((((4 * d(1, 2)) + (-4 * d(2, 2))) - d(3, 2)) + d(4, 2))) + (((((4 * d(1, 3)) + (-4 * d(2, 3))) - d(3, 3)) + d(4, 3)) * 2)) + ((((4 * d(1, 4)) + (-4 * d(2, 4))) - d(3, 4)) + d(4, 4)));
            X(2, 4) = ((((((((4 * d(1, 1)) + (-4 * d(2, 1))) - d(3, 1)) + d(4, 1)) * 2) - ((((4 * d(1, 2)) + (-4 * d(2, 2))) - d(3, 2)) + d(4, 2))) + (((((4 * d(1, 3)) + (-4 * d(2, 3))) - d(3, 3)) + d(4, 3)) * -2)) + ((((4 * d(1, 4)) + (-4 * d(2, 4))) - d(3, 4)) + d(4, 4)));
            X(2, 5) = (((((((4 * d(1, 1)) + (-4 * d(2, 1))) - d(3, 1)) + d(4, 1)) * 4) + (((((4 * d(1, 3)) + (-4 * d(2, 3))) - d(3, 3)) + d(4, 3)) * -5)) + ((((4 * d(1, 5)) + (-4 * d(2, 5))) - d(3, 5)) + d(4, 5)));
            X(3, 0) = (((((((-2 * d(1, 0)) - d(2, 0)) + (2 * d(3, 0))) + d(4, 0)) * 4) + (((((-2 * d(1, 2)) - d(2, 2)) + (2 * d(3, 2))) + d(4, 2)) * -5)) + ((((-2 * d(1, 4)) - d(2, 4)) + (2 * d(3, 4))) + d(4, 4)));
            X(3, 1) = ((((((((-2 * d(1, 1)) - d(2, 1)) + (2 * d(3, 1))) + d(4, 1)) * -4) + (((((-2 * d(1, 2)) - d(2, 2)) + (2 * d(3, 2))) + d(4, 2)) * -4)) + ((((-2 * d(1, 3)) - d(2, 3)) + (2 * d(3, 3))) + d(4, 3))) + ((((-2 * d(1, 4)) - d(2, 4)) + (2 * d(3, 4))) + d(4, 4)));
            X(3, 2) = ((((((((-2 * d(1, 1)) - d(2, 1)) + (2 * d(3, 1))) + d(4, 1)) * 4) + (((((-2 * d(1, 2)) - d(2, 2)) + (2 * d(3, 2))) + d(4, 2)) * -4)) - ((((-2 * d(1, 3)) - d(2, 3)) + (2 * d(3, 3))) + d(4, 3))) + ((((-2 * d(1, 4)) - d(2, 4)) + (2 * d(3, 4))) + d(4, 4)));
            X(3, 3) = ((((((((-2 * d(1, 1)) - d(2, 1)) + (2 * d(3, 1))) + d(4, 1)) * -2) - ((((-2 * d(1, 2)) - d(2, 2)) + (2 * d(3, 2))) + d(4, 2))) + (((((-2 * d(1, 3)) - d(2, 3)) + (2 * d(3, 3))) + d(4, 3)) * 2)) + ((((-2 * d(1, 4)) - d(2, 4)) + (2 * d(3, 4))) + d(4, 4)));
            X(3, 4) = ((((((((-2 * d(1, 1)) - d(2, 1)) + (2 * d(3, 1))) + d(4, 1)) * 2) - ((((-2 * d(1, 2)) - d(2, 2)) + (2 * d(3, 2))) + d(4, 2))) + (((((-2 * d(1, 3)) - d(2, 3)) + (2 * d(3, 3))) + d(4, 3)) * -2)) + ((((-2 * d(1, 4)) - d(2, 4)) + (2 * d(3, 4))) + d(4, 4)));
            X(3, 5) = (((((((-2 * d(1, 1)) - d(2, 1)) + (2 * d(3, 1))) + d(4, 1)) * 4) + (((((-2 * d(1, 3)) - d(2, 3)) + (2 * d(3, 3))) + d(4, 3)) * -5)) + ((((-2 * d(1, 5)) - d(2, 5)) + (2 * d(3, 5))) + d(4, 5)));
            X(4, 0) = (((((((2 * d(1, 0)) - d(2, 0)) + (-2 * d(3, 0))) + d(4, 0)) * 4) + (((((2 * d(1, 2)) - d(2, 2)) + (-2 * d(3, 2))) + d(4, 2)) * -5)) + ((((2 * d(1, 4)) - d(2, 4)) + (-2 * d(3, 4))) + d(4, 4)));
            X(4, 1) = ((((((((2 * d(1, 1)) - d(2, 1)) + (-2 * d(3, 1))) + d(4, 1)) * -4) + (((((2 * d(1, 2)) - d(2, 2)) + (-2 * d(3, 2))) + d(4, 2)) * -4)) + ((((2 * d(1, 3)) - d(2, 3)) + (-2 * d(3, 3))) + d(4, 3))) + ((((2 * d(1, 4)) - d(2, 4)) + (-2 * d(3, 4))) + d(4, 4)));
            X(4, 2) = ((((((((2 * d(1, 1)) - d(2, 1)) + (-2 * d(3, 1))) + d(4, 1)) * 4) + (((((2 * d(1, 2)) - d(2, 2)) + (-2 * d(3, 2))) + d(4, 2)) * -4)) - ((((2 * d(1, 3)) - d(2, 3)) + (-2 * d(3, 3))) + d(4, 3))) + ((((2 * d(1, 4)) - d(2, 4)) + (-2 * d(3, 4))) + d(4, 4)));
            X(4, 3) = ((((((((2 * d(1, 1)) - d(2, 1)) + (-2 * d(3, 1))) + d(4, 1)) * -2) - ((((2 * d(1, 2)) - d(2, 2)) + (-2 * d(3, 2))) + d(4, 2))) + (((((2 * d(1, 3)) - d(2, 3)) + (-2 * d(3, 3))) + d(4, 3)) * 2)) + ((((2 * d(1, 4)) - d(2, 4)) + (-2 * d(3, 4))) + d(4, 4)));
            X(4, 4) = ((((((((2 * d(1, 1)) - d(2, 1)) + (-2 * d(3, 1))) + d(4, 1)) * 2) - ((((2 * d(1, 2)) - d(2, 2)) + (-2 * d(3, 2))) + d(4, 2))) + (((((2 * d(1, 3)) - d(2, 3)) + (-2 * d(3, 3))) + d(4, 3)) * -2)) + ((((2 * d(1, 4)) - d(2, 4)) + (-2 * d(3, 4))) + d(4, 4)));
            X(4, 5) = (((((((2 * d(1, 1)) - d(2, 1)) + (-2 * d(3, 1))) + d(4, 1)) * 4) + (((((2 * d(1, 3)) - d(2, 3)) + (-2 * d(3, 3))) + d(4, 3)) * -5)) + ((((2 * d(1, 5)) - d(2, 5)) + (-2 * d(3, 5))) + d(4, 5)));
            X(5, 0) = ((((((4 * d(1, 0)) + (-5 * d(3, 0))) + d(5, 0)) * 4) + ((((4 * d(1, 2)) + (-5 * d(3, 2))) + d(5, 2)) * -5)) + (((4 * d(1, 4)) + (-5 * d(3, 4))) + d(5, 4)));
            X(5, 1) = (((((((4 * d(1, 1)) + (-5 * d(3, 1))) + d(5, 1)) * -4) + ((((4 * d(1, 2)) + (-5 * d(3, 2))) + d(5, 2)) * -4)) + (((4 * d(1, 3)) + (-5 * d(3, 3))) + d(5, 3))) + (((4 * d(1, 4)) + (-5 * d(3, 4))) + d(5, 4)));
            X(5, 2) = (((((((4 * d(1, 1)) + (-5 * d(3, 1))) + d(5, 1)) * 4) + ((((4 * d(1, 2)) + (-5 * d(3, 2))) + d(5, 2)) * -4)) - (((4 * d(1, 3)) + (-5 * d(3, 3))) + d(5, 3))) + (((4 * d(1, 4)) + (-5 * d(3, 4))) + d(5, 4)));
            X(5, 3) = (((((((4 * d(1, 1)) + (-5 * d(3, 1))) + d(5, 1)) * -2) - (((4 * d(1, 2)) + (-5 * d(3, 2))) + d(5, 2))) + ((((4 * d(1, 3)) + (-5 * d(3, 3))) + d(5, 3)) * 2)) + (((4 * d(1, 4)) + (-5 * d(3, 4))) + d(5, 4)));
            X(5, 4) = (((((((4 * d(1, 1)) + (-5 * d(3, 1))) + d(5, 1)) * 2) - (((4 * d(1, 2)) + (-5 * d(3, 2))) + d(5, 2))) + ((((4 * d(1, 3)) + (-5 * d(3, 3))) + d(5, 3)) * -2)) + (((4 * d(1, 4)) + (-5 * d(3, 4))) + d(5, 4)));
            X(5, 5) = ((((((4 * d(1, 1)) + (-5 * d(3, 1))) + d(5, 1)) * 4) + ((((4 * d(1, 3)) + (-5 * d(3, 3))) + d(5, 3)) * -5)) + (((4 * d(1, 5)) + (-5 * d(3, 5))) + d(5, 5)));
        }

        template <typename BlockType1, typename BlockType2>
        static inline void TransformInputBlock(const BlockType1& d, int blockSize, BlockType2& X)
        {
            // Compute B'dB
            for (int index = 0; index < blockSize; ++index)
            {
                X(0, 0, index) = ((((((4 * d(0, 0, index)) + (-5 * d(2, 0, index))) + d(4, 0, index)) * 4) + ((((4 * d(0, 2, index)) + (-5 * d(2, 2, index))) + d(4, 2, index)) * -5)) + (((4 * d(0, 4, index)) + (-5 * d(2, 4, index))) + d(4, 4, index)));
                X(0, 1, index) = (((((((4 * d(0, 1, index)) + (-5 * d(2, 1, index))) + d(4, 1, index)) * -4) + ((((4 * d(0, 2, index)) + (-5 * d(2, 2, index))) + d(4, 2, index)) * -4)) + (((4 * d(0, 3, index)) + (-5 * d(2, 3, index))) + d(4, 3, index))) + (((4 * d(0, 4, index)) + (-5 * d(2, 4, index))) + d(4, 4, index)));
                X(0, 2, index) = (((((((4 * d(0, 1, index)) + (-5 * d(2, 1, index))) + d(4, 1, index)) * 4) + ((((4 * d(0, 2, index)) + (-5 * d(2, 2, index))) + d(4, 2, index)) * -4)) - (((4 * d(0, 3, index)) + (-5 * d(2, 3, index))) + d(4, 3, index))) + (((4 * d(0, 4, index)) + (-5 * d(2, 4, index))) + d(4, 4, index)));
                X(0, 3, index) = (((((((4 * d(0, 1, index)) + (-5 * d(2, 1, index))) + d(4, 1, index)) * -2) - (((4 * d(0, 2, index)) + (-5 * d(2, 2, index))) + d(4, 2, index))) + ((((4 * d(0, 3, index)) + (-5 * d(2, 3, index))) + d(4, 3, index)) * 2)) + (((4 * d(0, 4, index)) + (-5 * d(2, 4, index))) + d(4, 4, index)));
                X(0, 4, index) = (((((((4 * d(0, 1, index)) + (-5 * d(2, 1, index))) + d(4, 1, index)) * 2) - (((4 * d(0, 2, index)) + (-5 * d(2, 2, index))) + d(4, 2, index))) + ((((4 * d(0, 3, index)) + (-5 * d(2, 3, index))) + d(4, 3, index)) * -2)) + (((4 * d(0, 4, index)) + (-5 * d(2, 4, index))) + d(4, 4, index)));
                X(0, 5, index) = ((((((4 * d(0, 1, index)) + (-5 * d(2, 1, index))) + d(4, 1, index)) * 4) + ((((4 * d(0, 3, index)) + (-5 * d(2, 3, index))) + d(4, 3, index)) * -5)) + (((4 * d(0, 5, index)) + (-5 * d(2, 5, index))) + d(4, 5, index)));
                X(1, 0, index) = (((((((-4 * d(1, 0, index)) + (-4 * d(2, 0, index))) + d(3, 0, index)) + d(4, 0, index)) * 4) + (((((-4 * d(1, 2, index)) + (-4 * d(2, 2, index))) + d(3, 2, index)) + d(4, 2, index)) * -5)) + ((((-4 * d(1, 4, index)) + (-4 * d(2, 4, index))) + d(3, 4, index)) + d(4, 4, index)));
                X(1, 1, index) = ((((((((-4 * d(1, 1, index)) + (-4 * d(2, 1, index))) + d(3, 1, index)) + d(4, 1, index)) * -4) + (((((-4 * d(1, 2, index)) + (-4 * d(2, 2, index))) + d(3, 2, index)) + d(4, 2, index)) * -4)) + ((((-4 * d(1, 3, index)) + (-4 * d(2, 3, index))) + d(3, 3, index)) + d(4, 3, index))) + ((((-4 * d(1, 4, index)) + (-4 * d(2, 4, index))) + d(3, 4, index)) + d(4, 4, index)));
                X(1, 2, index) = ((((((((-4 * d(1, 1, index)) + (-4 * d(2, 1, index))) + d(3, 1, index)) + d(4, 1, index)) * 4) + (((((-4 * d(1, 2, index)) + (-4 * d(2, 2, index))) + d(3, 2, index)) + d(4, 2, index)) * -4)) - ((((-4 * d(1, 3, index)) + (-4 * d(2, 3, index))) + d(3, 3, index)) + d(4, 3, index))) + ((((-4 * d(1, 4, index)) + (-4 * d(2, 4, index))) + d(3, 4, index)) + d(4, 4, index)));
                X(1, 3, index) = ((((((((-4 * d(1, 1, index)) + (-4 * d(2, 1, index))) + d(3, 1, index)) + d(4, 1, index)) * -2) - ((((-4 * d(1, 2, index)) + (-4 * d(2, 2, index))) + d(3, 2, index)) + d(4, 2, index))) + (((((-4 * d(1, 3, index)) + (-4 * d(2, 3, index))) + d(3, 3, index)) + d(4, 3, index)) * 2)) + ((((-4 * d(1, 4, index)) + (-4 * d(2, 4, index))) + d(3, 4, index)) + d(4, 4, index)));
                X(1, 4, index) = ((((((((-4 * d(1, 1, index)) + (-4 * d(2, 1, index))) + d(3, 1, index)) + d(4, 1, index)) * 2) - ((((-4 * d(1, 2, index)) + (-4 * d(2, 2, index))) + d(3, 2, index)) + d(4, 2, index))) + (((((-4 * d(1, 3, index)) + (-4 * d(2, 3, index))) + d(3, 3, index)) + d(4, 3, index)) * -2)) + ((((-4 * d(1, 4, index)) + (-4 * d(2, 4, index))) + d(3, 4, index)) + d(4, 4, index)));
                X(1, 5, index) = (((((((-4 * d(1, 1, index)) + (-4 * d(2, 1, index))) + d(3, 1, index)) + d(4, 1, index)) * 4) + (((((-4 * d(1, 3, index)) + (-4 * d(2, 3, index))) + d(3, 3, index)) + d(4, 3, index)) * -5)) + ((((-4 * d(1, 5, index)) + (-4 * d(2, 5, index))) + d(3, 5, index)) + d(4, 5, index)));
                X(2, 0, index) = (((((((4 * d(1, 0, index)) + (-4 * d(2, 0, index))) - d(3, 0, index)) + d(4, 0, index)) * 4) + (((((4 * d(1, 2, index)) + (-4 * d(2, 2, index))) - d(3, 2, index)) + d(4, 2, index)) * -5)) + ((((4 * d(1, 4, index)) + (-4 * d(2, 4, index))) - d(3, 4, index)) + d(4, 4, index)));
                X(2, 1, index) = ((((((((4 * d(1, 1, index)) + (-4 * d(2, 1, index))) - d(3, 1, index)) + d(4, 1, index)) * -4) + (((((4 * d(1, 2, index)) + (-4 * d(2, 2, index))) - d(3, 2, index)) + d(4, 2, index)) * -4)) + ((((4 * d(1, 3, index)) + (-4 * d(2, 3, index))) - d(3, 3, index)) + d(4, 3, index))) + ((((4 * d(1, 4, index)) + (-4 * d(2, 4, index))) - d(3, 4, index)) + d(4, 4, index)));
                X(2, 2, index) = ((((((((4 * d(1, 1, index)) + (-4 * d(2, 1, index))) - d(3, 1, index)) + d(4, 1, index)) * 4) + (((((4 * d(1, 2, index)) + (-4 * d(2, 2, index))) - d(3, 2, index)) + d(4, 2, index)) * -4)) - ((((4 * d(1, 3, index)) + (-4 * d(2, 3, index))) - d(3, 3, index)) + d(4, 3, index))) + ((((4 * d(1, 4, index)) + (-4 * d(2, 4, index))) - d(3, 4, index)) + d(4, 4, index)));
                X(2, 3, index) = ((((((((4 * d(1, 1, index)) + (-4 * d(2, 1, index))) - d(3, 1, index)) + d(4, 1, index)) * -2) - ((((4 * d(1, 2, index)) + (-4 * d(2, 2, index))) - d(3, 2, index)) + d(4, 2, index))) + (((((4 * d(1, 3, index)) + (-4 * d(2, 3, index))) - d(3, 3, index)) + d(4, 3, index)) * 2)) + ((((4 * d(1, 4, index)) + (-4 * d(2, 4, index))) - d(3, 4, index)) + d(4, 4, index)));
                X(2, 4, index) = ((((((((4 * d(1, 1, index)) + (-4 * d(2, 1, index))) - d(3, 1, index)) + d(4, 1, index)) * 2) - ((((4 * d(1, 2, index)) + (-4 * d(2, 2, index))) - d(3, 2, index)) + d(4, 2, index))) + (((((4 * d(1, 3, index)) + (-4 * d(2, 3, index))) - d(3, 3, index)) + d(4, 3, index)) * -2)) + ((((4 * d(1, 4, index)) + (-4 * d(2, 4, index))) - d(3, 4, index)) + d(4, 4, index)));
                X(2, 5, index) = (((((((4 * d(1, 1, index)) + (-4 * d(2, 1, index))) - d(3, 1, index)) + d(4, 1, index)) * 4) + (((((4 * d(1, 3, index)) + (-4 * d(2, 3, index))) - d(3, 3, index)) + d(4, 3, index)) * -5)) + ((((4 * d(1, 5, index)) + (-4 * d(2, 5, index))) - d(3, 5, index)) + d(4, 5, index)));
                X(3, 0, index) = (((((((-2 * d(1, 0, index)) - d(2, 0, index)) + (2 * d(3, 0, index))) + d(4, 0, index)) * 4) + (((((-2 * d(1, 2, index)) - d(2, 2, index)) + (2 * d(3, 2, index))) + d(4, 2, index)) * -5)) + ((((-2 * d(1, 4, index)) - d(2, 4, index)) + (2 * d(3, 4, index))) + d(4, 4, index)));
                X(3, 1, index) = ((((((((-2 * d(1, 1, index)) - d(2, 1, index)) + (2 * d(3, 1, index))) + d(4, 1, index)) * -4) + (((((-2 * d(1, 2, index)) - d(2, 2, index)) + (2 * d(3, 2, index))) + d(4, 2, index)) * -4)) + ((((-2 * d(1, 3, index)) - d(2, 3, index)) + (2 * d(3, 3, index))) + d(4, 3, index))) + ((((-2 * d(1, 4, index)) - d(2, 4, index)) + (2 * d(3, 4, index))) + d(4, 4, index)));
                X(3, 2, index) = ((((((((-2 * d(1, 1, index)) - d(2, 1, index)) + (2 * d(3, 1, index))) + d(4, 1, index)) * 4) + (((((-2 * d(1, 2, index)) - d(2, 2, index)) + (2 * d(3, 2, index))) + d(4, 2, index)) * -4)) - ((((-2 * d(1, 3, index)) - d(2, 3, index)) + (2 * d(3, 3, index))) + d(4, 3, index))) + ((((-2 * d(1, 4, index)) - d(2, 4, index)) + (2 * d(3, 4, index))) + d(4, 4, index)));
                X(3, 3, index) = ((((((((-2 * d(1, 1, index)) - d(2, 1, index)) + (2 * d(3, 1, index))) + d(4, 1, index)) * -2) - ((((-2 * d(1, 2, index)) - d(2, 2, index)) + (2 * d(3, 2, index))) + d(4, 2, index))) + (((((-2 * d(1, 3, index)) - d(2, 3, index)) + (2 * d(3, 3, index))) + d(4, 3, index)) * 2)) + ((((-2 * d(1, 4, index)) - d(2, 4, index)) + (2 * d(3, 4, index))) + d(4, 4, index)));
                X(3, 4, index) = ((((((((-2 * d(1, 1, index)) - d(2, 1, index)) + (2 * d(3, 1, index))) + d(4, 1, index)) * 2) - ((((-2 * d(1, 2, index)) - d(2, 2, index)) + (2 * d(3, 2, index))) + d(4, 2, index))) + (((((-2 * d(1, 3, index)) - d(2, 3, index)) + (2 * d(3, 3, index))) + d(4, 3, index)) * -2)) + ((((-2 * d(1, 4, index)) - d(2, 4, index)) + (2 * d(3, 4, index))) + d(4, 4, index)));
                X(3, 5, index) = (((((((-2 * d(1, 1, index)) - d(2, 1, index)) + (2 * d(3, 1, index))) + d(4, 1, index)) * 4) + (((((-2 * d(1, 3, index)) - d(2, 3, index)) + (2 * d(3, 3, index))) + d(4, 3, index)) * -5)) + ((((-2 * d(1, 5, index)) - d(2, 5, index)) + (2 * d(3, 5, index))) + d(4, 5, index)));
                X(4, 0, index) = (((((((2 * d(1, 0, index)) - d(2, 0, index)) + (-2 * d(3, 0, index))) + d(4, 0, index)) * 4) + (((((2 * d(1, 2, index)) - d(2, 2, index)) + (-2 * d(3, 2, index))) + d(4, 2, index)) * -5)) + ((((2 * d(1, 4, index)) - d(2, 4, index)) + (-2 * d(3, 4, index))) + d(4, 4, index)));
                X(4, 1, index) = ((((((((2 * d(1, 1, index)) - d(2, 1, index)) + (-2 * d(3, 1, index))) + d(4, 1, index)) * -4) + (((((2 * d(1, 2, index)) - d(2, 2, index)) + (-2 * d(3, 2, index))) + d(4, 2, index)) * -4)) + ((((2 * d(1, 3, index)) - d(2, 3, index)) + (-2 * d(3, 3, index))) + d(4, 3, index))) + ((((2 * d(1, 4, index)) - d(2, 4, index)) + (-2 * d(3, 4, index))) + d(4, 4, index)));
                X(4, 2, index) = ((((((((2 * d(1, 1, index)) - d(2, 1, index)) + (-2 * d(3, 1, index))) + d(4, 1, index)) * 4) + (((((2 * d(1, 2, index)) - d(2, 2, index)) + (-2 * d(3, 2, index))) + d(4, 2, index)) * -4)) - ((((2 * d(1, 3, index)) - d(2, 3, index)) + (-2 * d(3, 3, index))) + d(4, 3, index))) + ((((2 * d(1, 4, index)) - d(2, 4, index)) + (-2 * d(3, 4, index))) + d(4, 4, index)));
                X(4, 3, index) = ((((((((2 * d(1, 1, index)) - d(2, 1, index)) + (-2 * d(3, 1, index))) + d(4, 1, index)) * -2) - ((((2 * d(1, 2, index)) - d(2, 2, index)) + (-2 * d(3, 2, index))) + d(4, 2, index))) + (((((2 * d(1, 3, index)) - d(2, 3, index)) + (-2 * d(3, 3, index))) + d(4, 3, index)) * 2)) + ((((2 * d(1, 4, index)) - d(2, 4, index)) + (-2 * d(3, 4, index))) + d(4, 4, index)));
                X(4, 4, index) = ((((((((2 * d(1, 1, index)) - d(2, 1, index)) + (-2 * d(3, 1, index))) + d(4, 1, index)) * 2) - ((((2 * d(1, 2, index)) - d(2, 2, index)) + (-2 * d(3, 2, index))) + d(4, 2, index))) + (((((2 * d(1, 3, index)) - d(2, 3, index)) + (-2 * d(3, 3, index))) + d(4, 3, index)) * -2)) + ((((2 * d(1, 4, index)) - d(2, 4, index)) + (-2 * d(3, 4, index))) + d(4, 4, index)));
                X(4, 5, index) = (((((((2 * d(1, 1, index)) - d(2, 1, index)) + (-2 * d(3, 1, index))) + d(4, 1, index)) * 4) + (((((2 * d(1, 3, index)) - d(2, 3, index)) + (-2 * d(3, 3, index))) + d(4, 3, index)) * -5)) + ((((2 * d(1, 5, index)) - d(2, 5, index)) + (-2 * d(3, 5, index))) + d(4, 5, index)));
                X(5, 0, index) = ((((((4 * d(1, 0, index)) + (-5 * d(3, 0, index))) + d(5, 0, index)) * 4) + ((((4 * d(1, 2, index)) + (-5 * d(3, 2, index))) + d(5, 2, index)) * -5)) + (((4 * d(1, 4, index)) + (-5 * d(3, 4, index))) + d(5, 4, index)));
                X(5, 1, index) = (((((((4 * d(1, 1, index)) + (-5 * d(3, 1, index))) + d(5, 1, index)) * -4) + ((((4 * d(1, 2, index)) + (-5 * d(3, 2, index))) + d(5, 2, index)) * -4)) + (((4 * d(1, 3, index)) + (-5 * d(3, 3, index))) + d(5, 3, index))) + (((4 * d(1, 4, index)) + (-5 * d(3, 4, index))) + d(5, 4, index)));
                X(5, 2, index) = (((((((4 * d(1, 1, index)) + (-5 * d(3, 1, index))) + d(5, 1, index)) * 4) + ((((4 * d(1, 2, index)) + (-5 * d(3, 2, index))) + d(5, 2, index)) * -4)) - (((4 * d(1, 3, index)) + (-5 * d(3, 3, index))) + d(5, 3, index))) + (((4 * d(1, 4, index)) + (-5 * d(3, 4, index))) + d(5, 4, index)));
                X(5, 3, index) = (((((((4 * d(1, 1, index)) + (-5 * d(3, 1, index))) + d(5, 1, index)) * -2) - (((4 * d(1, 2, index)) + (-5 * d(3, 2, index))) + d(5, 2, index))) + ((((4 * d(1, 3, index)) + (-5 * d(3, 3, index))) + d(5, 3, index)) * 2)) + (((4 * d(1, 4, index)) + (-5 * d(3, 4, index))) + d(5, 4, index)));
                X(5, 4, index) = (((((((4 * d(1, 1, index)) + (-5 * d(3, 1, index))) + d(5, 1, index)) * 2) - (((4 * d(1, 2, index)) + (-5 * d(3, 2, index))) + d(5, 2, index))) + ((((4 * d(1, 3, index)) + (-5 * d(3, 3, index))) + d(5, 3, index)) * -2)) + (((4 * d(1, 4, index)) + (-5 * d(3, 4, index))) + d(5, 4, index)));
                X(5, 5, index) = ((((((4 * d(1, 1, index)) + (-5 * d(3, 1, index))) + d(5, 1, index)) * 4) + ((((4 * d(1, 3, index)) + (-5 * d(3, 3, index))) + d(5, 3, index)) * -5)) + (((4 * d(1, 5, index)) + (-5 * d(3, 5, index))) + d(5, 5, index)));
            }
        }

        template <typename MatrixType1, typename MatrixType2>
        static void TransformOutputTile(const MatrixType1& X, MatrixType2& result)
        {
            result(0, 0) = ((((((((X(0, 0) + X(1, 0)) + X(2, 0)) + X(3, 0)) + X(4, 0)) + ((((X(0, 1) + X(1, 1)) + X(2, 1)) + X(3, 1)) + X(4, 1))) + ((((X(0, 2) + X(1, 2)) + X(2, 2)) + X(3, 2)) + X(4, 2))) + ((((X(0, 3) + X(1, 3)) + X(2, 3)) + X(3, 3)) + X(4, 3))) + ((((X(0, 4) + X(1, 4)) + X(2, 4)) + X(3, 4)) + X(4, 4)));
            result(0, 1) = (((((((X(0, 1) + X(1, 1)) + X(2, 1)) + X(3, 1)) + X(4, 1)) - ((((X(0, 2) + X(1, 2)) + X(2, 2)) + X(3, 2)) + X(4, 2))) + (((((X(0, 3) + X(1, 3)) + X(2, 3)) + X(3, 3)) + X(4, 3)) * 2)) + (((((X(0, 4) + X(1, 4)) + X(2, 4)) + X(3, 4)) + X(4, 4)) * -2));
            result(0, 2) = (((((((X(0, 1) + X(1, 1)) + X(2, 1)) + X(3, 1)) + X(4, 1)) + ((((X(0, 2) + X(1, 2)) + X(2, 2)) + X(3, 2)) + X(4, 2))) + (((((X(0, 3) + X(1, 3)) + X(2, 3)) + X(3, 3)) + X(4, 3)) * 4)) + (((((X(0, 4) + X(1, 4)) + X(2, 4)) + X(3, 4)) + X(4, 4)) * 4));
            result(0, 3) = ((((((((X(0, 1) + X(1, 1)) + X(2, 1)) + X(3, 1)) + X(4, 1)) - ((((X(0, 2) + X(1, 2)) + X(2, 2)) + X(3, 2)) + X(4, 2))) + (((((X(0, 3) + X(1, 3)) + X(2, 3)) + X(3, 3)) + X(4, 3)) * 8)) + (((((X(0, 4) + X(1, 4)) + X(2, 4)) + X(3, 4)) + X(4, 4)) * -8)) + ((((X(0, 5) + X(1, 5)) + X(2, 5)) + X(3, 5)) + X(4, 5)));
            result(1, 0) = (((((((X(1, 0) - X(2, 0)) + (2 * X(3, 0))) + (-2 * X(4, 0))) + (((X(1, 1) - X(2, 1)) + (2 * X(3, 1))) + (-2 * X(4, 1)))) + (((X(1, 2) - X(2, 2)) + (2 * X(3, 2))) + (-2 * X(4, 2)))) + (((X(1, 3) - X(2, 3)) + (2 * X(3, 3))) + (-2 * X(4, 3)))) + (((X(1, 4) - X(2, 4)) + (2 * X(3, 4))) + (-2 * X(4, 4))));
            result(1, 1) = ((((((X(1, 1) - X(2, 1)) + (2 * X(3, 1))) + (-2 * X(4, 1))) - (((X(1, 2) - X(2, 2)) + (2 * X(3, 2))) + (-2 * X(4, 2)))) + ((((X(1, 3) - X(2, 3)) + (2 * X(3, 3))) + (-2 * X(4, 3))) * 2)) + ((((X(1, 4) - X(2, 4)) + (2 * X(3, 4))) + (-2 * X(4, 4))) * -2));
            result(1, 2) = ((((((X(1, 1) - X(2, 1)) + (2 * X(3, 1))) + (-2 * X(4, 1))) + (((X(1, 2) - X(2, 2)) + (2 * X(3, 2))) + (-2 * X(4, 2)))) + ((((X(1, 3) - X(2, 3)) + (2 * X(3, 3))) + (-2 * X(4, 3))) * 4)) + ((((X(1, 4) - X(2, 4)) + (2 * X(3, 4))) + (-2 * X(4, 4))) * 4));
            result(1, 3) = (((((((X(1, 1) - X(2, 1)) + (2 * X(3, 1))) + (-2 * X(4, 1))) - (((X(1, 2) - X(2, 2)) + (2 * X(3, 2))) + (-2 * X(4, 2)))) + ((((X(1, 3) - X(2, 3)) + (2 * X(3, 3))) + (-2 * X(4, 3))) * 8)) + ((((X(1, 4) - X(2, 4)) + (2 * X(3, 4))) + (-2 * X(4, 4))) * -8)) + (((X(1, 5) - X(2, 5)) + (2 * X(3, 5))) + (-2 * X(4, 5))));
            result(2, 0) = (((((((X(1, 0) + X(2, 0)) + (4 * X(3, 0))) + (4 * X(4, 0))) + (((X(1, 1) + X(2, 1)) + (4 * X(3, 1))) + (4 * X(4, 1)))) + (((X(1, 2) + X(2, 2)) + (4 * X(3, 2))) + (4 * X(4, 2)))) + (((X(1, 3) + X(2, 3)) + (4 * X(3, 3))) + (4 * X(4, 3)))) + (((X(1, 4) + X(2, 4)) + (4 * X(3, 4))) + (4 * X(4, 4))));
            result(2, 1) = ((((((X(1, 1) + X(2, 1)) + (4 * X(3, 1))) + (4 * X(4, 1))) - (((X(1, 2) + X(2, 2)) + (4 * X(3, 2))) + (4 * X(4, 2)))) + ((((X(1, 3) + X(2, 3)) + (4 * X(3, 3))) + (4 * X(4, 3))) * 2)) + ((((X(1, 4) + X(2, 4)) + (4 * X(3, 4))) + (4 * X(4, 4))) * -2));
            result(2, 2) = ((((((X(1, 1) + X(2, 1)) + (4 * X(3, 1))) + (4 * X(4, 1))) + (((X(1, 2) + X(2, 2)) + (4 * X(3, 2))) + (4 * X(4, 2)))) + ((((X(1, 3) + X(2, 3)) + (4 * X(3, 3))) + (4 * X(4, 3))) * 4)) + ((((X(1, 4) + X(2, 4)) + (4 * X(3, 4))) + (4 * X(4, 4))) * 4));
            result(2, 3) = (((((((X(1, 1) + X(2, 1)) + (4 * X(3, 1))) + (4 * X(4, 1))) - (((X(1, 2) + X(2, 2)) + (4 * X(3, 2))) + (4 * X(4, 2)))) + ((((X(1, 3) + X(2, 3)) + (4 * X(3, 3))) + (4 * X(4, 3))) * 8)) + ((((X(1, 4) + X(2, 4)) + (4 * X(3, 4))) + (4 * X(4, 4))) * -8)) + (((X(1, 5) + X(2, 5)) + (4 * X(3, 5))) + (4 * X(4, 5))));
            result(3, 0) = ((((((((X(1, 0) - X(2, 0)) + (8 * X(3, 0))) + (-8 * X(4, 0))) + X(5, 0)) + ((((X(1, 1) - X(2, 1)) + (8 * X(3, 1))) + (-8 * X(4, 1))) + X(5, 1))) + ((((X(1, 2) - X(2, 2)) + (8 * X(3, 2))) + (-8 * X(4, 2))) + X(5, 2))) + ((((X(1, 3) - X(2, 3)) + (8 * X(3, 3))) + (-8 * X(4, 3))) + X(5, 3))) + ((((X(1, 4) - X(2, 4)) + (8 * X(3, 4))) + (-8 * X(4, 4))) + X(5, 4)));
            result(3, 1) = (((((((X(1, 1) - X(2, 1)) + (8 * X(3, 1))) + (-8 * X(4, 1))) + X(5, 1)) - ((((X(1, 2) - X(2, 2)) + (8 * X(3, 2))) + (-8 * X(4, 2))) + X(5, 2))) + (((((X(1, 3) - X(2, 3)) + (8 * X(3, 3))) + (-8 * X(4, 3))) + X(5, 3)) * 2)) + (((((X(1, 4) - X(2, 4)) + (8 * X(3, 4))) + (-8 * X(4, 4))) + X(5, 4)) * -2));
            result(3, 2) = (((((((X(1, 1) - X(2, 1)) + (8 * X(3, 1))) + (-8 * X(4, 1))) + X(5, 1)) + ((((X(1, 2) - X(2, 2)) + (8 * X(3, 2))) + (-8 * X(4, 2))) + X(5, 2))) + (((((X(1, 3) - X(2, 3)) + (8 * X(3, 3))) + (-8 * X(4, 3))) + X(5, 3)) * 4)) + (((((X(1, 4) - X(2, 4)) + (8 * X(3, 4))) + (-8 * X(4, 4))) + X(5, 4)) * 4));
            result(3, 3) = ((((((((X(1, 1) - X(2, 1)) + (8 * X(3, 1))) + (-8 * X(4, 1))) + X(5, 1)) - ((((X(1, 2) - X(2, 2)) + (8 * X(3, 2))) + (-8 * X(4, 2))) + X(5, 2))) + (((((X(1, 3) - X(2, 3)) + (8 * X(3, 3))) + (-8 * X(4, 3))) + X(5, 3)) * 8)) + (((((X(1, 4) - X(2, 4)) + (8 * X(3, 4))) + (-8 * X(4, 4))) + X(5, 4)) * -8)) + ((((X(1, 5) - X(2, 5)) + (8 * X(3, 5))) + (-8 * X(4, 5))) + X(5, 5)));
        }

        template <typename BlockType1, typename BlockType2>
        static void TransformOutputBlock(const BlockType1& X, int blockSize, BlockType2& result)
        {
            for (int index = 0; index < blockSize; ++index)
            {
                result(0, 0, index) = ((((((((X(0, 0, index) + X(1, 0, index)) + X(2, 0, index)) + X(3, 0, index)) + X(4, 0, index)) + ((((X(0, 1, index) + X(1, 1, index)) + X(2, 1, index)) + X(3, 1, index)) + X(4, 1, index))) + ((((X(0, 2, index) + X(1, 2, index)) + X(2, 2, index)) + X(3, 2, index)) + X(4, 2, index))) + ((((X(0, 3, index) + X(1, 3, index)) + X(2, 3, index)) + X(3, 3, index)) + X(4, 3, index))) + ((((X(0, 4, index) + X(1, 4, index)) + X(2, 4, index)) + X(3, 4, index)) + X(4, 4, index)));
                result(0, 1, index) = (((((((X(0, 1, index) + X(1, 1, index)) + X(2, 1, index)) + X(3, 1, index)) + X(4, 1, index)) - ((((X(0, 2, index) + X(1, 2, index)) + X(2, 2, index)) + X(3, 2, index)) + X(4, 2, index))) + (((((X(0, 3, index) + X(1, 3, index)) + X(2, 3, index)) + X(3, 3, index)) + X(4, 3, index)) * 2)) + (((((X(0, 4, index) + X(1, 4, index)) + X(2, 4, index)) + X(3, 4, index)) + X(4, 4, index)) * -2));
                result(0, 2, index) = (((((((X(0, 1, index) + X(1, 1, index)) + X(2, 1, index)) + X(3, 1, index)) + X(4, 1, index)) + ((((X(0, 2, index) + X(1, 2, index)) + X(2, 2, index)) + X(3, 2, index)) + X(4, 2, index))) + (((((X(0, 3, index) + X(1, 3, index)) + X(2, 3, index)) + X(3, 3, index)) + X(4, 3, index)) * 4)) + (((((X(0, 4, index) + X(1, 4, index)) + X(2, 4, index)) + X(3, 4, index)) + X(4, 4, index)) * 4));
                result(0, 3, index) = ((((((((X(0, 1, index) + X(1, 1, index)) + X(2, 1, index)) + X(3, 1, index)) + X(4, 1, index)) - ((((X(0, 2, index) + X(1, 2, index)) + X(2, 2, index)) + X(3, 2, index)) + X(4, 2, index))) + (((((X(0, 3, index) + X(1, 3, index)) + X(2, 3, index)) + X(3, 3, index)) + X(4, 3, index)) * 8)) + (((((X(0, 4, index) + X(1, 4, index)) + X(2, 4, index)) + X(3, 4, index)) + X(4, 4, index)) * -8)) + ((((X(0, 5, index) + X(1, 5, index)) + X(2, 5, index)) + X(3, 5, index)) + X(4, 5, index)));
                result(1, 0, index) = (((((((X(1, 0, index) - X(2, 0, index)) + (2 * X(3, 0, index))) + (-2 * X(4, 0, index))) + (((X(1, 1, index) - X(2, 1, index)) + (2 * X(3, 1, index))) + (-2 * X(4, 1, index)))) + (((X(1, 2, index) - X(2, 2, index)) + (2 * X(3, 2, index))) + (-2 * X(4, 2, index)))) + (((X(1, 3, index) - X(2, 3, index)) + (2 * X(3, 3, index))) + (-2 * X(4, 3, index)))) + (((X(1, 4, index) - X(2, 4, index)) + (2 * X(3, 4, index))) + (-2 * X(4, 4, index))));
                result(1, 1, index) = ((((((X(1, 1, index) - X(2, 1, index)) + (2 * X(3, 1, index))) + (-2 * X(4, 1, index))) - (((X(1, 2, index) - X(2, 2, index)) + (2 * X(3, 2, index))) + (-2 * X(4, 2, index)))) + ((((X(1, 3, index) - X(2, 3, index)) + (2 * X(3, 3, index))) + (-2 * X(4, 3, index))) * 2)) + ((((X(1, 4, index) - X(2, 4, index)) + (2 * X(3, 4, index))) + (-2 * X(4, 4, index))) * -2));
                result(1, 2, index) = ((((((X(1, 1, index) - X(2, 1, index)) + (2 * X(3, 1, index))) + (-2 * X(4, 1, index))) + (((X(1, 2, index) - X(2, 2, index)) + (2 * X(3, 2, index))) + (-2 * X(4, 2, index)))) + ((((X(1, 3, index) - X(2, 3, index)) + (2 * X(3, 3, index))) + (-2 * X(4, 3, index))) * 4)) + ((((X(1, 4, index) - X(2, 4, index)) + (2 * X(3, 4, index))) + (-2 * X(4, 4, index))) * 4));
                result(1, 3, index) = (((((((X(1, 1, index) - X(2, 1, index)) + (2 * X(3, 1, index))) + (-2 * X(4, 1, index))) - (((X(1, 2, index) - X(2, 2, index)) + (2 * X(3, 2, index))) + (-2 * X(4, 2, index)))) + ((((X(1, 3, index) - X(2, 3, index)) + (2 * X(3, 3, index))) + (-2 * X(4, 3, index))) * 8)) + ((((X(1, 4, index) - X(2, 4, index)) + (2 * X(3, 4, index))) + (-2 * X(4, 4, index))) * -8)) + (((X(1, 5, index) - X(2, 5, index)) + (2 * X(3, 5, index))) + (-2 * X(4, 5, index))));
                result(2, 0, index) = (((((((X(1, 0, index) + X(2, 0, index)) + (4 * X(3, 0, index))) + (4 * X(4, 0, index))) + (((X(1, 1, index) + X(2, 1, index)) + (4 * X(3, 1, index))) + (4 * X(4, 1, index)))) + (((X(1, 2, index) + X(2, 2, index)) + (4 * X(3, 2, index))) + (4 * X(4, 2, index)))) + (((X(1, 3, index) + X(2, 3, index)) + (4 * X(3, 3, index))) + (4 * X(4, 3, index)))) + (((X(1, 4, index) + X(2, 4, index)) + (4 * X(3, 4, index))) + (4 * X(4, 4, index))));
                result(2, 1, index) = ((((((X(1, 1, index) + X(2, 1, index)) + (4 * X(3, 1, index))) + (4 * X(4, 1, index))) - (((X(1, 2, index) + X(2, 2, index)) + (4 * X(3, 2, index))) + (4 * X(4, 2, index)))) + ((((X(1, 3, index) + X(2, 3, index)) + (4 * X(3, 3, index))) + (4 * X(4, 3, index))) * 2)) + ((((X(1, 4, index) + X(2, 4, index)) + (4 * X(3, 4, index))) + (4 * X(4, 4, index))) * -2));
                result(2, 2, index) = ((((((X(1, 1, index) + X(2, 1, index)) + (4 * X(3, 1, index))) + (4 * X(4, 1, index))) + (((X(1, 2, index) + X(2, 2, index)) + (4 * X(3, 2, index))) + (4 * X(4, 2, index)))) + ((((X(1, 3, index) + X(2, 3, index)) + (4 * X(3, 3, index))) + (4 * X(4, 3, index))) * 4)) + ((((X(1, 4, index) + X(2, 4, index)) + (4 * X(3, 4, index))) + (4 * X(4, 4, index))) * 4));
                result(2, 3, index) = (((((((X(1, 1, index) + X(2, 1, index)) + (4 * X(3, 1, index))) + (4 * X(4, 1, index))) - (((X(1, 2, index) + X(2, 2, index)) + (4 * X(3, 2, index))) + (4 * X(4, 2, index)))) + ((((X(1, 3, index) + X(2, 3, index)) + (4 * X(3, 3, index))) + (4 * X(4, 3, index))) * 8)) + ((((X(1, 4, index) + X(2, 4, index)) + (4 * X(3, 4, index))) + (4 * X(4, 4, index))) * -8)) + (((X(1, 5, index) + X(2, 5, index)) + (4 * X(3, 5, index))) + (4 * X(4, 5, index))));
                result(3, 0, index) = ((((((((X(1, 0, index) - X(2, 0, index)) + (8 * X(3, 0, index))) + (-8 * X(4, 0, index))) + X(5, 0, index)) + ((((X(1, 1, index) - X(2, 1, index)) + (8 * X(3, 1, index))) + (-8 * X(4, 1, index))) + X(5, 1, index))) + ((((X(1, 2, index) - X(2, 2, index)) + (8 * X(3, 2, index))) + (-8 * X(4, 2, index))) + X(5, 2, index))) + ((((X(1, 3, index) - X(2, 3, index)) + (8 * X(3, 3, index))) + (-8 * X(4, 3, index))) + X(5, 3, index))) + ((((X(1, 4, index) - X(2, 4, index)) + (8 * X(3, 4, index))) + (-8 * X(4, 4, index))) + X(5, 4, index)));
                result(3, 1, index) = (((((((X(1, 1, index) - X(2, 1, index)) + (8 * X(3, 1, index))) + (-8 * X(4, 1, index))) + X(5, 1, index)) - ((((X(1, 2, index) - X(2, 2, index)) + (8 * X(3, 2, index))) + (-8 * X(4, 2, index))) + X(5, 2, index))) + (((((X(1, 3, index) - X(2, 3, index)) + (8 * X(3, 3, index))) + (-8 * X(4, 3, index))) + X(5, 3, index)) * 2)) + (((((X(1, 4, index) - X(2, 4, index)) + (8 * X(3, 4, index))) + (-8 * X(4, 4, index))) + X(5, 4, index)) * -2));
                result(3, 2, index) = (((((((X(1, 1, index) - X(2, 1, index)) + (8 * X(3, 1, index))) + (-8 * X(4, 1, index))) + X(5, 1, index)) + ((((X(1, 2, index) - X(2, 2, index)) + (8 * X(3, 2, index))) + (-8 * X(4, 2, index))) + X(5, 2, index))) + (((((X(1, 3, index) - X(2, 3, index)) + (8 * X(3, 3, index))) + (-8 * X(4, 3, index))) + X(5, 3, index)) * 4)) + (((((X(1, 4, index) - X(2, 4, index)) + (8 * X(3, 4, index))) + (-8 * X(4, 4, index))) + X(5, 4, index)) * 4));
                result(3, 3, index) = ((((((((X(1, 1, index) - X(2, 1, index)) + (8 * X(3, 1, index))) + (-8 * X(4, 1, index))) + X(5, 1, index)) - ((((X(1, 2, index) - X(2, 2, index)) + (8 * X(3, 2, index))) + (-8 * X(4, 2, index))) + X(5, 2, index))) + (((((X(1, 3, index) - X(2, 3, index)) + (8 * X(3, 3, index))) + (-8 * X(4, 3, index))) + X(5, 3, index)) * 8)) + (((((X(1, 4, index) - X(2, 4, index)) + (8 * X(3, 4, index))) + (-8 * X(4, 4, index))) + X(5, 4, index)) * -8)) + ((((X(1, 5, index) - X(2, 5, index)) + (8 * X(3, 5, index))) + (-8 * X(4, 5, index))) + X(5, 5, index)));
            }
        }
    };

    //
    // Helper class to implement Winograd convolution steps
    //
    template <typename ValueType, int tileSizeValue, int filterSizeValue, int blockSizeValue>
    struct FixedWinograd2D
    {
        static constexpr int tileSize = tileSizeValue;
        static constexpr int filterSize = filterSizeValue;
        static constexpr int windowSize = filterSize + tileSize - 1;
        static constexpr int blockSize = blockSizeValue;

        using TileArray = Fixed2DArray<ValueType, tileSize, tileSize>;
        using WindowArray = Fixed2DArray<ValueType, windowSize, windowSize>;
        using TileBlock = Fixed3DArray<ValueType, tileSize, tileSize, blockSize>;
        using WindowBlock = Fixed3DArray<ValueType, windowSize, windowSize, blockSize>;
        using Tensor = math::ChannelColumnRowTensor<ValueType>;
        using TensorReference = math::ChannelColumnRowTensorReference<ValueType>;
        using ConstTensorReference = math::ConstChannelColumnRowTensorReference<ValueType>;

        static void GetInputBlock(const ConstTensorReference& input,
                                  int tileRowIndex,
                                  int tileColumnIndex,
                                  int channelIndex,
                                  WindowBlock& d)
        {
            const auto rowIndex = tileRowIndex * tileSize;
            const auto columnIndex = tileColumnIndex * tileSize;
            const auto numChannels = static_cast<int>(input.NumChannels());
            const int channelsToCopy = (numChannels - channelIndex) < blockSize ? (numChannels - channelIndex) : blockSize;

            d.CopyFrom(input, rowIndex, columnIndex, channelIndex, windowSize, windowSize, channelsToCopy, numChannels * static_cast<int>(input.NumColumns()), numChannels);
        }

        static void GetPartialInputBlock(const ConstTensorReference& input,
                                         int tileRowIndex,
                                         int tileColumnIndex,
                                         int channelIndex,
                                         int rows,
                                         int columns,
                                         WindowBlock& d)
        {
            auto rowIndex = tileRowIndex * tileSize;
            auto columnIndex = tileColumnIndex * tileSize;
            const auto numChannels = static_cast<int>(input.NumChannels());
            const auto numColumns = static_cast<int>(input.NumColumns());
            const int channelsToCopy = (numChannels - channelIndex) < blockSize ? (numChannels - channelIndex) : blockSize;
            d.CopyFrom(input, rowIndex, columnIndex, channelIndex, rows, columns, channelsToCopy, numChannels * numColumns, numChannels);
        }

        static void ProcessInputBlock(const ConstTensorReference& input,
                                      int tileRowIndex,
                                      int tileColumnIndex,
                                      int channelStartIndex,
                                      int numTileRows,
                                      int numTileColumns,
                                      int numChannels,
                                      int thisBlockSize,
                                      WindowBlock& d,
                                      WindowBlock& X,
                                      Tensor& transformedInput)
        {
            // Get the input window
            GetInputBlock(input, tileRowIndex, tileColumnIndex, channelStartIndex, d);

            // Transform it
            FixedWinogradTransform2D<ValueType, tileSize, filterSize>::TransformInputBlock(d, thisBlockSize, X);

            // Now splat transformedInputWindow into transformedInput
            SplatTransformedInputBlock(X, tileRowIndex, tileColumnIndex, channelStartIndex, numTileRows, numTileColumns, numChannels, transformedInput);
        }

        static void ProcessPartialInputBlock(const ConstTensorReference& input,
                                             int tileRowIndex,
                                             int tileColumnIndex,
                                             int channelStartIndex,
                                             int numTileRows,
                                             int numTileColumns,
                                             int numChannels,
                                             int rows,
                                             int columns,
                                             int thisBlockSize,
                                             WindowBlock& d,
                                             WindowBlock& X,
                                             Tensor& transformedInput)
        {
            GetPartialInputBlock(input, tileRowIndex, tileColumnIndex, channelStartIndex, rows, columns, d);
            FixedWinogradTransform2D<ValueType, tileSize, filterSize>::TransformInputBlock(d, thisBlockSize, X);
            SplatTransformedInputBlock(X, tileRowIndex, tileColumnIndex, channelStartIndex, numTileRows, numTileColumns, numChannels, transformedInput);
        }

        static void TransformInput(const ConstTensorReference& input,
                                   int numOutputRows,
                                   int numOutputColumns,
                                   int numChannels,
                                   Tensor& transformedInput)
        {
            const auto numFullTileRows = numOutputRows / tileSize;
            const auto numFullTileColumns = numOutputColumns / tileSize;
            const auto numTileRows = ((numOutputRows - 1) / tileSize) + 1;
            const auto numTileColumns = ((numOutputColumns - 1) / tileSize) + 1;

            WindowBlock d;
            WindowBlock X;

            //
            // First, visit all fully-covered input tiles
            //
            for (int tileRowIndex = 0; tileRowIndex < numFullTileRows; ++tileRowIndex)
            {
                for (int tileColumnIndex = 0; tileColumnIndex < numFullTileColumns; ++tileColumnIndex)
                {
                    for (int channelStartIndex = 0; channelStartIndex < numChannels; channelStartIndex += blockSize)
                    {
                        int thisBlockSize = numChannels - channelStartIndex > blockSize ? blockSize : numChannels - channelStartIndex;
                        ProcessInputBlock(input, tileRowIndex, tileColumnIndex, channelStartIndex, numTileRows, numTileColumns, numChannels, thisBlockSize, d, X, transformedInput);
                    }
                }
            }

            //
            // Now go and fill in transformed data for tiles that aren't fully contained in the input image
            //

            const int lastTileRow = numFullTileRows;
            const auto lastRowStart = lastTileRow * tileSize;
            const int lastTileColumn = numFullTileColumns;
            const auto lastColumnStart = lastTileColumn * tileSize;

            // First, the bottom row
            if (numTileRows > numFullTileRows)
            {
                assert(numTileRows == numFullTileRows + 1);
                const auto rows = static_cast<int>(input.NumRows() - lastRowStart);
                for (int tileColumnIndex = 0; tileColumnIndex < numFullTileColumns; ++tileColumnIndex)
                {
                    for (int channelStartIndex = 0; channelStartIndex < numChannels; channelStartIndex += blockSize)
                    {
                        int thisBlockSize = numChannels - channelStartIndex > blockSize ? blockSize : numChannels - channelStartIndex;
                        ProcessPartialInputBlock(input, lastTileRow, tileColumnIndex, channelStartIndex, numTileRows, numTileColumns, numChannels, rows, windowSize, thisBlockSize, d, X, transformedInput);
                    }
                }
            }

            // Then the righthand column
            if (numTileColumns > numFullTileColumns)
            {
                assert(numTileColumns == numFullTileColumns + 1);
                const auto columns = static_cast<int>(input.NumColumns() - lastColumnStart);
                for (int tileRowIndex = 0; tileRowIndex < numFullTileRows; ++tileRowIndex)
                {
                    for (int channelStartIndex = 0; channelStartIndex < numChannels; channelStartIndex += blockSize)
                    {
                        int thisBlockSize = numChannels - channelStartIndex > blockSize ? blockSize : numChannels - channelStartIndex;
                        ProcessPartialInputBlock(input, tileRowIndex, lastTileColumn, channelStartIndex, numTileRows, numTileColumns, numChannels, windowSize, columns, thisBlockSize, d, X, transformedInput);
                    }
                }
            }

            // Finally, the lower-righthand corner
            if (numTileRows > numFullTileRows && numTileColumns > numFullTileColumns)
            {
                assert(numTileRows == numFullTileRows + 1);
                const auto rows = static_cast<int>(input.NumRows() - lastRowStart);
                const auto columns = static_cast<int>(input.NumColumns() - lastColumnStart);

                for (int channelStartIndex = 0; channelStartIndex < numChannels; channelStartIndex += blockSize)
                {
                    int thisBlockSize = numChannels - channelStartIndex > blockSize ? blockSize : numChannels - channelStartIndex;
                    ProcessPartialInputBlock(input, lastTileRow, lastTileColumn, channelStartIndex, numTileRows, numTileColumns, numChannels, rows, columns, thisBlockSize, d, X, transformedInput);
                }
            }
        }

        static void SplatTransformedInputBlock(const WindowBlock& dataTile, int tileRowIndex, int tileColumnIndex, int channelIndex, int numTileRows, int numTileColumns, int numChannels, Tensor& transformedInput)
        {
            const auto windowEntryStride = numChannels * numTileRows * numTileColumns;
            const auto tileOffset = (tileRowIndex * numTileColumns * numChannels) + (tileColumnIndex * numChannels) + channelIndex; // offset within the tile
            auto transformedInputPtr = transformedInput.GetDataPointer() + tileOffset;
            auto dataTilePtr = dataTile.GetDataPointer();
            const int channelsToCopy = (numChannels - channelIndex) < blockSize ? (numChannels - channelIndex) : blockSize;
            for (int entryIndex = 0; entryIndex < windowSize * windowSize; ++entryIndex)
            {
                std::copy(dataTilePtr, dataTilePtr + channelsToCopy, transformedInputPtr);
                dataTilePtr += blockSize;
                transformedInputPtr += windowEntryStride;
            }
        }

        static void GetTransformedOutputBlock(const math::ConstChannelColumnRowTensorReference<ValueType>& transformedOutput, int tileRowIndex, int tileColumnIndex, int filterStartIndex, int numTileRows, int numTileColumns, int numFilters, WindowBlock& transformedOutputBlock)
        {
            assert(filterStartIndex < numFilters);
            const int filtersToCopy = (numFilters - filterStartIndex) < blockSize ? (numFilters - filterStartIndex) : blockSize;

            const int windowEntryStride = numFilters * numTileRows * numTileColumns;
            const int tileIndex = tileRowIndex * numTileColumns + tileColumnIndex;
            const int offset = (tileIndex * numFilters) + filterStartIndex;
            auto outputPtr = transformedOutput.GetConstDataPointer() + offset;
            auto outputBlockPtr = transformedOutputBlock.GetDataPointer();
            for (int windowPos = 0; windowPos < windowSize * windowSize; ++windowPos)
            {
                std::copy(outputPtr, outputPtr + filtersToCopy, outputBlockPtr);
                outputPtr += windowEntryStride;
                outputBlockPtr += blockSize;
            }
        }

        // outputTile is a tr x tc matrix
        // output is a r x c x nf tensor
        static void SplatOutputTile(const TileArray& outputTile,
                                    int tileRowIndex,
                                    int tileColumnIndex,
                                    int filterIndex,
                                    TensorReference& output)
        {
            // iterate over entries in the tile
            for (int rowIndex = 0; rowIndex < tileSize; ++rowIndex)
            {
                for (int columnIndex = 0; columnIndex < tileSize; ++columnIndex)
                {
                    output((tileRowIndex * tileSize) + rowIndex, (tileColumnIndex * tileSize) + columnIndex, filterIndex) = outputTile(rowIndex, columnIndex);
                }
            }
        }

        // outputBlock is a tr x tc x nf block
        // output is a r x c x nf tensor
        static inline void SplatOutputBlock(const TileBlock& outputBlock,
                                            int tileRowIndex,
                                            int tileColumnIndex,
                                            int filterIndex,
                                            TensorReference& output)
        {
            const int numFilters = static_cast<int>(output.NumChannels());
            assert(filterIndex < numFilters);
            const int filtersToCopy = (numFilters - filterIndex) < blockSize ? (numFilters - filterIndex) : blockSize;

            const auto rowOffset = tileRowIndex * tileSize;
            const auto rowStride = output.NumChannels() * output.NumColumns();
            const auto columnOffset = tileColumnIndex * tileSize;
            const auto columnStride = output.NumChannels();
            const auto outputOffset = rowOffset * rowStride + columnOffset * columnStride + filterIndex;

            auto outputBlockPtr = outputBlock.GetDataPointer();
            auto outputPtr = output.GetDataPointer() + outputOffset;

            // iterate over entries in the tile
            for (int rowIndex = 0; rowIndex < tileSize; ++rowIndex)
            {
                auto outputRow = outputPtr;
                for (int columnIndex = 0; columnIndex < tileSize; ++columnIndex)
                {
                    std::copy(outputBlockPtr, outputBlockPtr + filtersToCopy, outputRow);
                    outputBlockPtr += blockSize;
                    outputRow += columnStride;
                }
                outputPtr += rowStride;
            }
        }

        // outputTile is a tr x tc matrix
        // output is a r x c x nf tensor
        static void SplatPartialOutputTile(const TileArray& outputTile,
                                           int tileRowIndex,
                                           int tileColumnIndex,
                                           int filterIndex,
                                           int rows,
                                           int columns,
                                           TensorReference& output)
        {
            // iterate over entries in the tile
            for (int rowIndex = 0; rowIndex < rows; ++rowIndex)
            {
                for (int columnIndex = 0; columnIndex < columns; ++columnIndex)
                {
                    output((tileRowIndex * tileSize) + rowIndex, (tileColumnIndex * tileSize) + columnIndex, filterIndex) = outputTile(rowIndex, columnIndex);
                }
            }
        }

        // outputTile is a tr x tc matrix
        // output is a r x c x nf tensor
        static void AccumulatePartialOutputTile(const TileArray& outputTile,
                                                int tileRowIndex,
                                                int tileColumnIndex,
                                                int filterIndex,
                                                int rows,
                                                int columns,
                                                TensorReference& output)
        {
            // iterate over entries in the tile
            for (int rowIndex = 0; rowIndex < rows; ++rowIndex)
            {
                for (int columnIndex = 0; columnIndex < columns; ++columnIndex)
                {
                    output((tileRowIndex * tileSize) + rowIndex, (tileColumnIndex * tileSize) + columnIndex, filterIndex) += outputTile(rowIndex, columnIndex);
                }
            }
        }

        // outputBlock is a tr x tc x blockSize matrix
        // output is a r x c x nf tensor
        static void SplatPartialOutputBlock(const TileBlock& outputBlock,
                                            int tileRowIndex,
                                            int tileColumnIndex,
                                            int filterIndex,
                                            int rows,
                                            int columns,
                                            TensorReference& output)
        {

            const int numFilters = static_cast<int>(output.NumChannels());
            assert(filterIndex < numFilters);
            const int filtersToCopy = (numFilters - filterIndex) < blockSize ? (numFilters - filterIndex) : blockSize;

            const auto rowOffset = tileRowIndex * tileSize;
            const auto rowStride = output.NumChannels() * output.NumColumns();
            const auto columnOffset = tileColumnIndex * tileSize;
            const auto columnStride = output.NumChannels();
            const auto outputOffset = rowOffset * rowStride + columnOffset * columnStride + filterIndex;

            auto outputBlockPtr = outputBlock.GetDataPointer();
            auto outputPtr = output.GetDataPointer() + outputOffset;

            // iterate over entries in the tile
            for (int rowIndex = 0; rowIndex < rows; ++rowIndex)
            {
                auto outputRow = outputPtr;
                auto outputRowPtr = outputBlockPtr;
                for (int columnIndex = 0; columnIndex < columns; ++columnIndex)
                {
                    std::copy(outputRowPtr, outputRowPtr + filtersToCopy, outputRow);

                    outputRowPtr += blockSize;
                    outputRow += columnStride;
                }
                outputBlockPtr += blockSize * outputBlock.columns;
                outputPtr += rowStride;
            }
        }

        static void TransformOutput(const Tensor& transformedOutput, TensorReference& output)
        {
            const auto numOutputRows = static_cast<int>(output.NumRows());
            const auto numOutputColumns = static_cast<int>(output.NumColumns());
            const auto numFilters = static_cast<int>(output.NumChannels());
            const auto numFullTileRows = numOutputRows / tileSize;
            const auto numFullTileColumns = numOutputColumns / tileSize;
            const auto numTileRows = ((numOutputRows - 1) / tileSize) + 1;
            const auto numTileColumns = ((numOutputColumns - 1) / tileSize) + 1;
            const auto lastTileRow = numFullTileRows;
            const auto lastTileColumn = numFullTileColumns;
            const auto lastRowStart = numFullTileRows * tileSize;
            const auto lastColumnStart = numFullTileColumns * tileSize;

            WindowBlock transformedOutputBlock;
            TileBlock outputBlock;

            // Un-transform the result, copying tiles into the output
            for (int tileRowIndex = 0; tileRowIndex < numFullTileRows; ++tileRowIndex)
            {
                for (int tileColumnIndex = 0; tileColumnIndex < numFullTileColumns; ++tileColumnIndex)
                {
                    for (int filterStartIndex = 0; filterStartIndex < numFilters; filterStartIndex += blockSize)
                    {
                        const int thisBlockSize = (numFilters - filterStartIndex) < blockSize ? (numFilters - filterStartIndex) : blockSize;
                        GetTransformedOutputBlock(transformedOutput, tileRowIndex, tileColumnIndex, filterStartIndex, numTileRows, numTileColumns, numFilters, transformedOutputBlock);
                        FixedWinogradTransform2D<ValueType, tileSize, filterSize>::TransformOutputBlock(transformedOutputBlock, thisBlockSize, outputBlock);
                        SplatOutputBlock(outputBlock, tileRowIndex, tileColumnIndex, filterStartIndex, output);
                    }
                }

                // Handle any partial tile in the righthand column
                if (numTileColumns > numFullTileColumns)
                {
                    assert(numTileColumns == numFullTileColumns + 1);
                    const auto columns = static_cast<int>(output.NumColumns() - lastColumnStart);
                    for (int filterStartIndex = 0; filterStartIndex < numFilters; filterStartIndex += blockSize)
                    {
                        const int thisBlockSize = (numFilters - filterStartIndex) < blockSize ? (numFilters - filterStartIndex) : blockSize;
                        GetTransformedOutputBlock(transformedOutput, tileRowIndex, lastTileColumn, filterStartIndex, numTileRows, numTileColumns, numFilters, transformedOutputBlock);
                        FixedWinogradTransform2D<ValueType, tileSize, filterSize>::TransformOutputBlock(transformedOutputBlock, thisBlockSize, outputBlock);
                        SplatPartialOutputBlock(outputBlock, tileRowIndex, lastTileColumn, filterStartIndex, tileSize, columns, output);
                    }
                }
            }

            // Handle partial output tiles on the bottom row
            if (numTileRows > numFullTileRows)
            {
                assert(numTileRows == numFullTileRows + 1);
                const auto rows = static_cast<int>(output.NumRows() - lastRowStart);
                for (int tileColumnIndex = 0; tileColumnIndex < numFullTileColumns; ++tileColumnIndex)
                {
                    for (int filterStartIndex = 0; filterStartIndex < numFilters; filterStartIndex += blockSize)
                    {
                        const int thisBlockSize = (numFilters - filterStartIndex) < blockSize ? (numFilters - filterStartIndex) : blockSize;
                        GetTransformedOutputBlock(transformedOutput, lastTileRow, tileColumnIndex, filterStartIndex, numTileRows, numTileColumns, numFilters, transformedOutputBlock);
                        FixedWinogradTransform2D<ValueType, tileSize, filterSize>::TransformOutputBlock(transformedOutputBlock, thisBlockSize, outputBlock);
                        SplatPartialOutputBlock(outputBlock, lastTileRow, tileColumnIndex, filterStartIndex, rows, tileSize, output);
                    }
                }
            }

            // Finally, the lower-righthand corner
            if (numTileRows > numFullTileRows && numTileColumns > numFullTileColumns)
            {
                assert(numTileRows == numFullTileRows + 1);
                assert(numTileColumns == numFullTileColumns + 1);
                const auto rows = static_cast<int>(output.NumRows() - lastRowStart);
                const auto columns = static_cast<int>(output.NumColumns() - lastColumnStart);
                for (int filterStartIndex = 0; filterStartIndex < numFilters; filterStartIndex += blockSize)
                {
                    const int thisBlockSize = (numFilters - filterStartIndex) < blockSize ? (numFilters - filterStartIndex) : blockSize;
                    GetTransformedOutputBlock(transformedOutput, lastTileRow, lastTileColumn, filterStartIndex, numTileRows, numTileColumns, numFilters, transformedOutputBlock);
                    FixedWinogradTransform2D<ValueType, tileSize, filterSize>::TransformOutputBlock(transformedOutputBlock, thisBlockSize, outputBlock);
                    SplatPartialOutputBlock(outputBlock, lastTileRow, lastTileColumn, filterStartIndex, rows, columns, output);
                }
            }
        }

        //
        // Straightforward implementation of Winograd algorithm, using separate matrix multiplies to transform
        // each tile. Handles the depthwise-separable case as well.
        //
        static void Convolve2DWinogradFiltersFirst(const Tensor& input,
                                                   const Tensor& transformedFilters,
                                                   int numFilters,
                                                   TensorReference& output)
        {
            using Matrix = math::RowMatrix<ValueType>;
            const int numInputRows = static_cast<int>(input.NumRows());
            const int numInputColumns = static_cast<int>(input.NumColumns());
            const int numChannels = static_cast<int>(input.NumChannels());
            const int numOutputRows = static_cast<int>(output.NumRows());
            const int numOutputColumns = static_cast<int>(output.NumColumns());
            assert(numFilters == static_cast<int>(output.NumChannels()));

            const int numTileRows = ((numOutputRows - 1) / tileSize) + 1;
            const int numTileColumns = ((numOutputColumns - 1) / tileSize) + 1;
            const int numFilterChannels = static_cast<int>(transformedFilters.NumColumns());
            const int filterStride = numFilterChannels * windowSize * windowSize;
            const int filterChannelStride = windowSize * windowSize;

            // Temporary values
            Matrix inputSlice(numInputRows, numInputColumns);
            WindowArray X;
            WindowArray d;
            TileArray outputTile;

            output.Fill(0);
            for (int filterIndex = 0; filterIndex < numFilters; ++filterIndex)
            {
                const int channelStart = (filterIndex * numFilterChannels) % numChannels;
                for (int filterChannel = 0; filterChannel < numFilterChannels; ++filterChannel)
                {
                    const int channelIndex = channelStart + filterChannel;
                    GetChannelSlice(input, channelIndex, inputSlice);
                    const auto filterPtr = transformedFilters.GetConstDataPointer() + filterIndex * filterStride + filterChannel * filterChannelStride;
                    for (int tileRowIndex = 0; tileRowIndex < numTileRows; ++tileRowIndex)
                    {
                        const auto rowIndex = tileRowIndex * tileSize;
                        for (int tileColumnIndex = 0; tileColumnIndex < numTileColumns; ++tileColumnIndex)
                        {
                            const auto columnIndex = tileColumnIndex * tileSize;
                            d.CopyFrom(inputSlice.GetConstDataPointer(), rowIndex, columnIndex, 0, numInputRows, numInputColumns, 1, numInputColumns);

                            // Compute X = B'dB
                            FixedWinogradTransform2D<ValueType, tileSize, filterSize>::TransformInputWindow(d, X);

                            ElementwiseMultiply(filterPtr, X.GetDataPointer(), windowSize * windowSize, X.GetDataPointer());

                            // Now compute output tile Y = At * X * A
                            FixedWinogradTransform2D<ValueType, 2, 3>::TransformOutputTile(X, outputTile);

                            // copy the tile into the output
                            const int outputTileRows = std::min(static_cast<int>(tileSize), numOutputRows - rowIndex);
                            const int outputTileColumns = std::min(static_cast<int>(tileSize), numOutputColumns - columnIndex);
                            AccumulatePartialOutputTile(outputTile, tileRowIndex, tileColumnIndex, filterIndex, outputTileRows, outputTileColumns, output);
                        }
                    }
                }
            }
        }

        //
        // More efficient version (especially when the number of channels and filters is large) that
        // pretransforms all of the input tiles, uses a series of GEMM calls to accumulate the channels
        // of the filtered output, and then transforms the output.
        //
        static void Convolve2DWinogradTilesFirst(const ConstTensorReference& input,
                                                 const Tensor& transformedFilters,
                                                 int numFilters,
                                                 Tensor& transformedInputScratch,
                                                 Tensor& transformedOutputScratch,
                                                 TensorReference& output)
        {
            // transformedFilters is a (windowRows*windowColumns) x numFilters x numChannels array
            // output is a rows x columns x numFilters tensor
            const auto numOutputRows = static_cast<int>(input.NumRows()) - filterSize + 1;
            const auto numOutputColumns = static_cast<int>(input.NumColumns()) - filterSize + 1;
            const auto numChannels = static_cast<int>(input.NumChannels());
            assert(numFilters == static_cast<int>(output.NumChannels()));

            // transformedInput is a (windowRows*windowColumns) x (numTileRows * numTileColumns) x (numChannels) tensor containing the entire transformed input signal
            TransformInput(input, numOutputRows, numOutputColumns, numChannels, transformedInputScratch);

            // transformedOutput is (windowRows*windowColumns) x (numTileRows * numTileColumns) x (numFilters)
            ComputeTransformedOutput(transformedInputScratch, transformedFilters, numOutputRows, numOutputColumns, numChannels, numFilters, tileSize, filterSize, transformedOutputScratch);

            // Un-transform convolved output and write into output image
            TransformOutput(transformedOutputScratch, output);
        }
    }; // End of FixedWinograd2D class

    template <typename ValueType>
    math::ChannelColumnRowTensor<ValueType> AllocateScratchInput(int numOutputRows, int numOutputColumns, int numChannels, int tileSize, int filterSize, WinogradFilterOrder order)
    {
        if (order != WinogradFilterOrder::tilesFirst)
        {
            return { 0, 0, 0 };
        }

        const auto windowSize = tileSize + filterSize - 1;
        const auto numTileRows = ((numOutputRows - 1) / tileSize) + 1;
        const auto numTileColumns = ((numOutputColumns - 1) / tileSize) + 1;
        assert(windowSize > 0);
        return { static_cast<size_t>(windowSize * windowSize), static_cast<size_t>(numTileRows * numTileColumns), static_cast<size_t>(numChannels) };
    }

    template <typename ValueType>
    math::ChannelColumnRowTensor<ValueType> AllocateScratchOutput(int numOutputRows, int numOutputColumns, int numFilters, int tileSize, int filterSize, WinogradFilterOrder order)
    {
        if (order == WinogradFilterOrder::filtersFirst)
        {
            return { 0, 0, 0 };
        }

        const auto windowSize = tileSize + filterSize - 1;
        const auto numTileRows = ((numOutputRows - 1) / tileSize) + 1;
        const auto numTileColumns = ((numOutputColumns - 1) / tileSize) + 1;
        return { static_cast<size_t>(windowSize * windowSize), static_cast<size_t>(numTileRows * numTileColumns), static_cast<size_t>(numFilters) };
    }

    //
    // Straightforward implementation of Winograd algorithm, using separate matrix multiplies to transform
    // each tile
    //

    template <typename ValueType>
    void Convolve2DWinogradFiltersFirst(const math::ConstChannelColumnRowTensorReference<ValueType>& input,
                                        const math::ConstChannelColumnRowTensorReference<ValueType>& transformedFilters,
                                        int numFilters,
                                        int tileSize,
                                        int filterSize,
                                        math::ChannelColumnRowTensor<ValueType>& output)
    {
        constexpr int blockSize = 64;
        if (tileSize == 2 && filterSize == 3)
        {
            FixedWinograd2D<ValueType, 2, 3, blockSize>::Convolve2DWinogradFiltersFirst(input, transformedFilters, numFilters, output);
        }
        else if (tileSize == 4 && filterSize == 3)
        {
            FixedWinograd2D<ValueType, 4, 3, blockSize>::Convolve2DWinogradFiltersFirst(input, transformedFilters, numFilters, output);
        }
        else
        {
            throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
        }
    }

    //
    // More efficient version (especially when the number of channels and filters is large) that
    // pretransforms all of the input tiles, uses a series of GEMM calls to accumulate the channels
    // of the filtered output, and then transforms the output.
    //
    template <typename ValueType>
    void Convolve2DWinogradTilesFirst(const math::ConstChannelColumnRowTensorReference<ValueType>& input,
                                      const math::ConstChannelColumnRowTensorReference<ValueType>& transformedFilters,
                                      int numFilters,
                                      int tileSize,
                                      int filterSize,
                                      math::ChannelColumnRowTensor<ValueType>& transformedInputScratch,
                                      math::ChannelColumnRowTensor<ValueType>& transformedOutputScratch,
                                      math::ChannelColumnRowTensor<ValueType>& output)
    {
        constexpr int blockSize = 64;
        if (tileSize == 2 && filterSize == 3)
        {
            FixedWinograd2D<ValueType, 2, 3, blockSize>::Convolve2DWinogradTilesFirst(input, transformedFilters, numFilters, transformedInputScratch, transformedOutputScratch, output);
        }
        else if (tileSize == 4 && filterSize == 3)
        {
            FixedWinograd2D<ValueType, 4, 3, blockSize>::Convolve2DWinogradTilesFirst(input, transformedFilters, numFilters, transformedInputScratch, transformedOutputScratch, output);
        }
        else
        {
            throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
        }
    }

    //
    // Actual API function implementations
    //

    // 1D
    template <typename ValueType>
    math::RowVector<ValueType> Convolve1DWinograd(const math::RowVector<ValueType>& input, const math::RowVector<ValueType>& filter)
    {
        const int tileSize = 2;
        return Convolve1DWinograd(input, filter, tileSize);
    }

    template <typename ValueType>
    math::RowVector<ValueType> Convolve1DWinograd(const math::RowVector<ValueType>& input, const math::RowVector<ValueType>& filter, int tileSize)
    {
        using namespace std::string_literals;
        const int filterSize = static_cast<int>(filter.Size());
        const int outputSize = static_cast<int>(input.Size()) - filterSize + 1;
        math::RowVector<ValueType> output(outputSize);
        if (tileSize == 2 && filterSize == 3)
        {
            FixedWinograd1D<ValueType, 2, 3>::Convolve(input, filter, output);
        }
        else
        {
            throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, "1D Winograd convolution not implemented for tile size "s + std::to_string(tileSize) + " and filter size " + std::to_string(filterSize));
        }
        return output;
    }

    // 2D
    template <typename ValueType>
    math::ChannelColumnRowTensor<ValueType> Convolve2DWinograd(const math::ConstChannelColumnRowTensorReference<ValueType>& input, const math::ConstChannelColumnRowTensorReference<ValueType>& filters, int numFilters, WinogradFilterOrder order)
    {
        const int tileSize = 2;
        return Convolve2DWinograd(input, filters, numFilters, tileSize, order);
    }

    // filters is a numFilters x filterSize x filterSize x numChannels tensor (represented in 3D as (numFilters * filterSize) x filterSize x numChannels)
    template <typename ValueType>
    math::ChannelColumnRowTensor<ValueType> Convolve2DWinograd(const math::ConstChannelColumnRowTensorReference<ValueType>& input, const math::ConstChannelColumnRowTensorReference<ValueType>& filters, int numFilters, int tileSize, WinogradFilterOrder order)
    {
        const auto filterSize = static_cast<int>(filters.NumRows()) / numFilters;
        assert(filterSize == static_cast<int>(filters.NumColumns()) && "Filters must be square");

        auto transformedFilters = GetTransformedFilters(filters, numFilters, tileSize, order);
        return Convolve2DWinogradPretransformed(input, transformedFilters, numFilters, tileSize, filterSize, order);
    }

    template <typename ValueType>
    math::ChannelColumnRowTensor<ValueType> Convolve2DWinogradPretransformed(const math::ConstChannelColumnRowTensorReference<ValueType>& input, const math::ConstChannelColumnRowTensorReference<ValueType>& transformedFilters, int numFilters, int tileSize, int filterSize, WinogradFilterOrder order)
    {
        const auto numOutputRows = static_cast<int>(input.NumRows()) - filterSize + 1;
        const auto numOutputColumns = static_cast<int>(input.NumColumns()) - filterSize + 1;
        const auto numChannels = static_cast<int>(input.NumChannels());
        math::ChannelColumnRowTensor<ValueType> output(numOutputRows, numOutputColumns, numFilters);

        switch (order)
        {
        case WinogradFilterOrder::filtersFirst: {
            Convolve2DWinogradFiltersFirst(input, transformedFilters, numFilters, tileSize, filterSize, output);
        }
        break;
        case WinogradFilterOrder::tilesFirst: {
            auto transformedInputScratch = AllocateScratchInput<ValueType>(numOutputRows, numOutputColumns, numChannels, tileSize, filterSize, order);
            auto transformedOutputScratch = AllocateScratchOutput<ValueType>(numOutputRows, numOutputColumns, numFilters, tileSize, filterSize, order);
            Convolve2DWinogradTilesFirst(input, transformedFilters, numFilters, tileSize, filterSize, transformedInputScratch, transformedOutputScratch, output);
        }
        break;
        default:
            throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
        }

        return output;
    }

    // filters is a numFilters x filterSize x filterSize tensor
    template <typename ValueType>
    math::ChannelColumnRowTensor<ValueType> Convolve2DWinogradDepthwiseSeparable(const math::ConstChannelColumnRowTensorReference<ValueType>& input, const math::ConstChannelColumnRowTensorReference<ValueType>& filters, int numFilters, WinogradFilterOrder order)
    {
        const int tileSize = 2;
        return Convolve2DWinogradDepthwiseSeparable(input, filters, numFilters, tileSize, order);
    }

    // filters is a numFilters x filterSize x filterSize tensor
    template <typename ValueType>
    math::ChannelColumnRowTensor<ValueType> Convolve2DWinogradDepthwiseSeparable(const math::ConstChannelColumnRowTensorReference<ValueType>& input, const math::ConstChannelColumnRowTensorReference<ValueType>& filters, int numFilters, int tileSize, WinogradFilterOrder order)
    {
        const auto filterSize = static_cast<int>(filters.NumColumns());
        // assert(filterSize == static_cast<int>(filters.NumChannels()) && "Filters must be square");
        assert(filters.NumChannels() == 1 && "Filters must be single-channel");
        assert(numFilters == static_cast<int>(input.NumChannels()) && "Must have same number of filters as input channels");

        auto transformedFilters = GetTransformedFilters(filters, numFilters, tileSize, order);
        return Convolve2DWinogradDepthwiseSeparablePretransformed(input, transformedFilters, numFilters, tileSize, filterSize, order);
    }

    template <typename ValueType>
    math::ChannelColumnRowTensor<ValueType> Convolve2DWinogradDepthwiseSeparablePretransformed(const math::ConstChannelColumnRowTensorReference<ValueType>& input, const math::ConstChannelColumnRowTensorReference<ValueType>& transformedFilters, int numFilters, int tileSize, int filterSize, WinogradFilterOrder order)
    {
        UNUSED(order);
        const auto numOutputRows = static_cast<int>(input.NumRows()) - filterSize + 1;
        const auto numOutputColumns = static_cast<int>(input.NumColumns()) - filterSize + 1;
        const auto numChannels = static_cast<int>(input.NumChannels());

        math::ChannelColumnRowTensor<ValueType> output(numOutputRows, numOutputColumns, numChannels);

        constexpr int blockSize = 1; // blockSize isn't really used in "filters first" mode
        if (tileSize == 2 && filterSize == 3)
        {
            FixedWinograd2D<ValueType, 2, 3, blockSize>::Convolve2DWinogradFiltersFirst(input, transformedFilters, numFilters, output);
        }
        else if (tileSize == 4 && filterSize == 3)
        {
            FixedWinograd2D<ValueType, 4, 3, blockSize>::Convolve2DWinogradFiltersFirst(input, transformedFilters, numFilters, output);
        }
        else
        {
            assert(false && "Tile and filter size not implemented");
        }

        return output;
    }

    //
    // Explicit instantiations
    //

    // Basic 1D entry points
    template math::RowVector<float> Convolve1DWinograd(const math::RowVector<float>& input, const math::RowVector<float>& filter);
    template math::RowVector<float> Convolve1DWinograd(const math::RowVector<float>& input, const math::RowVector<float>& filter, int tileSize);
    template math::RowVector<double> Convolve1DWinograd(const math::RowVector<double>& input, const math::RowVector<double>& filter);
    template math::RowVector<double> Convolve1DWinograd(const math::RowVector<double>& input, const math::RowVector<double>& filter, int tileSize);

    // Basic tensor-valued 2D entry points
    template math::ChannelColumnRowTensor<float> Convolve2DWinograd(const math::ConstChannelColumnRowTensorReference<float>& input, const math::ConstChannelColumnRowTensorReference<float>& filters, int numFilters, WinogradFilterOrder order);
    template math::ChannelColumnRowTensor<float> Convolve2DWinograd(const math::ConstChannelColumnRowTensorReference<float>& input, const math::ConstChannelColumnRowTensorReference<float>& filters, int numFilters, int tileSize, WinogradFilterOrder order);
    template math::ChannelColumnRowTensor<float> Convolve2DWinogradPretransformed(const math::ConstChannelColumnRowTensorReference<float>& input, const math::ConstChannelColumnRowTensorReference<float>& transformedFilters, int numFilters, int tileSize, int filterSize, WinogradFilterOrder order);
    template math::ChannelColumnRowTensor<double> Convolve2DWinograd(const math::ConstChannelColumnRowTensorReference<double>& input, const math::ConstChannelColumnRowTensorReference<double>& filters, int numFilters, WinogradFilterOrder order);
    template math::ChannelColumnRowTensor<double> Convolve2DWinograd(const math::ConstChannelColumnRowTensorReference<double>& input, const math::ConstChannelColumnRowTensorReference<double>& filters, int numFilters, int tileSize, WinogradFilterOrder order);
    template math::ChannelColumnRowTensor<double> Convolve2DWinogradPretransformed(const math::ConstChannelColumnRowTensorReference<double>& input, const math::ConstChannelColumnRowTensorReference<double>& transformedFilters, int numFilters, int tileSize, int filterSize, WinogradFilterOrder order);

    // Depthwise-separable versions
    template math::ChannelColumnRowTensor<float> Convolve2DWinogradDepthwiseSeparable(const math::ConstChannelColumnRowTensorReference<float>& input, const math::ConstChannelColumnRowTensorReference<float>& filters, int numFilters, WinogradFilterOrder order);
    template math::ChannelColumnRowTensor<float> Convolve2DWinogradDepthwiseSeparable(const math::ConstChannelColumnRowTensorReference<float>& input, const math::ConstChannelColumnRowTensorReference<float>& filters, int numFilters, int tileSize, WinogradFilterOrder order);
    template math::ChannelColumnRowTensor<float> Convolve2DWinogradDepthwiseSeparablePretransformed(const math::ConstChannelColumnRowTensorReference<float>& input, const math::ConstChannelColumnRowTensorReference<float>& transformedFilters, int numFilters, int tileSize, int filterSize, WinogradFilterOrder order);
    template math::ChannelColumnRowTensor<double> Convolve2DWinogradDepthwiseSeparable(const math::ConstChannelColumnRowTensorReference<double>& input, const math::ConstChannelColumnRowTensorReference<double>& filters, int numFilters, WinogradFilterOrder order);
    template math::ChannelColumnRowTensor<double> Convolve2DWinogradDepthwiseSeparable(const math::ConstChannelColumnRowTensorReference<double>& input, const math::ConstChannelColumnRowTensorReference<double>& filters, int numFilters, int tileSize, WinogradFilterOrder order);
    template math::ChannelColumnRowTensor<double> Convolve2DWinogradDepthwiseSeparablePretransformed(const math::ConstChannelColumnRowTensorReference<double>& input, const math::ConstChannelColumnRowTensorReference<double>& transformedFilters, int numFilters, int tileSize, int filterSize, WinogradFilterOrder order);

    // Winograd matrix functions
    template math::RowMatrix<float> GetLeftDataTransformMatrix(int tileSize, int filterSize);
    template math::RowMatrix<float> GetRightDataTransformMatrix(int tileSize, int filterSize);
    template math::RowMatrix<float> GetLeftFilterTransformMatrix(int tileSize, int filterSize);
    template math::RowMatrix<float> GetRightFilterTransformMatrix(int tileSize, int filterSize);
    template math::RowMatrix<float> GetLeftResultTransformMatrix(int tileSize, int filterSize);
    template math::RowMatrix<float> GetRightResultTransformMatrix(int tileSize, int filterSize);
    template math::RowMatrix<double> GetLeftDataTransformMatrix(int tileSize, int filterSize);
    template math::RowMatrix<double> GetRightDataTransformMatrix(int tileSize, int filterSize);
    template math::RowMatrix<double> GetLeftFilterTransformMatrix(int tileSize, int filterSize);
    template math::RowMatrix<double> GetRightFilterTransformMatrix(int tileSize, int filterSize);
    template math::RowMatrix<double> GetLeftResultTransformMatrix(int tileSize, int filterSize);
    template math::RowMatrix<double> GetRightResultTransformMatrix(int tileSize, int filterSize);

    // Obtaining pretransformed filters
    template math::ChannelColumnRowTensor<float> GetTransformedFilters(const math::ConstChannelColumnRowTensorReference<float>& filters, int numFilters, int tileSize, WinogradFilterOrder order);
    template void TransformFilters(const math::ConstChannelColumnRowTensorReference<float>& filters, int numFilters, int tileSize, WinogradFilterOrder order, math::ChannelColumnRowTensorReference<float>& transformedFilters);
    template math::ChannelColumnRowTensor<double> GetTransformedFilters(const math::ConstChannelColumnRowTensorReference<double>& filters, int numFilters, int tileSize, WinogradFilterOrder order);
    template void TransformFilters(const math::ConstChannelColumnRowTensorReference<double>& filters, int numFilters, int tileSize, WinogradFilterOrder order, math::ChannelColumnRowTensorReference<double>& transformedFilters);
} // namespace dsp
} // namespace ell
