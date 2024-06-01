////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     WinogradConvolutionNode.cpp (nodes)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WinogradConvolutionNode.h"
#include "ConstantNode.h"
#include "ReorderDataCodeNode.h"

#include <dsp/include/WinogradConvolution.h>

#include <math/include/Matrix.h>

#include <emitters/include/IRProfiler.h>

#include <utilities/include/Logger.h>
#include <utilities/include/Unused.h>

// #define PROFILE_REGIONS

namespace ell
{
namespace nodes
{
    using namespace logging;

    //
    // Internal utility routines and types
    //
    namespace
    {
        //
        // Useful debugging utilities
        //
        template <typename ValueType>
        void PrintBlock(emitters::IRFunctionEmitter& function,
                        emitters::LLVMValue block,
                        int rows,
                        int columns,
                        int channels)
        {
            auto blockArray = function.LocalMultidimArray(block, { rows, columns, channels });
            for (int rowIndex = 0; rowIndex < rows; ++rowIndex)
            {
                for (int columnIndex = 0; columnIndex < columns; ++columnIndex)
                {
                    for (int channelIndex = 0; channelIndex < channels; ++channelIndex)
                    {
                        emitters::IRLocalScalar val = blockArray({ rowIndex, columnIndex, channelIndex });
                        function.Printf("%f ", { function.CastValue<double>(val) });
                    }
                    function.Print("  ");
                }
                function.Print("\n");
            }
        }

        //
        // Types
        //

        // Holds a spatial location plus channel index in a 3D image
        struct ImageCoordinates
        {
            emitters::IRLocalScalar row;
            emitters::IRLocalScalar column;
            emitters::IRLocalScalar channel;
        };

        struct ConstImageSize
        {
            int rows;
            int columns;
            int channels;
        };

        struct ConstConvolutionSize
        {
            int rows;
            int columns;
            int inputChannels;
            int filterChannels;
            int outputChannels;

            ConstImageSize GetInputSize() const { return { rows, columns, inputChannels }; }
            ConstImageSize GetOutputSize() const { return { rows, columns, outputChannels }; }
        };

        using BlockRange = emitters::IRFunctionEmitter::BlockInterval; // (begin, end, size)

        struct ImageBlockRange
        {
            BlockRange rows;
            BlockRange columns;
            BlockRange channels;
        };

        // Holds image coordinates for an input image channel and and output filter
        struct ConvolutionCoordinates
        {
            emitters::IRLocalScalar row;
            emitters::IRLocalScalar column;
            emitters::IRLocalScalar channel;
            emitters::IRLocalScalar filter;

            ImageCoordinates GetInputCoordinates() const { return { row, column, channel }; }
            ImageCoordinates GetOutputCoordinates() const { return { row, column, filter }; }
        };

        // extents for the sub-block we're convolving at the moment
        struct ConvolutionBlockRanges
        {
            BlockRange inputRows; // input window rows
            BlockRange inputColumns; // input window columns
            BlockRange inputChannels; // input channels
            BlockRange filters; // filters (== input channels for depthwise-separable convolution)
            BlockRange filterChannels; // filter channels (== input channels for "full" 3D convolution)
            BlockRange outputRows; // output tile rows: shares same begin with input rows, different end, size
            BlockRange outputColumns; // output tile columns: shares same begin with input columns, different end, size
            BlockRange outputChannels; // output channels (== input channels for depthwise-separable convolution)

            ImageBlockRange GetInputBlockRange() const { return { inputRows, inputColumns, inputChannels }; } // an input "window" containing the values necessary to compute a tile of output
            ImageBlockRange GetOutputBlockRange() const { return { outputRows, outputColumns, outputChannels }; } // an output tile
        };

        struct WinogradScratchStorage
        {
            emitters::LLVMValue inputBlock;
            emitters::LLVMValue transformedInputBlock;
            emitters::LLVMValue transformedFilterBlock;
            emitters::LLVMValue transformedOutputBlock;
            emitters::LLVMValue outputTile;
        };

        //
        // Misc
        //
        std::string to_string(dsp::WinogradFilterOrder order)
        {
            switch (order)
            {
            case dsp::WinogradFilterOrder::filtersFirst:
                return "filtersFirst";
                break;
            case dsp::WinogradFilterOrder::tilesFirst:
                return "tilesFirst";
                break;
            default:
                throw utilities::DataFormatException(utilities::DataFormatErrors::illegalValue);
            }
        }

        dsp::WinogradFilterOrder filter_order_from_string(std::string name)
        {
            if (name == "filtersFirst")
            {
                return dsp::WinogradFilterOrder::filtersFirst;
            }
            else if (name == "tilesFirst")
            {
                return dsp::WinogradFilterOrder::tilesFirst;
            }
            else
            {
                throw utilities::DataFormatException(utilities::DataFormatErrors::illegalValue);
            }
        }

        //
        // IR arithmetic with simplification
        //
        emitters::IRLocalScalar AddAndSimplify(emitters::IRLocalScalar a, emitters::IRLocalScalar b)
        {
            if (auto constA = llvm::dyn_cast<llvm::ConstantFP>(a.value))
            {
                if (constA->isZero())
                {
                    return b;
                }
                if (auto constB = llvm::dyn_cast<llvm::ConstantFP>(b.value))
                {
                    // a and b both constants
                    const auto& aValue = constA->getValueAPF();
                    const auto& bValue = constB->getValueAPF();
                    return a.function.LocalScalar(llvm::ConstantFP::get(a.value->getContext(), aValue + bValue));
                }
            }
            if (auto constB = llvm::dyn_cast<llvm::ConstantFP>(b.value))
            {
                if (constB->isZero())
                {
                    return a;
                }
            }
            return a + b;
        }

        emitters::IRLocalScalar MultiplyAndSimplify(emitters::IRLocalScalar a, emitters::IRLocalScalar b)
        {
            auto& function = a.function;
            if (auto constA = llvm::dyn_cast<llvm::ConstantFP>(a.value))
            {
                if (auto constB = llvm::dyn_cast<llvm::ConstantFP>(b.value))
                {
                    // a and b both constants
                    const auto& aValue = constA->getValueAPF();
                    const auto& bValue = constB->getValueAPF();
                    return function.LocalScalar(llvm::ConstantFP::get(a.value->getContext(), aValue * bValue));
                }
                if (constA->isZero())
                {
                    return function.LocalScalar(llvm::ConstantFP::get(a.value->getType(), 0.0));
                }
                if (constA->isExactlyValue(1.0))
                {
                    return b;
                }
            }
            if (auto constB = llvm::dyn_cast<llvm::ConstantFP>(b.value))
            {
                if (constB->isZero())
                {
                    return function.LocalScalar(llvm::ConstantFP::get(a.value->getType(), 0.0));
                }
                if (constB->isExactlyValue(1.0))
                {
                    return a;
                }
            }

            return a * b;
        }

        // computes (a*b) + c, unless c is invalid, in which case return (a*b)
        emitters::IRLocalScalar MultiplyAddSafe(emitters::IRLocalScalar a, emitters::IRLocalScalar b, emitters::IRLocalScalar c)
        {
            auto product = MultiplyAndSimplify(a, b);
            return c.IsValid() ? AddAndSimplify(product, c) : product;
        }

        //
        // IRLocalValueMatrix is just a class that holds a rectangular array of LLVM IR values
        // so that we can do matrix algebra with them easily. The matrix doesn't correspond to
        // a region of memory on the target device, the elements are just expressions built up
        // at compile time.
        //
        class IRLocalValueMatrix
        {
        public:
            IRLocalValueMatrix(emitters::IRFunctionEmitter& function, int rows, int columns) :
                function(function),
                rows(rows),
                columns(columns)
            {
                for (int index = 0; index < (rows * columns); ++index)
                {
                    _data.push_back(function.LocalScalar());
                }
            }

            emitters::IRLocalScalar operator()(int row, int column) const
            {
                return _data[GetIndex(row, column)];
            }

            emitters::IRLocalScalar& operator()(int row, int column)
            {
                return _data[GetIndex(row, column)];
            }

            // properties
            emitters::IRFunctionEmitter& function;
            int rows = 0;
            int columns = 0;

        private:
            std::vector<emitters::IRLocalScalar> _data;
            int GetIndex(int row, int column) const
            {
                return row * columns + column;
            }
        };

        //
        // IRLocalValueMultidimArray is just a class that holds a multidimensional array of LLVM IR values.
        // The array doesn't correspond to a region of memory on the target device, the elements are just
        // expressions built up at compile time.
        //
        class IRLocalValueMultidimArray
        {
        public:
            IRLocalValueMultidimArray(emitters::IRFunctionEmitter& function, std::initializer_list<int> dimensions) :
                function(function)
            {
                strides.reserve(dimensions.size());
                std::copy(dimensions.begin() + 1, dimensions.end(), std::back_inserter(strides));
                strides.push_back(1);
                int currentStride = 1;
                int numEntries = 1;
                for (auto d : dimensions)
                {
                    numEntries *= d;
                }

                for (auto it = std::rbegin(strides); it != std::rend(strides); ++it)
                {
                    *it *= currentStride;
                    currentStride = *it;
                }

                for (int index = 0; index < numEntries; ++index)
                {
                    _data.push_back(function.LocalScalar());
                }
            }

            emitters::IRLocalScalar operator()(std::initializer_list<int> indices) const
            {
                return _data[GetIndex(indices)];
            }

            emitters::IRLocalScalar& operator()(std::initializer_list<int> indices)
            {
                return _data[GetIndex(indices)];
            }

            // properties
            emitters::IRFunctionEmitter& function;
            std::vector<int> dimensions;
            std::vector<int> strides;

        private:
            int GetIndex(std::initializer_list<int> indices) const
            {
                assert(indices.size() == strides.size());
                int offset = 0;
                auto stridesIt = strides.begin();
                for (auto& i : indices)
                {
                    offset += i * (*stridesIt);
                    ++stridesIt;
                }
                return offset;
            }

            std::vector<emitters::IRLocalScalar> _data;
        };

        template <typename ValueType>
        IRLocalValueMatrix GetLocalMatrix(emitters::IRFunctionEmitter& function, const math::RowMatrix<ValueType>& m)
        {
            auto numRows = static_cast<int>(m.NumRows());
            auto numColumns = static_cast<int>(m.NumColumns());
            auto result = IRLocalValueMatrix(function, numRows, numColumns);
            for (int i = 0; i < numRows; ++i)
            {
                for (int j = 0; j < numColumns; ++j)
                {
                    result(i, j) = function.LocalScalar<ValueType>(m(i, j));
                }
            }
            return result;
        }

        // Compute a * b * a', for each channel in a tensor b
        template <typename IRMatrixType1, typename IRTensorType2>
        IRLocalValueMultidimArray MatrixMatrixTransposeMultiply(const IRMatrixType1& a, const IRTensorType2& b, int blockSize)
        {
            // a: m x k,  b: k x k x blockSize,  result: m x m x blockSize
            auto m = a.rows;
            auto k = a.columns;

            auto result = IRLocalValueMultidimArray(a.function, { m, m, blockSize });
            for (int channel = 0; channel < blockSize; ++channel)
            {
                for (int ii = 0; ii < m; ++ii)
                {
                    for (int jj = 0; jj < m; ++jj)
                    {
                        auto sum = a.function.LocalScalar();
                        for (int kk = 0; kk < k; ++kk)
                        {
                            for (int ll = 0; ll < k; ++ll)
                            {
                                sum = MultiplyAddSafe(MultiplyAndSimplify(a(ii, ll), b({ ll, kk, channel })), a(jj, kk), sum);
                            }
                        }
                        result({ ii, jj, channel }) = sum;
                    }
                }
            }
            return result;
        }

        // Compute a * b * a', for each channel in b
        template <typename IRMatrixType1, typename IRTensorType2>
        void MatrixMatrixTransposeMultiplyInto(const IRMatrixType1& a, const IRTensorType2& b, int blockSize, emitters::IRLocalMultidimArray& c)
        {
            // a: m x k,  b: k x k x blockSize,  c: m x m x blockSize
            auto m = a.rows;
            auto k = a.columns;
            auto& function = a.function;

            // The size of the input blocks is generally larger than the blocks we want to operate on when
            // doing the transformation itself. So, we'll break the block up into sub-blocks
            function.For(blockSize, [=](emitters::IRFunctionEmitter& function, auto channelIndex) {
                // Compute a * b * a' for a sub-block of b
                auto inputSubBlock = function.PointerOffset(b.data, channelIndex);
                auto d = function.LocalTensor(inputSubBlock, { k, k, blockSize }, emitters::RowMajorTensorLayout);

                auto Xvals = MatrixMatrixTransposeMultiply(a, d, 1);

                for (int i = 0; i < m; ++i)
                {
                    for (int j = 0; j < m; ++j)
                    {
                        c({ function.LocalScalar(i), function.LocalScalar(j), channelIndex }) = Xvals({ i, j, 0 });
                    }
                }
            });
        }

        void ElementwiseMultiply(emitters::IRFunctionEmitter& function, emitters::LLVMValue AMem, emitters::LLVMValue BMem, int numEntries, emitters::LLVMValue CMem)
        {
#ifdef PROFILE_REGIONS
            auto region = emitters::IRProfileRegionBlock(function, "Winograd_FF_ElementwiseMultiply");
            UNUSED(region);
#endif
            auto A = function.LocalArray(AMem);
            auto B = function.LocalArray(BMem);
            auto C = function.LocalArray(CMem);

            const bool parallelize = false;
            if (parallelize)
            {
                function.ParallelFor(numEntries, { A, B, C }, [](emitters::IRFunctionEmitter& function, emitters::IRLocalScalar i, const std::vector<emitters::LLVMValue>& capturedValues) {
                    auto A = function.LocalArray(capturedValues[0]);
                    auto B = function.LocalArray(capturedValues[1]);
                    auto C = function.LocalArray(capturedValues[2]);
                    C[i] = A[i] * B[i];
                });
            }
            else
            {
                function.For(numEntries, [A, B, C](emitters::IRFunctionEmitter& function, auto i) {
                    C[i] = A[i] * B[i];
                });
            }
        }

        void ElementwiseMultiplyAccumulate(emitters::IRFunctionEmitter& function, emitters::LLVMValue filterMem, emitters::LLVMValue inputMem, int numEntries, int filterDepth, int channelDepth, emitters::LLVMValue outputMem)
        {
#ifdef PROFILE_REGIONS
            auto region = emitters::IRProfileRegionBlock(function, "Winograd_FF_ElementwiseMultiplyAccumulate");
            UNUSED(region);
#endif
            function.StoreZero(outputMem, numEntries * filterDepth);
            function.For(numEntries, [filterDepth, channelDepth, filterMem, inputMem, outputMem](emitters::IRFunctionEmitter& function, auto i) {
                auto input = function.LocalArray(function.PointerOffset(inputMem, i * channelDepth));
                auto output = function.LocalArray(function.PointerOffset(outputMem, i * filterDepth));
                function.For(filterDepth, [i, input, output, filterMem, filterDepth, channelDepth](emitters::IRFunctionEmitter& function, auto j) {
                    auto filter = function.LocalArray(function.PointerOffset(filterMem, i * (channelDepth * filterDepth) + j * channelDepth));
                    function.For(channelDepth, [j, filter, input, output](emitters::IRFunctionEmitter& function, auto k) {
                        output[j] = output[j] + (filter[k] * input[k]);
                    });
                });
            });
        }

        //
        // Winograd-specific routines
        //
        template <typename ValueType>
        void TransformInputBlock(emitters::IRFunctionEmitter& function, emitters::LLVMValue inputBlock, int tileSize, int filterSize, int blockSize, emitters::LLVMValue transformedInputBlock)
        {
#ifdef PROFILE_REGIONS
            auto region = emitters::IRProfileRegionBlock(function, "Winograd_FF_TransformInputBlock");
            UNUSED(region);
#endif

            const int windowSize = tileSize + filterSize - 1;
            auto Bt = GetLocalMatrix(function, dsp::GetLeftDataTransformMatrix<ValueType>(tileSize, filterSize));

            // Compute X = B'dB
            auto d = function.LocalMultidimArray(inputBlock, { windowSize, windowSize, blockSize });
            auto outputBlock = function.LocalMultidimArray(transformedInputBlock, { windowSize, windowSize, blockSize });
            MatrixMatrixTransposeMultiplyInto(Bt, d, blockSize, outputBlock);
        }

        template <typename ValueType>
        void TransformOutputBlock(emitters::IRFunctionEmitter& function, emitters::LLVMValue transformedOutputBlock, int tileSize, int filterSize, int blockSize, emitters::LLVMValue outputBlock)
        {
#ifdef PROFILE_REGIONS
            auto region = emitters::IRProfileRegionBlock(function, "Winograd_FF_TransformOutputBlock");
            UNUSED(region);
#endif
            const int windowSize = tileSize + filterSize - 1;
            auto At = GetLocalMatrix(function, dsp::GetLeftResultTransformMatrix<ValueType>(tileSize, filterSize));

            // Compute result tile At * X * A
            auto X = function.LocalMultidimArray(transformedOutputBlock, { windowSize, windowSize, blockSize });
            auto result = function.LocalMultidimArray(outputBlock, { tileSize, tileSize, blockSize });
            MatrixMatrixTransposeMultiplyInto(At, X, blockSize, result);
        }

        template <typename ValueType>
        void LoadInputBlock(emitters::IRFunctionEmitter& function,
                            emitters::IRLocalArray input,
                            const model::PortMemoryLayout& inputLayout,
                            ImageBlockRange inputRange,
                            int tileSize,
                            int filterSize,
                            emitters::IRLocalArray inputBlock)
        {
#ifdef PROFILE_REGIONS
            auto region = emitters::IRProfileRegionBlock(function, "Winograd_LoadInputBlock");
            UNUSED(region);
#endif

            const auto numChannels = inputLayout.GetLogicalDimensionActiveSize(2);
            UNUSED(numChannels);
            const int windowSize = tileSize + filterSize - 1;

            auto blockSize = inputRange.channels.size;
            bool channelMajor = inputLayout.GetLogicalDimensionOrder() == utilities::DimensionOrder({ 2, 0, 1 });
            if (channelMajor)
            {
                // input image: channels x rows x columns
                // Retrieve values from input into local matrix of size windowSize x windowSize x blockSize
                function.For(inputRange.channels.size, [blockSize, windowSize, inputRange, input, inputLayout, inputBlock](emitters::IRFunctionEmitter& function, auto channelIndex) {
                    function.For(inputRange.rows.size, [channelIndex, blockSize, windowSize, inputRange, input, inputLayout, inputBlock](emitters::IRFunctionEmitter& function, auto rowIndex) {
                        function.For(inputRange.columns.size, [rowIndex, channelIndex, blockSize, windowSize, inputRange, input, inputLayout, inputBlock](emitters::IRFunctionEmitter& function, auto columnIndex) {
                            const auto inputRowStride = static_cast<int>(inputLayout.GetLogicalDimensionIncrement(0));
                            const auto inputColumnStride = static_cast<int>(inputLayout.GetLogicalDimensionIncrement(1));
                            const auto inputChannelStride = static_cast<int>(inputLayout.GetLogicalDimensionIncrement(2));
                            auto windowLoc = (rowIndex * windowSize + columnIndex) * blockSize + channelIndex;
                            auto inputLoc = ((inputRange.rows.begin + rowIndex) * inputRowStride) + ((inputRange.columns.begin + columnIndex) * inputColumnStride) + ((inputRange.channels.begin + channelIndex) * inputChannelStride);
                            inputBlock[windowLoc] = input[inputLoc];
                        });
                    });
                });
            }
            else if (inputLayout.IsCanonicalOrder())
            {
                // input image: rows x columns x channels
                // Retrieve values from input into local matrix of size windowSize x windowSize x blockSize
                function.For(inputRange.rows.size, [blockSize, windowSize, inputRange, input, inputLayout, inputBlock](emitters::IRFunctionEmitter& function, auto rowIndex) {
                    function.For(inputRange.columns.size, [rowIndex, blockSize, windowSize, inputRange, input, inputLayout, inputBlock](emitters::IRFunctionEmitter& function, auto columnIndex) {
                        function.For(inputRange.channels.size, [rowIndex, columnIndex, blockSize, windowSize, inputRange, input, inputLayout, inputBlock](emitters::IRFunctionEmitter& function, auto channelIndex) {
                            const auto inputRowStride = static_cast<int>(inputLayout.GetLogicalDimensionIncrement(0));
                            const auto inputColumnStride = static_cast<int>(inputLayout.GetLogicalDimensionIncrement(1));
                            const auto inputChannelStride = static_cast<int>(inputLayout.GetLogicalDimensionIncrement(2));
                            auto windowLoc = (rowIndex * windowSize + columnIndex) * blockSize + channelIndex;
                            auto inputLoc = ((inputRange.rows.begin + rowIndex) * inputRowStride) + ((inputRange.columns.begin + columnIndex) * inputColumnStride) + ((inputRange.channels.begin + channelIndex) * inputChannelStride);
                            inputBlock[windowLoc] = input[inputLoc];
                        });
                    });
                });
            }
            else
            {
                throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "WinogradConvolutionComputeNode: Can only operate on row-major or channel-major inputs");
            }

            // input: inputImageRows x inputImageColumns x numChannels tensor
            // inputBlock: windowSize x windowSize x blockSize block extracted from the given channel with the given upper-left coordinates

            // check for row range size < tileSize
            auto windowRows = inputRange.rows.size.GetIntValue<int>(windowSize);
            auto windowColumns = inputRange.columns.size.GetIntValue<int>(windowSize);

            // Zero out unused parts
            // First, the righthand edge:
            if (windowSize - windowColumns > 0)
            {
                for (int rowIndex = 0; rowIndex < windowRows; ++rowIndex)
                {
                    assert(blockSize.IsConstantInt());
                    auto outputLoc = (rowIndex * windowSize + windowColumns) * blockSize.GetIntValue<int>();
                    function.StoreZero(function.PointerOffset(inputBlock, outputLoc), (windowSize - windowColumns) * blockSize.GetIntValue<int>());
                }
            }

            // Then the bottom part:
            if (windowSize - windowRows > 0)
            {
                assert(blockSize.IsConstantInt());
                auto outputLoc = windowRows * windowSize * blockSize.GetIntValue<int>();
                function.StoreZero(function.PointerOffset(inputBlock, outputLoc), (windowSize - windowRows) * windowSize * blockSize.GetIntValue<int>());
            }
        }

        template <typename ValueType>
        void LoadFilterBlock(emitters::IRFunctionEmitter& function,
                             emitters::LLVMValue filters,
                             const model::PortMemoryLayout& filterLayout,
                             BlockRange filterRange,
                             BlockRange filterChannelRange,
                             int tileSize,
                             int filterSize,
                             emitters::LLVMValue filterBlock)
        {
#ifdef PROFILE_REGIONS
            auto region = emitters::IRProfileRegionBlock(function, "Winograd_FF_GetFiltersBlock");
            UNUSED(region);
#endif
            int windowSize = tileSize + filterSize - 1;

            // Input filters in f x fc x r x c order
            auto filtersArray = function.LocalArray(filters);
            auto filterBlockArray = function.LocalArray(filterBlock);

            const int filterStride = filterLayout.GetCumulativeIncrement(0); // dimension 0 is "filters" in f x fc x r x c order
            const int filterChannelStride = filterLayout.GetCumulativeIncrement(1); // dimension 1 is "filter channels" in f x fc x r x c order
            const int filterRowStride = filterLayout.GetCumulativeIncrement(2); // dimension 2 is "rows" in f x fc x r x c order
            const int blockFilters = filterRange.size.GetIntValue<int>();
            const int blockFilterChannels = filterChannelRange.size.GetIntValue<int>();

            function.For(windowSize, [=](emitters::IRFunctionEmitter& function, emitters::IRLocalScalar rowIndex) {
                function.For(windowSize, [=](emitters::IRFunctionEmitter& function, emitters::IRLocalScalar columnIndex) {
                    function.For(filterRange.size, [=](emitters::IRFunctionEmitter& function, emitters::IRLocalScalar filterIndex) {
                        function.For(filterChannelRange.size, [=](emitters::IRFunctionEmitter& function, emitters::IRLocalScalar filterChannelIndex) {
                            auto filterLoc = (filterRange.begin + filterIndex) * filterStride + (filterChannelRange.begin + filterChannelIndex) * filterChannelStride + rowIndex * filterRowStride + columnIndex;
                            auto filterBlockLoc = rowIndex * windowSize * blockFilters * blockFilterChannels + columnIndex * blockFilters * blockFilterChannels + filterIndex * blockFilterChannels + filterChannelIndex;
                            filterBlockArray[filterBlockLoc] = filtersArray[filterLoc];
                        });
                    });
                });
            });
        }

        // transformedInput is a windowSize x windowSize x tileRows x tileColumns x numChannels  tensor containing the entire transformed input signal.
        // Think of it as (windowSize*windowSize) separate tileRows x tileColumns x numChannels tensors: one tileRows x tileColumns image for each position in the transformed window.
        // So, there's a tensor representing the upper-left window pixel for each transformed input tile, another representing the (0,1) pixel
        // of each transformed input tile, and so on.

        template <typename ValueType>
        void SplatTransformedInputBlock(emitters::IRFunctionEmitter& function,
                                        emitters::IRLocalArray transformedInputBlock,
                                        ImageBlockRange inputRange,
                                        int numTileRows,
                                        int numTileColumns,
                                        int numChannels,
                                        int tileSize,
                                        int filterSize,
                                        emitters::IRLocalArray transformedInput)
        {

            auto tileRow = inputRange.rows.index;
            auto tileColumn = inputRange.columns.index;
            auto channelIndex = inputRange.channels.begin; // #### TODO: verify we don't really want inputRange.channels.index here

            const int windowSize = tileSize + filterSize - 1;
            auto offset = (tileRow * numTileColumns * numChannels) + (tileColumn * numChannels) + channelIndex;
            auto entryStride = numTileRows * numTileColumns * numChannels;

            int numEntries = windowSize * windowSize;
            for (int windowLoc = 0; windowLoc < numEntries; ++windowLoc)
            {
                auto outputLoc = windowLoc * entryStride + offset;
                auto inputLoc = windowLoc * inputRange.channels.size;
                function.MemoryCopy<ValueType>(transformedInputBlock, inputLoc, transformedInput, outputLoc, inputRange.channels.size);
            }
        }

        template <typename ValueType>
        void ProcessInputBlock(emitters::IRFunctionEmitter& function,
                               emitters::IRLocalArray input,
                               const model::PortMemoryLayout& inputLayout,
                               ImageBlockRange inputRange,
                               int tileSize,
                               int filterSize,
                               emitters::IRLocalArray inputBlock,
                               emitters::IRLocalArray transformedInputBlock,
                               emitters::IRLocalArray transformedInput)
        {
            // input: inputImageRows x inputImageColumns x numChannels tensor
            // transformedInput is a windowSize x windowSize x tileRows x tileColumns x numChannels tensor containing the entire transformed input signal
            LoadInputBlock<ValueType>(function, input, inputLayout, inputRange, tileSize, filterSize, inputBlock);

            // TODO: fix TransformInputBlock to take ranges instead of tileSize/filterSize/blockSize
            auto blockSize = inputRange.channels.size.GetIntValue<int>();
            TransformInputBlock<ValueType>(function, inputBlock, tileSize, filterSize, blockSize, transformedInputBlock);

            const auto numOutputRows = inputLayout.GetLogicalDimensionActiveSize(0);
            const auto numOutputColumns = inputLayout.GetLogicalDimensionActiveSize(1);
            const auto numTileRows = ((numOutputRows - 1) / tileSize) + 1;
            const auto numTileColumns = ((numOutputColumns - 1) / tileSize) + 1;
            const auto numChannels = inputLayout.GetLogicalDimensionActiveSize(2);
            SplatTransformedInputBlock<ValueType>(function, transformedInputBlock, inputRange, numTileRows, numTileColumns, numChannels, tileSize, filterSize, transformedInput);
        }

        template <typename ValueType>
        void GetTransformedOutputBlock(emitters::IRFunctionEmitter& function, emitters::LLVMValue transformedOutput, emitters::IRLocalScalar tileRow, emitters::IRLocalScalar tileColumn, emitters::IRLocalScalar filterIndex, int numOutputRows, int numOutputColumns, int numFilters, int tileSize, int filterSize, int blockSize, emitters::LLVMValue transformedOutputBlock)
        {
            const auto windowSize = tileSize + filterSize - 1;
            const auto numTileRows = ((numOutputRows - 1) / tileSize) + 1;
            const auto numTileColumns = ((numOutputColumns - 1) / tileSize) + 1;
            const auto windowEntryStride = numFilters * numTileRows * numTileColumns;
            const auto tileIndex = tileRow * numTileColumns + tileColumn;
            const auto offset = (tileIndex * numFilters) + filterIndex;

            for (int windowLoc = 0; windowLoc < windowSize * windowSize; ++windowLoc)
            {
                auto inputLoc = windowLoc * windowEntryStride + offset;
                function.MemoryCopy<ValueType>(transformedOutput, inputLoc, transformedOutputBlock, function.Literal<int>(windowLoc * blockSize), function.Literal<int>(blockSize));
            }
        }

        template <typename ValueType>
        void SplatOutputTile(emitters::IRFunctionEmitter& function, emitters::LLVMValue outputTile, emitters::IRLocalScalar tileRow, emitters::IRLocalScalar tileColumn, emitters::IRLocalScalar filterIndex, int numOutputRows, int numOutputColumns, int numFilters, int tileSize, int blockSize, emitters::LLVMValue output)
        {
            auto tileRowSize = tileSize;
            auto tileColumnSize = tileSize;

            // check for constant tile row on last row
            if (tileRow.IsConstantInt())
            {
                auto tileStart = tileRow.GetIntValue<int>() * tileSize;
                tileRowSize = std::min(tileSize, numOutputRows - tileStart);
            }

            // check for constant tile column on last column
            if (tileColumn.IsConstantInt())
            {
                auto tileStart = tileColumn.GetIntValue<int>() * tileSize;
                tileColumnSize = std::min(tileSize, numOutputColumns - tileStart);
            }

            // rowIndex and columnIndex are the row and column indices within the tile
            const auto columnStride = numFilters;
            const auto rowStride = columnStride * numOutputColumns;
            auto outputStart = (tileSize * tileRow * rowStride) + (tileSize * tileColumn * columnStride) + filterIndex; // offset into the upper-left of the given tile
            for (int rowIndex = 0; rowIndex < tileRowSize; ++rowIndex)
            {
                for (int columnIndex = 0; columnIndex < tileColumnSize; ++columnIndex)
                {
                    auto inputLoc = ((rowIndex * tileSize) + columnIndex) * blockSize;
                    auto outputLoc = outputStart + (rowIndex * rowStride) + (columnIndex * columnStride);
                    function.MemoryCopy<ValueType>(outputTile, function.Literal<int>(inputLoc), output, outputLoc, function.Literal<int>(blockSize));
                }
            }
        }

        template <typename ValueType>
        void AccumulateOutputTile(emitters::IRFunctionEmitter& function,
                                  emitters::LLVMValue outputTile,
                                  ImageBlockRange outputRange,
                                  int tileSize,
                                  emitters::LLVMValue output,
                                  const model::PortMemoryLayout& outputLayout)
        {
#ifdef PROFILE_REGIONS
            auto region = emitters::IRProfileRegionBlock(function, "Winograd_FF_AccumulateOutputTile");
            UNUSED(region);
#endif
            int numOutputColumns = outputLayout.GetLogicalDimensionActiveSize(1);
            int numOutputChannels = outputLayout.GetLogicalDimensionActiveSize(2);

            const auto columnStride = numOutputChannels;
            const auto rowStride = columnStride * numOutputColumns;

            // outputTile is tileSize x tileSize x blockSize
            std::vector<emitters::IRFunctionEmitter::LoopRange> loopRanges = { { function.LocalScalar<int>(0), outputRange.rows.size },
                                                                               { function.LocalScalar<int>(0), outputRange.columns.size } };
            function.For(loopRanges, [rowStride, columnStride, outputRange, tileSize, outputTile, output](emitters::IRFunctionEmitter& function, auto indices) {
                auto rowIndex = indices[0];
                auto columnIndex = indices[1];
                auto blockSize = outputRange.channels.size;
                auto filterIndex = outputRange.channels.begin;
                auto inputLoc = ((rowIndex * tileSize) + columnIndex) * blockSize;
                auto outputLoc = ((rowIndex + outputRange.rows.begin) * rowStride) + ((columnIndex + outputRange.columns.begin) * columnStride) + filterIndex;

                // TODO: I think now the outputTile is always contiguous, so we can just iterate over it continuously
                auto outputTileArray = function.LocalArray(function.PointerOffset(outputTile, inputLoc));
                auto outputArray = function.LocalArray(function.PointerOffset(output, outputLoc));
                assert(blockSize.IsConstantInt());
                for (int blockEntryIndex = 0; blockEntryIndex < blockSize.GetIntValue<int>(); ++blockEntryIndex)
                {
                    outputArray[blockEntryIndex] = outputArray[blockEntryIndex] + outputTileArray[blockEntryIndex];
                }
            });
        }

        //
        // `ProcessOutputBlock()` copies data for a transformed output tile into the correct place in the output
        // If the tile row or column are passed in as compile-time constants, then partial tiles will be correctly
        // copied, otherwise this code assumes the tile is fully contained in the output.
        //
        template <typename ValueType>
        void ProcessOutputBlock(emitters::IRFunctionEmitter& function,
                                emitters::IRLocalArray transformedOutput,
                                emitters::IRLocalScalar tileRow,
                                emitters::IRLocalScalar tileColumn,
                                emitters::IRLocalScalar filterIndex,
                                int tileSize,
                                int filterSize,
                                int blockSize,
                                emitters::IRLocalArray transformedOutputBlock,
                                emitters::IRLocalArray outputTile,
                                emitters::IRLocalArray output,
                                const model::PortMemoryLayout& outputLayout)
        {
            const auto numOutputRows = outputLayout.GetLogicalDimensionActiveSize(0);
            const auto numOutputColumns = outputLayout.GetLogicalDimensionActiveSize(1);
            const auto numFilters = outputLayout.GetLogicalDimensionActiveSize(2);

            // transformedOutput is a numTileRows, numTileColumns, numFilters image tensor containing the convolution result
            GetTransformedOutputBlock<ValueType>(function, transformedOutput, tileRow, tileColumn, filterIndex, numOutputRows, numOutputColumns, numFilters, tileSize, filterSize, blockSize, transformedOutputBlock);
            TransformOutputBlock<ValueType>(function, transformedOutputBlock, tileSize, filterSize, blockSize, outputTile);

            // outputTile is the tile block at (tileRow, tileColumn, filterIndex) of the output
            SplatOutputTile<ValueType>(function, outputTile, tileRow, tileColumn, filterIndex, numOutputRows, numOutputColumns, numFilters, tileSize, blockSize, output);
        }

        //
        // `ConvolveAccumulateBlock()` performs the Winograd convolution for a single block of the input data. This function is used
        // in 'filtersFirst' and separable convolutions.
        //
        template <typename ValueType>
        void ConvolveAccumulateBlock(emitters::IRFunctionEmitter& function,

                                     // input data
                                     emitters::IRLocalArray input,
                                     const model::PortMemoryLayout& inputLayout,

                                     // filters
                                     emitters::IRLocalArray transformedFilters,
                                     const model::PortMemoryLayout& transformedFilterLayout,

                                     ConvolutionBlockRanges tileRange,

                                     // problem size
                                     ConstConvolutionSize problemSize,

                                     // Winograd-specific parameters
                                     int tileSize,
                                     int filterSize,

                                     // Temporary storage
                                     WinogradScratchStorage scratch,

                                     // output image
                                     emitters::IRLocalArray output,
                                     const model::PortMemoryLayout& outputLayout)
        {
            const auto windowSize = filterSize + tileSize - 1;
            const auto numChannels = problemSize.inputChannels;
            const auto numFilters = problemSize.outputChannels;
            const auto numFilterChannels = problemSize.filterChannels;
            const bool isSeparable = (numFilterChannels == 1 && numFilters == numChannels);
            auto inputBlock = function.LocalArray(scratch.inputBlock);
            auto transformedInputBlock = function.LocalArray(scratch.transformedInputBlock);
            auto transformedFilterBlock = function.LocalArray(scratch.transformedFilterBlock);
            auto transformedOutputBlock = function.LocalArray(scratch.transformedOutputBlock);
            auto outputTile = function.LocalArray(scratch.outputTile);

            // NOTE: in this function, the filter values in main memory are stored in a (numFilters x numChannels x windowSize x windowSize) tensor

            // inputBlock has dimensions given by tileRange.GetInputBlockRange()
            // transformedInputBlock and transformedOutputBlock have dimensions windowSize x windowSize x tileRange.channel.size
            // outputTile has dimensions tileSize x tileSize x tileRange.outputChannel.size, to be copied into the output image range described by tileRange.GetOutputBlockRange()

            const int inputChannelBlockDepth = tileRange.inputChannels.size.GetIntValue<int>();
            const int filterBlockDepth = tileRange.filters.size.GetIntValue<int>();
            const int filterChannelBlockDepth = tileRange.filterChannels.size.GetIntValue<int>();
            const int outputChannelBlockDepth = tileRange.outputChannels.size.GetIntValue<int>();

            // Fetch a (windowSize x windowSize x tileRange.inputChannels.size) block of data from the input image
            auto inputBlockRange = tileRange.GetInputBlockRange();
            inputBlockRange.channels.size = function.LocalScalar(inputChannelBlockDepth);
            inputBlockRange.channels.end = inputBlockRange.channels.begin + inputChannelBlockDepth;

            // Load a block of input image data into local memory block
            LoadInputBlock<ValueType>(function, input, inputLayout, inputBlockRange, tileSize, filterSize, inputBlock);

            // Compute X = B'dB, a block of the same dimensions as inputBlock
            TransformInputBlock<ValueType>(function, inputBlock, tileSize, filterSize, inputChannelBlockDepth, transformedInputBlock);

            const auto useFilterBlock = (filterBlockDepth > 1 || filterChannelBlockDepth > 1);
            if (useFilterBlock)
            {
                if (isSeparable)
                {
                    ElementwiseMultiply(function, transformedFilterBlock, transformedInputBlock, windowSize * windowSize * filterBlockDepth * filterChannelBlockDepth, transformedOutputBlock);
                }
                else
                {
                    ElementwiseMultiplyAccumulate(function, transformedFilterBlock, transformedInputBlock, windowSize * windowSize, filterBlockDepth, filterChannelBlockDepth, transformedOutputBlock);
                }
            }
            else
            {
                function.For(tileRange.outputChannels.begin, tileRange.outputChannels.end, [tileRange, numFilterChannels, windowSize, transformedFilters, transformedInputBlock, transformedOutputBlock](emitters::IRFunctionEmitter& function, auto filterIndex) {
                    const int filterStride = numFilterChannels * windowSize * windowSize;
                    const int filterChannelStride = windowSize * windowSize;

                    auto filterChannelStart = tileRange.filterChannels.begin;
                    auto filterChannelPtr = function.PointerOffset(transformedFilters, filterIndex * filterStride + filterChannelStart * filterChannelStride);
                    ElementwiseMultiply(function, filterChannelPtr, transformedInputBlock, windowSize * windowSize, transformedOutputBlock);
                });
            }

            // Now compute output tile Y = At * X * A, a (tileSize x tileSize x tileRange.outputChannels.size) tensor
            TransformOutputBlock<ValueType>(function, transformedOutputBlock, tileSize, filterSize, outputChannelBlockDepth, outputTile);

            // TODO: if we're not separable, we want to reduce first and accumulate a single output channel (== filter index)
            AccumulateOutputTile<ValueType>(function, outputTile, tileRange.GetOutputBlockRange(), tileSize, output, outputLayout);
        }

        //
        // Core algorithm parts
        //
        template <typename ValueType>
        void TransformInput(emitters::IRFunctionEmitter& function,
                            emitters::IRLocalArray input,
                            const model::PortMemoryLayout& inputLayout,
                            int tileSize,
                            int filterSize,
                            int blockSize,
                            emitters::IRLocalArray transformedInput)
        {
#ifdef PROFILE_REGIONS
            auto region = emitters::IRProfileRegionBlock(function, "Winograd_TF_TransformInput");
            UNUSED(region);
#endif

            const int windowSize = tileSize + filterSize - 1;
            const auto windowPadding = function.LocalScalar(filterSize - 1); // This is just the amount by which "windows" (== input tiles) are bigger than output tiles

            // scratch space for conversion
            const auto valueType = emitters::GetVariableType<ValueType>();
            auto inputBlock = function.LocalArray(function.Variable(valueType, windowSize * windowSize * blockSize));
            auto transformedInputBlock = function.LocalArray(function.Variable(valueType, windowSize * windowSize * blockSize));
            const auto numOutputRows = inputLayout.GetLogicalDimensionActiveSize(0);
            const auto numOutputColumns = inputLayout.GetLogicalDimensionActiveSize(1);
            const auto numChannels = inputLayout.GetLogicalDimensionActiveSize(2);
            ;

            auto loopRanges = std::vector<emitters::IRFunctionEmitter::ConstTiledLoopRange>{ { 0, numOutputRows, tileSize },
                                                                                             { 0, numOutputColumns, tileSize },
                                                                                             { 0, numChannels, blockSize } };
            function.For(loopRanges, [=](emitters::IRFunctionEmitter& function, auto loopRanges) {
                BlockRange windowRowRange{ loopRanges[0].begin, AddAndSimplify(loopRanges[0].end, windowPadding), AddAndSimplify(loopRanges[0].size, windowPadding), loopRanges[0].index };
                BlockRange windowColumnRange{ loopRanges[1].begin, AddAndSimplify(loopRanges[1].end, windowPadding), AddAndSimplify(loopRanges[1].size, windowPadding), loopRanges[1].index };

                ProcessInputBlock<ValueType>(function,
                                             input,
                                             inputLayout,
                                             { windowRowRange, windowColumnRange, loopRanges[2] },
                                             tileSize,
                                             filterSize,
                                             inputBlock,
                                             transformedInputBlock,
                                             transformedInput);
            });
        }

        // Apply the (transformed) filters to the transformed input to produce the transformed output
        template <typename ValueType>
        void ComputeTransformedOutput(emitters::IRFunctionEmitter& function,
                                      emitters::LLVMValue transformedInput,
                                      emitters::LLVMValue transformedFilters,
                                      int numOutputRows,
                                      int numOutputColumns,
                                      int numChannels,
                                      int numFilters,
                                      int tileSize,
                                      int filterSize,
                                      emitters::LLVMValue transformedOutput)
        {
#ifdef PROFILE_REGIONS
            auto region = emitters::IRProfileRegionBlock(function, "Winograd_TF_ComputeTransformedOutput");
            UNUSED(region);
#endif
            // Do a matrix multiply to reduce many entries in parallel
            //
            // transformedSignal is a (windowRows*windowColumns) x (tr * tc) x (numChannels) tensor containing the entire transformed input signal
            // transformedFilters is a (windowRows*windowColumns) x (numFilters) x (numChannels) tensor
            // transformedOutput is a (windowRows*windowColumns) x (tr * tc) x (numFilters) tensor containing the entire transformed output signal

            const auto windowSize = filterSize + tileSize - 1;
            const auto numTileRows = ((numOutputRows - 1) / tileSize) + 1;
            const auto numTileColumns = ((numOutputColumns - 1) / tileSize) + 1;

            // These strides are the distance between spatially-adjacent window entries in the various data structures
            int numOutputTiles = numTileRows * numTileColumns;
            int transformedInputStride = numOutputTiles * numChannels;
            int transformedFiltersStride = numFilters * numChannels;
            int transformedOutputStride = numOutputTiles * numFilters;

            // Each window pixel position has a separate matrix of values to transform via a matrix multiply
            for (int windowPosition = 0; windowPosition < windowSize * windowSize; ++windowPosition)
            {
                // Compute the offsets to the particular (wr, wc) matrix we want
                auto transformedInputMatrix = function.PointerOffset(transformedInput, windowPosition * transformedInputStride);
                auto transformedFiltersMatrix = function.PointerOffset(transformedFilters, windowPosition * transformedFiltersStride);
                auto transformedOutputMatrix = function.PointerOffset(transformedOutput, windowPosition * transformedOutputStride);

                // filter: m x k, input: k x n, output: m x n
                // transformedOutput = transformedFilter * transformedInput
                const int m = numOutputTiles;
                const int n = numFilters;
                const int k = numChannels;
                const int lda = numChannels;
                const int ldb = numChannels;
                const int ldc = numFilters;

                // Now do a matrix multiply to reduce many entries in parallel
                function.CallGEMM<ValueType>(false, true, m, n, k, transformedInputMatrix, lda, transformedFiltersMatrix, ldb, transformedOutputMatrix, ldc);
            }
        }

        template <typename ValueType>
        void TransformOutput(emitters::IRFunctionEmitter& function,
                             emitters::IRLocalArray transformedOutput,
                             int tileSize,
                             int filterSize,
                             int blockSize,
                             emitters::IRLocalArray output,
                             const model::PortMemoryLayout& outputLayout)
        {
#ifdef PROFILE_REGIONS
            auto region = emitters::IRProfileRegionBlock(function, "Winograd_TF_TransformOutput");
            UNUSED(region);
#endif

            const int windowSize = tileSize + filterSize - 1;
            const auto numOutputRows = outputLayout.GetLogicalDimensionActiveSize(0);
            const auto numOutputColumns = outputLayout.GetLogicalDimensionActiveSize(1);
            const auto numFilters = outputLayout.GetLogicalDimensionActiveSize(2);

            const auto valueType = emitters::GetVariableType<ValueType>();
            auto transformedOutputBlock = function.LocalArray(function.Variable(valueType, windowSize * windowSize * blockSize));
            auto outputTile = function.LocalArray(function.Variable(valueType, tileSize * tileSize * blockSize));

            auto loopRanges = std::vector<emitters::IRFunctionEmitter::ConstTiledLoopRange>{ { 0, numFilters, blockSize },
                                                                                             { 0, numOutputRows, tileSize },
                                                                                             { 0, numOutputColumns, tileSize } };
            function.For(loopRanges, [=](emitters::IRFunctionEmitter& function, auto loopRanges) {
                auto filterIndex = loopRanges[0].begin;
                auto rowTileIndex = loopRanges[1].index;
                auto columnTileIndex = loopRanges[2].index;
                auto thisBlockSize = loopRanges[0].size.template GetIntValue<int>();

                ProcessOutputBlock<ValueType>(function,
                                              transformedOutput,
                                              rowTileIndex,
                                              columnTileIndex,
                                              filterIndex,
                                              tileSize,
                                              filterSize,
                                              thisBlockSize,
                                              transformedOutputBlock,
                                              outputTile,
                                              output,
                                              outputLayout);
            });
        }
    } // end anonymous namespace

    //
    // WinogradConvolutionNode
    //

    template <typename ValueType>
    WinogradConvolutionNode<ValueType>::WinogradConvolutionNode() :
        CompilableNode({ &_input }, { &_output }),
        _input(this, {}, defaultInputPortName),
        _output(this, defaultOutputPortName, 0)
    {
    }

    template <typename ValueType>
    WinogradConvolutionNode<ValueType>::WinogradConvolutionNode(const WinogradConvolutionNode<ValueType>& other, const model::OutputPort<ValueType>& input) :
        CompilableNode({ &_input }, { &_output }),
        _input(this, input, defaultInputPortName),
        _output(this, defaultOutputPortName, other.GetOutputMemoryLayout()),
        _inputMemoryLayout(other._inputMemoryLayout),
        _filterWeights(other._filterWeights),
        _stride(other._stride),
        _tileSize(other._tileSize),
        _filterSize(other._filterSize),
        _order(other._order)
    {
    }

    template <typename ValueType>
    WinogradConvolutionNode<ValueType>::WinogradConvolutionNode(const model::OutputPort<ValueType>& input,
                                                                const model::PortMemoryLayout& inputMemoryLayout,
                                                                const model::PortMemoryLayout& outputMemoryLayout,
                                                                const ConstTensorReferenceType& filterWeights,
                                                                int stride) :
        CompilableNode({ &_input }, { &_output }),
        _input(this, input, defaultInputPortName),
        _output(this, defaultOutputPortName, outputMemoryLayout),
        _inputMemoryLayout(inputMemoryLayout),
        _stride(stride)
    {
        using FilterOrder = typename WinogradConvolutionNode<ValueType>::FilterOrder;

        _tileSize = 2;
        const int numFilters = outputMemoryLayout.GetLogicalDimensionActiveSize(2);
        const int numFilterChannels = static_cast<int>(filterWeights.NumChannels());
        const int filtersFirstThreshold = 4; // empirically determined
        _order = (numFilterChannels <= filtersFirstThreshold) ? FilterOrder::filtersFirst : FilterOrder::tilesFirst;

        _filterSize = filterWeights.NumColumns();
        if (filterWeights.NumRows() != static_cast<size_t>(_filterSize * numFilters))
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "WinogradConvolutionComputeNode filterWeights.NumRows() != _filterSize * numFilters");
        }
        _filterWeights = dsp::GetTransformedFilters(filterWeights, numFilters, _tileSize, _order);
    }

    template <typename ValueType>
    WinogradConvolutionNode<ValueType>::WinogradConvolutionNode(const model::OutputPort<ValueType>& input,
                                                                const model::PortMemoryLayout& inputMemoryLayout,
                                                                const model::PortMemoryLayout& outputMemoryLayout,
                                                                const ConstTensorReferenceType& filterWeights,
                                                                int stride,
                                                                int tileSize,
                                                                FilterOrder order) :
        CompilableNode({ &_input }, { &_output }),
        _input(this, input, defaultInputPortName),
        _output(this, defaultOutputPortName, outputMemoryLayout),
        _inputMemoryLayout(inputMemoryLayout),
        _stride(stride),
        _tileSize(tileSize),
        _order(order)
    {
        const int numFilters = outputMemoryLayout.GetLogicalDimensionActiveSize(2);
        _filterSize = filterWeights.NumColumns();
        if (filterWeights.NumRows() != static_cast<size_t>(_filterSize * numFilters))
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "WinogradConvolutionComputeNode filterWeights.NumRows() != _filterSize * numFilters");
        }
        _filterWeights = dsp::GetTransformedFilters(filterWeights, numFilters, _tileSize, _order);
    }

    template <typename ValueType>
    void WinogradConvolutionNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInput = transformer.GetCorrespondingInputs(_input);
        auto newNode = transformer.AddNode<WinogradConvolutionNode<ValueType>>(*this, newInput);
        transformer.MapNodeOutput(this->output, newNode->output);
    }

    template <typename ValueType>
    bool WinogradConvolutionNode<ValueType>::Refine(model::ModelTransformer& transformer) const
    {
        const int windowSize = _tileSize + _filterSize - 1;
        const auto* newInput = &transformer.GetCorrespondingInputs(this->input);

        const auto& weightsMatrix = _filterWeights.ReferenceAsMatrix();
        const auto weightsValues = weightsMatrix.ToArray();
        const int numFilters = GetOutputMemoryLayout().GetLogicalDimensionActiveSize(2);

        model::MemoryShape weightsShape;
        switch (_order)
        {
        case FilterOrder::tilesFirst: {
            // 'tilesFirst': (windowRows * windowColumns) x (numFilters) x (numChannels)
            auto numFilterChannels = static_cast<int>(_filterWeights.NumChannels());
            weightsShape = model::MemoryShape({ windowSize, windowSize, numFilters, numFilterChannels });
        }
        break;
        case FilterOrder::filtersFirst: {
            // 'filtersFirst': (numFilters) x (numChannels) x (windowRows * windowColumns)
            auto numFilterChannels = static_cast<int>(_filterWeights.NumColumns());
            weightsShape = model::MemoryShape({ numFilters, numFilterChannels, windowSize, windowSize });
        }
        break;
        default:
            throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState, "WinogradConvolutionNode: illegal value for _order");
        }

        const auto& weights = Constant(transformer, weightsValues, weightsShape);
        const auto numFilterChannels = _order == FilterOrder::tilesFirst ? _filterWeights.NumChannels() : _filterWeights.NumColumns();
        auto convInputLayout = _inputMemoryLayout;
        if (numFilterChannels == 1 && _order == FilterOrder::filtersFirst)
        {
            // add a ReorderDataNode to convert to channel-major, which is more efficient in this case
            auto orderArr = utilities::ChannelMajorTensorOrder;
            const auto& reorderedData = ReorderDataWithCodeNode(*newInput, convInputLayout, convInputLayout, utilities::DimensionOrder{ orderArr });
            newInput = &reorderedData;
            convInputLayout = reorderedData.GetMemoryLayout();
        }

        auto convNode = transformer.AddNode<WinogradConvolutionComputeNode<ValueType>>(*newInput, weights, convInputLayout, GetOutputMemoryLayout(), _stride, _tileSize, _filterSize, _order, static_cast<int>(numFilterChannels));
        convNode->GetMetadata() = GetMetadata();
        transformer.MapNodeOutput(this->output, convNode->output);
        return true;
    }

    template <typename ValueType>
    void WinogradConvolutionNode<ValueType>::Compute() const
    {
        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
    }

    template <typename ValueType>
    void WinogradConvolutionNode<ValueType>::WriteToArchive(utilities::Archiver& archiver) const
    {
        model::CompilableNode::WriteToArchive(archiver);
        archiver[defaultInputPortName] << _input;
        archiver["inputLayout"] << _inputMemoryLayout;
        archiver["outputLayout"] << GetOutputMemoryLayout();
        archiver["tileSize"] << _tileSize;
        archiver["filterSize"] << _filterSize;
        archiver["stride"] << _stride;
        archiver["order"] << to_string(_order);
        math::TensorArchiver::Write(_filterWeights, "weights", archiver);
    }

    template <typename ValueType>
    void WinogradConvolutionNode<ValueType>::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        model::CompilableNode::ReadFromArchive(archiver);
        archiver[defaultInputPortName] >> _input;
        archiver["inputLayout"] >> _inputMemoryLayout;
        model::PortMemoryLayout outputMemoryLayout;
        archiver["outputLayout"] >> outputMemoryLayout;
        _output.SetMemoryLayout(outputMemoryLayout);
        archiver["tileSize"] >> _tileSize;
        archiver["filterSize"] >> _filterSize;
        archiver["stride"] >> _stride;
        std::string orderName;
        archiver["order"] >> orderName;
        _order = filter_order_from_string(orderName);
        math::TensorArchiver::Read(_filterWeights, "weights", archiver);
    }

    //
    // WinogradConvolutionComputeNode
    //

    template <typename ValueType>
    WinogradConvolutionComputeNode<ValueType>::WinogradConvolutionComputeNode() :
        CompilableNode({ &_input, &_filterWeights }, { &_output }),
        _input(this, {}, defaultInputPortName),
        _filterWeights(this, {}, filterWeightsPortName),
        _output(this, defaultOutputPortName, 0)
    {
    }

    template <typename ValueType>
    WinogradConvolutionComputeNode<ValueType>::WinogradConvolutionComputeNode(const model::OutputPort<ValueType>& input,
                                                                              const model::OutputPort<ValueType>& filterWeights,
                                                                              const model::PortMemoryLayout& inputMemoryLayout,
                                                                              const model::PortMemoryLayout& outputMemoryLayout,
                                                                              int stride,
                                                                              int tileSize,
                                                                              int filterSize,
                                                                              FilterOrder order,
                                                                              int numFilterChannels) :
        CompilableNode({ &_input, &_filterWeights }, { &_output }),
        _input(this, input, defaultInputPortName),
        _filterWeights(this, filterWeights, filterWeightsPortName),
        _output(this, defaultOutputPortName, outputMemoryLayout),
        _inputMemoryLayout(inputMemoryLayout),
        _stride(stride),
        _tileSize(tileSize),
        _filterSize(filterSize),
        _order(order),
        _numFilterChannels(numFilterChannels)
    {
        const auto numChannels = inputMemoryLayout.GetLogicalDimensionActiveSize(2);
        const int numFilters = outputMemoryLayout.GetLogicalDimensionActiveSize(2);
        _inputBlockSize = std::min(64, numChannels);
        _outputBlockSize = std::min(64, numFilters);
    }

    template <typename ValueType>
    WinogradConvolutionComputeNode<ValueType>::WinogradConvolutionComputeNode(const WinogradConvolutionComputeNode<ValueType>& other,
                                                                              const model::OutputPort<ValueType>& input,
                                                                              const model::OutputPort<ValueType>& filterWeights) :
        CompilableNode({ &_input, &_filterWeights }, { &_output }),
        _input(this, input, defaultInputPortName),
        _filterWeights(this, filterWeights, filterWeightsPortName),
        _output(this, defaultOutputPortName, other.GetOutputMemoryLayout()),
        _inputMemoryLayout(other._inputMemoryLayout),
        _stride(other._stride),
        _tileSize(other._tileSize),
        _filterSize(other._filterSize),
        _order(other._order),
        _numFilterChannels(other._numFilterChannels),
        _inputBlockSize(other._inputBlockSize),
        _outputBlockSize(other._outputBlockSize)
    {
    }

    template <typename ValueType>
    void WinogradConvolutionComputeNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInput = transformer.GetCorrespondingInputs(_input);
        const auto& newFilterWeights = transformer.GetCorrespondingInputs(_filterWeights);
        auto newNode = transformer.AddNode<WinogradConvolutionComputeNode<ValueType>>(*this, newInput, newFilterWeights);
        transformer.MapNodeOutput(this->output, newNode->output);
    }

    template <typename ValueType>
    void WinogradConvolutionComputeNode<ValueType>::Compute() const
    {
        throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
    }

    template <typename ValueType>
    void WinogradConvolutionComputeNode<ValueType>::Compile(model::IRMapCompiler& compiler, emitters::IRFunctionEmitter& function)
    {
        auto input = function.LocalArray(compiler.EnsurePortEmitted(this->input));
        auto transformedFilters = function.LocalArray(compiler.EnsurePortEmitted(this->filterWeights));
        auto output = function.LocalArray(compiler.EnsurePortEmitted(this->output));

        switch (_order)
        {
        case FilterOrder::filtersFirst:
            CompileFiltersFirst(compiler, function, input, transformedFilters, output);
            break;
        case FilterOrder::tilesFirst:
            CompileTilesFirst(compiler, function, input, transformedFilters, output);
            break;
        default:
            throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState, "WinogradConvolutionComputeNode has illegal order");
        }
    }

    template <typename ValueType>
    void WinogradConvolutionComputeNode<ValueType>::CompileTilesFirst(model::IRMapCompiler& compiler,
                                                                      emitters::IRFunctionEmitter& function,
                                                                      emitters::IRLocalArray input,
                                                                      emitters::IRLocalArray transformedFilters,
                                                                      emitters::IRLocalArray output)
    {
        auto& module = function.GetModule();

        // Input data parameters
        const auto inputLayout = this->GetInputMemoryLayout();
        const auto numChannels = inputLayout.GetLogicalDimensionActiveSize(2);

        if (inputLayout.GetLogicalDimensionOffset(0) != _filterSize / 2)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Padding must be filterSize/2");
        }

        // Output data parameters
        const auto outputLayout = this->GetOutputMemoryLayout();
        const int numOutputRows = outputLayout.GetLogicalDimensionActiveSize(0);
        const int numOutputColumns = outputLayout.GetLogicalDimensionActiveSize(1);
        const int numFilters = outputLayout.GetLogicalDimensionActiveSize(2);

        // Winograd-specific stuff
        const auto windowSize = _filterSize + _tileSize - 1;
        const auto numTileRows = (numOutputRows + _tileSize - 1) / _tileSize;
        const auto numTileColumns = (numOutputColumns + _tileSize - 1) / _tileSize;

        // Allocate scratch space to hold transformed input and output
        int transformedInputSize = windowSize * windowSize * numTileRows * numTileColumns * numChannels;
        int transformedOutputSize = windowSize * windowSize * numTileRows * numTileColumns * numFilters;
        auto transformedInput = function.LocalArray(module.GlobalArray<ValueType>(GetInternalStateIdentifier() + "_transformedInput", transformedInputSize));
        auto transformedOutput = function.LocalArray(module.GlobalArray<ValueType>(GetInternalStateIdentifier() + "_transformedOutput", transformedOutputSize));

        // transformedInput is (windowSize*windowSize) x (tileRows * tileColumns) x numChannels
        // transformedFilters is (windowSize*windowSize) x numFilters x numChannels
        // transformedOutput is (windowSize*windowSize) x (tileRows * tileColumns) x numFilters

        // TODO: eventually, pass this in to ComputeTransformedOutput, instead of just assuming a layout there
        const model::PortMemoryLayout transformedFilterLayout(model::MemoryShape{ windowSize, windowSize, numFilters, numChannels });
        UNUSED(transformedFilterLayout);

        // Clear output buffer
        function.StoreZero(output, outputLayout.NumElements());

        // This is the core of the Winograd convolution algorithm: transform the input, perform an elementwise multiply between it an the transformed filter, and transform it back
        TransformInput<ValueType>(function, input, inputLayout, _tileSize, _filterSize, _inputBlockSize, transformedInput);
        ComputeTransformedOutput<ValueType>(function, transformedInput, transformedFilters, numOutputRows, numOutputColumns, numChannels, numFilters, _tileSize, _filterSize, transformedOutput);
        TransformOutput<ValueType>(function, transformedOutput, _tileSize, _filterSize, _outputBlockSize, output, outputLayout);
    }

    template <typename ValueType>
    void WinogradConvolutionComputeNode<ValueType>::CompileFiltersFirst(model::IRMapCompiler& compiler,
                                                                        emitters::IRFunctionEmitter& function,
                                                                        emitters::IRLocalArray input,
                                                                        emitters::IRLocalArray transformedFilters,
                                                                        emitters::IRLocalArray output)
    {
        const auto windowSize = _filterSize + _tileSize - 1;

        // Input data parameters
        const auto inputLayout = this->GetInputMemoryLayout();
        const auto numChannels = inputLayout.GetLogicalDimensionActiveSize(2);

        if (inputLayout.GetLogicalDimensionOffset(0) != _filterSize / 2)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "WinogradConvolutionComputeNode: padding must be filterSize/2");
        }

        // Filter data parameters
        const int numFilterChannels = _numFilterChannels;

        // Output data parameters
        const auto outputLayout = this->GetOutputMemoryLayout();
        const int numOutputRows = outputLayout.GetLogicalDimensionActiveSize(0);
        const int numOutputColumns = outputLayout.GetLogicalDimensionActiveSize(1);
        const int numFilters = outputLayout.GetLogicalDimensionActiveSize(2);

        const model::PortMemoryLayout transformedFilterLayout(model::MemoryShape{ numFilters, numFilterChannels, windowSize, windowSize });

        // When blockSize > 1, the inner loop reads in a windowSize x windowSize x blockSize block of
        // input image data, transforms it, multiplies it, post-transforms it, and writes to output image tile
        // All this can happen inside ConvolveAccumulateBlock()

        const bool isSeparable = (numFilterChannels == 1 && numFilters == numChannels);
        if (!isSeparable && numFilterChannels != numChannels)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "WinogradConvolutionComputeNode: filters must be depthwise-separable or full-channel");
        }

        const int defaultSeparableBlockDepth = 8;
        const int defaultNonseparableFilterBlockDepth = 2;
        const int defaultNonseparableChannelDepth = 4;
        auto separableBlockDepth = compiler.GetModelOptimizerOptions(*this).template GetEntry<int>("separableBlockDepth", defaultSeparableBlockDepth);
        auto nonseparableFilterBlockDepth = compiler.GetModelOptimizerOptions(*this).template GetEntry<int>("nonseparableFilterBlockDepth", defaultNonseparableFilterBlockDepth);
        auto nonseparableChannelDepth = compiler.GetModelOptimizerOptions(*this).template GetEntry<int>("nonseparableChannelDepth", defaultNonseparableChannelDepth);

        const int maxFilterChannelBlockDepth = isSeparable ? 1 : nonseparableChannelDepth;
        const int maxFilterBlockDepth = isSeparable ? separableBlockDepth : nonseparableFilterBlockDepth;
        const int maxInputBlockDepth = isSeparable ? maxFilterBlockDepth : maxFilterChannelBlockDepth;
        const int maxOutputBlockDepth = maxFilterBlockDepth;

        // Temporaries
        // TODO: allocate a set of temporaries per thread and parallelize the big outer loop
        WinogradScratchStorage scratch;
        const auto valueType = emitters::GetVariableType<ValueType>();
        const auto inputBlockElements = windowSize * windowSize * maxInputBlockDepth;
        const auto transformedInputBlockElements = windowSize * windowSize * maxInputBlockDepth;
        const auto transformedFilterBlockElements = windowSize * windowSize * maxFilterBlockDepth * maxFilterChannelBlockDepth;
        const auto transformedOutputBlockElements = windowSize * windowSize * maxOutputBlockDepth;
        const auto outputTileElements = _tileSize * _tileSize * maxOutputBlockDepth;

        // It turns out we can alias inputBlock with transformedOutputBlock and transformedInputBlock with outputTile
        scratch.inputBlock = function.Variable(valueType, std::max(inputBlockElements, transformedOutputBlockElements));
        scratch.transformedInputBlock = function.Variable(valueType, std::max(transformedInputBlockElements, outputTileElements));
        scratch.transformedFilterBlock = function.Variable(valueType, transformedFilterBlockElements); // Not aliased with anything
        scratch.transformedOutputBlock = scratch.inputBlock;
        scratch.outputTile = scratch.transformedInputBlock;

        ConstConvolutionSize problemSize = { numOutputRows, numOutputColumns, numChannels, numFilterChannels, numFilters };
        auto windowPadding = function.LocalScalar(windowSize - _tileSize); // This is just the amount by which "windows" (== input tiles) are bigger than output tiles

        // Clear output buffer
        function.StoreZero(output, outputLayout.NumElements());
        std::vector<emitters::IRFunctionEmitter::ConstTiledLoopRange> outerLoopRanges = { { 0, numFilters, maxFilterBlockDepth },
                                                                                          { 0, numFilterChannels, maxFilterChannelBlockDepth } };
        function.For(outerLoopRanges, [problemSize, numChannels, numFilterChannels, isSeparable, windowPadding, scratch, input, inputLayout, transformedFilters, transformedFilterLayout, output, outputLayout, this](emitters::IRFunctionEmitter& function, std::vector<emitters::IRFunctionEmitter::BlockInterval> loopRanges) {
            auto filterRange = loopRanges[0];
            auto filterChannelRange = loopRanges[1];
            int filterBlockDepth = filterRange.size.template GetIntValue<int>();
            int filterChannelBlockDepth = filterChannelRange.size.template GetIntValue<int>();
            const auto useFilterBlock = (filterBlockDepth > 1 || filterChannelBlockDepth > 1);
            if (useFilterBlock)
            {
                LoadFilterBlock<ValueType>(function, transformedFilters, transformedFilterLayout, filterRange, filterChannelRange, this->_tileSize, this->_filterSize, scratch.transformedFilterBlock);
            }

            std::vector<emitters::IRFunctionEmitter::ConstTiledLoopRange> innerLoopRanges = { { 0, problemSize.rows, _tileSize },
                                                                                              { 0, problemSize.columns, _tileSize } };
            function.For(innerLoopRanges, [filterRange, filterChannelRange, problemSize, numChannels, numFilterChannels, isSeparable, windowPadding, scratch, input, inputLayout, transformedFilters, transformedFilterLayout, output, outputLayout, this](emitters::IRFunctionEmitter& function, std::vector<emitters::IRFunctionEmitter::BlockInterval> loopRanges) {
                auto tileRowRange = loopRanges[0];
                auto tileColumnRange = loopRanges[1];

                auto channelStart = (filterRange.begin * numFilterChannels) % numChannels;
                // TODO: See about removing the "% numChannels" if we can know that it's unnecessary at compile-time (e.g., if numFilterChannels == N*numChannels) for N > 0
                // TODO: assert (channelStart + filterRange.size * numFilterChannels) < numChannels) --- make sure it doesn't wrap around while processing a block
                // TODO: add a comment describing the logic behind setting channelIndex (it allows us to unify depthwise-separabale and non-separable logic)
                auto inputChannelBegin = channelStart + filterChannelRange.begin; // deal with depthwise-separable filters
                auto inputChannelRangeSize = isSeparable ? filterRange.size : filterChannelRange.size;
                auto inputChannelRangeBlockIndex = isSeparable ? filterRange.index : filterChannelRange.index;
                BlockRange windowRowRange{ tileRowRange.begin, AddAndSimplify(tileRowRange.end, windowPadding), AddAndSimplify(tileRowRange.size, windowPadding), tileRowRange.index };
                BlockRange windowColumnRange{ tileColumnRange.begin, AddAndSimplify(tileColumnRange.end, windowPadding), AddAndSimplify(tileColumnRange.size, windowPadding), tileColumnRange.index };
                BlockRange inputChannelRange = { inputChannelBegin, inputChannelBegin + inputChannelRangeSize, inputChannelRangeSize, inputChannelRangeBlockIndex };
                BlockRange outputChannelRange = filterRange;
                ConvolutionBlockRanges ranges{ windowRowRange, windowColumnRange, inputChannelRange, filterRange, filterChannelRange, tileRowRange, tileColumnRange, outputChannelRange }; // filterRange == output channel range

                ConvolveAccumulateBlock<ValueType>(function, input, inputLayout, transformedFilters, transformedFilterLayout, ranges, problemSize, _tileSize, _filterSize, scratch, output, outputLayout);
            });
        });
    }

    template <typename ValueType>
    void WinogradConvolutionComputeNode<ValueType>::WriteToArchive(utilities::Archiver& archiver) const
    {
        model::CompilableNode::WriteToArchive(archiver);
        archiver[defaultInputPortName] << _input;
        archiver["weights"] << _filterWeights;
        archiver["inputLayout"] << _inputMemoryLayout;
        archiver["outputLayout"] << GetOutputMemoryLayout();
        archiver["tileSize"] << _tileSize;
        archiver["filterSize"] << _filterSize;
        archiver["stride"] << _stride;
        archiver["order"] << to_string(_order);
        archiver["filterChannels"] << _numFilterChannels;
    }

    template <typename ValueType>
    void WinogradConvolutionComputeNode<ValueType>::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        model::CompilableNode::ReadFromArchive(archiver);
        archiver[defaultInputPortName] >> _input;
        archiver["weights"] >> _filterWeights;
        archiver["inputLayout"] >> _inputMemoryLayout;
        model::PortMemoryLayout outputMemoryLayout;
        archiver["outputLayout"] >> outputMemoryLayout;
        _output.SetMemoryLayout(outputMemoryLayout);
        archiver["tileSize"] >> _tileSize;
        archiver["filterSize"] >> _filterSize;
        archiver["stride"] >> _stride;
        std::string orderName;
        archiver["order"] >> orderName;
        _order = filter_order_from_string(orderName);
        archiver["filterChannels"] >> _numFilterChannels;
    }

    // Explicit specializations
    template class WinogradConvolutionNode<float>;
    template class WinogradConvolutionNode<double>;
    template class WinogradConvolutionComputeNode<float>;
    template class WinogradConvolutionComputeNode<double>;
} // namespace nodes
} // namespace ell
