////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CachingStrategies.cpp (value)
//  Authors:  Kern Handa, Mason Remy
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <Array.h>
#include <CachingStrategies.h>
#include <LoopNests.h>
#include <Matrix.h>
#include <Reference.h>
#include <Scalar.h>
#include <Tensor.h>
#include <loopnests/Kernel.h>
#include <loopnests/LoopNest.h>

#include <utilities/include/Exception.h>
#include <utilities/include/TypeAliases.h>
#include <utilities/include/Unused.h>

#include "ComputeContext.h"
#include "CppEmitterContext.h"
#include "LLVMContext.h"

#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/Target/TargetMachine.h>

#include <cmath>
#include <queue>
#include <set>

#if 1 // DEBUGGING
#include <iostream>
#endif

namespace ell
{
namespace value
{
    using namespace utilities;

    // TODO : Generalize to machine characteristics and move out of CachingStrategies
    struct RegisterCharacteristics
    {
        unsigned NumberOfSIMDRegisters;
        unsigned NumberOfElementsPerSIMDRegister;
    };

    template <typename ValueType>
    RegisterCharacteristics GetRegisterCharacteristics()
    {
        RegisterCharacteristics characteristics;
        // Set some defaults for non-LLVMContext
        characteristics.NumberOfSIMDRegisters = 8;
        characteristics.NumberOfElementsPerSIMDRegister = 4;
        InvokeForContext<LLVMContext>([&](LLVMContext& context) {
            auto targetMachine = context.GetModuleEmitter().GetTargetMachine();
            auto fn = context.GetFunctionEmitter().GetFunction();
            auto info = targetMachine->getTargetTransformInfo(*fn);
            // See https://llvm.org/doxygen/classllvm_1_1TargetTransformInfo.html for the big list of amazing things you can get from this TargetMachineInfo object
            characteristics.NumberOfSIMDRegisters = info.getNumberOfRegisters(true);
            auto SIMDRegisterBitWidth = info.getRegisterBitWidth(true);

            auto bytesPerElement = context.GetModuleEmitter().GetIREmitter().SizeOf<ValueType>();
            auto bitsPerElement = 8 * bytesPerElement;
            characteristics.NumberOfElementsPerSIMDRegister = SIMDRegisterBitWidth / bitsPerElement;
        });
        return characteristics;
    }

    RegisterCharacteristics GetRegisterCharacteristics(ValueType type)
    {
        switch (type)
        {
        case ValueType::Void:
            return GetRegisterCharacteristics<void>();
            break;
        case ValueType::Boolean:
            return GetRegisterCharacteristics<bool>();
            break;
        case ValueType::Char8:
            return GetRegisterCharacteristics<char>();
            break;
        case ValueType::Byte:
            return GetRegisterCharacteristics<uint8_t>();
            break;
        case ValueType::Int16:
            return GetRegisterCharacteristics<short>();
            break;
        case ValueType::Int32:
            return GetRegisterCharacteristics<int>();
            break;
        case ValueType::Int64:
            return GetRegisterCharacteristics<int64_t>();
            break;
        case ValueType::Float:
            return GetRegisterCharacteristics<float>();
            break;
        case ValueType::Double:
            return GetRegisterCharacteristics<double>();
            break;
        default:
            throw InputException(InputExceptionErrors::invalidArgument, "Unrecognized or unsupported ValueType");
        }
    }

    void CopyReduce(Scalar baseValue, Scalar cacheValue)
    {
        baseValue = cacheValue;
    }

    void SumReduce(Scalar baseValue, Scalar cacheValue)
    {
        baseValue += cacheValue;
    }

    // Makes a vector of all integers that are a power of base that are strictly less than N, ordered in decreasing value
    std::vector<int> GetTelescopingSizes(int N, int base = 2)
    {
        int maxPower = std::log2(N);
        if (std::pow(base, maxPower) == N)
        {
            // If N is already a power of base, dont add it to the vector
            maxPower--;
        }
        std::vector<int> result;
        result.reserve(maxPower);
        for (int power = maxPower; power >= 0; --power)
        {
            result.push_back(static_cast<int>(std::pow(base, power)));
        }
        return result;
    }

    int RoundUpToMultiple(int input, int factor)
    {
        int remainder = input % factor;
        return remainder > 0 ? input + (factor - remainder) : input;
    }

    static inline void ValidateInputDimensionality(const Value& value, const MemoryShape& cacheSize, const DimensionOrder& order)
    {
        if (cacheSize.NumDimensions() != value.GetLayout().NumDimensions())
        {
            throw LogicException(LogicExceptionErrors::illegalState, "Dimensionality of data-to-be-cached must match shape of requested cache size");
        }
        if (cacheSize.NumDimensions() != order.NumDimensions())
        {
            throw LogicException(LogicExceptionErrors::illegalState, "Dimensionality of dimension order must match shape of requested cache size");
        }

        if (value.GetLayout().NumDimensions() != 2)
        {
            throw LogicException(LogicExceptionErrors::notImplemented, "Only matrix source data is supported at this time");
        }
    }

    // TODO move to Array slice code and generalize
    Array SliceArray4_1(Array array, Scalar firstIndex)
    {
        Value indexedValue = array.GetValue().Offset({ firstIndex, 0, 0, 0 });
        auto currentLayout = array.GetValue().GetLayout();

        indexedValue.SetLayout(currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(0)));

        return indexedValue;
    }

    Array SliceArray4_1_offset(Array array, Scalar firstIndex)
    {
        auto currentLayout = array.GetValue().GetLayout();
        auto memoryOffsets = currentLayout.GetOffset();
        const auto memoryOrder = currentLayout.GetLogicalDimensionOrder();

        // TODO : replace memory offsets with absolute offset support
        Value indexedValue = array.GetValue().Offset({ firstIndex - memoryOffsets[memoryOrder[0]],
                                                       0 - memoryOffsets[memoryOrder[1]],
                                                       0 - memoryOffsets[memoryOrder[2]],
                                                       0 - memoryOffsets[memoryOrder[3]] });
        indexedValue.SetLayout(currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(0)));

        return indexedValue;
    }
    Matrix SliceArray4_2(Array array, Scalar firstIndex, Scalar secondIndex)
    {
        Value indexedValue = array.GetValue().Offset({ firstIndex, secondIndex, 0, 0 });
        auto currentLayout = array.GetValue().GetLayout();

        auto newLayout = currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(0));
        newLayout = newLayout.GetSliceLayout(newLayout.GetPhysicalDimension(0));

        indexedValue.SetLayout(newLayout);
        return indexedValue;
    }
    Array SliceArray6_2(Array array, Scalar firstIndex, Scalar secondIndex)
    {
        Value indexedValue = array.GetValue().Offset({ firstIndex, secondIndex, 0, 0, 0, 0 });
        auto currentLayout = array.GetValue().GetLayout();

        auto newLayout = currentLayout.GetSliceLayout(currentLayout.GetPhysicalDimension(0));
        newLayout = newLayout.GetSliceLayout(newLayout.GetPhysicalDimension(0));

        indexedValue.SetLayout(newLayout);
        return indexedValue;
    }
    Array SliceArray6_4(Array array, Scalar firstIndex, Scalar secondIndex, Scalar thirdIndex, Scalar fourthIndex)
    {
        Value indexedValue = array.GetValue().Offset({ firstIndex, secondIndex, thirdIndex, fourthIndex, 0, 0 });
        auto newLayout = array.GetValue().GetLayout();
        newLayout = newLayout.GetSliceLayout(newLayout.GetPhysicalDimension(0));
        newLayout = newLayout.GetSliceLayout(newLayout.GetPhysicalDimension(0));
        newLayout = newLayout.GetSliceLayout(newLayout.GetPhysicalDimension(0));
        newLayout = newLayout.GetSliceLayout(newLayout.GetPhysicalDimension(0));

        indexedValue.SetLayout(newLayout);
        return indexedValue;
    }

    void CopyInputCopyOutput::HandleCachingImpl(LoopNest& nest)
    {
        // throw LogicException(LogicExceptionErrors::notImplemented);
    }

    void CopyInputNoOutput::HandleCachingImpl(LoopNest& nest)
    {
        ValidateInputDimensionality(_value, _shape, _order);

        // _shape is specified in logical dimensions, if _order is not canonical order then we need to reorder the layout
        auto canonicalLayout = MemoryLayout{ _shape };
        auto orderedLayout = canonicalLayout.ReorderedCopy(_order);

        auto cacheName = UniqueName("copyInputNoOutputCache");
        auto cacheValue = StaticAllocate(cacheName, _value.GetBaseType(), orderedLayout, AllocateFlags::ThreadLocal);
        cacheValue.SetName(cacheName);
        auto cacheRef = cacheValue.Reference();
        cacheRef.SetName(cacheName + "Ref");

        [[maybe_unused]] IntPtrT origAddress{};
        InvokeForContext<ComputeContext>([&] { origAddress = std::get<IntPtrT*>(cacheRef.GetUnderlyingData())[0]; });

        auto copyInputKernel = loopnests::Kernel(cacheName + "_Init_Kernel")
                                   .Inputs(_value, cacheRef)
                                   .Indices(_kernelIndices)
                                   .Define([origAddress, orderedLayout](value::Matrix input, value::Value cacheRef, value::Scalar i, value::Scalar j) {
                                       DEBUG_USED(origAddress);
                                       InvokeForContext<ComputeContext>([&] {
                                           [[maybe_unused]] auto addr = std::get<IntPtrT*>(cacheRef.GetUnderlyingData())[0];
                                           assert(addr == origAddress);
                                       });

                                       Matrix cacheMatrix = cacheRef.Dereference();
                                       int M = static_cast<int>(input.Rows());
                                       int N = static_cast<int>(input.Columns());
                                       Scalar cacheRows = value::Min(M - i, orderedLayout.GetLogicalDimensionActiveSize(0));
                                       Scalar cacheColumns = value::Min(N - j, orderedLayout.GetLogicalDimensionActiveSize(1));

                                       if (input.GetMatrixLayout() == Matrix::MatrixLayout::rowMajor)
                                       {
                                           ForRange(cacheRows, [&](Scalar i_inner) {
                                               ForRange(cacheColumns, [&](Scalar j_inner) {
                                                   cacheMatrix(i_inner, j_inner) = input(i + i_inner, j + j_inner);
                                               });
                                           });
                                       }
                                       else
                                       {
                                           ForRange(cacheColumns, [&](Scalar j_inner) {
                                               ForRange(cacheRows, [&](Scalar i_inner) {
                                                   cacheMatrix(i_inner, j_inner) = input(i + i_inner, j + j_inner);
                                               });
                                           });
                                       }
                                       auto offsetCacheValue = cacheMatrix.GetValue().Offset({ -1 * i, -1 * j });
                                       offsetCacheValue.SetLayout(orderedLayout);
                                       cacheRef = offsetCacheValue.Reference();
                                   });

        auto resetOffsetKernel = loopnests::Kernel(cacheName + "_Reset_Kernel")
                                     .Inputs(cacheRef)
                                     .Indices(_kernelIndices)
                                     .Define([orderedLayout](value::Value cacheRef, value::Scalar i, value::Scalar j) {
                                         Matrix cacheMatrix = cacheRef.Dereference();
                                         auto offsetCacheValue = cacheMatrix.GetValue().Offset({ i, j });
                                         offsetCacheValue.SetLayout(orderedLayout);
                                         cacheRef = offsetCacheValue.Reference();
                                     });

        auto& underlyingNest = nest.GetUnderlyingLoopNest();
        underlyingNest.AddKernel(copyInputKernel, loopnests::CodePositionConstraints{ loopnests::LoopFragmentType::prologue, _atIndices, {} });
        underlyingNest.AddKernel(resetOffsetKernel, loopnests::CodePositionConstraints{ loopnests::LoopFragmentType::epilogue, _atIndices, {} });
        underlyingNest.RenameVariable(_value, cacheRef, _atIndices, { copyInputKernel });
    }

    void ZeroInputReduceOutput::HandleCachingImpl(LoopNest& nest)
    {
        ValidateInputDimensionality(_value, _shape, _order);
        auto canonicalLayout = MemoryLayout{ _shape };
        auto orderedLayout = canonicalLayout.ReorderedCopy(_order);

        auto cacheName = UniqueName("empyInputCopyOutputCache");
        auto tempValue = StaticAllocate(cacheName, _value.GetBaseType(), orderedLayout, AllocateFlags::ThreadLocal);
        tempValue.SetName(cacheName);
        Matrix temp(tempValue);
        auto cacheRef = tempValue.Reference();
        cacheRef.SetName(cacheName + "Ref");
        [[maybe_unused]] IntPtrT origAddress{};
        InvokeForContext<ComputeContext>([&] { origAddress = std::get<IntPtrT*>(cacheRef.GetUnderlyingData())[0]; });

        auto kernel3 = loopnests::Kernel(cacheName + "_Init_Kernel")
                           .Inputs(cacheRef)
                           .Indices(_kernelIndices)
                           .Define([shape = orderedLayout](value::Value temp, value::Scalar i, value::Scalar j) {
                               Matrix tempMatrix = temp.Dereference();

                               value::For(tempMatrix, [&](value::Scalar i_inner, value::Scalar j_inner) {
                                   tempMatrix(i_inner, j_inner) = Cast(0, tempMatrix.Type());
                               });

                               // Update cacheRef so that global (i, k) index into the corect spot in the cache
                               auto cacheTmpOffset = tempMatrix.GetValue().Offset({ -1 * i, -1 * j });
                               cacheTmpOffset.SetLayout(shape);
                               temp = cacheTmpOffset.Reference();
                           });

        auto& underlyingNest = nest.GetUnderlyingLoopNest();
        underlyingNest.AddKernel(kernel3, loopnests::CodePositionConstraints{ loopnests::LoopFragmentType::prologue, _atIndices, {} });

        auto kernel2 = loopnests::Kernel(cacheName + "_Reduce_Kernel")
                           .Inputs(_value, cacheRef)
                           .Indices(_kernelIndices)
                           .Define([shape = orderedLayout](value::Matrix C, value::Value temp, value::Scalar i, value::Scalar j) {
                               auto cacheTmpOffset = temp.Dereference().Offset({ i, j });
                               cacheTmpOffset.SetLayout(shape);
                               temp = cacheTmpOffset.Reference();
                               auto cache = value::Matrix(temp.Dereference());

                               int M = static_cast<int>(C.Rows());
                               int N = static_cast<int>(C.Columns());
                               Scalar extraM = value::Min(M - i, shape.GetLogicalDimensionActiveSize(0));
                               Scalar extraN = value::Min(N - j, shape.GetLogicalDimensionActiveSize(1));

                               ForRange(extraM, [&](Scalar i_inner) {
                                   ForRange(extraN, [&](Scalar j_inner) {
                                       C(i + i_inner, j + j_inner) += cache(i_inner, j_inner);
                                   });
                               });
                           });
        underlyingNest.AddKernel(kernel2, loopnests::CodePositionConstraints{ loopnests::LoopFragmentType::epilogue, _atIndices, {} });
        underlyingNest.RenameVariable(_value, cacheRef, _atIndices, { kernel2, kernel3 });
    } // namespace value

    void BLASTCopy::HandleCachingImpl(LoopNest& nest)
    {
        /* BLAS T COPY:
        suppose input matrix is M x N, cache size is M' x N', stripeSize = 4
        so cache successive M'x4 row-major submatrices from the input matrix

         0  1  2  3 16 17 18 19      0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 ...
         4  5  6  7 20 21 22 23 ->  16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
         8  9 10 11 24 25 26 27
        12 13 14 15 28 29 30 31

        Need 2 layers of caching:
        at M x N level, build up cache values
        at stripeSize level, set up pointer and memory layout
         */

        ValidateInputDimensionality(_value, _shape, _order);

        // get block size, stripe size, and stripe slitting index from extras
        auto extraParams = std::any_cast<std::tuple<int, Index, BoundaryConditionHandling>>(_extra);
        int stripeSize;
        Index stripeSplitIndex;
        BoundaryConditionHandling boundaryHandling;
        std::tie(stripeSize, stripeSplitIndex, boundaryHandling) = extraParams;

        if (boundaryHandling == BoundaryConditionHandling::ZeroPadding && _shape[1] % stripeSize != 0)
        {
            // To avoid an odd repeated edge case, enforce that the number of cache columns is a multiple of the stripe size
            // So the base 3D cache view can represent the full cache
            throw InputException(InputExceptionErrors::invalidSize, "The number of cache columns must be a multiple of the cache stripe size");
        }

        // Cache structure:
        // Lift the 2D submatrix into a 3D array to set up the cache simply
        // The first dimension identifies which cached column block to use
        // The second two dimensions identify the element inside of that cached submatrix block
        // Index mapping: input ( i, j ) -> cache ( j / stripeSize, i, j % stripeSize )
        //                cache ( i, j, k ) -> input ( j, i * stripeSize + k )

        // Boundary handling
        // There are 4 boundary scenarios (possibly all 4 can happen in a single input matrix + cache size combination
        // while iterating over the matrix):
        //     |-------N-------|
        //     |----N'---|
        // _ _ *---------------*
        // | | |         |     |
        // | M'|    1    |  2  |
        // | | |         |     |
        // M _ |_________|_____|
        // |   |    3    |  4  |
        // |   |         |     |
        // _   *---------------*

        // 1 : The cache has exactly as many rows and columns as the input matrix chunk
        //     - This is the simple case, leave the cache as { M' x N' }
        // 2 : The cache has more columns than the input matrix but fewer rows
        //     - re-view the cache to be { M' x remainingColumns }
        // 3 : The cache has more rows than the input matrix but fewer columns
        //     - re-view the cache to be { remainingRows x N' }
        // 4 : The cache has more rows and columns than the input matrix
        //     - re-view the cache to be { remainingRows x remainingColumns }
        // Note: it is assumed that the input matrix is stepped over in splits based on the
        //     cache size given, so the cache can never be smaller than the input matrix chunk

        // Since the matrix and cache sizes are known ahead of time, we can compute all of the boundary
        // condition layouts that are needed:
        // remainingRows = M % M'
        // remainingColumns = N % N'

        auto inputMatrix = Matrix(_value);
        int inputRows = inputMatrix.Rows();
        int inputCols = inputMatrix.Columns();
        int remainingRows = inputRows % _shape[0];
        int remainingCols = inputCols % _shape[1];
        int roundedRemainingCols = RoundUpToMultiple(remainingCols, stripeSize);
        // we don't need to round up remainingRows since stripe size only applies to columns in BLASTCopy

        auto generateTCOPYCacheLayout = [stripeSize](int rows, int cols) {
            auto cacheDimOrder = DimensionOrder{ 0, 1, 2 };
            auto liftedShape = MemoryShape{ cols / stripeSize, rows, stripeSize };
            auto cacheLayout = MemoryLayout{ liftedShape, cacheDimOrder };
            return cacheLayout;
        };
        auto generateTCOPYCacheViewLayout = [stripeSize](int rows, int cols) {
            auto cacheViewLayout = MemoryLayout{ { rows, stripeSize }, RowMajorMatrixOrder };
            return cacheViewLayout;
        };

        auto baseCacheLayout = generateTCOPYCacheLayout(_shape[0], _shape[1]); // The non-boundary-case 3D lifted shape
        auto baseCacheViewLayout = generateTCOPYCacheViewLayout(_shape[0], _shape[1]);

        // "Boundary" condition 1 is the general case (i.e. non-boundary case)
        auto boundaryConditionCacheLayout1 = baseCacheLayout;
        auto cacheViewLayout1 = baseCacheViewLayout;

        // Boundary condition 2, re-view to M' x remainingColumns
        auto boundaryConditionCacheLayout2 = generateTCOPYCacheLayout(_shape[0], roundedRemainingCols);
        auto cacheViewLayout2 = generateTCOPYCacheViewLayout(_shape[0], roundedRemainingCols);

        // Boundary condition 3, re-view to remainingRows x N'
        auto boundaryConditionCacheLayout3 = generateTCOPYCacheLayout(remainingRows, _shape[1]);
        auto cacheViewLayout3 = generateTCOPYCacheViewLayout(remainingRows, _shape[1]);

        // Boundary condition 4, re-view to remainingRows x remainingColumns
        auto boundaryConditionCacheLayout4 = generateTCOPYCacheLayout(remainingRows, roundedRemainingCols);
        auto cacheViewLayout4 = generateTCOPYCacheViewLayout(remainingRows, roundedRemainingCols);

        auto cacheName = UniqueName("BLASTCopyCache");
        _rawCache = StaticAllocate(cacheName, _value.GetBaseType(), baseCacheLayout);
        Array liftedCache(_rawCache);

        auto cacheRef = _rawCache.Reference();
        cacheRef.SetLayout(baseCacheViewLayout);
        cacheRef.SetName(cacheName + "_Ref");

        auto cacheFillKernel = loopnests::Kernel(cacheName + "_Fill_Cache_Kernel")
                                   .Inputs(_value, liftedCache)
                                   .Indices(_kernelIndices)
                                   .Define([remainingRows, remainingCols, stripeSize, shape = _shape, inputRows, inputCols, boundaryConditionCacheLayout1, boundaryConditionCacheLayout2, boundaryConditionCacheLayout3, boundaryConditionCacheLayout4](value::Matrix input, value::Array cache, value::Scalar i, value::Scalar j) {
                                       // We may need to re-view the cache to a smaller layout if we have less
                                       // data to cache than we have available space in the cache.
                                       // If we re-view the cache then we can keep the smaller cached data
                                       // physically contiguous while still using the same looping APIs
                                       Scalar kernelRemainingRows = inputRows - i;
                                       Scalar kernelRemainingCols = inputCols - j;
                                       Scalar notEnoughRows = shape[0] > kernelRemainingRows;
                                       Scalar notEnoughCols = shape[1] > kernelRemainingCols;
                                       ZeroMemory(cache);

                                       // Generate the cache fill loop in a parameterized lambda so we can emit the different layout versions independently
                                       auto cacheFillLoop = [&](MemoryLayout cacheFillLayout, int rows, int cols) {
                                           auto cacheFillView = cache.GetValue();
                                           cacheFillView.SetLayout(cacheFillLayout);
                                           auto reViewedCache = Array(cacheFillView);

                                           ForRange(Scalar{ 0 }, Scalar{ cols / stripeSize }, [&](Scalar stripeColumnChunk) {
                                               ForRange(Scalar{ 0 }, Scalar{ rows }, [&](Scalar row) {
                                                   ForRange(Scalar{ 0 }, Scalar{ stripeSize }, [&](Scalar stripeColumn) {
                                                       reViewedCache({ stripeColumnChunk, row, stripeColumn }) = input(i + row, j + stripeColumnChunk * stripeSize + stripeColumn);
                                                   });
                                               });
                                           });
                                           auto finalColumnChunk = Scalar{ cols / stripeSize };
                                           ForRange(Scalar{ 0 }, Scalar{ rows }, [&](Scalar row) {
                                               ForRange(Scalar{ 0 }, Scalar{ cols % stripeSize }, [&](Scalar stripeColumn) {
                                                   reViewedCache({ finalColumnChunk, row, stripeColumn }) = input(i + row, j + finalColumnChunk * stripeSize + stripeColumn);
                                               });
                                           });
                                       };

                                       // Emit all of the different loops individually since the cache layouts are set at emit-time
                                       If(notEnoughRows,
                                          [&]() {
                                              If(notEnoughCols,
                                                 [&]() {
                                                     // Boundary condition 4
                                                     cacheFillLoop(boundaryConditionCacheLayout4, remainingRows, remainingCols);
                                                 })
                                                  .Else(
                                                      [&]() {
                                                          // Boundary condition 3
                                                          cacheFillLoop(boundaryConditionCacheLayout3, remainingRows, shape[1]);
                                                      });
                                          })
                                           .ElseIf(notEnoughCols,
                                                   [&]() {
                                                       // Boundary condition 2
                                                       cacheFillLoop(boundaryConditionCacheLayout2, shape[0], remainingCols);
                                                   })
                                           .Else(
                                               [&]() {
                                                   // Boundary condition 1
                                                   cacheFillLoop(boundaryConditionCacheLayout1, shape[0], shape[1]);
                                               });
                                   });

        auto& underlyingNest = nest.GetUnderlyingLoopNest();
        underlyingNest.AddKernel(cacheFillKernel, loopnests::CodePositionConstraints{ loopnests::LoopFragmentType::prologue, _atIndices, {} });

        std::vector<Index> viewInitKernelIndices;
        viewInitKernelIndices.assign(_kernelIndices.begin(), _kernelIndices.end());
        viewInitKernelIndices.push_back(stripeSplitIndex);
        auto viewInitKernel = loopnests::Kernel(cacheName + "_View_Init_Kernel")
                                  .Inputs(liftedCache, cacheRef)
                                  .Indices(viewInitKernelIndices)
                                  .Define([shape = _shape, stripeSize, inputRows, inputCols, cacheViewLayout1, cacheViewLayout2, cacheViewLayout3, cacheViewLayout4, boundaryConditionCacheLayout1, boundaryConditionCacheLayout2, boundaryConditionCacheLayout3, boundaryConditionCacheLayout4](value::Array cache, value::Value cacheRef, value::Scalar i, value::Scalar j, value::Scalar jStripe) {
                                      // To set up the view for the kernel to use, we need to set up the cacheRef reference
                                      // so that a kernel indexing with (i, j) winds up in the right spot, pointing into the
                                      // cached row-major submatrix that is the (j / stripeSize, ALL, ALL) slice of the cache array

                                      // We may need to re-view the cache view to a smaller layout if we are in one of the boundary conditions
                                      Scalar remainingRows = inputRows - i;
                                      Scalar remainingCols = inputCols - j;
                                      Scalar notEnoughRows = shape[0] > remainingRows;
                                      Scalar notEnoughCols = shape[1] > remainingCols;

                                      auto cacheViewFn = [&](MemoryLayout cacheLayout, MemoryLayout viewLayout) {
                                          // Re-View the cache so we can index into the correct cached stripe
                                          auto cacheView = cache.GetValue();
                                          cacheView.SetLayout(cacheLayout);
                                          auto cacheStripe = jStripe % shape[1]; // If N > N', make sure we index into the re-initialized cache position
                                          auto indexedCacheView = cacheView.Offset({ cacheStripe / stripeSize, 0, 0 });

                                          // Re-View the indexed cache as a 2-D matrix so we can position the offset pointer for use in the inner kernels
                                          indexedCacheView.SetLayout(viewLayout);
                                          auto offsetIndexedCacheView = indexedCacheView.Offset({ -1 * i, -1 * j });
                                          offsetIndexedCacheView.SetLayout(viewLayout);
                                          cacheRef.SetLayout(viewLayout);
                                          cacheRef = offsetIndexedCacheView.Reference();
                                      };

                                      // Emit all of the views and offsets individually since the cache layouts are set at emit-time
                                      If(notEnoughRows,
                                         [&]() {
                                             If(notEnoughCols,
                                                [&]() {
                                                    // Boundary condition 4
                                                    cacheViewFn(boundaryConditionCacheLayout4, cacheViewLayout4);
                                                })
                                                 .Else(
                                                     [&]() {
                                                         // Boundary condition 3
                                                         cacheViewFn(boundaryConditionCacheLayout3, cacheViewLayout3);
                                                     });
                                         })
                                          .ElseIf(notEnoughCols,
                                                  [&]() {
                                                      // Boundary condition 2
                                                      cacheViewFn(boundaryConditionCacheLayout2, cacheViewLayout2);
                                                  })
                                          .Else(
                                              [&]() {
                                                  // Boundary condition 1
                                                  cacheViewFn(boundaryConditionCacheLayout1, cacheViewLayout1);
                                              });
                                  });
        underlyingNest.AddKernel(viewInitKernel, loopnests::CodePositionConstraints{ loopnests::LoopFragmentType::prologue, { stripeSplitIndex }, {} });
        underlyingNest.RenameVariable(_value, cacheRef, _atIndices, { cacheFillKernel, viewInitKernel });
    }

    // namespace value

    // Helper class to hold a binary tree with a MemoryLayout at each leaf node corresponding to a different
    // boundary condition and with a number of levels equal to the number of dimensions in a cache layout
    class BoundaryConditionMemoryLayoutHelper
    {
        // A multi-dimensional cache memory layout with N dimensions can have 2^N different boundary condition
        // layouts since each dimension of the cache memory layout could either be in a:
        // - general case - the number of elements in that dimension in this particular slice in the
        //                  cache layout is less than or equal to the number of elements remaining in the input
        //                  for that dimension
        // - boundary case - the number of elements in that dimension in this particular slice in the cache
        //                  layout is greater than the number of elements remaining in the input for that dimension
        //
        // We must generate two different types of things for these cases:
        // 1) a set of memory layouts for each possible scenario - at emit time we can know all of the general
        //      or boundary cases that we will hit. We need the shape of the input region, the input fill region,
        //      the cache layout, and the cache fill layout.
        // 2) a nested set of emitted If/Else switches that will switch on the remaining size of the input for
        //      each cache dimension and call a given lambda with the appropriate boundary condition memory layout
    public:
        BoundaryConditionMemoryLayoutHelper(MemoryShape inputShape,
                                            const std::vector<int>& orderedIndexSizes,
                                            const std::vector<int>& logicalDimensionMapping,
                                            const std::vector<int>& splitIndexScaleFactors,
                                            unsigned cacheFillThresholdIdxOffset,
                                            unsigned cacheViewThresholdIdxOffset) :
            _inputShape(inputShape),
            _orderedIndexSizes(orderedIndexSizes),
            _logicalDimensionMapping(logicalDimensionMapping),
            _splitIndexScaleFactors(splitIndexScaleFactors),
            _cacheFillThresholdIdxOffset(cacheFillThresholdIdxOffset),
            _cacheViewThresholdIdx(cacheViewThresholdIdxOffset)
        {
            if (orderedIndexSizes.size() != logicalDimensionMapping.size())
            {
                throw InputException(InputExceptionErrors::invalidSize, "Need to provide the same number of ordered index sizes as logical dimension mappings");
            }
            if (orderedIndexSizes.size() != splitIndexScaleFactors.size())
            {
                throw InputException(InputExceptionErrors::invalidSize, "Need to provide the same number of ordered index sizes as split index scale factors mappings");
            }
            if (orderedIndexSizes.empty())
            {
                throw InputException(InputExceptionErrors::invalidSize, "Need to provide at least one index size");
            }
            FillTree();
        }

        template <typename Fn>
        void EmitBoundarySwitches(const std::vector<Scalar>& compositeIndices, Fn&& func) const
        {
            unsigned inputLogicalDimensionCount = _inputShape.NumDimensions();
            if (compositeIndices.size() != inputLogicalDimensionCount)
            {
                throw InputException(InputExceptionErrors::invalidSize, "Need to provide one scalar index value per logical dimension in the input");
            }

            // Compute how many elements are remaining in each logical dimension
            std::vector<Scalar> remainingElements;
            remainingElements.reserve(inputLogicalDimensionCount);
            for (size_t logicalDimension = 0; logicalDimension < inputLogicalDimensionCount; ++logicalDimension)
            {
                Scalar remaining = _inputShape[logicalDimension] - compositeIndices[logicalDimension];
                remainingElements.push_back(remaining);
            }

            // Determine which levels of the tree are going to be in a boundary condition based on the remaining elements
            std::vector<Scalar> isBoundaryCase;
            isBoundaryCase.reserve(_logicalDimensionMapping.size());
            for (unsigned idx = 0; idx < _logicalDimensionMapping.size(); ++idx)
            {
                isBoundaryCase.push_back(_orderedIndexSizes[idx] > remainingElements[_logicalDimensionMapping[idx]]);
            }

            // Run a depth-first traversal of the tree to emit the nested If/Else switches to handle all the boundary conditions
            RecursiveEmitHelper(_tree, isBoundaryCase, 0, func);
        }

    private:
        struct BoundaryConditionTreeNode
        {
            BoundaryConditionTreeNode(const std::vector<int>& logicalDimensionSizes) :
                cacheLogicalDimensionSizes(logicalDimensionSizes.size()),
                cacheFillLogicalDimensionSizes(logicalDimensionSizes.size()),
                subLogicalDimensionSizes(logicalDimensionSizes)
            {}
            BoundaryConditionTreeNode(const std::shared_ptr<BoundaryConditionTreeNode>& parent,
                                      int newCacheSize,
                                      int newInputSize,
                                      int logicalDimension,
                                      const std::vector<int>& splitIndexScaleFactors,
                                      bool isCacheFillIdx) :
                cacheSizes(parent->cacheSizes),
                cacheFillSizes(parent->cacheFillSizes),
                cacheLogicalDimensionSizes(parent->cacheLogicalDimensionSizes),
                cacheFillLogicalDimensionSizes(parent->cacheFillLogicalDimensionSizes),
                subLogicalDimensionSizes(parent->subLogicalDimensionSizes)
            {
                cacheSizes.push_back(newCacheSize);

                if (cacheLogicalDimensionSizes[logicalDimension] == 0)
                {
                    cacheLogicalDimensionSizes[logicalDimension] = newInputSize;
                }

                if (isCacheFillIdx)
                {
                    cacheFillSizes.push_back(newCacheSize);
                    if (cacheFillLogicalDimensionSizes[logicalDimension] == 0)
                    {
                        cacheFillLogicalDimensionSizes[logicalDimension] = newInputSize;
                    }
                }
                subLogicalDimensionSizes[logicalDimension] = newInputSize;
                ComputeShape(splitIndexScaleFactors);
            }

            void ComputeShape(const std::vector<int>& splitIndexScaleFactors)
            {
                // Create a cache shape for this level
                std::vector<int> shardSizes;
                shardSizes.reserve(cacheSizes.size());

                std::vector<int> fillShardSizes;
                fillShardSizes.reserve(cacheFillSizes.size());
                unsigned fillOffset = cacheSizes.size() - cacheFillSizes.size();
                for (unsigned idx = 0; idx < cacheSizes.size(); ++idx)
                {
                    int shardSize = cacheSizes[idx] / splitIndexScaleFactors[idx];
                    if (cacheSizes[idx] % splitIndexScaleFactors[idx] != 0)
                    {
                        // Account for partial shards
                        shardSize++;
                    }
                    shardSizes.push_back(shardSize);
                    if (idx >= fillOffset)
                    {
                        fillShardSizes.push_back(shardSize);
                    }
                }

                cacheShape = { shardSizes };
                cacheFillShape = { fillShardSizes };
                inputRegionShape = { cacheLogicalDimensionSizes };
                inputRegionFillShape = { cacheFillLogicalDimensionSizes };
            }

            // Use shared_ptr instead of unique_ptr since we need to be able to copy these helper objects into multiple lambdas
            std::shared_ptr<BoundaryConditionTreeNode> generalCase;
            std::shared_ptr<BoundaryConditionTreeNode> boundaryCase;
            std::vector<int> cacheSizes;
            std::vector<int> cacheFillSizes;
            std::vector<int> cacheLogicalDimensionSizes; // logical dimension sizes represented by the full cache
            std::vector<int> cacheFillLogicalDimensionSizes; // logical dimension sizes represented by the fill view of the cache
            std::vector<int> subLogicalDimensionSizes; // logical input dimension sizes represented by this portion of the tree
            MemoryShape cacheShape;
            MemoryShape cacheFillShape;
            MemoryShape inputRegionShape;
            MemoryShape inputRegionFillShape;
        };

        void FillTree()
        {
            int logicalDimensionCount = _inputShape.NumDimensions();
            std::vector<int> baseLogicalDimensionCacheSizes;
            baseLogicalDimensionCacheSizes.reserve(logicalDimensionCount);
            for (int logicalDimension = 0; logicalDimension < logicalDimensionCount; ++logicalDimension)
            {
                baseLogicalDimensionCacheSizes.push_back(_inputShape[logicalDimension]);
            }
            _tree = std::make_shared<BoundaryConditionTreeNode>(baseLogicalDimensionCacheSizes);
            std::queue<std::shared_ptr<BoundaryConditionTreeNode>> activeNodes;
            activeNodes.push(_tree);

            // Compute the minimum sizes for remainders / boundary cases for each level
            // A remainder / boundary case needs to cover an integer number of the splits that
            // occur later on in that logical dimension, so we compute the possible sizes for
            // the remainders in each cache dimension up front
            // E.g. if we have a cache that is 4x4, with a split of 2 in the column dimension,
            //      ordered at {1,0,1} so that the cache size is {2,4,2}, but our input is
            //      4x3, we need to zero-pad the innermost dimension since we need to keep
            //      an integer number of them, and thus behave as though it's still a 4x4
            //      input region and we have a {2,4,2} cache.
            //      If instead we have 4x2 input, we can reduce the outermost dimension shard
            //      count by 1 and still cover an integer number of the inner splits with a
            //      {1,4,2} cache
            std::vector<int> remainderMinimumSizes(_logicalDimensionMapping.size());
            std::map<int, int> logicalDimensionWorkingSizes;
            // loop from the innermost split dimension to the outermost
            for (unsigned idx = _logicalDimensionMapping.size() - 1; _logicalDimensionMapping.size() > idx; --idx)
            {
                int logicalDimension = _logicalDimensionMapping[idx];
                auto workingSizeIter = logicalDimensionWorkingSizes.find(logicalDimension);
                int size = _orderedIndexSizes[idx];
                if (workingSizeIter == logicalDimensionWorkingSizes.end())
                {
                    remainderMinimumSizes[idx] = 1;
                }
                else
                {
                    remainderMinimumSizes[idx] = logicalDimensionWorkingSizes[logicalDimension];
                }
                logicalDimensionWorkingSizes[logicalDimension] = size;
            }

            for (unsigned idx = 0; idx < _logicalDimensionMapping.size(); ++idx)
            {
                int logicalDimension = _logicalDimensionMapping[idx];
                int cacheSplitSize = _orderedIndexSizes[idx];
                size_t numNodesInLevel = activeNodes.size();
                for (unsigned nodeIdx = 0; nodeIdx < numNodesInLevel; ++nodeIdx)
                {
                    auto currentNode = activeNodes.front();
                    activeNodes.pop();

                    int baseLogicalInputSize = currentNode->subLogicalDimensionSizes[logicalDimension];
                    int baseRemainderSize = baseLogicalInputSize % cacheSplitSize;

                    // round up the logical input size based on the remainder minimum size for this dimension
                    int logicalInputSize = RoundUpToMultiple(baseLogicalInputSize, remainderMinimumSizes[idx]);
                    int remainderSize = logicalInputSize % cacheSplitSize;

                    if (idx > _cacheViewThresholdIdx || remainderSize == 0)
                    {
                        // We can't reshape the cache view, so if we're inside of the view portion
                        // of the cache we need to zero-pad
                        // As a half-step to keep the cache as dense as possible when we're in a boundary condition,
                        // we let the first cacheViewThresholdIdx be shrunk for the purposes of creating
                        // the cache layout, since this idx is definitely in the most-major dimension of
                        // the cache view as it is the farthest out. Therefore we only consider if
                        // idx > _cacheViewThresholdIdx instead of idx >= _cacheViewThresholdIdx

                        // Additionally, if after rounding up the logical input size we've produced
                        // an integer multiple of cacheSplitSize, we need to generate a boundary condition
                        // branch with the full cacheSplitSize as the cache size, but with the base remainder
                        // size as the input size
                        remainderSize = cacheSplitSize;
                    }

                    if (cacheSplitSize <= logicalInputSize)
                    {
                        currentNode->generalCase = std::make_shared<BoundaryConditionTreeNode>(currentNode,
                                                                                               cacheSplitSize,
                                                                                               cacheSplitSize,
                                                                                               logicalDimension,
                                                                                               _splitIndexScaleFactors,
                                                                                               idx >= _cacheFillThresholdIdxOffset);
                        activeNodes.push(currentNode->generalCase);
                    }
                    if (baseRemainderSize > 0)
                    {
                        currentNode->boundaryCase = std::make_shared<BoundaryConditionTreeNode>(currentNode,
                                                                                                remainderSize,
                                                                                                baseRemainderSize,
                                                                                                logicalDimension,
                                                                                                _splitIndexScaleFactors,
                                                                                                idx >= _cacheFillThresholdIdxOffset);
                        activeNodes.push(currentNode->boundaryCase);
                    }
                }
            }
        }

        template <typename Fn>
        void RecursiveEmitHelper(const std::shared_ptr<BoundaryConditionTreeNode>& currentNode, const std::vector<Scalar>& isBoundaryCase, unsigned currentIdx, Fn&& func) const
        {
            if (currentNode->generalCase == nullptr && currentNode->boundaryCase == nullptr)
            {
                // Base case, call the given function with our cache shape for this leaf of the tree
                func(currentNode->inputRegionShape, currentNode->inputRegionFillShape, currentNode->cacheShape, currentNode->cacheFillShape);
            }
            else if (currentNode->generalCase == nullptr)
            {
                // we only have a boundary case, so don't emit an If/Else but instead just recurse to the boundary case
                RecursiveEmitHelper(currentNode->boundaryCase, isBoundaryCase, currentIdx + 1, func);
            }
            else if (currentNode->boundaryCase == nullptr)
            {
                // we only have a general case, so don't emit an If/Else but instead just recurse to the general case
                RecursiveEmitHelper(currentNode->generalCase, isBoundaryCase, currentIdx + 1, func);
            }
            else
            {
                // We have both a general case and a boundary case, so emit an If/Else switch to emit both cases
                If(isBoundaryCase[currentIdx], [&] {
                    RecursiveEmitHelper(currentNode->boundaryCase, isBoundaryCase, currentIdx + 1, func);
                }).Else([&] {
                    RecursiveEmitHelper(currentNode->generalCase, isBoundaryCase, currentIdx + 1, func);
                });
            }
        }

        MemoryShape _inputShape;
        std::vector<int> _orderedIndexSizes;
        std::vector<int> _logicalDimensionMapping;
        std::vector<int> _splitIndexScaleFactors;
        unsigned _cacheFillThresholdIdxOffset;
        unsigned _cacheViewThresholdIdx;
        std::shared_ptr<BoundaryConditionTreeNode> _tree; // Use shared_ptr's since we need to be able to copy these helper objects into multiple lambdas
    };

    std::pair<MemoryLayout, unsigned int> ComputeCacheView(MemoryLayout cacheLayout,
                                                           const std::vector<int>& cacheLogicalDimensionMapping,
                                                           int logicalDimensionCount)
    {
        std::vector<int> cacheViewSizes;
        cacheViewSizes.reserve(logicalDimensionCount);
        std::vector<int> dimensionOrdering(logicalDimensionCount, -1); // initialize the dimensionOrdering since we will be filling it out-of-order

        // Iterate the cacheLogicalDimensionMapping from back to front in order to walk the shape
        // from the inner splits to the outer splits
        int previousLogicalDimension = -1; // -1 == sentinel uninitialized value, not any of the logical dimensions
        unsigned cacheViewThresholdIdx = cacheLogicalDimensionMapping.size() - 1; // index in the logical dimension mapping / split indices vector to start the cache view at
        unsigned currentDimensionOrderingIdx = dimensionOrdering.size() - 1;
        std::set<int> seenLogicalDimensions;
        std::map<int, int> logicalDimensionToCacheViewSize; // maps from logical dimension to the cache view size
        for (unsigned idx = cacheLogicalDimensionMapping.size() - 1; cacheLogicalDimensionMapping.size() > idx; --idx)
        {
            int logicalDimension = cacheLogicalDimensionMapping[idx];
            if (previousLogicalDimension != logicalDimension)
            {
                // This is different from the previous logical dimension that we were collapsing
                if (seenLogicalDimensions.find(logicalDimension) != seenLogicalDimensions.end())
                {
                    // If we've seen this logical dimension before and we aren't currently collapsing it then this is a repeat
                    // that prompts us to stop building up the cache view
                    break;
                }
                else
                {
                    // this is the first time we've seen this dimension, so insert it into the dimension ordering outside of the
                    // dimensions we've already seen
                    dimensionOrdering[currentDimensionOrderingIdx--] = logicalDimension;
                    seenLogicalDimensions.insert(logicalDimension);
                    previousLogicalDimension = logicalDimension;
                }
            }
            cacheViewThresholdIdx = idx;
        }

        // Now we know the sizes of all the dimensions in the view and we need to fill the remainder of
        // the dimension ordering with any logical dimensions in the input that aren't part of the cache view.
        // Any logical dimensions that aren't part of the cache view have a cache view size of 1, and thus
        // the ordering of them doesn't really matter since we'll re-view the cache before changing the
        // index in that dimension we examine
        for (int logicalDimension = 0; logicalDimension < logicalDimensionCount; ++logicalDimension)
        {
            // Set the size to 1, for logical dimensions that are also in the view, we will multiply
            // this value by the shard sizes in the view
            logicalDimensionToCacheViewSize[logicalDimension] = 1;
            if (seenLogicalDimensions.find(logicalDimension) == seenLogicalDimensions.end())
            {
                // This dimension isn't part of the view, so insert it in the dimension ordering
                // outside of the dimensions that are in the view
                dimensionOrdering[currentDimensionOrderingIdx--] = logicalDimension;
            }
        }

        // Now we need to build up the sizes of the view dimensions by taking the product of cache dimension
        // sizes within each logical dimension after the point in the cache hierarchy where the view starts.
        // We take the product of the sizes because the active sizes at each cache dimension represent the
        // number of shards in that split dimension, not necessarily element count in that logical dimension.
        // In the innermost split in each logical dimension the shareds are all of size 1 and therefore
        // shard count == element count
        for (unsigned idx = cacheViewThresholdIdx; idx < cacheLogicalDimensionMapping.size(); ++idx)
        {
            int logicalDimension = cacheLogicalDimensionMapping[idx];
            logicalDimensionToCacheViewSize[logicalDimension] *= cacheLayout.GetActiveSize(idx);
        }

        // Now that we have the full view dimension ordering and a map from logical dimension to view size,
        // fill out the ordered view sizes vector
        for (unsigned idx = 0; idx < dimensionOrdering.size(); ++idx)
        {
            int logicalDimension = dimensionOrdering[idx];
            cacheViewSizes.push_back(logicalDimensionToCacheViewSize[logicalDimension]);
        }

        return { MemoryLayout{ MemoryShape{ cacheViewSizes }, DimensionOrder{ dimensionOrdering } }, cacheViewThresholdIdx };
    }

    void GeneralCachingStrategy::HandleCachingImpl(LoopNest& nest)
    {
        // General caching strategy:
        // Given:
        //     - input value
        //     - top level indices that the input uses
        //     - name for the cache
        //     - size of the cache to use in # of elements
        //     - # of elements to cache at a time ( < size of cache for progressive caching, > size of cache is an error)
        //     - Input / InputOutput / Output designation
        //     - Reduce function operating on individual Scalars
        //
        // Set up 3-4 kernels:
        //     - Cache flushing kernel
        //     - Cache filling kernel if Input/InputOutput
        //     - Cache viewing kernel (based on the shape of the input value)
        //     - Cache reduce kernel if InputOutput/Output

        auto extraParams = std::any_cast<std::tuple<value::ArgumentType,
                                                    std::string,
                                                    size_t,
                                                    size_t,
                                                    std::function<ReduceFunctionType>,
                                                    bool>>(_extra);
        value::ArgumentType argType;
        std::string baseName;
        size_t maxCacheElts;
        size_t fillThreshold; // fillThreshold <= maxCacheElts
        std::function<ReduceFunctionType> reduceFunction;
        bool accumulateReduce;
        std::tie(argType,
                 baseName,
                 maxCacheElts,
                 fillThreshold,
                 reduceFunction,
                 accumulateReduce) = extraParams;

        // Read target machine characteristics for number of SIMD registers and the size of the registers
        RegisterCharacteristics registerCharacteristics = GetRegisterCharacteristics(_value.GetBaseType());

        // Determine kernels needed
        bool useFillKernel = (argType == value::ArgumentType::Input || argType == value::ArgumentType::InputOutput);
        bool useViewKernel = true; // always include view kernel for simplicity for now, even if the re-viewing winds up being redundant
        bool useReduceKernel = (argType == value::ArgumentType::Output || argType == value::ArgumentType::InputOutput);

        size_t bufferAlignment = 16 * sizeof(float);

        InvokeForContext<CppEmitterContext>([&] {
            // TODO : Support buffer alignment in CppEmitterContext
            bufferAlignment = 0;
        });

        auto inputArray = Array(_value);
        int logicalDimensionCount = _value.GetLayout().NumDimensions();
        int compositeIndexCount = _kernelIndices.size();
        auto& underlyingNest = nest.GetUnderlyingLoopNest();

        const auto& loopSequence = underlyingNest.GetLoopSequence();
        std::vector<Index> orderedIndices;
        for (const auto& index : loopSequence)
        {
            const auto& dimensionIndex = underlyingNest.GetDimensionRange(index).GetDimensionIndex();
            auto indexIter = std::find(_kernelIndices.begin(), _kernelIndices.end(), dimensionIndex);
            if (indexIter != _kernelIndices.end())
            {
                orderedIndices.push_back(index);
            }
        }

        // Ensure we have some indices
        if (orderedIndices.empty())
        {
            throw InputException(InputExceptionErrors::invalidSize, "Don't have any indices relevant to this input for this loop nest");
        }

        // If there are no _atIndices specified, default to the outermost orderedIndices index
        if (_atIndices.empty())
        {
            _atIndices.push_back(orderedIndices.front());
        }

        // Compute the mapping between the orderedIndices list and the logical input dimensions
        std::vector<int> logicalDimensionMapping;
        logicalDimensionMapping.reserve(orderedIndices.size());

        // Determine the size for each split for each logical dimension
        // We only care about the split indices that are passed in as part of
        // orderedIndices, so instead of recording the sizes of those indices,
        // instead record the size of the full index range followed by the increments
        // of the each of the orderedIndices
        std::map<int, std::vector<int>> logicalDimensionSplitSizes;
        for (int logicalDimension = 0; logicalDimension < logicalDimensionCount; ++logicalDimension)
        {
            logicalDimensionSplitSizes[logicalDimension].push_back(_value.GetLayout().GetActiveSize(logicalDimension));
        }

        // Determine the increments for each split index in the orderedIndices
        // The cache dimensions all operate with logical increments of 1, so when we are mapping between input space and cache space
        // we need to scale appropriately by the split index increments for each split index
        std::vector<int> orderedIndexIncrements;
        orderedIndexIncrements.reserve(orderedIndices.size());

        for (const auto& index : orderedIndices)
        {
            // Compute the logical dimension mapping
            const auto& dimensionIndex = underlyingNest.GetDimensionRange(index).GetDimensionIndex();
            auto indexIter = std::find(_kernelIndices.begin(), _kernelIndices.end(), dimensionIndex);
            // Here we assume:
            //  - _kernelIndices is a vector or similar, so (iteror - begin) == idx of iterator
            //  - _kernelIndices is arranged in logical dimension ordering for this input
            int logicalDimension = indexIter - _kernelIndices.begin();
            logicalDimensionMapping.push_back(logicalDimension);

            // Find the index increment for this index to use for scaling index values to
            // convert between cache dimensions and input indices
            // Also use this for the logical dimension split sizes
            auto increment = underlyingNest.GetIndexRange(index).Increment();
            orderedIndexIncrements.push_back(increment);
            logicalDimensionSplitSizes[logicalDimension].push_back(increment);
        }

        // Compute the memory shape for the cache based on the index sizes in each logical
        // dimension. Each MemoryShape dimension counts the number of shards of the cache
        // that dimension indexes over, so the size of each MemoryShape dimension ought to be
        // the size of the index divided by the size of the next split index in the same
        // logical input dimension.
        // e.g. if Index i ranges over [0,64), and is split by 32, then by 16, then by 4
        //      we will have split indices [0,64,32), [0,32,16), [0,16,4), and [0,4,1),
        //      but suppose a cache doesn't use the second index, i.e. it only uses
        //      [0,64,32), [0,16,4), and [0,4,1), then the memory shape (for split dimensions
        //      in the i logical dimension) should be { 4, 4, 4 } since the outer index
        //      ranging from 0 to 64 accounts for 4 shards of 16
        //      and the next index ranging from 0 to 16 accounts for 4 shards of 4
        //      and the next index ranging from 0 to 4 accounts for 4 shards of 1
        //
        // Now that we have the base dimension size and all the increments for the indices we're using
        // we can compute the shard sizes for each logical dimension by dividing each dimension split
        // size we accumulated above with the size that comes after it, indicating how many instnaces of
        // the next shard occur within the current shard
        std::map<int, std::queue<int>> logicalIndexToShardSizes;
        std::map<int, std::queue<int>> logicalIndexToSizes; // Full element counts, not shard counts
        for (int logicalDimension = 0; logicalDimension < logicalDimensionCount; ++logicalDimension)
        {
            const auto& splitSizes = logicalDimensionSplitSizes[logicalDimension];
            for (unsigned splitIdx = 0; splitIdx < splitSizes.size() - 1; ++splitIdx)
            {
                int currentSize = splitSizes[splitIdx];
                int nextSize = splitSizes[splitIdx + 1];
                int shardSize = currentSize / nextSize;
                if (currentSize % nextSize != 0)
                {
                    // Round up to account for partial shards
                    shardSize++;
                }
                logicalIndexToShardSizes[logicalDimension].push(shardSize);
                logicalIndexToSizes[logicalDimension].push(currentSize);
            }
        }

        // Now that we have the shard sizes grouped by logical dimension, arrange them to match
        // the orderedIndices
        std::vector<int> orderedIndexShardSizes;
        std::vector<int> orderedIndexSizes; // Full element counts, not shard counts
        orderedIndexShardSizes.reserve(orderedIndices.size());
        orderedIndexSizes.reserve(orderedIndices.size());
        for (unsigned idx = 0; idx < logicalDimensionMapping.size(); ++idx)
        {
            int logicalDimension = logicalDimensionMapping[idx];

            orderedIndexShardSizes.push_back(logicalIndexToShardSizes[logicalDimension].front());
            logicalIndexToShardSizes[logicalDimension].pop();

            orderedIndexSizes.push_back(logicalIndexToSizes[logicalDimension].front());
            logicalIndexToSizes[logicalDimension].pop();
        }

        // Create a MemoryShape for the cache based on the shard counts
        // This isn't the final cache shape and layout yet - we may need to shrink it to fit the number
        // of elements requested in the cache
        MemoryShape fullInputShape = { orderedIndexShardSizes };
        MemoryLayout fullInputLayout = { fullInputShape };

        // Physical Cache
        // Determine how large the physical cache ought to be by trying to cover complete view
        // dimensions without exceeding maxCacheElts elements in size.
        // e.g. if the full view has 5 dimensions, and our maxCacheElts only covers the inner most two dimensions,
        //      then the cache size is set to that size and we create our "fill" and "reduce" kernels accordingly
        // To achieve this, start from the base full cache layout and slice off physical dimensions going from the
        // outermost to the innermost until the full extent has no more than maxCacheElts elements
        MemoryLayout cacheLayout = fullInputLayout;
        unsigned cacheThresholdIdx = 0;
        while (cacheLayout.GetMemorySize() > maxCacheElts)
        {
            cacheLayout = cacheLayout.GetSliceLayout(0);
            cacheThresholdIdx++;
        }
        if (cacheLayout.NumElements() == 0)
        {
            throw InputException(InputExceptionErrors::invalidSize, "Specified cache size isn't large enough to cover the smallest dimension of the cache layout");
        }
        std::vector<int> cacheOrderedIndexSizes(orderedIndexSizes.begin() + cacheThresholdIdx, orderedIndexSizes.end());
        std::vector<int> cacheLogicalDimensionMapping(logicalDimensionMapping.begin() + cacheThresholdIdx, logicalDimensionMapping.end());
        std::vector<int> cacheOrderedIndexIncrements(orderedIndexIncrements.begin() + cacheThresholdIdx, orderedIndexIncrements.end());
        auto cacheName = UniqueName(baseName);
        _rawCache = StaticAllocate(cacheName, _value.GetBaseType(), cacheLayout);

        // Progresive Caching
        // To enable progressive caching, where a subset of the full physical cache is
        // filled and used, then later the next chunk of the physical cache is filled
        // and used, we need to find the dimension split at which fillThreshold elements
        // is surpassed and set up a fill kernel at that point
        // If fillThreshold == maxCacheElts or they are both exceeded in the same
        // split, then ensure that the fill kernel occurs after the cache emptying kernel
        if (fillThreshold > maxCacheElts)
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Fill threshold can't be larger than the max cache size");
        }
        unsigned cacheFillThresholdIdx = cacheThresholdIdx;
        MemoryLayout cacheFillLayout = cacheLayout;
        while (cacheFillLayout.GetMemorySize() > fillThreshold)
        {
            cacheFillLayout = cacheFillLayout.GetSliceLayout(0);
            cacheFillThresholdIdx++;
        }
        if (cacheFillLayout.NumElements() == 0)
        {
            throw InputException(InputExceptionErrors::invalidSize, "Specified cache fill threshold size isn't large enough to cover the smallest dimension of the cache layout");
        }
        std::vector<int> cacheFillOrderedIndexSizes(orderedIndexSizes.begin() + cacheFillThresholdIdx, orderedIndexSizes.end());
        std::vector<int> cacheFillLogicalDimensionMapping(logicalDimensionMapping.begin() + cacheFillThresholdIdx, logicalDimensionMapping.end());
        std::vector<int> cacheFillOrderedIndexIncrements(orderedIndexIncrements.begin() + cacheFillThresholdIdx, orderedIndexIncrements.end());

        // Cache View
        // The cache view needs to have the same number of dimensions as the input value
        // but cover an area that is a subset of the full cache and represents one cache
        // dimension per logical input dimension.
        // This may mean that for some of the logical input dimensions, the cache view
        // size is 1, e.g. suppose a 3-D input is cached where the inner 3 dimensions of
        // the cache only operate over two of the logical dimensions of the input while the
        // two innermost dimensions of those operate over the two distinct input logical
        // dimensions. In that case the cache view would cover the inner two cache dimensions
        // and have a 1 for the third dimension size.
        // In general, the cache view needs to cover an area of the cache that can be
        // contiguously represented like the logical input value.

        // To build up the cache view layout, start from the innermost dimension of the
        // cache layout and accumulate dimensions going outward until either all of the
        // logical input dimensions are accounted for or one of the logical input dimensions
        // repeats. However, when a single dimension is repeated multiple times in a row,
        // those repeats can be collapsed into a single visiting of that dimension. These
        // can be collapsed because the logical behavior is the same regardless of whether
        // the split that produced the repeated dimension was made or not.
        // E.g. suppose your dimensions are { 0, 0, 1, 1, 1, 0, 0 }, then the first pair of
        //      0's can be collapsed and treated like a single visiting of that dimension,
        //      the set of 3 1's can be collapsed, and the final pair of 0's can be collapsed,
        //      producing a collapsed dimension ordering of { 0, 1, 0 }. With a collapsed
        //      dimension ordering of { 0, 1, 0 }, the cache view needs to break at the inner
        //      { 1, 0 }, because after that a dimension (the 0 dimension) will repeat.
        MemoryLayout baseCacheViewLayout;
        unsigned cacheViewThresholdIdxOffset;
        std::tie(baseCacheViewLayout, cacheViewThresholdIdxOffset) = ComputeCacheView(cacheFillLayout,
                                                                                      cacheFillLogicalDimensionMapping,
                                                                                      logicalDimensionCount);
        unsigned cacheViewThresholdIdx = cacheFillThresholdIdx + cacheViewThresholdIdxOffset;

        auto cacheRef = _rawCache.Reference();
        cacheRef.SetLayout(baseCacheViewLayout);

        // Boundary Conditions
        // Boundary conditions occur when the region of the input value that we want
        // to cache does not fill the physical cache,
        // e.g. for a matrix cache there are 4 cases, 3 of which are considered boundary condition cases:
        //      Suppose the matrix is M x N and the physical cache is sized to hold M' x N' elements,
        //      where M / 2 < M' < M, N / 2 < N' < N
        //     |-------N-------|
        //     |----N'---|----N'---|
        // _ _ *---------------*
        // | | |         |     |
        // | M'|    1    |  2  |
        // | | |         |     |
        // M _ |_________|_____|
        // | | |    3    |  4  |
        // | M'|         |     |
        // _ | *---------------*
        //   _
        // 1 : The cache has exactly as many rows and columns as the input matrix chunk
        // 2 : The cache has more columns than the matrix chunk but just as many rows
        // 3 : The cache has more rows than the matrix chunk but just as many columns
        // 4 : The cache has more rows and columns than the matrix chunk
        //
        // One possible solution is to zero-pad the cache and keep the layout as-is. This would certainly work
        //
        // However, in order to maximize data locality in the cache (which is the purpose of the cache),
        // we would prefer it if the cache were reshaped such that the input value chunk
        // fills the cache from the beginning until the end of the chunk without any gaps.
        // This reshape amounts to shrinking the cache sizes in some dimensions, however to preserve
        // vectorization behavior we avoid shrinking the innermost dimension and instead zero-pad
        // that dimension
        unsigned cacheFillThresholdIdxOffset = cacheFillThresholdIdx - cacheThresholdIdx;
        unsigned cacheViewThresholdIdxCacheOffset = cacheViewThresholdIdxOffset + cacheFillThresholdIdxOffset;
        BoundaryConditionMemoryLayoutHelper boundaryConditionCacheHelper(_value.GetLayout().GetActiveSize(), cacheOrderedIndexSizes, cacheLogicalDimensionMapping, cacheOrderedIndexIncrements, cacheFillThresholdIdxOffset, cacheViewThresholdIdxCacheOffset);

        std::vector<loopnests::Kernel> cachingKernels;

        {
            // Flush the cache to implicitly zero-pad any regions of the cache we don't fill later
            std::vector<Index> cacheFlushPosition(orderedIndices.begin(), orderedIndices.begin() + cacheThresholdIdx);
            auto cacheEmptyKernel = loopnests::Kernel(cacheName + "_Empty_Cache_Kernel")
                                        .Inputs(_rawCache)
                                        .Indices()
                                        .Define([](Value cache) {
                                            // TODO : determine if a vectorized approach is worthwhile here
                                            ZeroMemory(cache);
                                        });

            underlyingNest.AddKernel(cacheEmptyKernel, loopnests::CodePositionConstraints{ loopnests::LoopFragmentType::prologue, cacheFlushPosition, {} });
            cachingKernels.push_back(cacheEmptyKernel);
        }
        if (useFillKernel)
        {
            std::vector<Index> cacheFillPosition(orderedIndices.begin(), orderedIndices.begin() + cacheFillThresholdIdx);
            std::vector<Index> cacheFillIndices(_kernelIndices.begin(), _kernelIndices.end());
            cacheFillIndices.insert(cacheFillIndices.end(), cacheFillPosition.begin(), cacheFillPosition.end());

            auto cacheFillKernel = loopnests::Kernel(cacheName + "_Fill_Cache_Kernel")
                                       .Inputs(_value, _rawCache)
                                       .Indices(cacheFillIndices)
                                       .DefineEx([=](std::vector<Value> values, std::vector<Scalar> indices) {
                                           auto& input = values[0];
                                           auto& cache = values[1];
                                           std::vector<Scalar> compositeIndexValues(indices.begin(), indices.begin() + compositeIndexCount);
                                           std::vector<Scalar> splitIndexValues(indices.begin() + compositeIndexCount, indices.end());

                                           auto offsetInput = input.Offset(compositeIndexValues);
                                           offsetInput.SetLayout(input.GetLayout());
                                           auto offsetInputArrayView = Array(offsetInput);

                                           boundaryConditionCacheHelper.EmitBoundarySwitches(compositeIndexValues, [=](MemoryLayout inputRegionShape, MemoryLayout inputRegionFillShape, MemoryLayout boundaryCacheLayout, MemoryLayout boundaryCacheFillLayout) {
                                               // Offset the cache write head based on the where we're at in the progressive caching
                                               // Since fillThreshold <= maxCacheElts, we may run this kernel multiple times filling
                                               // different portions of the cache, so we look at the indices between the
                                               // cacheThresholdIdx and the cacheFillThresholdIdx to find what position we need to
                                               // offset to
                                               // these indices all map in order to the dimensions that are in the cache and outside
                                               // the fill region since the cache memory ordering is based on these indices in this order

                                               auto cacheView = cache;
                                               cacheView.SetLayout(boundaryCacheLayout);
                                               std::vector<Scalar> cacheOffsetIndices;
                                               cacheOffsetIndices.reserve(boundaryCacheLayout.NumDimensions());

                                               // Note: if cacheThresholdIdx == cacheFillThresholdIdx (i.e. if there is no progressive caching)
                                               // Then the first loop is skipped and no offsetting occurs, and therefore filling the cache from
                                               // the beginning every time this kernel is run
                                               for (unsigned idx = cacheThresholdIdx; idx < cacheFillThresholdIdx; ++idx)
                                               {
                                                   // Mapping loopnest indices (input space) -> cache offsets (cache space) so divide by split index increment
                                                   cacheOffsetIndices.push_back(splitIndexValues[idx] / orderedIndexIncrements[idx]);
                                               }
                                               for (unsigned idx = cacheFillThresholdIdx; idx < static_cast<unsigned>(fullInputLayout.NumDimensions()); ++idx)
                                               {
                                                   cacheOffsetIndices.push_back(Scalar{ 0 });
                                               }
                                               auto offsetCache = cacheView.Offset(cacheOffsetIndices);
                                               offsetCache.SetLayout(boundaryCacheFillLayout);
                                               auto cacheFillArrayView = Array(offsetCache);

                                               // Prefer input-oriented loops to maximize locality as the input
                                               // is likely to be larger than the cache in most cases
                                               // Based on the element size and counts in different dimensions,
                                               // we will split and unroll some of the inner loops in order to maximize
                                               // vectorization.
                                               // In order to get appropriate utilization of all the SIMD
                                               // registers, we will need to use a temporary buffer (which we expect
                                               // the compiler to optimize away) with a size equal to the total number
                                               // of elements that can be held in all of the SIMD registers.
                                               // The filling of this temporary buffer from the input needs to be an
                                               // unrolled operation and the filling of the cache from the temporary
                                               // buffer also needs to be an unrolled operation that happens after
                                               // the full temporary buffer has been filled.
                                               // Therefore, we need multiple levels of loopnests so that the area
                                               // outside of the temporary buffer's addressable region can be looped
                                               // over, and the area inside the temporary buffer region can have two
                                               // sequential fully unrolled loopnests.
                                               // new loopnest (outer):
                                               // For ... {
                                               //   For ... {
                                               //       // start of outer loopnest prologue kernel
                                               //       // Fill temp buf
                                               //       new loopnest (inner #1):
                                               //       For ... (unroll) {
                                               //           For ... (unroll) {
                                               //               ... {
                                               //                   // start of inner loopnest #1 kernel
                                               //                   tempBuf(tempBufIndices) = input(inputIndices)
                                               //                   // end of inner loopnest #1 kernel
                                               //               }
                                               //               ...
                                               //           }
                                               //       }
                                               //       // Fill cache
                                               //       new loopnest (inner #2):
                                               //       For ... (unroll) {
                                               //           For ... (unroll) {
                                               //               ... {
                                               //                   // start of inner loopnest #2 kernel
                                               //                   cache(cacheIndices) = tempBuf(tempBufIndices)
                                               //                   // end of inner loopnest #2 kernel
                                               //               }
                                               //               ...
                                               //           }
                                               //       }
                                               //       // end of outer loopnest kernel
                                               //   }
                                               // }

                                               std::vector<loopnests::Index> fillIndices;
                                               fillIndices.reserve(inputRegionFillShape.NumDimensions());
                                               for (int idx = 0; idx < inputRegionFillShape.NumDimensions(); ++idx)
                                               {
                                                   fillIndices.push_back(loopnests::Index("fillIdx_" + std::to_string(idx)));
                                               }

                                               // Define LoopNest
                                               auto fillNest = Using({ offsetInputArrayView }, ArgumentType::Input)
                                                                   .Using({ cacheFillArrayView }, ArgumentType::Output);
                                               for (int idx = 0; idx < inputRegionFillShape.NumDimensions(); ++idx)
                                               {
                                                   fillNest.ForAll(fillIndices[idx], 0, inputRegionFillShape.GetActiveSize(idx));
                                               }

                                               const int VectorizationSize = registerCharacteristics.NumberOfElementsPerSIMDRegister;
                                               int maximumElementsInTempBuf = registerCharacteristics.NumberOfSIMDRegisters * VectorizationSize;
                                               std::vector<int> indexSplitSizes(fillIndices.size());
                                               std::vector<int> tmpBufDimensionMapping(indexSplitSizes.size());

                                               // Handle the innermost input dimension differently since we'll be counting elements there instead of shards of a memory layout
                                               int shardSize = VectorizationSize;
                                               int totalElementsPerShard = VectorizationSize;
                                               for (unsigned idx = fillIndices.size() - 1; fillIndices.size() > idx; --idx)
                                               {
                                                   int availableShardsInTmpBuf = maximumElementsInTempBuf / totalElementsPerShard;
                                                   int inputDimAvailableShards = inputRegionFillShape.GetActiveSize(idx) / shardSize;
                                                   int numShards = std::min(availableShardsInTmpBuf, inputDimAvailableShards);
                                                   tmpBufDimensionMapping[idx] = inputRegionFillShape.GetLogicalDimension(idx);
                                                   if (numShards > 1)
                                                   {
                                                       indexSplitSizes[idx] = numShards * shardSize;
                                                       shardSize = 1; // After the initial vectorization size, we target units of entire memory layout shards
                                                       totalElementsPerShard *= numShards; // The number of elements represented by a target scales with the number of inner targets it represents
                                                   }
                                                   else
                                                   {
                                                       indexSplitSizes[idx] = 1;
                                                   }
                                               }
                                               // The index split sizes are measured in input-space, so no scaling is needed
                                               std::vector<int> tmpBufScaleFactors(indexSplitSizes.size(), 1);

                                               BoundaryConditionMemoryLayoutHelper fillKernelBoundaryHelper(inputRegionFillShape.GetActiveSize(),
                                                                                                            indexSplitSizes,
                                                                                                            tmpBufDimensionMapping,
                                                                                                            tmpBufScaleFactors,
                                                                                                            0, // Fill index doesn't matter for this usage
                                                                                                            tmpBufDimensionMapping.size()); // Shrink any index split sizes needed since we don't have a "view" to worry about

                                               auto cacheFillInternalKernel = loopnests::Kernel("Internal_Fill_Cache_Outer_Kernel")
                                                                                  .Inputs(offsetInputArrayView, cacheFillArrayView)
                                                                                  .Indices(fillIndices)
                                                                                  .DefineEx([=](std::vector<Value> values, std::vector<Scalar> innerIndices) {
                                                                                      Array offsetInput = values[0];
                                                                                      Array cacheFillView = values[1];

                                                                                      Value offsetInputInnerVal = offsetInput.GetValue().Offset(innerIndices);
                                                                                      offsetInputInnerVal.SetLayout(offsetInput.GetValue().GetLayout());
                                                                                      Array offsetInputInner = offsetInputInnerVal;

                                                                                      std::vector<Scalar> cacheIndices;
                                                                                      cacheIndices.reserve(boundaryCacheFillLayout.NumDimensions());
                                                                                      for (int cacheDimIdx = 0; cacheDimIdx < boundaryCacheFillLayout.NumDimensions(); ++cacheDimIdx)
                                                                                      {
                                                                                          unsigned baseDimIdx = cacheFillThresholdIdx + cacheDimIdx;
                                                                                          int logicalDimension = logicalDimensionMapping[baseDimIdx];
                                                                                          // Mapping loopnest indices (input space) -> cache indices (cache space) so divide by split index increment
                                                                                          cacheIndices.push_back((innerIndices[logicalDimension] / orderedIndexIncrements[baseDimIdx]) % boundaryCacheFillLayout.GetActiveSize(cacheDimIdx));
                                                                                      }
                                                                                      Value offsetCacheInnerVal = cacheFillView.GetValue().Offset(cacheIndices);
                                                                                      offsetCacheInnerVal.SetLayout(cacheFillView.GetValue().GetLayout());
                                                                                      Array offsetCacheInner = offsetCacheInnerVal;

                                                                                      fillKernelBoundaryHelper.EmitBoundarySwitches(innerIndices, [=](MemoryLayout fillRegionShape, MemoryLayout, MemoryLayout boundaryTempBufLayout, MemoryLayout) {
                                                                                          Array tmpBuf = Allocate(offsetInput.Type(), boundaryTempBufLayout, bufferAlignment);

                                                                                          std::vector<loopnests::Index> tmpBufInputIndices;

                                                                                          tmpBufInputIndices.reserve(fillRegionShape.NumDimensions());
                                                                                          for (int idx = 0; idx < fillRegionShape.NumDimensions(); ++idx)
                                                                                          {
                                                                                              tmpBufInputIndices.push_back(loopnests::Index("tmpBuf_FillIdx_" + std::to_string(idx)));
                                                                                          }

                                                                                          auto tmpBufFillNest = Using({ offsetInputInner }, ArgumentType::Input)
                                                                                                                    .Using({ tmpBuf }, ArgumentType::Output);
                                                                                          for (int idx = 0; idx < fillRegionShape.NumDimensions(); ++idx)
                                                                                          {
                                                                                              tmpBufFillNest.ForAll(tmpBufInputIndices[idx], 0, fillRegionShape.GetActiveSize(idx));
                                                                                          }

                                                                                          auto tmpBufFill = loopnests::Kernel("Internal_TmpBuf_FillTmpBuf_Kernel")
                                                                                                                .Inputs(offsetInputInner, tmpBuf)
                                                                                                                .Indices(tmpBufInputIndices)
                                                                                                                .DefineEx([=](std::vector<Value> tmpBufValues, std::vector<Scalar> tmpBufInputIndices) {
                                                                                                                    Array offsetInputInner = tmpBufValues[0];
                                                                                                                    Array tmpBuf = tmpBufValues[1];

                                                                                                                    tmpBuf(tmpBufInputIndices) = offsetInputInner(tmpBufInputIndices);
                                                                                                                });
                                                                                          tmpBufFillNest.Do(tmpBufFill);
                                                                                          auto& tmpBufFillSchedule = tmpBufFillNest.GetSchedule();
                                                                                          // unroll everything
                                                                                          for (unsigned idx = 0; idx < tmpBufInputIndices.size(); ++idx)
                                                                                          {
                                                                                              tmpBufFillSchedule.Unroll(tmpBufInputIndices[idx]);
                                                                                          }
                                                                                          tmpBufFillNest.Run();

                                                                                          // Cache fill from tmp buf
                                                                                          auto cacheFillNest = Using({ tmpBuf }, ArgumentType::Input)
                                                                                                                   .Using({ offsetCacheInner }, ArgumentType::Output);
                                                                                          for (int idx = 0; idx < tmpBuf.GetValue().GetLayout().NumDimensions(); ++idx)
                                                                                          {
                                                                                              cacheFillNest.ForAll(tmpBufInputIndices[idx], 0, tmpBuf.GetValue().GetLayout().GetActiveSize(idx));
                                                                                          }

                                                                                          auto cacheFill = loopnests::Kernel("Internal_TmpBuf_FillCache_Kernel")
                                                                                                               .Inputs(tmpBuf, offsetCacheInner)
                                                                                                               .Indices(tmpBufInputIndices)
                                                                                                               .DefineEx([=](std::vector<Value> tmpBufValues, std::vector<Scalar> tmpBufIndices) {
                                                                                                                   Array tmpBuf = tmpBufValues[0];
                                                                                                                   Array offsetCacheInner = tmpBufValues[1];

                                                                                                                   int cacheDimensions = offsetCacheInner.GetValue().GetLayout().NumDimensions();
                                                                                                                   std::vector<Scalar> cacheIndices;
                                                                                                                   cacheIndices.reserve(cacheDimensions);
                                                                                                                   for (int cacheDimIdx = 0; cacheDimIdx < cacheDimensions; ++cacheDimIdx)
                                                                                                                   {
                                                                                                                       unsigned baseDimIdx = cacheFillThresholdIdx + cacheDimIdx;
                                                                                                                       int logicalDimension = logicalDimensionMapping[baseDimIdx];
                                                                                                                       // Mapping loopnest indices (input space) -> cache indices (cache space) so divide by split index increment
                                                                                                                       cacheIndices.push_back((tmpBufIndices[logicalDimension] / orderedIndexIncrements[baseDimIdx]) % boundaryCacheFillLayout.GetActiveSize(cacheDimIdx));
                                                                                                                   }
                                                                                                                   offsetCacheInner(cacheIndices) = tmpBuf(tmpBufIndices);
                                                                                                               });
                                                                                          cacheFillNest.Do(cacheFill);
                                                                                          auto& cacheFillSchedule = cacheFillNest.GetSchedule();
                                                                                          for (unsigned idx = 0; idx < tmpBufInputIndices.size(); ++idx)
                                                                                          {
                                                                                              cacheFillSchedule.Unroll(tmpBufInputIndices[idx]);
                                                                                          }
                                                                                          cacheFillNest.Run();
                                                                                      });
                                                                                  });

                                               auto& schedule = fillNest.GetSchedule();
                                               std::vector<loopnests::Index> splitOuterIndices;
                                               for (unsigned idx = 0; idx < fillIndices.size(); ++idx)
                                               {
                                                   if (indexSplitSizes[idx] > 1)
                                                   {
                                                       splitOuterIndices.push_back(schedule.Split(fillIndices[idx], indexSplitSizes[idx]));
                                                   }
                                                   else
                                                   {
                                                       splitOuterIndices.push_back(fillIndices[idx]);
                                                   }
                                               }

                                               fillNest.Do(cacheFillInternalKernel, splitOuterIndices);

                                               fillNest.Run();
                                           });
                                       });

            underlyingNest.AddKernel(cacheFillKernel, loopnests::CodePositionConstraints{ loopnests::LoopFragmentType::prologue, cacheFillPosition, {} });
            cachingKernels.push_back(cacheFillKernel);
        }

        if (useViewKernel)
        {
            // The cache view indices are all of the indices that occur before the cacheViewThresholdIdx
            std::vector<Index> cacheViewPosition(orderedIndices.begin(), orderedIndices.begin() + cacheViewThresholdIdx);
            std::vector<Index> cacheViewIndices(_kernelIndices.begin(), _kernelIndices.end());
            cacheViewIndices.insert(cacheViewIndices.end(), cacheViewPosition.begin(), cacheViewPosition.end());

            auto cacheViewKernel = loopnests::Kernel(cacheName + "_View_Cache_Kernel")
                                       .Inputs(_rawCache, cacheRef)
                                       .Indices(cacheViewIndices)
                                       .DefineEx([boundaryConditionCacheHelper, compositeIndexCount, fullInputLayout, cacheLayout, baseCacheViewLayout, cacheLogicalDimensionMapping, logicalDimensionMapping, orderedIndices, orderedIndexIncrements, cacheThresholdIdx, cacheViewThresholdIdx, logicalDimensionCount](std::vector<Value> values, std::vector<Scalar> indices) {
                                           auto& cache = values[0];
                                           auto& cacheRef = values[1];
                                           std::vector<Scalar> compositeIndexValues(indices.begin(), indices.begin() + compositeIndexCount);
                                           std::vector<Scalar> splitIndexValues(indices.begin() + compositeIndexCount, indices.end());

                                           boundaryConditionCacheHelper.EmitBoundarySwitches(compositeIndexValues, [&](MemoryLayout inputRegionShape, MemoryLayout inputRegionFillShape, MemoryLayout boundaryCacheLayout, MemoryLayout boundaryCacheFillLayout) {
                                               // Find the view slice in the cache for this offset
                                               // The indices in [cacheThresoldIdx, cacheViewThresholdIdx) in the indices determine which slice to use
                                               std::vector<Scalar> cacheOffsetIndices;
                                               cacheOffsetIndices.reserve(cacheLayout.NumDimensions());

                                               // Note: if cacheThresholdIdx == cacheViewThresholdIdx (i.e. if there is no repeated re-viewing of the cache)
                                               // Then the first loop is skipped and no offsetting occurs
                                               auto cacheView = cache;
                                               for (unsigned idx = cacheThresholdIdx; idx < cacheViewThresholdIdx; ++idx)
                                               {
                                                   // Mapping loopnest indices (input space) -> cache offsets (cache space) so divide by split index increment
                                                   cacheOffsetIndices.push_back(splitIndexValues[idx] / orderedIndexIncrements[idx]);
                                               }
                                               for (unsigned idx = cacheViewThresholdIdx; idx < static_cast<unsigned>(fullInputLayout.NumDimensions()); ++idx)
                                               {
                                                   cacheOffsetIndices.push_back(Scalar{ 0 });
                                               }

                                               cacheView.SetLayout(boundaryCacheLayout);
                                               auto offsetCache = cacheView.Offset(cacheOffsetIndices);
                                               offsetCache.SetLayout(baseCacheViewLayout);

                                               // Offset the cache ref from the base cache such that indexing with the current loop values
                                               // would offset a pointer to the beginning of this view of the cache
                                               std::vector<Scalar> offsetIndices(logicalDimensionCount);
                                               for (int idx = 0; idx < logicalDimensionCount; ++idx)
                                               {
                                                   offsetIndices[idx] -= compositeIndexValues[idx];
                                               }

                                               auto offsetCacheView = offsetCache.Offset(offsetIndices);
                                               offsetCacheView.SetLayout(baseCacheViewLayout);
                                               cacheRef.SetLayout(baseCacheViewLayout);
                                               cacheRef = offsetCacheView.Reference();
                                           });
                                       });

            underlyingNest.AddKernel(cacheViewKernel, loopnests::CodePositionConstraints{ loopnests::LoopFragmentType::prologue, cacheViewPosition, {} });
            cachingKernels.push_back(cacheViewKernel);
        }

        if (useReduceKernel)
        {
            // The cache reduce indices are all of the indices that occur before the cacheThresholdIdx
            // Because the reduce is symmetric with the cache non-progressive fill / flush level of a loop nest
            std::vector<Index> cacheReducePosition(orderedIndices.begin(), orderedIndices.begin() + cacheThresholdIdx);
            std::vector<Index> cacheReduceIndices(_kernelIndices.begin(), _kernelIndices.end());
            cacheReduceIndices.insert(cacheReduceIndices.end(), cacheReducePosition.begin(), cacheReducePosition.end());

            auto cacheReduceKernel = loopnests::Kernel(cacheName + "_Reduce_Kernel")
                                         .Inputs(_value, _rawCache)
                                         .Indices(cacheReduceIndices)
                                         .DefineEx([=](std::vector<Value> values, std::vector<Scalar> indices) {
                                             auto& input = values[0];
                                             auto& cache = values[1];
                                             std::vector<Scalar> compositeIndexValues(indices.begin(), indices.begin() + compositeIndexCount);
                                             std::vector<Scalar> splitIndexValues(indices.begin() + compositeIndexCount, indices.end());

                                             auto offsetInput = input.Offset(compositeIndexValues);
                                             offsetInput.SetLayout(input.GetLayout());
                                             auto offsetInputArrayView = Array(offsetInput);

                                             boundaryConditionCacheHelper.EmitBoundarySwitches(compositeIndexValues, [=](MemoryLayout inputRegionShape, MemoryLayout, MemoryLayout boundaryCacheLayout, MemoryLayout) {
                                                 auto cacheArrayView = Array(cache);

                                                 // Prefer input-oriented loops to maximize locality as the input
                                                 // is likely to be larger than the cache in most cases
                                                 // Based on the element size and counts in different dimensions,
                                                 // we will split and unroll some of the inner loops in order to maximize
                                                 // vectorization.
                                                 // In order to get appropriate utilization of all the SIMD
                                                 // registers, we will need to use a temporary buffer (which we expect
                                                 // the compiler to optimize away) with a size equal to the total number
                                                 // of elements that can be held in all of the SIMD registers.
                                                 // The filling of this temporary buffer from the cache needs to be an
                                                 // unrolled operation and the reducing of the output from the temporary
                                                 // buffer also needs to be an unrolled operation that happens after
                                                 // the full temporary buffer has been filled.
                                                 // If the reduce operation is a SumReduce operation, then we need
                                                 // a third loop in the middle which accumulates the current value
                                                 // from the output into the temporary buffer, then have the
                                                 // third loop copy the temporary buffer to the output
                                                 // Therefore, we need multiple levels of loopnests so that the area
                                                 // outside of the temporary buffer's addressable region can be looped
                                                 // over, and the area inside the temporary buffer region can have two
                                                 // or three sequential fully unrolled loopnests.
                                                 // new loopnest (outer):
                                                 // For ... {
                                                 //   For ... {
                                                 //       // start of outer loopnest prologue kernel
                                                 //       // Fill temp buf with cache data
                                                 //       new loopnest (inner #1):
                                                 //       For ... (unroll) {
                                                 //           For ... (unroll) {
                                                 //               ... {
                                                 //                   // start of inner loopnest #1 kernel
                                                 //                   tempBuf(tempBufIndices) = cache(cacheIndices)
                                                 //                   // end of inner loopnest #1 kernel
                                                 //               }
                                                 //               ...
                                                 //           }
                                                 //       }
                                                 //       // if reduceFunction == SumReduce
                                                 //       // Apply the reduce function to reduce elements of the output into the temp buf
                                                 //       new loopnest (inner #2):
                                                 //       For ... (unroll) {
                                                 //           For ... (unroll) {
                                                 //               ... {
                                                 //                   // start of inner loopnest #2 kernel
                                                 //                   tempBuf(tempBufIndices) += input(inputIndices)
                                                 //                   // end of inner loopnest #2 kernel
                                                 //               }
                                                 //               ...
                                                 //           }
                                                 //       }
                                                 //       // Copy temp buf to output
                                                 //       new loopnest (inner #3):
                                                 //       For ... (unroll) {
                                                 //           For ... (unroll) {
                                                 //               ... {
                                                 //                   // start of inner loopnest #3 kernel
                                                 //                   input(inputIndices) = tempBuf(tempBufIndices)
                                                 //                   // end of inner loopnest #3 kernel
                                                 //               }
                                                 //               ...
                                                 //           }
                                                 //       }
                                                 //       // end of outer loopnest kernel
                                                 //   }
                                                 // }

                                                 std::vector<loopnests::Index> reduceIndices;
                                                 reduceIndices.reserve(inputRegionShape.NumDimensions());
                                                 for (int idx = 0; idx < inputRegionShape.NumDimensions(); ++idx)
                                                 {
                                                     reduceIndices.push_back(loopnests::Index("reduceIdx_" + std::to_string(idx)));
                                                 }

                                                 // Define LoopNest
                                                 auto reduceNest = Using({ offsetInputArrayView }, ArgumentType::Input)
                                                                       .Using({ cacheArrayView }, ArgumentType::Output);
                                                 for (int idx = 0; idx < inputRegionShape.NumDimensions(); ++idx)
                                                 {
                                                     reduceNest.ForAll(reduceIndices[idx], 0, inputRegionShape.GetActiveSize(idx));
                                                 }

                                                 const int VectorizationSize = registerCharacteristics.NumberOfElementsPerSIMDRegister;
                                                 int maximumElementsInTempBuf = registerCharacteristics.NumberOfSIMDRegisters * VectorizationSize;
                                                 std::vector<int> indexSplitSizes(reduceIndices.size());
                                                 std::vector<int> tmpBufDimensionMapping(indexSplitSizes.size());

                                                 // Handle the innermost input dimension differently since we'll be counting elements there instead of shards of a memory layout
                                                 int shardSize = VectorizationSize;
                                                 int totalElementsPerShard = VectorizationSize;
                                                 for (unsigned idx = reduceIndices.size() - 1; reduceIndices.size() > idx; --idx)
                                                 {
                                                     int availableShardsInTmpBuf = maximumElementsInTempBuf / totalElementsPerShard;
                                                     int inputDimAvailableShards = inputRegionShape.GetActiveSize(idx) / shardSize;
                                                     int numShards = std::min(availableShardsInTmpBuf, inputDimAvailableShards);
                                                     tmpBufDimensionMapping[idx] = inputRegionShape.GetLogicalDimension(idx);
                                                     if (numShards > 1)
                                                     {
                                                         indexSplitSizes[idx] = numShards * shardSize;
                                                         shardSize = 1; // After the initial vectorization size, we target units of entire memory layout shards
                                                         totalElementsPerShard *= numShards; // The number of elements represented by a target scales with the number of inner targets it represents
                                                     }
                                                     else
                                                     {
                                                         indexSplitSizes[idx] = 1;
                                                     }
                                                 }
                                                 // The index split sizes are measured in input-space, so no scaling is needed
                                                 std::vector<int> tmpBufScaleFactors(indexSplitSizes.size(), 1);

                                                 BoundaryConditionMemoryLayoutHelper reduceKernelBoundaryHelper(inputRegionShape.GetActiveSize(),
                                                                                                                indexSplitSizes,
                                                                                                                tmpBufDimensionMapping,
                                                                                                                tmpBufScaleFactors,
                                                                                                                0, // Fill index doesn't matter for this usage
                                                                                                                tmpBufDimensionMapping.size()); // Shrink any index split sizes needed since we don't have a "view" to worry about

                                                 auto cacheReduceInternalKernel = loopnests::Kernel("Internal_Reduce_Cache_Outer_Kernel")
                                                                                      .Inputs(offsetInputArrayView, cacheArrayView)
                                                                                      .Indices(reduceIndices)
                                                                                      .DefineEx([=](std::vector<Value> values, std::vector<Scalar> innerIndices) {
                                                                                          Array offsetInput = values[0];
                                                                                          Array cacheView = values[1];

                                                                                          Value offsetInputInnerVal = offsetInput.GetValue().Offset(innerIndices);
                                                                                          offsetInputInnerVal.SetLayout(offsetInput.GetValue().GetLayout());
                                                                                          Array offsetInputInner = offsetInputInnerVal;

                                                                                          std::vector<Scalar> cacheIndices;
                                                                                          cacheIndices.reserve(boundaryCacheLayout.NumDimensions());
                                                                                          for (int cacheDimIdx = 0; cacheDimIdx < boundaryCacheLayout.NumDimensions(); ++cacheDimIdx)
                                                                                          {
                                                                                              unsigned baseDimIdx = cacheThresholdIdx + cacheDimIdx;
                                                                                              int logicalDimension = logicalDimensionMapping[baseDimIdx];
                                                                                              // Mapping loopnest indices (input space) -> cache indices (cache space) so divide by split index increment
                                                                                              cacheIndices.push_back((innerIndices[logicalDimension] / orderedIndexIncrements[baseDimIdx]) % boundaryCacheLayout.GetActiveSize(cacheDimIdx));
                                                                                          }
                                                                                          Value offsetCacheInnerVal = cacheView.GetValue().Offset(cacheIndices);
                                                                                          offsetCacheInnerVal.SetLayout(cacheView.GetValue().GetLayout());
                                                                                          Array offsetCacheInner = offsetCacheInnerVal;

                                                                                          reduceKernelBoundaryHelper.EmitBoundarySwitches(innerIndices, [=](MemoryLayout reduceRegionShape, MemoryLayout, MemoryLayout boundaryTempBufLayout, MemoryLayout) {
                                                                                              Array tmpBuf = Allocate(offsetInput.Type(), boundaryTempBufLayout, bufferAlignment);

                                                                                              std::vector<loopnests::Index> tmpBufInputIndices;

                                                                                              tmpBufInputIndices.reserve(reduceRegionShape.NumDimensions());
                                                                                              for (int idx = 0; idx < reduceRegionShape.NumDimensions(); ++idx)
                                                                                              {
                                                                                                  tmpBufInputIndices.push_back(loopnests::Index("tmpBuf_ReduceIdx_" + std::to_string(idx)));
                                                                                              }

                                                                                              auto tmpBufFillFromCacheNest = Using({ offsetCacheInner }, ArgumentType::Input)
                                                                                                                                 .Using({ tmpBuf }, ArgumentType::Output);
                                                                                              for (int idx = 0; idx < reduceRegionShape.NumDimensions(); ++idx)
                                                                                              {
                                                                                                  tmpBufFillFromCacheNest.ForAll(tmpBufInputIndices[idx], 0, reduceRegionShape.GetActiveSize(idx));
                                                                                              }

                                                                                              // Fill tmp buf from cache
                                                                                              auto tmpBufFillFromCache = loopnests::Kernel("Internal_TmpBuf_FillTmpBuf_Kernel")
                                                                                                                             .Inputs(offsetCacheInner, tmpBuf)
                                                                                                                             .Indices(tmpBufInputIndices)
                                                                                                                             .DefineEx([=](std::vector<Value> tmpBufValues, std::vector<Scalar> tmpBufInputIndices) {
                                                                                                                                 Array offsetCacheInner = tmpBufValues[0];
                                                                                                                                 Array tmpBuf = tmpBufValues[1];

                                                                                                                                 int cacheDimensions = offsetCacheInner.GetValue().GetLayout().NumDimensions();
                                                                                                                                 std::vector<Scalar> cacheIndices;
                                                                                                                                 cacheIndices.reserve(cacheDimensions);
                                                                                                                                 for (int cacheDimIdx = 0; cacheDimIdx < cacheDimensions; ++cacheDimIdx)
                                                                                                                                 {
                                                                                                                                     unsigned baseDimIdx = cacheFillThresholdIdx + cacheDimIdx;
                                                                                                                                     int logicalDimension = logicalDimensionMapping[baseDimIdx];
                                                                                                                                     // Mapping loopnest indices (input space) -> cache indices (cache space) so divide by split index increment
                                                                                                                                     cacheIndices.push_back((tmpBufInputIndices[logicalDimension] / orderedIndexIncrements[baseDimIdx]) % boundaryCacheLayout.GetActiveSize(cacheDimIdx));
                                                                                                                                 }
                                                                                                                                 tmpBuf(tmpBufInputIndices) = offsetCacheInner(cacheIndices);
                                                                                                                             });
                                                                                              tmpBufFillFromCacheNest.Do(tmpBufFillFromCache);
                                                                                              auto& tmpBufFillSchedule = tmpBufFillFromCacheNest.GetSchedule();
                                                                                              // unroll everything
                                                                                              for (unsigned idx = 0; idx < tmpBufInputIndices.size(); ++idx)
                                                                                              {
                                                                                                  tmpBufFillSchedule.Unroll(tmpBufInputIndices[idx]);
                                                                                              }
                                                                                              tmpBufFillFromCacheNest.Run();

                                                                                              if (accumulateReduce)
                                                                                              {
                                                                                                  // Reduce the current input/output contents into the temp buffer
                                                                                                  auto tmpBufReduceNest = Using({ offsetInputInner }, ArgumentType::Input)
                                                                                                                              .Using({ tmpBuf }, ArgumentType::Output);
                                                                                                  for (int idx = 0; idx < tmpBuf.GetValue().GetLayout().NumDimensions(); ++idx)
                                                                                                  {
                                                                                                      tmpBufReduceNest.ForAll(tmpBufInputIndices[idx], 0, tmpBuf.GetValue().GetLayout().GetActiveSize(idx));
                                                                                                  }

                                                                                                  auto tmpBufReduce = loopnests::Kernel("Internal_TmpBuf_ReduceOutput_Kernel")
                                                                                                                          .Inputs(tmpBuf, offsetInputInner)
                                                                                                                          .Indices(tmpBufInputIndices)
                                                                                                                          .DefineEx([=](std::vector<Value> tmpBufValues, std::vector<Scalar> tmpBufInputIndices) {
                                                                                                                              Array tmpBuf = tmpBufValues[0];
                                                                                                                              Array offsetInputInner = tmpBufValues[1];

                                                                                                                              reduceFunction(tmpBuf(tmpBufInputIndices), offsetInputInner(tmpBufInputIndices));
                                                                                                                          });
                                                                                                  tmpBufReduceNest.Do(tmpBufReduce);
                                                                                                  auto& tmpBufReduceSchedule = tmpBufReduceNest.GetSchedule();
                                                                                                  for (unsigned idx = 0; idx < tmpBufInputIndices.size(); ++idx)
                                                                                                  {
                                                                                                      tmpBufReduceSchedule.Unroll(tmpBufInputIndices[idx]);
                                                                                                  }
                                                                                                  tmpBufReduceNest.Run();

                                                                                                  // Copy temp buffer contents to input/output
                                                                                                  auto storeOutNest = Using({ tmpBuf }, ArgumentType::Input)
                                                                                                                          .Using({ offsetInputInner }, ArgumentType::Output);
                                                                                                  for (int idx = 0; idx < tmpBuf.GetValue().GetLayout().NumDimensions(); ++idx)
                                                                                                  {
                                                                                                      storeOutNest.ForAll(tmpBufInputIndices[idx], 0, tmpBuf.GetValue().GetLayout().GetActiveSize(idx));
                                                                                                  }

                                                                                                  auto storeOut = loopnests::Kernel("Internal_TmpBuf_CopyOutput_Kernel")
                                                                                                                      .Inputs(tmpBuf, offsetInputInner)
                                                                                                                      .Indices(tmpBufInputIndices)
                                                                                                                      .DefineEx([=](std::vector<Value> tmpBufValues, std::vector<Scalar> tmpBufInputIndices) {
                                                                                                                          Array tmpBuf = tmpBufValues[0];
                                                                                                                          Array offsetInputInner = tmpBufValues[1];

                                                                                                                          offsetInputInner(tmpBufInputIndices) = tmpBuf(tmpBufInputIndices);
                                                                                                                      });
                                                                                                  storeOutNest.Do(storeOut);
                                                                                                  auto& storeOutSchedule = storeOutNest.GetSchedule();
                                                                                                  for (unsigned idx = 0; idx < tmpBufInputIndices.size(); ++idx)
                                                                                                  {
                                                                                                      storeOutSchedule.Unroll(tmpBufInputIndices[idx]);
                                                                                                  }
                                                                                                  storeOutNest.Run();
                                                                                              }
                                                                                              else
                                                                                              {
                                                                                                  // Reduce the temp buffer into input/output
                                                                                                  auto outputReduceNest = Using({ tmpBuf }, ArgumentType::Input)
                                                                                                                              .Using({ offsetInputInner }, ArgumentType::Output);
                                                                                                  for (int idx = 0; idx < tmpBuf.GetValue().GetLayout().NumDimensions(); ++idx)
                                                                                                  {
                                                                                                      outputReduceNest.ForAll(tmpBufInputIndices[idx], 0, tmpBuf.GetValue().GetLayout().GetActiveSize(idx));
                                                                                                  }

                                                                                                  auto outputReduce = loopnests::Kernel("Internal_TmpBuf_ReduceOutput_Kernel")
                                                                                                                          .Inputs(tmpBuf, offsetInputInner)
                                                                                                                          .Indices(tmpBufInputIndices)
                                                                                                                          .DefineEx([=](std::vector<Value> tmpBufValues, std::vector<Scalar> tmpBufInputIndices) {
                                                                                                                              Array tmpBuf = tmpBufValues[0];
                                                                                                                              Array offsetInputInner = tmpBufValues[1];

                                                                                                                              reduceFunction(offsetInputInner(tmpBufInputIndices), tmpBuf(tmpBufInputIndices));
                                                                                                                          });
                                                                                                  outputReduceNest.Do(outputReduce);
                                                                                                  auto& outputReduceSchedule = outputReduceNest.GetSchedule();
                                                                                                  for (unsigned idx = 0; idx < tmpBufInputIndices.size(); ++idx)
                                                                                                  {
                                                                                                      outputReduceSchedule.Unroll(tmpBufInputIndices[idx]);
                                                                                                  }
                                                                                                  outputReduceNest.Run();
                                                                                              }
                                                                                          });
                                                                                      });

                                                 auto& schedule = reduceNest.GetSchedule();
                                                 std::vector<loopnests::Index> splitOuterIndices;
                                                 for (unsigned idx = 0; idx < reduceIndices.size(); ++idx)
                                                 {
                                                     if (indexSplitSizes[idx] > 1)
                                                     {
                                                         splitOuterIndices.push_back(schedule.Split(reduceIndices[idx], indexSplitSizes[idx]));
                                                     }
                                                 }

                                                 reduceNest.Do(cacheReduceInternalKernel, splitOuterIndices);

                                                 reduceNest.Run();
                                             });
                                         });

            underlyingNest.AddKernel(cacheReduceKernel, loopnests::CodePositionConstraints{ loopnests::LoopFragmentType::epilogue, cacheReducePosition, {} });
            cachingKernels.push_back(cacheReduceKernel);
        }

        underlyingNest.RenameVariable(_value, cacheRef, _atIndices, cachingKernels);
    }

} // namespace value
} // namespace ell
