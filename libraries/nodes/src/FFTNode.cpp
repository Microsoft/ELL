////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     FFTNode.cpp (nodes)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FFTNode.h"

#include <emitters/include/EmitterException.h>
#include <emitters/include/EmitterTypes.h>
#include <emitters/include/IRLocalValue.h>
#include <emitters/include/IRMath.h>
#include <emitters/include/LLVMUtilities.h>

#include <math/include/MathConstants.h>

#include <dsp/include/FFT.h>

#include <llvm/IR/Type.h>

#include <cmath>

#define USE_STORED_TWIDDLE_FACTORS 1
#define USE_FIXED_SMALL_FFT 1
#define MAX_INLINE_FFT_SIZE 0 // 0 to disable inlining FFT code
#define USE_REAL_FFT 0 // The real-valued FFT is slower than the complex one. Disable it for now.

namespace ell
{
namespace nodes
{
    namespace detail
    {
        //
        // Complex functions for emitters
        //
        template <typename ValueType>
        llvm::StructType* GetComplexType(emitters::IRModuleEmitter& module)
        {
            auto& emitter = module.GetIREmitter();
            auto valueType = emitter.Type(emitters::GetVariableType<ValueType>());
            return module.GetAnonymousStructType({ valueType, valueType }, true);
        }

        inline llvm::StructType* GetComplexType(emitters::IRModuleEmitter& module, emitters::LLVMType valueType)
        {
            return module.GetAnonymousStructType({ valueType, valueType }, true);
        }

        inline emitters::IRLocalValue ComplexAdd(emitters::IRLocalValue a, emitters::IRLocalValue b)
        {
            if (!(a.value->getType()->isStructTy() && a.value->getType()->getNumContainedTypes() == 2 && a.value->getType() == b.value->getType()))
            {
                throw emitters::EmitterException(emitters::EmitterError::valueTypeNotSupported);
            }

            // c.re = a.re + b.re
            // c.im = a.im + b.im
            auto& function = a.function;
            auto a_re = function.LocalScalar(function.ExtractStructField(a, 0));
            auto a_im = function.LocalScalar(function.ExtractStructField(a, 1));
            auto b_re = function.LocalScalar(function.ExtractStructField(b, 0));
            auto b_im = function.LocalScalar(function.ExtractStructField(b, 1));
            auto result = function.Variable(a.value->getType(), "c_add");
            function.FillStruct(result, { a_re + b_re, a_im + b_im });
            return { function, function.Load(result) };
        }

        inline emitters::IRLocalValue RealComplexAdd(emitters::IRLocalScalar a_re, emitters::IRLocalValue b)
        {
            if (!(b.value->getType()->isStructTy() && b.value->getType()->getNumContainedTypes() == 2 && a_re.value->getType() == b.value->getType()->getContainedType(0)))
            {
                throw emitters::EmitterException(emitters::EmitterError::valueTypeNotSupported);
            }

            // c.re = a.re + b.re
            // c.im = b.im
            auto& function = a_re.function;
            auto b_re = function.LocalScalar(function.ExtractStructField(b, 0));
            auto b_im = function.LocalScalar(function.ExtractStructField(b, 1));
            auto result = function.Variable(b.value->getType(), "rc_add");
            function.FillStruct(result, { a_re + b_re, b_im });
            return { function, function.Load(result) };
        }

        inline emitters::IRLocalValue ComplexSubtract(emitters::IRLocalValue a, emitters::IRLocalValue b)
        {
            if (!(a.value->getType()->isStructTy() && a.value->getType()->getNumContainedTypes() == 2 && a.value->getType() == b.value->getType()))
            {
                throw emitters::EmitterException(emitters::EmitterError::valueTypeNotSupported);
            }

            // c.re = a.re - b.re
            // c.im = a.im - b.im
            auto& function = a.function;
            auto a_re = function.LocalScalar(function.ExtractStructField(a, 0));
            auto a_im = function.LocalScalar(function.ExtractStructField(a, 1));
            auto b_re = function.LocalScalar(function.ExtractStructField(b, 0));
            auto b_im = function.LocalScalar(function.ExtractStructField(b, 1));
            auto result = function.Variable(a.value->getType(), "c_sub");
            function.FillStruct(result, { a_re - b_re, a_im - b_im });
            return { function, function.Load(result) };
        }

        inline emitters::IRLocalValue RealComplexSubtract(emitters::IRLocalScalar a_re, emitters::IRLocalValue b)
        {
            if (!(b.value->getType()->isStructTy() && b.value->getType()->getNumContainedTypes() == 2 && a_re.value->getType() == b.value->getType()->getContainedType(0)))
            {
                throw emitters::EmitterException(emitters::EmitterError::valueTypeNotSupported);
            }

            // c.re = a.re - b.re
            // c.im = b.im
            auto& function = a_re.function;
            auto b_re = function.LocalScalar(function.ExtractStructField(b, 0));
            auto b_im = function.LocalScalar(function.ExtractStructField(b, 1));
            auto result = function.Variable(b.value->getType(), "rc_sub");
            function.FillStruct(result, { a_re - b_re, b_im });
            return { function, function.Load(result) };
        }

        inline emitters::IRLocalValue ComplexMultiply(emitters::IRLocalValue a, emitters::IRLocalValue b)
        {
            if (!(a.value->getType()->isStructTy() && a.value->getType()->getNumContainedTypes() == 2 && a.value->getType() == b.value->getType()))
            {
                throw emitters::EmitterException(emitters::EmitterError::valueTypeNotSupported);
            }

            // c.re = a.re * b.re - a.im * b.im
            // c.im = a.im * b.re + a.re * b.im
            auto& function = a.function;
            auto a_re = function.LocalScalar(function.ExtractStructField(a, 0));
            auto a_im = function.LocalScalar(function.ExtractStructField(a, 1));
            auto b_re = function.LocalScalar(function.ExtractStructField(b, 0));
            auto b_im = function.LocalScalar(function.ExtractStructField(b, 1));
            auto result = function.Variable(a.value->getType(), "c_mul");
            function.FillStruct(result, { (a_re * b_re) - (a_im * b_im), (a_im * b_re) + (a_re * b_im) });
            return { function, function.Load(result) };
        }

        template <typename ValueType>
        inline emitters::IRLocalValue TimesI(emitters::IRLocalValue a)
        {
            if (!(a.value->getType()->isStructTy() && a.value->getType()->getNumContainedTypes() == 2))
            {
                throw emitters::EmitterException(emitters::EmitterError::valueTypeNotSupported);
            }

            // c.re = -a.im
            // c.im = i*a.re
            auto& function = a.function;
            auto a_re = function.LocalScalar(function.ExtractStructField(a, 0));
            auto a_im = function.LocalScalar(function.ExtractStructField(a, 1));
            auto result = function.Variable(a.value->getType(), "c_times_i");
            function.FillStruct(result, { -a_im, a_re });
            return { function, function.Load(result) };
        }

        inline emitters::IRLocalValue ComplexRealMultiply(emitters::IRLocalValue a, emitters::IRLocalScalar b_re)
        {
            if (!(a.value->getType()->isStructTy() && a.value->getType()->getNumContainedTypes() == 2 && a.value->getType()->getContainedType(0) == b_re.value->getType()))
            {
                throw emitters::EmitterException(emitters::EmitterError::valueTypeNotSupported);
            }

            // c.re = a.re * b
            // c.im = a.im * b
            auto& function = a.function;
            auto a_re = function.LocalScalar(function.ExtractStructField(a, 0));
            auto a_im = function.LocalScalar(function.ExtractStructField(a, 1));
            auto result = function.Variable(a.value->getType(), "rc_mul");
            function.FillStruct(result, { (a_re * b_re), (a_im * b_re) });
            return { function, function.Load(result) };
        }

        inline emitters::IRLocalValue ComplexAbs(emitters::IRLocalValue a)
        {
            if (!(a.value->getType()->isStructTy() && a.value->getType()->getNumContainedTypes() == 2))
            {
                throw emitters::EmitterException(emitters::EmitterError::valueTypeNotSupported);
            }
            auto& function = a.function;
            auto a_re = function.LocalScalar(function.ExtractStructField(a, 0));
            auto a_im = function.LocalScalar(function.ExtractStructField(a, 1));
            // result = sqrt(real^2 + imag^2)
            // Note, this is more accurately computed as std::hypot(std::real(z), std::imag(z))
            auto magSquared = (a_re * a_re) + (a_im * a_im);
            return { function, Sqrt(magSquared) };
        }

        inline emitters::IRLocalValue ImaginaryExp(emitters::IRLocalScalar x)
        {
            auto& function = x.function;
            auto& module = function.GetModule();
            auto type = x.value->getType();
            auto complexType = GetComplexType(module, type);
            // returns e ^ ix == cos(x) + isin(x)
            auto result = function.Variable(complexType, "i_exp");
            function.FillStruct(result, { Cos(x), Sin(x) });
            return { function, function.Load(result) };
        }

        //
        // FFT-specific functions
        //
        template <typename ValueType>
        std::vector<std::complex<ValueType>> GetTwiddleFactors(size_t size)
        {
            // size == FFT length / 2
            const auto pi = math::Constants<ValueType>::pi;
            std::vector<std::complex<ValueType>> result(size);
            for (size_t k = 0; k < size; ++k)
            {
                result[k] = std::exp(std::complex<ValueType>(0, pi * k / size));
            }
            return result;
        }

        template <typename ValueType>
        std::string GetFFTFunctionName(size_t length)
        {
            // function name: FFTC_<T>_<N>  (e.g., FFTC_float_2)
            // function signature: void FFTC(complex<T>*, complex<T>*)
            return std::string("FFTC_") + utilities::GetTypeName<ValueType>() + "_" + std::to_string(length);
        }

        template <typename ValueType>
        std::string GetRealFFTFunctionName(size_t length)
        {
            // function name: FFTR_<T>_<N>  (e.g., FFT_float_32)
            // function signature: void FFTR(T*, T*, complex<T>*)
            return std::string("FFTR_") + utilities::GetTypeName<ValueType>() + "_" + std::to_string(length);
        }

        template <typename ValueType>
        std::vector<emitters::LLVMType> GetFFTFunctionArguments(emitters::IRModuleEmitter& module)
        {
            auto complexType = detail::GetComplexType<ValueType>(module);
            auto complexPtrType = complexType->getPointerTo();
            return { complexPtrType, complexPtrType };
        }

        template <typename ValueType>
        std::vector<emitters::LLVMType> GetRealFFTFunctionArguments(emitters::IRModuleEmitter& module)
        {
            auto& emitter = module.GetIREmitter();
            auto valueType = emitter.Type(emitters::GetVariableType<ValueType>());
            auto valuePtrType = valueType->getPointerTo();
            auto complexType = detail::GetComplexType(module, valueType);
            auto complexPtrType = complexType->getPointerTo();
            return { valuePtrType, valuePtrType, complexPtrType };
        }

        template <typename ValueType>
        emitters::IRFunctionEmitter GetFFTFunctionEmitter(emitters::IRModuleEmitter& module, size_t length)
        {
            auto& context = module.GetLLVMContext();
            auto voidType = llvm::Type::getVoidTy(context);

            std::string functionName = GetFFTFunctionName<ValueType>(length);
            auto argumentTypes = detail::GetFFTFunctionArguments<ValueType>(module);
            emitters::IRFunctionEmitter function = module.BeginFunction(functionName, voidType, argumentTypes);
            function.SetAttributeForArguments(emitters::IRFunctionEmitter::Attributes::NoAlias);

            return function;
        }

        template <typename ValueType>
        emitters::IRFunctionEmitter GetRealFFTFunctionEmitter(emitters::IRModuleEmitter& module, size_t length)
        {
            auto& context = module.GetLLVMContext();
            auto voidType = llvm::Type::getVoidTy(context);

            std::string functionName = GetRealFFTFunctionName<ValueType>(length);
            auto argumentTypes = detail::GetRealFFTFunctionArguments<ValueType>(module);
            emitters::IRFunctionEmitter function = module.BeginFunction(functionName, voidType, argumentTypes);
            function.SetAttributeForArguments(emitters::IRFunctionEmitter::Attributes::NoAlias);

            return function;
        }

    } // namespace detail

    template <typename ValueType>
    FFTNode<ValueType>::FFTNode() :
        CompilableNode({ &_input }, { &_output }),
        _input(this, {}, defaultInputPortName),
        _output(this, defaultOutputPortName, 0),
        _fftSize(0)
    {
    }

    template <typename ValueType>
    FFTNode<ValueType>::FFTNode(const model::OutputPort<ValueType>& input) :
        CompilableNode({ &_input }, { &_output }),
        _input(this, input, defaultInputPortName),
        _output(this, defaultOutputPortName, 0),
        _fftSize(0)
    {
        double nearestPowerOf2Size = std::pow(2, std::ceil(std::log2(input.Size())));
        _fftSize = static_cast<size_t>(nearestPowerOf2Size);
        _output.SetSize(_fftSize / 2 + 1);
    }

    template <typename ValueType>
    FFTNode<ValueType>::FFTNode(const model::OutputPort<ValueType>& input, size_t fftSize) :
        CompilableNode({ &_input }, { &_output }),
        _input(this, input, defaultInputPortName),
        _output(this, defaultOutputPortName, fftSize / 2 + 1),
        _fftSize(fftSize)
    {
        if (fftSize == 0)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "fftSize must be greater than zero");
        }
        double power = std::log2(fftSize);
        if (std::floor(power) != power)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "fftSize must be a power of 2");
        }
    }

    inline void Deinterleave(emitters::IRFunctionEmitter& function, emitters::LLVMValue array, emitters::LLVMValue halfLength, emitters::LLVMValue scratch)
    {
        auto halfN = function.LocalScalar(halfLength);
        function.For(halfN, [&scratch, &array](emitters::IRFunctionEmitter& function, auto index) {
            auto evenIndex = index * 2;
            auto oddIndex = evenIndex + 1;
            function.SetValueAt(scratch, index, function.ValueAt(array, oddIndex));
            function.SetValueAt(array, index, function.ValueAt(array, evenIndex));
        });

        function.For(halfN, [&scratch, &array, halfN](emitters::IRFunctionEmitter& function, auto index) {
            function.SetValueAt(array, index + halfN, function.ValueAt(scratch, index));
        });
    }

    inline void Deinterleave(emitters::IRFunctionEmitter& function, emitters::LLVMValue array, int halfN, emitters::LLVMValue scratch)
    {
        Deinterleave(function, array, function.Literal(halfN), scratch);
    }

    // FFT1 twiddle factors: [1]
    // FFT2 twiddle factors: [1, -1]
    // FFT4 twiddle factors: [1, i, -1, -i]
    // FFT8 twiddle factors: [1, sqrt2/2+i*sqrt2/2, i, -sqrt2/2+i*sqrt2/2, -1, -sqrt2/2-i*sqrt2/2, -i, sqrt2/2-i*sqrt2/2]

    template <typename ValueType>
    void FFTNode<ValueType>::EmitFFT_2(emitters::IRFunctionEmitter& function, emitters::LLVMValue input)
    {
        // FFT of length 2: x' = [x0+x1, x0-x1]
        auto x0 = function.LocalScalar(function.ValueAt(input, 0));
        auto x1 = function.LocalScalar(function.ValueAt(input, 1));
        function.SetValueAt(input, 0, detail::ComplexAdd(x0, x1));
        function.SetValueAt(input, 1, detail::ComplexSubtract(x0, x1));
    }

    template <typename ValueType>
    emitters::LLVMFunction FFTNode<ValueType>::GetFFTFunction_2(emitters::IRModuleEmitter& module)
    {
        const auto length = 2;
        auto functionName = detail::GetFFTFunctionName<ValueType>(length);
        auto existingFunction = module.GetFunction(functionName);
        if (existingFunction != nullptr)
        {
            return existingFunction;
        }

        emitters::IRFunctionEmitter function = detail::GetFFTFunctionEmitter<ValueType>(module, length);
        {
            auto arguments = function.Arguments().begin();
            auto input = function.LocalScalar(&(*arguments++));
            EmitFFT_2(function, input);
        }
        module.EndFunction();
        return function.GetFunction();
    }

    template <typename ValueType>
    void FFTNode<ValueType>::EmitFFT_4(emitters::IRFunctionEmitter& function, emitters::LLVMValue input)
    {
        // FFT of length 4: X = [x0+x1+x2+x3, x0+ix1-x2-ix3, x0-x1+x2-x3, x0-ix1-x2+ix3]
        // Input x = {x0, x1, x2, x3}
        auto x0 = function.LocalScalar(function.ValueAt(input, 0));
        auto x1 = function.LocalScalar(function.ValueAt(input, 1));
        auto x2 = function.LocalScalar(function.ValueAt(input, 2));
        auto x3 = function.LocalScalar(function.ValueAt(input, 3));

        // deinterleave -> {x0, x2, x1, x3}
        // call fft2 on each half:
        //   -> {x0+x2, x0-x2, x1+x3, x1-x3}
        // apply twiddle factor w_0 = 1:
        //  -> { (x0+x2) + (x1+x3), ..., (x0+x2) - (x1+x3), ...}
        // apply twiddle factor w_1 = i:
        //  -> { (x0+x2) + (x1+x3), (x0-x2) + i(x1-x3), (x0+x2) - (x1+x3), (x0-x2) - i(x1-x3)}

        auto x0px2 = detail::ComplexAdd(x0, x2);
        auto x0mx2 = detail::ComplexSubtract(x0, x2);
        auto x1px3 = detail::ComplexAdd(x1, x3);
        auto x1mx3 = detail::ComplexSubtract(x1, x3);
        function.SetValueAt(input, 0, detail::ComplexAdd(x0px2, x1px3));
        function.SetValueAt(input, 1, detail::ComplexAdd(x0mx2, detail::TimesI<ValueType>(x1mx3)));
        function.SetValueAt(input, 2, detail::ComplexSubtract(x0px2, x1px3));
        function.SetValueAt(input, 3, detail::ComplexSubtract(x0mx2, detail::TimesI<ValueType>(x1mx3)));
    }

    template <typename ValueType>
    emitters::LLVMFunction FFTNode<ValueType>::GetFFTFunction_4(emitters::IRModuleEmitter& module)
    {
        const auto length = 4;
        auto functionName = detail::GetFFTFunctionName<ValueType>(length);
        auto existingFunction = module.GetFunction(functionName);
        if (existingFunction != nullptr)
        {
            return existingFunction;
        }

        emitters::IRFunctionEmitter function = detail::GetFFTFunctionEmitter<ValueType>(module, length);
        {
            auto arguments = function.Arguments().begin();
            auto input = function.LocalScalar(&(*arguments++));

            EmitFFT_4(function, input);
        }
        module.EndFunction();
        return function.GetFunction();
    }

    // Fixed-size FFT function implementation: size is known at compile time
    template <typename ValueType>
    void FFTNode<ValueType>::EmitFFT(emitters::IRFunctionEmitter& function, size_t length, emitters::LLVMValue input, emitters::LLVMValue scratch)
    {
#if (USE_FIXED_SMALL_FFT)
        if (length == 2)
        {
            EmitFFT_2(function, input);
        }
        if (length == 4)
        {
            EmitFFT_4(function, input);
        }
#endif // USE_FIXED_SMALL_FFT

        // TODO: assert(bitcount(length) == 1)  (i.e., length is a power of 2)
        auto& module = function.GetModule();
        auto& emitter = module.GetIREmitter();
        auto valueType = emitter.Type(emitters::GetVariableType<ValueType>());
        auto complexType = detail::GetComplexType(module, valueType);
        auto complexPtrType = complexType->getPointerTo();

        auto halfN = length / 2;
        assert(halfN >= 1);

        Deinterleave(function, input, halfN, scratch);
        auto evens = input;
        auto odds = function.PointerOffset(evens, halfN);

        if (halfN > 1) // call recursive case if necessary
        {
            DoFFT(function, halfN, evens, scratch);
            DoFFT(function, halfN, odds, scratch);
        }

#if (USE_STORED_TWIDDLE_FACTORS)
        auto twiddleFactors = detail::GetTwiddleFactors<ValueType>(halfN);
        std::vector<ValueType> twiddleFactorsUnwrapped(twiddleFactors.size() * 2);
        ValueType* dataPtr = reinterpret_cast<ValueType*>(twiddleFactors.data());
        std::copy(dataPtr, dataPtr + twiddleFactorsUnwrapped.size(), twiddleFactorsUnwrapped.begin());
        auto twiddleFactorsUnwrappedVar = module.ConstantArray(std::string("twiddles_") + std::to_string(halfN), twiddleFactorsUnwrapped); // TODO: encode type name in variable name
        auto twiddleFactorsVar = function.CastPointer(twiddleFactorsUnwrappedVar, complexPtrType);
#else
        bool twiddleFactorsVar = false;
#endif

        function.For(halfN, [&evens, &odds, &twiddleFactorsVar](emitters::IRFunctionEmitter& function, auto k) {

#if (USE_STORED_TWIDDLE_FACTORS)
            auto w = function.LocalScalar(function.ValueAt(twiddleFactorsVar, k));
#else
                // w = e^i(2*pi*k/N)
                auto kValue = function.LocalScalar(function.CastValue<ValueType>(k));
                auto w = detail::ImaginaryExp(function.LocalScalar(pi / halfN) * kValue);
#endif

            auto e = function.LocalScalar(function.ValueAt(evens, k));
            auto o = function.LocalScalar(function.ValueAt(odds, k));
            auto wo = detail::ComplexMultiply(w, o); // wo = w*o
            function.SetValueAt(evens, k, detail::ComplexAdd(e, wo)); // even
            function.SetValueAt(odds, k, detail::ComplexSubtract(e, wo)); // odd
        });
    }

    // Fixed-size FFT function implementation: size is known at compile time
    template <typename ValueType>
    emitters::LLVMFunction FFTNode<ValueType>::GetFFTFunction(emitters::IRModuleEmitter& module, size_t length)
    {
#if (USE_FIXED_SMALL_FFT)
        if (length == 2)
        {
            return GetFFTFunction_2(module);
        }
        if (length == 4)
        {
            return GetFFTFunction_4(module);
        }
#endif // USE_FIXED_SMALL_FFT

        auto functionName = detail::GetFFTFunctionName<ValueType>(length);
        auto existingFunction = module.GetFunction(functionName);
        if (existingFunction != nullptr)
        {
            return existingFunction;
        }

        // TODO: assert(bitcount(length) == 1)  (i.e., length is a power of 2)
        emitters::IRFunctionEmitter function = detail::GetFFTFunctionEmitter<ValueType>(module, length);
        {
            auto arguments = function.Arguments().begin();
            auto input = function.LocalScalar(&(*arguments++));
            auto scratch = function.LocalScalar(&(*arguments++));

            EmitFFT(function, length, input, scratch);
        }
        module.EndFunction();
        return function.GetFunction();
    }

    // Real-valued fixed-size FFT function implementation: size is known at compile time
    template <typename ValueType>
    void FFTNode<ValueType>::EmitRealFFT(emitters::IRFunctionEmitter& function, size_t length, emitters::LLVMValue input, emitters::LLVMValue scratch, emitters::LLVMValue complexInput)
    {
        // TODO: assert(bitcount(length) == 1)  (i.e., length is a power of 2)
        auto& module = function.GetModule();
        auto complexType = detail::GetComplexType<ValueType>(module);
        auto complexPtrType = complexType->getPointerTo();

        auto halfN = length / 2;
        assert(halfN >= 1);

        Deinterleave(function, input, halfN, scratch);
        auto evens = input;
        auto odds = function.PointerOffset(evens, halfN);
        auto complexEvens = complexInput;
        auto complexOdds = function.PointerOffset(complexEvens, halfN);

        if (halfN > 1) // call recursive case if necessary
        {
            DoRealFFT(function, halfN, evens, scratch, complexEvens);
            DoRealFFT(function, halfN, odds, scratch, complexOdds);
        }
        else // here halfN == 1
        {
            // Base case: copy from real to complex
            function.FillStruct(complexEvens, { function.Load(evens), function.Literal<ValueType>(0) });
            function.FillStruct(complexOdds, { function.Load(odds), function.Literal<ValueType>(0) });
        }

#if (USE_STORED_TWIDDLE_FACTORS)
        auto twiddleFactors = detail::GetTwiddleFactors<ValueType>(halfN);
        std::vector<ValueType> twiddleFactorsUnwrapped(twiddleFactors.size() * 2);
        ValueType* dataPtr = reinterpret_cast<ValueType*>(twiddleFactors.data());
        std::copy(dataPtr, dataPtr + twiddleFactorsUnwrapped.size(), twiddleFactorsUnwrapped.begin());
        auto twiddleFactorsUnwrappedVar = module.ConstantArray(std::string("twiddles_") + std::to_string(halfN), twiddleFactorsUnwrapped);
        auto twiddleFactorsVar = function.CastPointer(twiddleFactorsUnwrappedVar, complexPtrType);
#else
        bool twiddleFactorsVar = false; // Just here to appease the compiler
#endif

        function.For(halfN, [&evens, &odds, &complexEvens, complexOdds, &twiddleFactorsVar](emitters::IRFunctionEmitter& function, auto k) {

#if (USE_STORED_TWIDDLE_FACTORS)
            auto w = function.LocalScalar(function.ValueAt(twiddleFactorsVar, k));
#else
                // w = e^i(2*pi*k/N)
                auto kValue = function.LocalScalar(function.CastValue<ValueType>(k));
                auto w = detail::ImaginaryExp(function.LocalScalar(pi / halfN) * kValue);
#endif

            auto e = function.LocalScalar(function.ValueAt(evens, k));
            auto o = function.LocalScalar(function.ValueAt(odds, k));
            auto wo = detail::ComplexRealMultiply(w, o); // wo = w*o
            function.SetValueAt(complexEvens, k, detail::RealComplexAdd(e, wo)); // even
            function.SetValueAt(complexOdds, k, detail::RealComplexSubtract(e, wo)); // odd
        });
    }

    // Real-valued fixed-size FFT function implementation: size is known at compile time
    template <typename ValueType>
    emitters::LLVMFunction FFTNode<ValueType>::GetRealFFTFunction(emitters::IRModuleEmitter& module, size_t length)
    {
        auto functionName = detail::GetRealFFTFunctionName<ValueType>(length);
        auto existingFunction = module.GetFunction(functionName);
        if (existingFunction != nullptr)
        {
            return existingFunction;
        }

        emitters::IRFunctionEmitter function = detail::GetFFTFunctionEmitter<ValueType>(module, length);
        {
            auto arguments = function.Arguments().begin();
            auto input = function.LocalScalar(&(*arguments++));
            auto scratch = function.LocalScalar(&(*arguments++));
            auto complexInput = function.LocalScalar(&(*arguments++));

            EmitRealFFT(function, length, input, scratch, complexInput);
        }
        module.EndFunction();
        return function.GetFunction();
    }

    // Perform fixed-size FFT: size is known at compile time
    template <typename ValueType>
    void FFTNode<ValueType>::DoFFT(emitters::IRFunctionEmitter& function, size_t length, emitters::LLVMValue input, emitters::LLVMValue scratch)
    {
        const bool inlineFFT = length <= MAX_INLINE_FFT_SIZE;
        if (inlineFFT)
        {
            EmitFFT(function, length, input, scratch);
        }
        else
        {
            auto& module = function.GetModule();
            auto fftFunction = GetFFTFunction(module, length);
            function.Call(fftFunction, { input, scratch });
        }
    }

    // Fixed-size FFT function implementation: size is known at compile time
    template <typename ValueType>
    void FFTNode<ValueType>::DoRealFFT(emitters::IRFunctionEmitter& function, size_t length, emitters::LLVMValue input, emitters::LLVMValue scratch, emitters::LLVMValue complexInput)
    {
        const bool inlineFFT = length <= MAX_INLINE_FFT_SIZE;
        if (inlineFFT)
        {
            EmitRealFFT(function, length, input, scratch, complexInput);
        }
        else
        {
            auto& module = function.GetModule();
            auto fftFunction = GetRealFFTFunction(module, length);
            function.Call(fftFunction, { input, scratch, complexInput });
        }
    }

    template <typename ValueType>
    void FFTNode<ValueType>::Compute() const
    {
        std::vector<ValueType> temp = _input.GetValue();
        if (_fftSize != temp.size())
        {
            temp.resize(_fftSize);
        }
        dsp::FFT(temp);
        temp.resize(output.Size());
        _output.SetOutput(temp);
    };

    template <typename ValueType>
    void FFTNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInputs = transformer.GetCorrespondingInputs(_input);
        auto newNode = transformer.AddNode<FFTNode<ValueType>>(newInputs, _fftSize);
        transformer.MapNodeOutput(output, newNode->output);
    }

    template <typename ValueType>
    void FFTNode<ValueType>::Compile(model::IRMapCompiler& compiler, emitters::IRFunctionEmitter& function)
    {
        auto& module = function.GetModule();
        auto& emitter = module.GetIREmitter();
        auto valueType = emitter.Type(emitters::GetVariableType<ValueType>());
        auto complexType = detail::GetComplexType(module, valueType);

        auto inputSize = input.Size();
        auto outputSize = output.Size();

        // Get port variables
        emitters::LLVMValue pInput = compiler.EnsurePortEmitted(input);
        emitters::LLVMValue pOutput = compiler.EnsurePortEmitted(output);

        // Buffer for complex data
        emitters::LLVMValue complexBuffer = function.Variable(complexType, _fftSize);

        if (inputSize > _fftSize)
        {
            // truncate input to _fftSize
            inputSize = _fftSize;
        }

#if (USE_REAL_FFT)

        emitters::LLVMValue scratch = function.Variable(valueType, _fftSize / 2);
        DoRealFFT(function, _fftSize, pInput, scratch, complexBuffer);

#else // Complex-input FFT

        emitters::LLVMValue scratch = function.Variable(complexType, _fftSize / 2);
        emitters::LLVMValue temp = function.Variable(complexType, "temp");

        // Convert real-valued data to complex
        function.For(inputSize, [pInput, complexBuffer, temp](emitters::IRFunctionEmitter& function, auto index) {
            function.FillStruct(temp, { function.ValueAt(pInput, index), function.Literal<ValueType>(0) });
            function.SetValueAt(complexBuffer, index, function.Load(temp));
        });

        if (_fftSize > inputSize)
        {
            // zero-pad up to _fftSize
            emitters::LLVMValue zero_complex = function.Variable(complexType, "zero_complex");
            function.FillStruct(zero_complex, { function.Literal<ValueType>(0), function.Literal<ValueType>(0) });
            function.For(inputSize, _fftSize, [zero_complex, complexBuffer](emitters::IRFunctionEmitter& function, auto index) {
                function.SetValueAt(complexBuffer, index, function.Load(zero_complex));
            });
        }

        DoFFT(function, _fftSize, complexBuffer, scratch);

#endif // USE_REAL_FFT

        function.For(outputSize, [pOutput, complexBuffer](emitters::IRFunctionEmitter& function, auto index) {
            auto complexValue = function.LocalScalar(function.ValueAt(complexBuffer, index));
            auto absValue = detail::ComplexAbs(complexValue);
            function.SetValueAt(pOutput, index, absValue);
        });
    }

    template <typename ValueType>
    void FFTNode<ValueType>::WriteToArchive(utilities::Archiver& archiver) const
    {
        Node::WriteToArchive(archiver);
        archiver[defaultInputPortName] << _input;
        archiver["fftSize"] << _fftSize;
    }

    template <typename ValueType>
    void FFTNode<ValueType>::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        Node::ReadFromArchive(archiver);
        archiver[defaultInputPortName] >> _input;
        archiver["fftSize"] >> _fftSize;
        _output.SetSize(_fftSize / 2 + 1);
    }

    // Explicit instantiations
    template class FFTNode<float>;
    template class FFTNode<double>;
} // namespace nodes
} // namespace ell
