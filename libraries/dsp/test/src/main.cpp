////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     main.cpp (dsp)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ConvolutionTest.h"
#include "DCTTest.h"
#include "DSPTestData.h"
#include "FFTTest.h"
#include "FilterTest.h"
#include "MelTest.h"
#include "WindowTest.h"

#include <dsp/include/Convolution.h>
#include <dsp/include/FFT.h>
#include <dsp/include/IIRFilter.h>
#include <dsp/include/WindowFunctions.h>

#include <utilities/include/Files.h>
#include <utilities/include/Unused.h>

#include <testing/include/testing.h>

#include <complex>
#include <iostream>
#include <vector>

using namespace ell;
using namespace dsp;

void DspTest(const std::string& path)
{
    //
    // Testing
    //

    // 1D Convolution
    TestConv1D<float>(ConvolutionMethodOption::simple);
    TestConv1D<float>(ConvolutionMethodOption::winograd);
    TestConv1DVsSimple<float>(32, 3, ConvolutionMethodOption::winograd);
    TestConv1DVsSimple<float>(33, 3, ConvolutionMethodOption::winograd);

    // 2D Convolution

    // Simple
    TestConv2D<float>(ConvolutionMethodOption::simple);

    // Unrolled
    TestConv2D<float>(ConvolutionMethodOption::unrolled);

    // stride == 1
    TestConv2DVsSimple<float>(4, 4, 8, 3, 16, 1, ConvolutionMethodOption::unrolled);
    TestConv2DVsSimple<float>(6, 6, 8, 3, 16, 1, ConvolutionMethodOption::unrolled);
    TestConv2DVsSimple<float>(120, 80, 8, 3, 16, 1, ConvolutionMethodOption::unrolled);
    TestConv2DVsSimple<float>(121, 80, 8, 3, 16, 1, ConvolutionMethodOption::unrolled);
    TestConv2DVsSimple<float>(122, 80, 8, 3, 16, 1, ConvolutionMethodOption::unrolled);
    TestConv2DVsSimple<float>(121, 81, 8, 3, 16, 1, ConvolutionMethodOption::unrolled);
    TestConv2DVsSimple<float>(60, 40, 64, 3, 128, 1, ConvolutionMethodOption::unrolled);
    TestConv2DVsSimple<float>(129, 129, 128, 3, 128, 1, ConvolutionMethodOption::unrolled);

    // stride == 2
    TestConv2DVsSimple<float>(4, 4, 8, 3, 16, 2, ConvolutionMethodOption::unrolled);
    TestConv2DVsSimple<float>(6, 6, 8, 3, 16, 2, ConvolutionMethodOption::unrolled);
    TestConv2DVsSimple<float>(120, 80, 8, 3, 16, 2, ConvolutionMethodOption::unrolled);
    TestConv2DVsSimple<float>(121, 80, 8, 3, 16, 2, ConvolutionMethodOption::unrolled);
    TestConv2DVsSimple<float>(122, 80, 8, 3, 16, 2, ConvolutionMethodOption::unrolled);
    TestConv2DVsSimple<float>(121, 81, 8, 3, 16, 2, ConvolutionMethodOption::unrolled);
    TestConv2DVsSimple<float>(60, 40, 64, 3, 128, 2, ConvolutionMethodOption::unrolled);

    // Winograd
    TestConv2D<float>(ConvolutionMethodOption::winograd);
    TestConv2DVsSimple<float>(4, 4, 1, 3, 1, 1, ConvolutionMethodOption::winograd);
    TestConv2DVsSimple<float>(4, 4, 8, 3, 16, 1, ConvolutionMethodOption::winograd);
    TestConv2DVsSimple<float>(6, 6, 8, 3, 16, 1, ConvolutionMethodOption::winograd);
    TestConv2DVsSimple<float>(120, 80, 8, 3, 16, 1, ConvolutionMethodOption::winograd);
    TestConv2DVsSimple<float>(121, 80, 8, 3, 16, 1, ConvolutionMethodOption::winograd);
    TestConv2DVsSimple<float>(122, 80, 8, 3, 16, 1, ConvolutionMethodOption::winograd);
    TestConv2DVsSimple<float>(121, 81, 8, 3, 16, 1, ConvolutionMethodOption::winograd);
    TestConv2DVsSimple<float>(60, 40, 64, 3, 128, 1, ConvolutionMethodOption::winograd);
    TestConv2DVsSimple<float>(129, 129, 128, 3, 128, 1, ConvolutionMethodOption::winograd);

    // Depthwise-separable 2D convolution
    // Winograd
    TestConv2DSeparable<float>(ConvolutionMethodOption::winograd);
    TestConv2DSeparableVsSimple<float>(4, 4, 1, 3, 1, ConvolutionMethodOption::winograd);
    TestConv2DSeparableVsSimple<float>(4, 4, 8, 3, 1, ConvolutionMethodOption::winograd);
    TestConv2DSeparableVsSimple<float>(6, 6, 8, 3, 1, ConvolutionMethodOption::winograd);
    TestConv2DSeparableVsSimple<float>(120, 80, 8, 3, 1, ConvolutionMethodOption::winograd);
    TestConv2DSeparableVsSimple<float>(121, 80, 8, 3, 1, ConvolutionMethodOption::winograd);
    TestConv2DSeparableVsSimple<float>(122, 80, 8, 3, 1, ConvolutionMethodOption::winograd);
    TestConv2DSeparableVsSimple<float>(121, 81, 8, 3, 1, ConvolutionMethodOption::winograd);
    TestConv2DSeparableVsSimple<float>(60, 40, 64, 3, 1, ConvolutionMethodOption::winograd);
    TestConv2DSeparableVsSimple<float>(129, 129, 128, 3, 1, ConvolutionMethodOption::winograd);

    // FFT
    TestFFT<float>(16);
    TestFFT<double>(16);
    VerifyFFT<float>();
    VerifyFFT<double>();

    // Filters
    TestIIRFilter<float>();
    TestIIRFilterMultiSample<float>();
    TestIIRFilterImpulse<float>();

    // Window functions
    TestHammingWindow<float>();
    TestHammingWindow<double>();
    TestHammingWindow2<float>();
    TestHammingWindow2<double>();
    TestHannWindow<float>();
    TestHannWindow<double>();

    // Mel filterbank
    TestMelFilterBank();
    // TestMelFilterBank2(); // Commented out because our implementation rounds filter centers to integer locations, and the reference (librosa) doesn't

    // DCT
    TestDCT();
}

int main(int argc, char* argv[])
{
    UNUSED(argc);
    std::string path = utilities::GetDirectoryPath(argv[0]);

    try
    {
        DspTest(path);
    }
    catch (const utilities::Exception& exception)
    {
        std::cerr << "ERROR, got ELL exception. Message: " << exception.GetMessage() << std::endl;
        throw;
    }

    return testing::DidTestFail() ? 1 : 0;
}
