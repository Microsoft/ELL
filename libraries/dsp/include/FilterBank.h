////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     FilterBank.h (dsp)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <utilities/include/Archiver.h>
#include <utilities/include/IArchivable.h>

#include <cmath>
#include <complex>
#include <vector>

namespace ell
{
namespace dsp
{
    /// <summary> A class representing a simple triangular filter with a height of 1.0. </summary>
    /// The filter is nonzero only over a limited range, which can be found with the `GetStart` and `GetEnd` functions.
    /// The representation of these filters makes them generally only suitable for filtering a signal in the frequency domain ---
    /// that is to say, elementwise multiplying the filter values by the frequency components of the signal (say, from the output of an FFT).
    class TriangleFilter
    {
    public:
        /// <summary> Construct a filter given the indices of its start, high point, end, and overall size. </summary>
        ///
        /// <param name="lowBin"> The index of the beginning of the filter's nonzero support. </param>
        /// <param name="centerBin"> The index of the "center point" of the filter --- the location of the filter's maximum value. </param>
        /// <param name="highBin"> The index of the end of the filter's nonzero support. </param>
        /// <param name="size"> The total size of this filter (including zero support). </param>
        /// <param name="offset"> A number between 0 and 1 and shifts the filter position slightly. </param>
        TriangleFilter(size_t lowBin, size_t centerBin, size_t highBin, size_t size, double offset);

        /// <summary> Get the filter coefficient at the given index. </summary>
        ///
        /// <param name="index"> The index at which to retrieve the filter coefficient. </param>
        ///
        /// <returns> The filter coefficient at the given index. </returns>
        double operator[](size_t index);

        /// <summary> Return the length of the filter. </summary>
        ///
        /// <returns> The length of the filter. </returns>
        size_t Size() { return _size; }

        /// <summary> Get the start point of the filter: the first index for which the support is nonzero. </summary>
        ///
        /// <returns> The start point of the filter. </returns>
        size_t GetStart() const { return _lowBin; }

        /// <summary> Get the center of the filter: the index for which the support is maximum (1.0). </summary>
        ///
        /// <returns> The center of the filter. </returns>
        size_t GetCenter() const { return _centerBin; }

        /// <summary> Get the endpoint of the filter: the last index for which the support is nonzero. </summary>
        ///
        /// <returns> The endpoint of the filter. </returns>
        size_t GetEnd() const { return _highBin; }

        /// <summary> Return a dense array of filter coefficients. </summary>
        ///
        /// <returns> A dense array of filter coefficients. </returns>
        std::vector<double> ToArray() const;

    private:
        size_t _lowBin;
        size_t _centerBin;
        size_t _highBin;
        size_t _size;
        double _offset;
    };

    /// <summary> Base class for an arbitrary set of triangular filters. </summary>
    class TriangleFilterBank : public utilities::IArchivable
    {
    public:
        /// <summary> Apply the filter bank to the real-valued magnitudes of an FFT. </summary>
        ///
        /// <param name="frequencyMagnitudes"> The array of frequency magnitudes to apply the filters to. </param>
        ///
        /// <returns> The filtered array of frequency magnitudes </returns>
        template <typename ValueType>
        std::vector<ValueType> FilterFrequencyMagnitudes(const std::vector<ValueType>& frequencyMagnitudes) const;

        /// <summary> Apply the filter bank to the output of an FFT, returning the filtered magnitudes of the frequencies. </summary>
        ///
        /// <param name="fourierCoefficients"> The array of fourier coefficients to apply the filters to. </param>
        ///
        /// <returns> The filtered array of frequency magnitudes for the given fourier coefficients </returns>
        template <typename ValueType>
        std::vector<ValueType> FilterFourierCoefficients(const std::vector<std::complex<ValueType>>& fourierCoefficients) const;

        /// <summary> Apply the filters to the output of an FFT, returning a fast approximation of the filtered magnitudes of the frequencies. </summary>
        ///
        /// <param name="fourierCoefficients"> The array of fourier coefficients to apply the filters to. </param>
        ///
        /// <returns> The approximated filtered array of frequency magnitudes for the given fourier coefficients </returns>
        template <typename ValueType>
        std::vector<ValueType> FilterFourierCoefficientsFast(const std::vector<std::complex<ValueType>>& fourierCoefficients) const;

        /// <summary> Return the number of active filters in the filter bank. </summary>
        ///
        /// <returns> The number of active filters in the filter bank. </returns>
        size_t NumActiveFilters() const { return _endFilter - _beginFilter; }

        /// <summary> Return the total number of filters in the filter bank. </summary>
        ///
        /// <returns> The total number of filters in the filter bank. </returns>
        size_t NumFilters() const { return _numFilters; }

        /// <summary> Return a `TriangleFilter` object representing one of the filters in the filter bank. </summary>
        TriangleFilter GetFilter(size_t filterIndex) const;

        /// <summary> Get the length of the signal to filter. </summary>
        size_t GetWindowSize() const { return _windowSize; }

        /// <summary> Get the index of the first active filter. </summary>
        size_t GetBeginFilter() const { return _beginFilter; }

        /// <summary> Get the index one beyond the last active filter. </summary>
        size_t GetEndFilter() const { return _endFilter; }

        /// <summary> Get the sample rate of the input signal. </summary>
        double GetSampleRate() const { return _sampleRate; }

        /// <summary> Get the offset. </summary>
        double GetOffset() const { return _offset; }

        /// <summary> Gets the name of this type. </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "TriangleFilterBank"; }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

    protected:
        TriangleFilterBank() = default;

        /// <summary>The offset can be from 0 to 1 and shifts where we sample the triangle.  For example take this
        /// triangle filter of width 2 centered on input index 3:
        ///
        /// |           ^               |
        /// |          /|\              |
        /// |         / | \             |
        /// |        /  |  \            |
        /// |       /   |   \           |
        /// |---|---|-.-|-.-|-.-|---|---|
        /// 0   1   2   3   4   5   6   7
        ///
        /// with offset 0 the result would be (I3 * 1) but with offset 0.5 it would be (I2 * 0.5) + (I3 * 0.5).</summary>
        TriangleFilterBank(size_t windowSize, double sampleRate, size_t numFilters, double offset = 0);
        TriangleFilterBank(size_t windowSize, double sampleRate, size_t numFilters, size_t numFiltersToUse, double offset = 0);
        TriangleFilterBank(size_t windowSize, double sampleRate, size_t numFilters, size_t beginFilter, size_t endFilter, double offset = 0);
        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;
        void SetBins(const std::vector<size_t>& bins);

    private:
        size_t _windowSize = 0;
        double _sampleRate = 0;
        size_t _numFilters = 0; // total number of divisions
        size_t _beginFilter = 0; // index of first filter to use
        size_t _endFilter = 0; // index of last filter to use
        std::vector<size_t> _bins;
        double _offset;
    };

    /// <summary> A set of linearly-spaced triangular filters. </summary>
    class LinearFilterBank : public TriangleFilterBank
    {
    public:
        LinearFilterBank() = default;

        /// <summary> Constructor </summary>
        ///
        /// <param name="windowSize"> The length of the signal to filter. </param>
        /// <param name="sampleRate"> The sample rate of the input signal. </param>
        /// <param name="numFilters"> The total number of filters in the filter bank. </param>
        /// <param name="offset"> The offset. </param>
        LinearFilterBank(size_t windowSize, double sampleRate, size_t numFilters, double offset = 0);

        /// <summary> Constructor </summary>
        ///
        /// <param name="windowSize"> The length of the signal to filter. </param>
        /// <param name="sampleRate"> The sample rate of the input signal. </param>
        /// <param name="numFilters"> The total number of filters in the filter bank. </param>
        /// <param name="numFiltersToUse"> The number of active filters to use. The first N filters will be active. </param>
        /// <param name="offset"> The offset. </param>
        LinearFilterBank(size_t windowSize, double sampleRate, size_t numFilters, size_t numFiltersToUse, double offset = 0);

        /// <summary> Constructor </summary>
        ///
        /// <param name="windowSize"> The length of the signal to filter. </param>
        /// <param name="sampleRate"> The sample rate of the input signal. </param>
        /// <param name="numFilters"> The total number of filters in the filter bank. </param>
        /// <param name="beginFilter"> The index of the first active filter. </param>
        /// <param name="endFilter"> The index one beyond the last active filter. </param>
        /// <param name="offset"> The offset. </param>
        LinearFilterBank(size_t windowSize, double sampleRate, size_t numFilters, size_t beginFilter, size_t endFilter, double offset = 0);

        /// <summary> Gets the name of this type. </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "LinearFilterBank"; }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

    protected:
        void ReadFromArchive(utilities::Unarchiver& archiver) override;
        void InitializeBins();
    };

    //
    // MelFilterBank
    //
    class MelFilterBank : public TriangleFilterBank
    {
    public:
        MelFilterBank() = default;

        /// <summary> Constructor </summary>
        ///
        /// <param name="windowSize"> The length of the signal to filter. </param>
        /// <param name="sampleRate"> The sample rate of the input signal. </param>
        /// <param name="fftSize"> The size of the FFT. </param>
        /// <param name="numFilters"> The total number of filters in the filter bank. </param>
        /// <param name="offset"> The offset. </param>
        MelFilterBank(size_t windowSize, double sampleRate, size_t fftSize, size_t numFilters, double offset = 0);

        /// <summary> Constructor </summary>
        ///
        /// <param name="windowSize"> The length of the signal to filter. </param>
        /// <param name="sampleRate"> The sample rate of the input signal. </param>
        /// <param name="fftSize"> The size of the FFT. </param>
        /// <param name="numFilters"> The total number of filters in the filter bank. </param>
        /// <param name="numFiltersToUse"> The number of active filters to use. The first N filters will be active. </param>
        /// <param name="offset"> The offset. </param>
        MelFilterBank(size_t windowSize, double sampleRate, size_t fftSize, size_t numFilters, size_t numFiltersToUse, double offset = 0);

        /// <summary> Constructor </summary>
        ///
        /// <param name="windowSize"> The length of the signal to filter. </param>
        /// <param name="sampleRate"> The sample rate of the input signal. </param>
        /// <param name="fftSize"> The size of the FFT. </param>
        /// <param name="numFilters"> The total number of filters in the filter bank. </param>
        /// <param name="beginFilter"> The index of the first active filter. </param>
        /// <param name="endFilter"> The index one beyond the last active filter. </param>
        /// <param name="offset"> The offset. </param>
        MelFilterBank(size_t windowSize, double sampleRate, size_t fftSize, size_t numFilters, size_t beginFilter, size_t endFilter, double offset = 0);

        /// <summary> Gets the name of this type. </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "MelFilterBank"; }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

    protected:
        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;
        void InitializeBins();

    private:
        size_t _fftSize;
    };

    // Helper functions

    /// <summary> Convert a frequency (in Hz) to the mel scale. </summary>
    double FreqToMel(double freq);

    /// <summary> Convert a mel-scale frequency to Hz. </summary>
    double MelToFreq(double mel);
} // namespace dsp
} // namespace ell
