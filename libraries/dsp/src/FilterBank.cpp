////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     FilterBank.cpp (dsp)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FilterBank.h"

#include <utilities/include/StringUtil.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace ell
{
namespace dsp
{
    double FreqToMel(double freq)
    {
        // Formula for converting from Hz scale to mel
        // https://en.wikipedia.org/wiki/Mel_scale
        //
        // m = 1127*ln(1+f/700)
        return 1127 * std::log1p(freq / 700);
    }

    double MelToFreq(double mel)
    {
        // Formula for converting from mel scale to Hz
        // https://en.wikipedia.org/wiki/Mel_scale
        //
        // f = 700*(exp(m/1127) - 1)
        return 700 * (std::exp(mel / 1127) - 1);
    }

    //
    // TriangleFilter
    //

    TriangleFilter::TriangleFilter(size_t lowBin, size_t centerBin, size_t highBin, size_t size, double offset) :
        _lowBin(lowBin),
        _centerBin(centerBin),
        _highBin(highBin),
        _size(size),
        _offset(offset)
    {}

    double TriangleFilter::operator[](size_t index)
    {
        if (index < _lowBin)
        {
            return 0;
        }
        else if (index < _centerBin && _lowBin < _centerBin)
        {
            return static_cast<double>(index - _lowBin + _offset) / (_centerBin - _lowBin);
        }
        else if (index < _highBin && _centerBin < _highBin)
        {
            return static_cast<double>(_highBin - index - _offset) / (_highBin - _centerBin);
        }
        else
        {
            return 0;
        }
    }

    std::vector<double> TriangleFilter::ToArray() const
    {
        std::vector<double> result(_size, 0);
        for (size_t freqIndex = _lowBin; freqIndex < _centerBin; ++freqIndex) // between lo and center
        {
            result[freqIndex] = static_cast<double>(freqIndex - _lowBin) / (_centerBin - _lowBin);
        }
        for (size_t freqIndex = _centerBin; freqIndex < _highBin; ++freqIndex) // between center and hi
        {
            result[freqIndex] = static_cast<double>(_highBin - freqIndex) / (_highBin - _centerBin);
        }
        return result;
    }

    //
    // TriangleFilterBank
    //
    TriangleFilterBank::TriangleFilterBank(size_t windowSize, double sampleRate, size_t numFilters, double offset) :
        TriangleFilterBank(windowSize, sampleRate, numFilters, 0, numFilters, offset)
    {
        // Note: Subclass implementations must explicitly call SetBins() in order to properly initialize the object
    }

    TriangleFilterBank::TriangleFilterBank(size_t windowSize, double sampleRate, size_t numFilters, size_t numFiltersToUse, double offset) :
        TriangleFilterBank(windowSize, sampleRate, numFilters, 0, numFiltersToUse, offset)
    {
        // Note: Subclass implementations must explicitly call SetBins() in order to properly initialize the object
    }

    TriangleFilterBank::TriangleFilterBank(size_t windowSize, double sampleRate, size_t numFilters, size_t beginFilter, size_t endFilter, double offset) :
        _windowSize(windowSize),
        _sampleRate(sampleRate),
        _numFilters(numFilters),
        _beginFilter(beginFilter),
        _endFilter(endFilter),
        _offset(offset)
    {
        // Note: Subclass implementations must explicitly call SetBins() in order to properly initialize the object
    }

    TriangleFilter TriangleFilterBank::GetFilter(size_t filterIndex) const
    {
        const size_t filterLength = _windowSize / 2 + 1;

        const auto lowBin = _bins[filterIndex];
        const auto centerBin = _bins[filterIndex + 1];
        const auto highBin = _bins[filterIndex + 2];
        return { lowBin, centerBin, highBin, filterLength, _offset };
    }

    template <typename ValueType>
    std::vector<ValueType> TriangleFilterBank::FilterFourierCoefficients(const std::vector<std::complex<ValueType>>& fourierCoefficients) const
    {
        std::vector<ValueType> magnitudes(_windowSize / 2);
        for (size_t k = 0; k < _windowSize / 2; k++)
        {
            magnitudes[k] = std::abs(fourierCoefficients[k]);
        }

        return FilterFrequencyMagnitudes(magnitudes);
    }

    template <typename ValueType>
    std::vector<ValueType> TriangleFilterBank::FilterFourierCoefficientsFast(const std::vector<std::complex<ValueType>>& fourierCoefficients) const
    {
        // Approximate |X| = alpha * max(R{x}, I{x}) + beta * min(R{x}, I{x})
        const ValueType alpha = 1;
        const ValueType beta = 1;

        std::vector<ValueType> magnitudes(_windowSize / 2);
        for (size_t k = 0; k < _windowSize / 2; k++)
        {
            auto realMag = std::abs(fourierCoefficients[k].real());
            auto imagMag = std::abs(fourierCoefficients[k].imag());
            magnitudes[k] = alpha * std::max(realMag, imagMag) + beta * std::min(realMag, imagMag);
        }

        return FilterFrequencyMagnitudes(magnitudes);
    }

    template <typename ValueType>
    std::vector<ValueType> TriangleFilterBank::FilterFrequencyMagnitudes(const std::vector<ValueType>& frequencyMagnitudes) const
    {
        // Apply filters to frequency magnitudes --- just elementwise multiplication, then sum
        //        N/2
        // Y[i] = sum((|X[k]| sqrt(H_i[k]) ^ 2)
        //        k = 0
        auto numOutputs = _endFilter - _beginFilter;
        std::vector<ValueType> result(numOutputs);
        for (size_t filterIndex = _beginFilter; filterIndex < _endFilter; ++filterIndex)
        {
            auto filter = GetFilter(filterIndex);
            const size_t begin = filter.GetStart();
            const size_t end = filter.GetEnd();
            ValueType sum = 0;
            for (size_t k = begin; k < end; ++k)
            {
                sum += static_cast<ValueType>(frequencyMagnitudes[k] * filter[k]);
            }

            result[filterIndex] = sum;
        }
        return result;
    }

    void TriangleFilterBank::WriteToArchive(utilities::Archiver& archiver) const
    {
        archiver["windowSize"] << _windowSize;
        archiver["rate"] << _sampleRate;
        archiver["numFilters"] << _numFilters;
        archiver["begin"] << _beginFilter;
        archiver["end"] << _endFilter;
        archiver["offset"] << _offset;
    }

    void TriangleFilterBank::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        archiver["windowSize"] >> _windowSize;
        archiver["rate"] >> _sampleRate;
        archiver["numFilters"] >> _numFilters;
        archiver["begin"] >> _beginFilter;
        archiver["end"] >> _endFilter;
        archiver["offset"] >> _offset;
    }

    void TriangleFilterBank::SetBins(const std::vector<size_t>& bins)
    {
        _bins = bins;
        for (auto v : bins)
        {
            if (v > _windowSize)
            {
                throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange,
                                                utilities::FormatString("TriangleFilterBank::SetBins received a value %d that is outside the _windowSize %d", (int)v, (int)_windowSize));
            }
        }
    }

    //
    // LinearFilterBank
    //

    LinearFilterBank::LinearFilterBank(size_t windowSize, double sampleRate, size_t numFilters, double offset) :
        LinearFilterBank(windowSize, sampleRate, numFilters, 0, numFilters, offset)
    {
    }

    LinearFilterBank::LinearFilterBank(size_t windowSize, double sampleRate, size_t numFilters, size_t numFiltersToUse, double offset) :
        LinearFilterBank(windowSize, sampleRate, numFilters, 0, numFiltersToUse, offset)
    {
    }

    LinearFilterBank::LinearFilterBank(size_t windowSize, double sampleRate, size_t numFilters, size_t beginFilter, size_t endFilter, double offset) :
        TriangleFilterBank(windowSize, sampleRate, numFilters, beginFilter, endFilter, offset)
    {
        LinearFilterBank::InitializeBins();
    }

    void LinearFilterBank::InitializeBins()
    {
        const auto windowSize = GetWindowSize();
        const auto numFilters = NumFilters();
        const auto sampleRate = GetSampleRate();

        const double loFreq = 0;
        const double hiFreq = (sampleRate / 2); // Nyquist frequency
        const double freqInc = (hiFreq - loFreq) / (numFilters + 1);

        std::vector<size_t> bins(numFilters + 2);
        for (size_t index = 0; index < numFilters + 2; ++index)
        {
            auto freqPoint = index * freqInc + loFreq;
            bins[index] = static_cast<int>(freqPoint * (windowSize + 1) / sampleRate);
        }

        SetBins(bins);
    }

    void LinearFilterBank::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        TriangleFilterBank::ReadFromArchive(archiver);
        InitializeBins();
    }

    //
    // MelFilterBank
    //

    MelFilterBank::MelFilterBank(size_t windowSize, double sampleRate, size_t fftSize, size_t numFilters, double offset) :
        MelFilterBank(windowSize, sampleRate, fftSize, numFilters, 0, numFilters, offset)
    {
    }

    MelFilterBank::MelFilterBank(size_t windowSize, double sampleRate, size_t fftSize, size_t numFilters, size_t numFiltersToUse, double offset) :
        MelFilterBank(windowSize, sampleRate, fftSize, numFilters, 0, numFiltersToUse, offset)
    {
    }

    MelFilterBank::MelFilterBank(size_t windowSize, double sampleRate, size_t fftSize, size_t numFilters, size_t beginFilter, size_t endFilter, double offset) :
        TriangleFilterBank(windowSize, sampleRate, numFilters, beginFilter, endFilter, offset), _fftSize(fftSize)
    {
        MelFilterBank::InitializeBins();
    }

    void MelFilterBank::InitializeBins()
    {
        const auto numFilters = NumFilters();
        const auto sampleRate = GetSampleRate();

        // const double loFreq = 0;
        const double hiFreq = (sampleRate / 2); // Nyquist frequency
        const double loMel = 0; //FreqToMel(loFreq);
        const double hiMel = FreqToMel(hiFreq);
        const double melInc = (hiMel - loMel) / (numFilters + 1);

        std::vector<size_t> bins(numFilters + 2);
        for (size_t index = 0; index < numFilters + 2; ++index)
        {
            auto melPoint = index * melInc + loMel;
            auto freqPoint = MelToFreq(melPoint);
            bins[index] = static_cast<int>(freqPoint * (_fftSize + 1) / sampleRate);
        }

        SetBins(bins);
    }

    void MelFilterBank::WriteToArchive(utilities::Archiver& archiver) const
    {
        TriangleFilterBank::WriteToArchive(archiver);
        archiver["fftSize"] << _fftSize;
    }

    void MelFilterBank::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        TriangleFilterBank::ReadFromArchive(archiver);
        archiver["fftSize"] >> _fftSize;
        InitializeBins();
    }

    //
    // Explicit instantiations
    //
    template std::vector<float> TriangleFilterBank::FilterFrequencyMagnitudes<float>(const std::vector<float>&) const;
    template std::vector<double> TriangleFilterBank::FilterFrequencyMagnitudes<double>(const std::vector<double>&) const;
    template std::vector<float> TriangleFilterBank::FilterFourierCoefficients(const std::vector<std::complex<float>>& fourierCoefficients) const;
    template std::vector<double> TriangleFilterBank::FilterFourierCoefficients(const std::vector<std::complex<double>>& fourierCoefficients) const;
    template std::vector<float> TriangleFilterBank::FilterFourierCoefficientsFast(const std::vector<std::complex<float>>& fourierCoefficients) const;
    template std::vector<double> TriangleFilterBank::FilterFourierCoefficientsFast(const std::vector<std::complex<double>>& fourierCoefficients) const;
} // namespace dsp
} // namespace ell
