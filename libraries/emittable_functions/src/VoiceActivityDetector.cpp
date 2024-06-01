////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     VoiceActivityDetector.cpp (dsp)
//  Authors:  Chris Lovett, Ivan Tashev
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "VoiceActivityDetector.h"

#include <value/include/EmitterContext.h>

#include <algorithm>
#include <cassert>
#include <cstdint>

using namespace ell::value;
using namespace ell::utilities;

namespace ell
{
namespace emittable_functions
{
    class ActivityTracker
    {
        Scalar _lastLevel;
        Scalar _lastTime;
        Scalar _signal;

    public:
        const double _tauUp;
        const double _tauDown;
        const double _largeInput;
        const double _gainAtt;
        const double _thresholdUp;
        const double _thresholdDown;
        const double _levelThreshold;

        ActivityTracker(double tauUp,
                        double tauDown,
                        double largeInput,
                        double gainAtt,
                        double thresholdUp,
                        double thresholdDown,
                        double levelThreshold) :
            _tauUp(tauUp),
            _tauDown(tauDown),
            _largeInput(largeInput),
            _gainAtt(gainAtt),
            _thresholdUp(thresholdUp),
            _thresholdDown(thresholdDown),
            _levelThreshold(levelThreshold)
        {}

        void Reset()
        {
            _lastLevel = Cast(0.1, _lastLevel.GetType());
            _lastTime = Cast(0, _lastLevel.GetType());
            _signal = 0;
        }

        /// <summary> compute the next signal state given input time and power levels </summary>
        Scalar Classify(Scalar time, Scalar inputLevel)
        {
            auto dataType = inputLevel.GetType();
            if (dataType == ValueType::Double)
            {
                _lastLevel = StaticAllocate("lastLevel", 0.1);
                _lastTime = StaticAllocate("lastTime", 0.0);
            }
            else
            {
                _lastLevel = StaticAllocate("lastLevel", 0.1f);
                _lastTime = StaticAllocate("lastTime", 0.0f);
            }
            _signal = StaticAllocate("signal", 0);

            Scalar level = inputLevel;
            Scalar timeDelta = time - _lastTime;
            Scalar levelDelta = level - _lastLevel;
            Scalar tauDown = Cast(_tauDown, dataType);
            Scalar tauUp = Cast(_tauUp, dataType);
            Scalar gainAtt = Cast(_gainAtt, dataType);
            Scalar thresholdUp = Cast(_thresholdUp, dataType);
            Scalar levelThreshold = Cast(_levelThreshold, dataType);
            Scalar thresholdDown = Cast(_thresholdDown, dataType);
            Scalar largeInput = Cast(_largeInput, dataType);

            If(level < _lastLevel,
               [&] {
                   _lastLevel = _lastLevel + timeDelta / tauDown * levelDelta;

                   If(_lastLevel < level,
                      [&] {
                          _lastLevel = level;
                      });
               })
                .ElseIf(level > largeInput * _lastLevel,
                        [&] {
                            _lastLevel = _lastLevel + gainAtt * timeDelta / tauUp * levelDelta;

                            If(_lastLevel > level,
                               [&] {
                                   _lastLevel = level;
                               });
                        })
                .Else([&] {
                    _lastLevel = _lastLevel + timeDelta / tauUp * levelDelta;

                    If(_lastLevel > level,
                       [&] {
                           _lastLevel = level;
                       });
                });

            // bugbug: need && operator
            If(level > (thresholdUp * _lastLevel),
               [&] {
                   If(level > levelThreshold, [&] {
                       _signal = 1;
                   });
               });

            If(level < (thresholdDown * _lastLevel),
               [&] {
                   _signal = 0;
               });

            _lastTime = time;
            return _signal;
        };
    };

    /// <summary> internal use only - this class builds a message weight lookup table for given sampleRate and windowSize </summary>
    class CMessageWeights
    {
        const int _maxFreq = 4000;
        std::vector<double> _weights;

    public:
        CMessageWeights(double sampleRate, double windowSize) { Generate(sampleRate, windowSize); }

        double GenWeight(double freq)
        {
            // clang-format off
            static int freqMap[] = { 60,   100,  200,  300,  400,  500,  600,  700,  800,  900,  1000,
                                     1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100,
                                     2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200,
                                     3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000 };
            static double msgWeights[] = { -54.65, -41.71, -25.17, -16.64, -11.29, -7.55, -4.75, -2.66, -1.19,
                                           -0.32,  0.03,   0.03,   -0.17,  -0.44,  -0.71, -0.94, -1.12, -1.24,
                                           -1.32,  -1.36,  -1.38,  -1.39,  -1.41,  -1.44, -1.50, -1.60, -1.76,
                                           -1.97,  -2.26,  -2.62,  -3.09,  -3.66,  -4.35, -5.18, -6.18, -7.36,
                                           -8.75,  -10.36, -12.12, -13.72, -14.43 };
            // clang-format on

            size_t tableSize = sizeof(freqMap) / sizeof(int);
            size_t f = 0;
            for (size_t i = 0; i < tableSize; ++i)
            {
                if (freq > freqMap[i])
                {
                    f = i + 1;
                }
            }
            double cmessw = 0;
            tableSize = sizeof(msgWeights) / sizeof(double);
            if (f >= tableSize)
            {
                cmessw = msgWeights[f - 1];
            }
            else if (f > 0)
            {
                cmessw = Interpolate(freq, freqMap[f - 1], freqMap[f], msgWeights[f - 1], msgWeights[f]);
            }
            return cmessw;
        }

        /// <summary> generates a lookup table of size windowSize </summary>
        void Generate(double sampleRate, double windowSize)
        {
            _weights.resize(static_cast<int>(windowSize));
            double div = sampleRate / _maxFreq;
            double freq_step = sampleRate / windowSize / div;
            for (int i = 0; i < windowSize; i++)
            {
                double w = GenWeight(i * freq_step);
                if (w != 0)
                {
                    w = pow(10, w / 20);
                    _weights[i] = w * w;
                }
            }
        }

        /// <summary> lookup the weight for given bin number out of windowSize bins </summary>
        double GetWeight(int bin)
        {
            assert(bin < static_cast<int>(_weights.size()));
            return _weights[bin];
        }

        /// <summary> for x in the range [x1, x2], interpolate the corresponding value of y in the range [y1, y2]  </summary>
        double Interpolate(double x, double x1, double x2, double y1, double y2)
        {
            const double epsilon = 1e-6;
            double diff = std::abs(x2 - x1);
            if (diff < epsilon)
            {
                return (y1 + y2) / 2.0;
            }
            double proportion = (x - x1) / (x2 - x1);
            return y1 + (y2 - y1) * proportion;
        }

        const std::vector<double>& GetWeights() const { return _weights; }
    };

    struct VoiceActivityDetectorImpl
    {
        CMessageWeights _cmw;
        ActivityTracker _tracker;
        const double _frameDuration;
        const double _sampleRate;
        const double _windowSize;
        Scalar _time;

        VoiceActivityDetectorImpl(double sampleRate,
                                  double windowSize,
                                  double frameDuration,
                                  double tauUp,
                                  double tauDown,
                                  double largeInput,
                                  double gainAtt,
                                  double thresholdUp,
                                  double thresholdDown,
                                  double levelThreshold) :
            _cmw(sampleRate, windowSize),
            _tracker(tauUp, tauDown, largeInput, gainAtt, thresholdUp, thresholdDown, levelThreshold),
            _frameDuration(frameDuration),
            _sampleRate(sampleRate),
            _windowSize(windowSize)
        {}
    };

    VoiceActivityDetector::VoiceActivityDetector() = default;

    VoiceActivityDetector::VoiceActivityDetector(double sampleRate,
                                                 double windowSize,
                                                 double frameDuration,
                                                 double tauUp,
                                                 double tauDown,
                                                 double largeInput,
                                                 double gainAtt,
                                                 double thresholdUp,
                                                 double thresholdDown,
                                                 double levelThreshold) :
        _impl(std::make_unique<VoiceActivityDetectorImpl>(sampleRate,
                                                          windowSize,
                                                          frameDuration,
                                                          tauUp,
                                                          tauDown,
                                                          largeInput,
                                                          gainAtt,
                                                          thresholdUp,
                                                          thresholdDown,
                                                          levelThreshold))
    {}

    VoiceActivityDetector::~VoiceActivityDetector() = default;

    void VoiceActivityDetector::Reset()
    {
        _impl->_time = int64_t{ 0 };
        _impl->_tracker.Reset();
    }

    double VoiceActivityDetector::GetSampleRate() const
    {
        return _impl->_sampleRate;
    }

    double VoiceActivityDetector::GetWindowSize() const
    {
        return _impl->_windowSize;
    }

    double VoiceActivityDetector::GetFrameDuration() const
    {
        return _impl->_frameDuration;
    }

    double VoiceActivityDetector::GetTauUp() const
    {
        return _impl->_tracker._tauUp;
    }

    double VoiceActivityDetector::GetTauDown() const
    {
        return _impl->_tracker._tauDown;
    }

    double VoiceActivityDetector::GetLargeInput() const
    {
        return _impl->_tracker._largeInput;
    }

    double VoiceActivityDetector::GetGainAtt() const
    {
        return _impl->_tracker._gainAtt;
    }

    double VoiceActivityDetector::GetThresholdUp() const
    {
        return _impl->_tracker._thresholdUp;
    }

    double VoiceActivityDetector::GetThresholdDown() const
    {
        return _impl->_tracker._thresholdDown;
    }

    double VoiceActivityDetector::GetLevelThreshold() const
    {
        return _impl->_tracker._levelThreshold;
    }

    Scalar VoiceActivityDetector::Process(const Vector data)
    {
        if (data.Size() != static_cast<size_t>(_impl->_windowSize))
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument,
                                            "data length should match windowSize");
        }

        _impl->_time = StaticAllocate("time", int64_t{ 0 });

        auto dataType = data.GetType();
        Vector weights = GetWeights();
        Scalar windowSize = Cast(_impl->_windowSize, dataType);
        Scalar frameDuration = Cast(_impl->_frameDuration, dataType);

        auto level = Dot(data, Cast(weights, dataType));

        level /= windowSize;

        Scalar castedTime = Cast(_impl->_time, dataType);
        Scalar t = castedTime * frameDuration;
        ++_impl->_time;

        Scalar signal = _impl->_tracker.Classify(t, level);
        return signal;
    }

    const std::vector<double>& VoiceActivityDetector::GetWeights() const
    {
        return _impl->_cmw.GetWeights();
    }

    bool VoiceActivityDetector::Equals(const VoiceActivityDetector& other) const
    {
        if (_impl.get() == nullptr)
        {
            return other._impl.get() == nullptr;
        }
        if (other._impl.get() == nullptr)
        {
            return false;
        }
        return _impl->_windowSize == other._impl->_windowSize && _impl->_sampleRate == other._impl->_sampleRate &&
               _impl->_frameDuration == other._impl->_frameDuration &&
               _impl->_tracker._tauUp == other._impl->_tracker._tauUp &&
               _impl->_tracker._tauDown == other._impl->_tracker._tauDown &&
               _impl->_tracker._largeInput == other._impl->_tracker._largeInput &&
               _impl->_tracker._gainAtt == other._impl->_tracker._gainAtt &&
               _impl->_tracker._thresholdUp == other._impl->_tracker._thresholdUp &&
               _impl->_tracker._thresholdDown == other._impl->_tracker._thresholdDown &&
               _impl->_tracker._levelThreshold == other._impl->_tracker._levelThreshold;
    }

    void VoiceActivityDetector::WriteToArchive(utilities::Archiver& archiver) const
    {
        archiver["windowSize"] << _impl->_windowSize;
        archiver["sampleRate"] << _impl->_sampleRate;
        archiver["frameDuration"] << _impl->_frameDuration;
        archiver["tauUp"] << _impl->_tracker._tauUp;
        archiver["tauDown"] << _impl->_tracker._tauDown;
        archiver["largeInput"] << _impl->_tracker._largeInput;
        archiver["gainAtt"] << _impl->_tracker._gainAtt;
        archiver["thresholdUp"] << _impl->_tracker._thresholdUp;
        archiver["thresholdDown"] << _impl->_tracker._thresholdDown;
        archiver["levelThreshold"] << _impl->_tracker._levelThreshold;
    }

    void VoiceActivityDetector::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        double windowSize;
        double sampleRate;
        double frameDuration;
        double tauUp;
        double tauDown;
        double largeInput;
        double gainAtt;
        double thresholdUp;
        double thresholdDown;
        double levelThreshold;
        archiver["windowSize"] >> windowSize;
        archiver["sampleRate"] >> sampleRate;
        archiver["frameDuration"] >> frameDuration;
        archiver["tauUp"] >> tauUp;
        archiver["tauDown"] >> tauDown;
        archiver["largeInput"] >> largeInput;
        archiver["gainAtt"] >> gainAtt;
        archiver["thresholdUp"] >> thresholdUp;
        archiver["thresholdDown"] >> thresholdDown;
        archiver["levelThreshold"] >> levelThreshold;
        _impl = std::make_unique<VoiceActivityDetectorImpl>(sampleRate,
                                                            windowSize,
                                                            frameDuration,
                                                            tauUp,
                                                            tauDown,
                                                            largeInput,
                                                            gainAtt,
                                                            thresholdUp,
                                                            thresholdDown,
                                                            levelThreshold);
    }

} // namespace emittable_functions
} // namespace ell
