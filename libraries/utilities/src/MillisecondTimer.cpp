////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     MillisecondTimer.cpp (utilities)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MillisecondTimer.h"

namespace ell
{
namespace utilities
{
    MillisecondTimer::MillisecondTimer() :
        _start(std::chrono::system_clock::now()),
        _elapsedTime(std::chrono::system_clock::duration::zero()),
        _running(true)
    {}

    void MillisecondTimer::Start()
    {
        Reset();
        _running = true;
    }

    void MillisecondTimer::Stop()
    {
        _elapsedTime += TimeSinceStart();
        _running = false;
    }

    void MillisecondTimer::Restart()
    {
        _start = Now();
        _running = true;
    }

    void MillisecondTimer::Reset()
    {
        _start = Now();
        _elapsedTime = _elapsedTime.zero();
    }

    std::chrono::milliseconds::rep MillisecondTimer::Elapsed()
    {
        auto elapsed = _running ? TimeSinceStart() + _elapsedTime : _elapsedTime;
        return std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    }

    std::chrono::system_clock::time_point MillisecondTimer::Now()
    {
        return std::chrono::system_clock::now();
    }

    std::chrono::system_clock::duration MillisecondTimer::TimeSinceStart()
    {
        return Now() - _start;
    }
} // namespace utilities
} // namespace ell
