////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Range.h (value)
//  Authors:  Chuck Jacobs, Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "loopnests/Range.h"

#include <ostream>

namespace ell
{
namespace value
{
    namespace loopnests
    {
        Range::Range(int begin, int end, int increment) :
            _begin(begin),
            _end(end),
            _increment(increment)
        {}

        int Range::Begin() const
        {
            return _begin;
        }

        int Range::End() const
        {
            return _end;
        }

        int Range::Size() const
        {
            return _end - _begin;
        }

        int Range::Increment() const
        {
            return _increment;
        }

        std::ostream& operator<<(std::ostream& os, const Range& r)
        {
            os << "[" << r.Begin() << "," << r.End() << ":" << r.Increment() << ")";
            return os;
        }

        bool operator==(const Range& i1, const Range& i2)
        {
            return (i1.Begin() == i2.Begin()) && (i1.End() == i2.End()) && (i1.Increment() == i2.Increment());
        }

        bool operator!=(const Range& i1, const Range& i2)
        {
            return (i1.Begin() != i2.Begin()) || (i1.End() != i2.End()) || (i1.Increment() != i2.Increment());
        }

        bool operator<(const Range& i1, const Range& i2)
        {
            if (i1.Begin() != i2.Begin())
            {
                return i1.Begin() < i2.Begin();
            }
            else if (i1.End() != i2.End())
            {
                return i1.End() < i2.End();
            }
            else
            {
                return i1.Increment() < i2.Increment();
            }
        }
    } // namespace loopnests
} // namespace value
} // namespace ell
