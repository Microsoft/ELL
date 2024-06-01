////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Boolean.cpp (utilities)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Boolean.h"

namespace ell
{
namespace utilities
{

    Boolean::Boolean() = default;

    Boolean::Boolean(bool value) :
        value(value)
    {}

    bool operator==(Boolean b1, Boolean b2)
    {
        return static_cast<bool>(b1) == static_cast<bool>(b2);
    }

    bool operator==(bool b1, Boolean b2)
    {
        return b1 == static_cast<bool>(b2);
    }

    bool operator==(Boolean b1, bool b2)
    {
        return static_cast<bool>(b1) == b2;
    }

    bool operator!=(Boolean b1, Boolean b2)
    {
        return static_cast<bool>(b1) != static_cast<bool>(b2);
    }

    bool operator!=(bool b1, Boolean b2)
    {
        return b1 != static_cast<bool>(b2);
    }

    bool operator!=(Boolean b1, bool b2)
    {
        return static_cast<bool>(b1) != b2;
    }

} // namespace utilities
} // namespace ell
