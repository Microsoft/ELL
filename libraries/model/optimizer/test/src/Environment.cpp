////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Environment.cpp (model/optimizer)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Environment.h"

#include <utilities/include/Exception.h>

namespace ell
{
namespace model
{
    namespace optimizer
    {
        Environment::Environment() :
            _targetDevice(std::nullopt)
        {
        }

        Environment::Environment(const emitters::TargetDevice& targetDevice) :
            _targetDevice(targetDevice)
        {
        }

        bool Environment::HasTargetDevice() const
        {
            return static_cast<bool>(_targetDevice);
        }

        const emitters::TargetDevice& Environment::GetTargetDevice() const
        {
            if (!HasTargetDevice())
            {
                throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Environment doesn't have a target device");
            }
            return _targetDevice.value();
        }
    } // namespace optimizer
} // namespace model
} // namespace ell
