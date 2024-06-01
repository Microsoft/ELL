////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Reference.h (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Reference.h"

namespace ell
{
namespace value
{
    Ref<Value>::Ref(Value value) :
        _value(value.Reference())
    {}

    Ref<Value>::Ref(std::in_place_t, Value value) :
        _value(value)
    {}

    Ref<Value>::Ref(const Ref& other) :
        _value(other._value)
    {}

    Ref<Value>::Ref(Ref&& other) noexcept :
        _value(std::move(other._value))
    {}

    Ref<Value>& Ref<Value>::operator=(const Ref& other)
    {
        if (this != &other)
        {
            _value = other._value;
        }
        return *this;
    }

    Ref<Value>& Ref<Value>::operator=(Ref&& other)
    {
        if (this != &other)
        {
            _value = std::move(other._value);
        }
        return *this;
    }

    Ref<Value>::~Ref() = default;

    Value Ref<Value>::operator*() const
    {
        return _value.Dereference();
    }

    Value Ref<Value>::GetValue() const
    {
        return _value;
    }

    void Ref<Value>::SetName(const std::string& name)
    {
        _value.SetName(name);
    }

    std::string Ref<Value>::GetName() const
    {
        return _value.GetName();
    }

} // namespace value
} // namespace ell
