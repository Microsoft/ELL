////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ObjectArchiver.cpp (utilities)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ObjectArchiver.h"
#include "IArchivable.h"
#include "Unused.h"

#include <string>
#include <vector>

namespace ell
{
namespace utilities
{
    ObjectArchiver::ObjectArchiver(SerializationContext context) :
        Unarchiver(context)
    {
    }

    ObjectArchiver::ObjectArchiver(const ObjectArchive& objectDescription, SerializationContext context) :
        Unarchiver(std::move(context)),
        _objectDescription(objectDescription)
    {
    }

//
// Serialization
//
#define ARCHIVE_TYPE_OP(t) IMPLEMENT_ARCHIVE_VALUE(ObjectArchiver, t);
    ARCHIVABLE_TYPES_LIST
#undef ARCHIVE_TYPE_OP

    // strings
    void ObjectArchiver::ArchiveValue(const char* name, const std::string& value)
    {
        if (std::string{ "" } == name)
        {
            _objectDescription.SetValue(value);
            _objectDescription << value;
        }
        else
        {
            _objectDescription[name] << value;
        }
    }

    void ObjectArchiver::ArchiveNull(const char* name)
    {
        if (std::string{ "" } != name)
        {
            _objectDescription[name].SetNull();
        }
    }

    // IArchivable

    void ObjectArchiver::ArchiveObject(const char* name, const IArchivable& value)
    {
        if (std::string{ "" } == name)
        {
            Archiver::ArchiveObject(name, value);
        }
        else
        {
            // need to create new object description archiver here, I guess
            SerializationContext context;
            ObjectArchiver archiver(context);
            archiver << value;
            _objectDescription[name] = archiver.GetObjectArchive();
        }
    }

    bool ObjectArchiver::HasNextPropertyName(const std::string& name)
    {
        return _objectDescription.HasProperty(name);
    }

//
// Arrays
//
#define ARCHIVE_TYPE_OP(t) IMPLEMENT_ARCHIVE_ARRAY(ObjectArchiver, t);
    ARCHIVABLE_TYPES_LIST
#undef ARCHIVE_TYPE_OP

    void ObjectArchiver::ArchiveArray(const char* name, const std::vector<std::string>& array)
    {
        WriteArray(name, array);
    }

    void ObjectArchiver::ArchiveArray(const char* name, const std::string& baseTypeName, const std::vector<const IArchivable*>& array)
    {
        UNUSED(baseTypeName);
        if (std::string{ "" } == name)
        {
            _objectDescription.SetValue(array);
            _objectDescription << array;
        }
        else
        {
            _objectDescription[name] << array;
        }
    }

//
// Deserialization
//
#define ARCHIVE_TYPE_OP(t) IMPLEMENT_UNARCHIVE_VALUE(ObjectArchiver, t);
    ARCHIVABLE_TYPES_LIST
#undef ARCHIVE_TYPE_OP

    // strings
    void ObjectArchiver::UnarchiveValue(const char* name, std::string& value)
    {
        if (std::string{ "" } == name)
        {
            _objectDescription >> value;
        }
        else
        {
            _objectDescription[name] >> value;
        }
    }

    bool ObjectArchiver::UnarchiveNull(const char* name)
    {
        if (std::string{ "" } == name)
        {
            return !_objectDescription.HasValue();
        }
        else
        {
            return !_objectDescription[name].HasValue();
        }
    }

    // IArchivable
    void ObjectArchiver::UnarchiveObject(const char* name, IArchivable& value)
    {
        if (std::string{ "" } == name)
        {
            Unarchiver::UnarchiveObject(name, value);
        }
        else
        {
            ObjectArchiver propertyUnarchiver(_objectDescription[name], GetContext());
            propertyUnarchiver.UnarchiveObject("", value);
        }
    }

//
// Arrays
//
#define ARCHIVE_TYPE_OP(t) IMPLEMENT_UNARCHIVE_ARRAY(ObjectArchiver, t);
    ARCHIVABLE_TYPES_LIST
#undef ARCHIVE_TYPE_OP

    void ObjectArchiver::UnarchiveArray(const char* name, std::vector<std::string>& array)
    {
        if (std::string{ "" } == name)
        {
            _objectDescription >> array;
        }
        else
        {
            _objectDescription[name] >> array;
        }
    }

    bool ObjectArchiver::BeginUnarchiveArrayItem(const std::string& typeName)
    {
        UNUSED(typeName);
        return true;
    }

    void ObjectArchiver::EndUnarchiveArrayItem(const std::string& typeName)
    {
        UNUSED(typeName);
    }
} // namespace utilities
} // namespace ell
