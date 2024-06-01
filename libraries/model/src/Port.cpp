////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Port.cpp (model)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Port.h"
#include "Node.h"

#include <utilities/include/Boolean.h>
#include <utilities/include/StringUtil.h>

#include <cctype>

namespace ell
{
namespace model
{
    namespace
    {
        //
        // Relevant archive format versions
        //
        constexpr utilities::ArchiveVersion noMetadataArchiveVersion = { utilities::ArchiveVersionNumbers::v0_initial };
        constexpr utilities::ArchiveVersion metadataArchiveVersion = { utilities::ArchiveVersionNumbers::v3_model_metadata };
    } // namespace

    //
    // Specializations of GetPortType
    //
    template <>
    Port::PortType Port::GetPortType<float>()
    {
        return Port::PortType::smallReal;
    }

    template <>
    Port::PortType Port::GetPortType<double>()
    {
        return Port::PortType::real;
    }

    template <>
    Port::PortType Port::GetPortType<int>()
    {
        return Port::PortType::integer;
    }

    template <>
    Port::PortType Port::GetPortType<int64_t>()
    {
        return Port::PortType::bigInt;
    }

    template <>
    Port::PortType Port::GetPortType<bool>()
    {
        return Port::PortType::boolean;
    }

    template <>
    Port::PortType Port::GetPortType<utilities::Boolean>()
    {
        return Port::PortType::boolean;
    }

    std::string Port::GetFullName() const
    {
        return GetNode()->GetId().ToString() + "." + GetName();
    }

    std::string Port::GetVariableName(const std::string& defaultName) const
    {
        // Find a friendly name in the node's metadata.  If the user bothered to name the node in
        // the model then this name probably has semantic significance worthy of reflecting in this port
        // friendly name and then emitting in generated function parameters.
        std::string name = this->GetNode()->GetFriendlyName();
        if (!name.empty())
        {
            if (std::isdigit(name[0]))
            {
                // not a valid identifier, so use the defaultName instead.
                name = defaultName;
            }
        }
        else
        {
            name = defaultName;
        }
        // If this port has an interesting non-default name then append it. This can happen
        // in the case where a node has multiple ports with interesting names.
        if (this->_name != Node::defaultOutputPortName && this->_name != Node::defaultInputPortName)
        {
            name += "_" + this->_name;
        }
        return utilities::MakeValidIdentifier(name);
    }

    utilities::ArchiveVersion Port::GetArchiveVersion() const
    {
        if (_metadata.IsEmpty())
        {
            return noMetadataArchiveVersion;
        }
        else
        {
            return metadataArchiveVersion;
        }
    }

    bool Port::CanReadArchiveVersion(const utilities::ArchiveVersion& version) const
    {
        return version >= noMetadataArchiveVersion && version <= metadataArchiveVersion;
    }

    void Port::WriteToArchive(utilities::Archiver& archiver) const
    {
        archiver["nodeId"] << _node->GetId();
        archiver["name"] << _name;
        archiver["type"] << static_cast<int>(_type);
        if (!_metadata.IsEmpty())
        {
            archiver["metadata"] << _metadata;
        }
    }

    void Port::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        Node::NodeId id;
        archiver["nodeId"] >> id;
        archiver["name"] >> _name;
        int typeCode;
        archiver["type"] >> typeCode;
        _type = static_cast<PortType>(typeCode);
        archiver.OptionalProperty("metadata") >> _metadata;
    }

    std::string GetPortCTypeName(ell::model::Port::PortType type)
    {
        switch (type)
        {
        case ell::model::Port::PortType::none:
            return "void";
        case ell::model::Port::PortType::smallReal:
            return "float";
        case ell::model::Port::PortType::real:
            return "double";
        case ell::model::Port::PortType::integer:
            return "int";
        case ell::model::Port::PortType::bigInt:
            return "int64_t";
        case ell::model::Port::PortType::categorical:
            return "int";
        case ell::model::Port::PortType::boolean:
            return "bool"; // ???
        default:
            return "Unknown";
        };
    }
} // namespace model
} // namespace ell
