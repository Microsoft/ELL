////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     PortElements.cpp (model)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "PortElements.h"
#include "Model.h"
#include "Node.h"

#include <utilities/include/Exception.h>
#include <utilities/include/Hash.h>
#include <utilities/include/Tokenizer.h>

#include <sstream>

namespace ell
{
namespace model
{
    namespace
    {
        MemoryShape GetShapeSuffix(const MemoryShape& shape)
        {
            if (shape.NumDimensions() == 0)
            {
                return shape;
            }
            auto result = shape.ToVector();
            return { std::vector<int>(result.begin() + 1, result.end()) };
        }

        // Concatenate two layouts, if they're compatible.
        // Concatenating anything with the empty layout returns the original layout.
        PortMemoryLayout ConcatenateLayouts(const PortMemoryLayout& layout1, const PortMemoryLayout& layout2)
        {
            if (layout1.NumDimensions() == 0)
            {
                return layout2;
            }

            if (layout2.NumDimensions() == 0)
            {
                return layout1;
            }

            if (layout1.NumDimensions() != layout2.NumDimensions())
            {
                throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Can't concatenate layouts of different dimensions");
            }
            auto layout1Size = layout1.GetActiveSize();
            auto layout2Size = layout2.GetActiveSize();
            if (GetShapeSuffix(layout1Size) != GetShapeSuffix(layout2Size))
            {
                throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Can't concatenate layouts of different suffix sizes");
            }

            auto layout1Stride = layout1.GetExtent();
            auto layout2Stride = layout2.GetExtent();
            if (GetShapeSuffix(layout1Stride) != GetShapeSuffix(layout2Stride))
            {
                throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Can't concatenate layouts of different suffix strides");
            }

            auto layout1Order = layout1.GetLogicalDimensionOrder();
            auto layout2Order = layout2.GetLogicalDimensionOrder();
            if (layout1Order != layout2Order)
            {
                throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Can't concatenate layouts of different orders");
            }

            auto resultSize = layout1Size;
            resultSize[0] += layout2Size[0];
            return { resultSize, layout1Order };
        }

        //
        // Helper functions for parsing and stringifying proxies
        //
        std::string GetRangeString(const PortRangeProxy& range)
        {
            using std::to_string;
            std::string prefix = to_string(range.GetNodeId()) + "." + range.GetPortName();
            if (!range.IsFixedSize())
            {
                return prefix;
            }
            else if (range.Size() == 1)
            {
                return prefix + "[" + to_string(range.GetStartIndex()) + "]";
            }
            return prefix + "[" + to_string(range.GetStartIndex()) + ":" + to_string(range.GetStartIndex() + range.Size()) + "]";
        }

        PortRangeProxy ParseRange(utilities::Tokenizer& tokenizer)
        {
            auto nodeIdStr = tokenizer.ReadNextToken();
            auto nodeId = utilities::UniqueId(nodeIdStr);
            tokenizer.MatchToken(".");
            auto portName = tokenizer.ReadNextToken();
            // now check for element/element slice
            if (tokenizer.PeekNextToken() == "[")
            {
                tokenizer.MatchToken("[");
                auto token = tokenizer.ReadNextToken();
                size_t startIndex = std::stoi(token);
                size_t size = 1;
                if (tokenizer.PeekNextToken() == ":")
                {
                    tokenizer.MatchToken(":");
                    auto endIndex = std::stoi(tokenizer.ReadNextToken());
                    size = endIndex - startIndex;
                }

                tokenizer.MatchToken("]");
                return { nodeId, portName, startIndex, size };
            }
            else
            {
                return { nodeId, portName };
            }
        }

        PortRangeProxy ParseRange(const std::string& str)
        {
            std::stringstream stream(str);
            std::string delimiters = "{}[],.:";
            utilities::Tokenizer tokenizer(stream, delimiters);
            return ParseRange(tokenizer);
        }

        std::vector<PortRangeProxy> ParseRangeList(utilities::Tokenizer& tokenizer)
        {
            std::vector<PortRangeProxy> result;
            while (true)
            {
                // read a range
                result.push_back(ParseRange(tokenizer));
                if (tokenizer.PeekNextToken() != ",")
                {
                    break;
                }
                tokenizer.MatchToken(",");
            }
            return result;
        }

        std::string GetPortElementsString(const PortElementsProxy& elements)
        {
            if (elements.NumRanges() == 0)
            {
                return "{}";
            }

            if (elements.NumRanges() == 1)
            {
                return GetRangeString(elements.GetRanges()[0]);
            }

            std::string valueString = "{";
            bool first = true;
            for (const auto& range : elements.GetRanges())
            {
                if (first)
                {
                    first = false;
                }
                else
                {
                    valueString += ", ";
                }
                valueString += GetRangeString(range);
            }

            valueString += "}";
            return valueString;
        }

        // Create a PortElementsProxy from a tokenizer
        PortElementsProxy ParsePortElements(utilities::Tokenizer& tokenizer)
        {
            auto t = tokenizer.PeekNextToken();
            if (t == "{")
            {
                tokenizer.MatchToken(t);
                std::vector<PortRangeProxy> ranges;
                if (tokenizer.PeekNextToken() != "}")
                {
                    ranges = ParseRangeList(tokenizer);
                }
                tokenizer.MatchToken("}");
                return PortElementsProxy(ranges);
            }
            else
            {
                return PortElementsProxy(ParseRange(tokenizer));
            }
        }

        // Create a PortElementsProxy from a string
        PortElementsProxy ParsePortElements(const std::string& str)
        {
            std::stringstream stream(str);
            std::string delimiters = "{}[],.:";
            utilities::Tokenizer tokenizer(stream, delimiters);
            return ParsePortElements(tokenizer);
        }

        PortRange PortRangeFromArchivedProxy(utilities::Unarchiver& archiver, const PortRangeProxy& proxy)
        {
            auto& context = archiver.GetContext();
            ModelSerializationContext& modelContext = dynamic_cast<ModelSerializationContext&>(context);
            Node* node = modelContext.GetNodeFromSerializedId(proxy.GetNodeId());
            if (node == nullptr)
            {
                throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState, "Could not find archived node.");
            }

            OutputPortBase* newPort = node->GetOutputPort(proxy.GetPortName());
            if (newPort == nullptr)
            {
                throw utilities::InputException(utilities::InputExceptionErrors::nullReference, std::string("Couldn't unarchive PortRange port -- bad port name \"") + proxy.GetPortName() + "\" for port " + proxy.GetNodeId().ToString());
            }

            if (proxy.IsFixedSize())
            {
                return PortRange(*newPort, proxy.GetStartIndex(), proxy.Size());
            }
            else
            {
                return PortRange(*newPort);
            }
        }

    } // end anonymous namespace

    //
    // PortElementBase::Iterator
    //
    void PortElementsBase::Iterator::Next()
    {
        if (_ranges.empty())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange);
        }

        if (_ranges[0].Size() == 1)
        {
            _ranges.erase(_ranges.begin());
        }
        else
        {
            auto oldRange = _ranges[0];
            _ranges[0] = { *oldRange.ReferencedPort(), oldRange.GetStartIndex() + 1, oldRange.Size() - 1 };
        }

        ++_index;
    }

    data::IndexValue PortElementsBase::Iterator::Get()
    {
        if (!IsValid())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange);
        }

        return { _index, _ranges[0].ReferencedPort()->GetDoubleOutput(_ranges[0].GetStartIndex()) };
    }

    PortElementsBase::Iterator::Iterator(const std::vector<PortRange>& ranges) :
        _ranges(ranges.begin(), ranges.end())
    {
    }

    //
    // PortRange
    //
    PortRange::PortRange(const OutputPortBase& port) :
        _referencedPort(&port),
        _startIndex(0),
        _sliceSize(port.Size()),
        _isFixedSize(false)
    {}

    PortRange::PortRange(const OutputPortBase& port, size_t index) :
        _referencedPort(&port),
        _startIndex(index),
        _sliceSize(1),
        _isFixedSize(true)
    {}

    PortRange::PortRange(const OutputPortBase& port, size_t startIndex, size_t numValues) :
        _referencedPort(&port),
        _startIndex(startIndex),
        _sliceSize(numValues),
        _isFixedSize(true)
    {}

    PortRange::PortRange(const PortElementBase& element) :
        _referencedPort(element.ReferencedPort()),
        _startIndex(element.GetIndex()),
        _sliceSize(1),
        _isFixedSize(true)
    {}

    size_t PortRange::Size() const
    {
        if (_isFixedSize)
        {
            return _sliceSize;
        }
        else
        {
            return _referencedPort->Size();
        }
    }

    PortMemoryLayout PortRange::GetMemoryLayout() const
    {
        auto portLayout = ReferencedPort()->GetMemoryLayout();

        if (IsFullPortRange())
        {
            return portLayout;
        }

        // return a layout that represents the slice along the major axis for this range
        auto stride = portLayout.GetExtent();
        auto activeSize = portLayout.GetActiveSize();
        auto offset = portLayout.GetOffset();
        auto order = portLayout.GetLogicalDimensionOrder();
        activeSize[0] = Size();
        offset[0] += GetStartIndex();
        return { activeSize, stride, offset, order };
    }

    bool PortRange::IsFullPortRange() const
    {
        return GetStartIndex() == 0 && Size() == ReferencedPort()->Size();
    }

    void PortRange::WriteToArchive(utilities::Archiver& archiver) const
    {
        PortRangeProxy proxy(*this);
        archiver << GetRangeString(proxy);
    }

    void PortRange::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        std::string str;
        archiver >> str;
        auto proxy = ParseRange(str);
        auto portRange = PortRangeFromArchivedProxy(archiver, proxy);
        *this = portRange;
    }

    bool PortRange::IsAdjacent(const PortRange& other) const
    {
        return (ReferencedPort() == other.ReferencedPort()) && (GetStartIndex() + Size() == other.GetStartIndex());
    }

    void PortRange::Append(const PortRange& other)
    {
        if (IsAdjacent(other) && _isFixedSize)
        {
            _sliceSize += other.Size();
        }
    }

    bool PortRange::operator==(const PortRange& other) const
    {
        return (_referencedPort == other._referencedPort) && (_startIndex == other._startIndex) && (_sliceSize == other._sliceSize);
    }

    bool PortRange::operator!=(const PortRange& other) const
    {
        return !(*this == other);
    }

    //
    // PortElementBase
    //
    PortElementBase::PortElementBase(const OutputPortBase& port, size_t index) :
        _referencedPort(&port),
        _index(index)
    {}

    //
    // PortElementsBase
    //
    PortElementsBase::PortElementsBase(const OutputPortBase& port)
    {
        _ranges.emplace_back(port);
        ComputeSize();
    }

    PortElementsBase::PortElementsBase(const OutputPortBase& port, size_t startIndex)
    {
        _ranges.emplace_back(port, startIndex);
        ComputeSize();
    }

    PortElementsBase::PortElementsBase(const OutputPortBase& port, size_t startIndex, size_t numValues)
    {
        _ranges.emplace_back(port, startIndex, numValues);
        ComputeSize();
    }

    PortElementsBase::PortElementsBase(const PortElementBase& element)
    {
        _ranges.push_back(PortRange{ element });
        _size = 1;
    }

    PortElementsBase::PortElementsBase(const PortRange& range)
    {
        _ranges.emplace_back(range);
        _size = range.Size();
    }

    PortElementsBase::PortElementsBase(const std::vector<PortRange>& ranges)
    {
        _ranges.insert(_ranges.end(), ranges.begin(), ranges.end());
        ComputeSize();
    }

    void PortElementsBase::Append(const PortElementsBase& other)
    {
        for (const auto& range : other.GetRanges())
        {
            AddRange(range);
        }
    }

    void PortElementsBase::Reset(const PortElementsBase& other)
    {
        _ranges = other._ranges;
        ComputeSize();
    }

    bool PortElementBase::operator==(const PortElementBase& other) const
    {
        return (_referencedPort == other._referencedPort) && (_index == other._index);
    }

    bool PortElementBase::operator!=(const PortElementBase& other) const
    {
        return !(*this == other);
    }

    Port::PortType PortElementsBase::GetPortType() const
    {
        if (Size() == 0)
        {
            return Port::PortType::none;
        }

        return _ranges[0].GetPortType();
    }

    PortMemoryLayout PortElementsBase::GetMemoryLayout() const
    {
        // Concatenate layout along largest (first) dimension
        PortMemoryLayout result = {};
        for (const auto& range : _ranges)
        {
            // for non-simple ranges, the size and strides of all but the first dimension must match
            auto rangeLayout = range.GetMemoryLayout();
            result = ConcatenateLayouts(result, rangeLayout);
        }

        return result;
    }

    void PortElementsBase::Reserve(size_t numRanges)
    {
        _ranges.reserve(numRanges);
    }

    void PortElementsBase::AddRange(const PortRange& range)
    {
        // Check if range is contiguous with _ranges.back(), and if so, just add range.Size() to ranges.back()
        if (_ranges.size() > 0 && _ranges.back().IsAdjacent(range))
        {
            _ranges.back().Append(range);
        }
        else
        {
            _ranges.push_back(range);
        }
        _size += range.Size();
    }

    PortElementBase PortElementsBase::GetElement(size_t index) const
    {
        size_t sumRangeSizesSoFar = 0;
        for (const auto& range : _ranges)
        {
            if (index < sumRangeSizesSoFar + range.Size())
            {
                return PortElementBase(*range.ReferencedPort(), range.GetStartIndex() + index - sumRangeSizesSoFar);
            }
            sumRangeSizesSoFar += range.Size();
        }
        throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange, "index value " + std::to_string(index) + " exceeds PortElements range");
    }

    void PortElementsBase::ComputeSize()
    {
        _size = 0;
        for (const auto& range : _ranges)
        {
            _size += range.Size();
        }
    }

    void PortElementsBase::Consolidate()
    {
        if (_ranges.size() > 1)
        {
            // For each range, check if it's adjacent to the next one. If so, combine them
            std::vector<PortRange> newRanges;
            newRanges.push_back(_ranges[0]);
            auto numRanges = _ranges.size();
            for (size_t index = 1; index < numRanges; ++index)
            {
                const auto r = _ranges[index];
                if (newRanges.back().IsAdjacent(r))
                {
                    newRanges.back().Append(r);
                }
                else
                {
                    newRanges.push_back(r);
                }
            }
            _ranges = newRanges;

            ComputeSize();
        }
    }

    bool PortElementsBase::operator==(const PortElementsBase& other) const
    {
        if ((GetPortType() != other.GetPortType()) || (NumRanges() != other.NumRanges()))
        {
            return false;
        }

        const auto numRanges = static_cast<int>(NumRanges());
        const auto& ranges = GetRanges();
        const auto& otherRanges = other.GetRanges();
        for (int index = 0; index < numRanges; ++index)
        {
            if (ranges[index] != otherRanges[index])
            {
                return false;
            }
        }
        return true;
    }

    bool PortElementsBase::operator!=(const PortElementsBase& other) const
    {
        return !(*this == other);
    }

    void PortElementsBase::WriteToArchive(utilities::Archiver& archiver) const
    {
        PortElementsProxy proxy(*this);
        archiver << GetPortElementsString(proxy);
    }

    void PortElementsBase::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        std::string str;
        archiver >> str;
        auto proxy = ParsePortElements(str);
        for (const auto& rangeProxy : proxy.GetRanges())
        {
            _ranges.push_back(PortRangeFromArchivedProxy(archiver, rangeProxy));
        }
        ComputeSize();
    }

    //
    // PortRangeProxy
    //

    PortRangeProxy::PortRangeProxy(Node::NodeId nodeId, std::string portName) :
        _nodeId(nodeId),
        _portName(portName),
        _portType(Port::PortType::none),
        _startIndex(0),
        _sliceSize(0),
        _isFixedSize(false)
    {
    }

    PortRangeProxy::PortRangeProxy(Node::NodeId nodeId, std::string portName, size_t startIndex) :
        _nodeId(nodeId),
        _portName(portName),
        _portType(Port::PortType::none),
        _startIndex(startIndex),
        _sliceSize(0),
        _isFixedSize(true)
    {
    }

    PortRangeProxy::PortRangeProxy(Node::NodeId nodeId, std::string portName, size_t startIndex, size_t numValues) :
        _nodeId(nodeId),
        _portName(portName),
        _portType(Port::PortType::none),
        _startIndex(startIndex),
        _sliceSize(numValues),
        _isFixedSize(true)
    {
    }

    PortRangeProxy::PortRangeProxy(Node::NodeId nodeId, std::string portName, Port::PortType portType, size_t startIndex, size_t numValues) :
        _nodeId(nodeId),
        _portName(portName),
        _portType(portType),
        _startIndex(startIndex),
        _sliceSize(numValues),
        _isFixedSize(true)
    {
    }

    PortRangeProxy::PortRangeProxy(const PortRange& range)
    {
        if (range.ReferencedPort() == nullptr || range.ReferencedPort()->GetNode() == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::nullReference);
        }

        _nodeId = range.ReferencedPort()->GetNode()->GetId();
        _portName = range.ReferencedPort()->GetName();
        _portType = range.GetPortType();
        _startIndex = range.GetStartIndex();
        _sliceSize = range.Size();
        _isFixedSize = range.IsFixedSize();
    }

    void PortRangeProxy::WriteToArchive(utilities::Archiver& archiver) const
    {
        archiver << GetRangeString(*this);
    }

    void PortRangeProxy::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        std::string str;
        archiver >> str;
        auto proxy = ParseRange(str);
        *this = proxy;
    }

    //
    // PortElementsProxy
    //

    PortElementsProxy::PortElementsProxy() :
        _portType(Port::PortType::none)
    {
    }

    PortElementsProxy::PortElementsProxy(Port::PortType portType) :
        _portType(portType)
    {
    }

    PortElementsProxy::PortElementsProxy(const PortElementsBase& elements)
    {
        for (auto r : elements.GetRanges())
        {
            Append(r);
        }
    }

    PortElementsProxy::PortElementsProxy(const PortRangeProxy& range)
    {
        Append(range);
    }

    PortElementsProxy::PortElementsProxy(const std::vector<PortRangeProxy>& ranges)
    {
        for (auto r : ranges)
        {
            Append(r);
        }
    }

    void PortElementsProxy::Append(const PortRangeProxy& range)
    {
        if (_portType == Port::PortType::none)
        {
            _portType = range.GetPortType();
        }
        else if (_portType != range.GetPortType())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::typeMismatch);
        }
        _ranges.push_back(range);
    }

    void PortElementsProxy::WriteToArchive(utilities::Archiver& archiver) const
    {
        archiver << GetPortElementsString(*this);
    }

    void PortElementsProxy::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        std::string str;
        archiver >> str;
        auto proxy = ParsePortElements(str);
        *this = proxy;
    }

    //
    // Helper functions
    //

    PortElementsProxy ParsePortElementsProxy(std::string str)
    {
        return ParsePortElements(str);
    }

    PortElementsBase ProxyToPortElements(const Model& model, const PortElementsProxy& proxy)
    {
        PortElementsBase elements;
        for (auto rangeProxy : proxy.GetRanges())
        {
            const Node* node = model.GetNode(rangeProxy.GetNodeId());
            if (node == nullptr)
            {
                throw utilities::InputException(utilities::InputExceptionErrors::nullReference, std::string("Couldn't unarchive PortRange port -- bad node id \"") + rangeProxy.GetNodeId().ToString() + "\"");
            }

            const OutputPortBase* port = node->GetOutputPort(rangeProxy.GetPortName());
            if (port == nullptr)
            {
                throw utilities::InputException(utilities::InputExceptionErrors::nullReference, std::string("Couldn't unarchive PortRange port -- bad port name \"") + rangeProxy.GetPortName() + "\" for port " + rangeProxy.GetNodeId().ToString());
            }

            if (rangeProxy.GetPortType() != Port::PortType::none && port->GetType() != rangeProxy.GetPortType())
            {
                throw utilities::InputException(utilities::InputExceptionErrors::typeMismatch);
            }

            // Transform to real range
            auto size = rangeProxy.Size();
            if (!rangeProxy.IsFixedSize())
            {
                size = port->Size();
            }
            elements.Append(PortRange(*port, rangeProxy.GetStartIndex(), size));
        }
        return elements;
    }
} // namespace model
} // namespace ell

//
// hash functions for PortElementBase and PortRange
//
std::hash<ell::model::PortElementBase>::result_type std::hash<ell::model::PortElementBase>::operator()(const argument_type& element) const
{
    using ::ell::utilities::HashCombine;

    size_t hash = 0;
    HashCombine(hash, element.ReferencedPort());
    HashCombine(hash, element.GetIndex());

    return hash;
}

std::hash<ell::model::PortRange>::result_type std::hash<ell::model::PortRange>::operator()(const argument_type& range) const
{
    using ::ell::utilities::HashCombine;

    size_t hash = 0;
    HashCombine(hash, range.ReferencedPort());
    HashCombine(hash, range.Size());
    HashCombine(hash, range.GetStartIndex());

    return hash;
}

std::hash<ell::model::PortElementsBase>::result_type std::hash<ell::model::PortElementsBase>::operator()(const argument_type& elements) const
{
    return ::ell::utilities::HashValue(elements.GetRanges());
}
