////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     MemoryLayout.cpp (utilities)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MemoryLayout.h"
#include "Exception.h"
#include "Hash.h"
#include "Unused.h"

#include <cassert>
#include <numeric>
#include <sstream>

namespace ell
{
namespace utilities
{
    namespace
    {
        MemoryShape Permute(const MemoryShape& shape, const DimensionOrder& order)
        {
            const int numDimensions = shape.NumDimensions();
            std::vector<int> result(numDimensions);
            for (int index = 0; index < numDimensions; ++index)
            {
                result[index] = shape[order[index]];
            }
            return { result };
        }

        MemoryCoordinates Permute(const MemoryCoordinates& coordinates, const DimensionOrder& order)
        {
            const int numDimensions = coordinates.NumDimensions();
            std::vector<int> result(numDimensions);
            for (int index = 0; index < numDimensions; ++index)
            {
                result[index] = coordinates[order[index]];
            }
            return { result };
        }

        MemoryShape ReversePermute(const MemoryShape& shape, const DimensionOrder& order)
        {
            const int numDimensions = shape.NumDimensions();
            std::vector<int> result(numDimensions);
            for (int index = 0; index < numDimensions; ++index)
            {
                result[order[index]] = shape[index];
            }
            return { result };
        }

        MemoryCoordinates ReversePermute(const MemoryCoordinates& coordinates, const DimensionOrder& order)
        {
            const int numDimensions = coordinates.NumDimensions();
            std::vector<int> result(numDimensions);
            for (int index = 0; index < numDimensions; ++index)
            {
                result[order[index]] = coordinates[index];
            }
            return { result };
        }

        MemoryShape ContiguousCumulativeIncrement(const MemoryShape& extent)
        {
            const auto numDimensions = extent.NumDimensions();
            std::vector<int> result(numDimensions);
            int prevScale = 1;
            for (int index = numDimensions - 1; index >= 0; --index)
            {
                result[index] = prevScale;
                prevScale = prevScale * extent[index];
            }
            return { result };
        }

        MemoryShape CreateExtent(const MemoryShape& physicalSize, const MemoryShape& physicalPadding)
        {
            assert(physicalSize.NumDimensions() == physicalPadding.NumDimensions());
            std::vector<int> extent(physicalSize.NumDimensions());
            std::transform(physicalSize.begin(),
                           physicalSize.end(),
                           physicalPadding.begin(),
                           extent.begin(),
                           [](int size, int padding) { return size + (2 * padding); });

            return extent;
        }
    } // namespace

    //
    // DimensionVector
    //
    DimensionVector::DimensionVector(const std::vector<size_t>& elements) :
        _data(elements.size())
    {
        std::transform(elements.begin(), elements.end(), _data.begin(), [](size_t x) { return static_cast<int>(x); });
    }

    DimensionVector::DimensionVector(const std::initializer_list<size_t>& elements) :
        _data(elements.size())
    {
        std::transform(elements.begin(), elements.end(), _data.begin(), [](size_t x) { return static_cast<int>(x); });
    }

    //
    // DimensionOrder
    //

    DimensionOrder::DimensionOrder(int numDimensions) :
        DimensionVector(std::vector<int>(numDimensions))
    {
        std::iota(_data.begin(), _data.end(), 0);
    }

    DimensionOrder::DimensionOrder(const std::vector<int>& order) :
        DimensionVector(order)
    {
        std::vector<int> test(order.size());
        std::iota(test.begin(), test.end(), 0);
        if (!std::is_permutation(order.begin(), order.end(), test.begin()))
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Dimension order must be a valid permutation vector.");
        }
    }

    DimensionOrder::DimensionOrder(const std::initializer_list<int>& order) :
        DimensionVector(order)
    {
        std::vector<int> test(order.size());
        std::iota(test.begin(), test.end(), 0);
        if (!std::is_permutation(order.begin(), order.end(), test.begin()))
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Dimension order must be a valid permutation vector.");
        }
    }

    int DimensionOrder::operator[](int index) const
    {
        return DimensionVector::operator[](index);
    }

    bool DimensionOrder::IsCanonicalOrder() const
    {
        for (int index = 0; index < NumDimensions(); ++index)
        {
            if (index != (*this)[index])
            {
                return false;
            }
        }
        return true;
    }

    std::string DimensionOrder::ToString() const
    {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    //
    // MemoryShape / Coordinates
    //

    int MemoryShape::NumElements() const
    {
        return std::accumulate(_data.begin(), _data.end(), 1, std::multiplies<int>());
    }

    void MemoryShape::Resize(int numDimensions)
    {
        if (numDimensions > static_cast<int>(_data.size()))
        {
            int extraDimensions = numDimensions - static_cast<int>(_data.size());
            _data.insert(_data.begin(), extraDimensions, 1);
        }
        while (numDimensions < static_cast<int>(_data.size()))
        {
            _data[1] *= _data[0];
            _data.erase(_data.begin());
        }
    }

    std::string MemoryShape::ToString() const
    {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    std::string MemoryCoordinates::ToString() const
    {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    //
    // MemoryLayout
    //
    MemoryLayout::MemoryLayout(const MemoryShape& physicalDimensionSize) :
        MemoryLayout(physicalDimensionSize, physicalDimensionSize, MemoryShape(std::vector<int>(physicalDimensionSize.NumDimensions(), 0)))
    {}

    MemoryLayout::MemoryLayout(const MemoryShape& physicalDimensionSize, const MemoryShape& physicalDimensionPadding) :
        MemoryLayout(physicalDimensionSize, CreateExtent(physicalDimensionSize, physicalDimensionPadding), physicalDimensionPadding)
    {
        for (int index = 0; index < _size.NumDimensions(); ++index)
        {
            if (physicalDimensionPadding[index] < 0)
            {
                throw InputException(InputExceptionErrors::invalidArgument, "Padding must be positive or zero.");
            }
        }
    }

    MemoryLayout::MemoryLayout(const MemoryShape& physicalDimensionSize, const MemoryShape& physicalDimensionExtent, const MemoryShape& physicalDimensionOffset) :
        MemoryLayout(physicalDimensionSize, physicalDimensionExtent, physicalDimensionOffset, ContiguousCumulativeIncrement(physicalDimensionExtent))
    {}

    MemoryLayout::MemoryLayout(const MemoryShape& physicalDimensionSize, const MemoryShape& physicalDimensionExtent, const MemoryShape& physicalDimensionOffset, const MemoryShape& physicalDimensionIncrement) :
        MemoryLayout(physicalDimensionSize, physicalDimensionExtent, physicalDimensionOffset, physicalDimensionIncrement, physicalDimensionSize.NumDimensions())
    {}

    MemoryLayout::MemoryLayout(const MemoryShape& physicalDimensionSize, const DimensionOrder& order) :
        MemoryLayout(physicalDimensionSize, physicalDimensionSize, MemoryShape(std::vector<int>(physicalDimensionSize.NumDimensions(), 0)), order)
    {}

    MemoryLayout::MemoryLayout(const MemoryShape& physicalDimensionSize, const MemoryShape& physicalDimensionPadding, const DimensionOrder& order) :
        MemoryLayout(physicalDimensionSize, CreateExtent(physicalDimensionSize, physicalDimensionPadding), physicalDimensionPadding, order)
    {
        for (int index = 0; index < _size.NumDimensions(); ++index)
        {
            if (physicalDimensionPadding[index] < 0)
            {
                throw InputException(InputExceptionErrors::invalidArgument, "Padding must be positive or zero.");
            }
        }
    }

    MemoryLayout::MemoryLayout(const MemoryShape& physicalDimensionSize, const MemoryShape& physicalDimensionExtent, const MemoryShape& physicalDimensionOffset, const DimensionOrder& order) :
        MemoryLayout(physicalDimensionSize, physicalDimensionExtent, physicalDimensionOffset, ContiguousCumulativeIncrement(physicalDimensionExtent), order)
    {}

    MemoryLayout::MemoryLayout(const MemoryShape& physicalDimensionSize, const MemoryShape& physicalDimensionExtent, const MemoryShape& physicalDimensionOffset, const MemoryShape& physicalDimensionIncrement, const DimensionOrder& order) :
        _size(physicalDimensionSize),
        _extent(physicalDimensionExtent),
        _offset(physicalDimensionOffset),
        _increment(physicalDimensionIncrement),
        _dimensionOrder(order)
    {
        for (int index = 0; index < _size.NumDimensions(); ++index)
        {
            if (_size[index] + _offset[index] > _extent[index])
            {
                throw InputException(InputExceptionErrors::invalidArgument,
                                     "Extent must be larger or equal to the size plus offset.");
            }
        }
    }

    bool MemoryLayout::HasPadding() const
    {
        return _size != _extent;
    }

    int MemoryLayout::GetActiveSize(size_t index) const
    {
        BoundsCheckDimensionIndex(index);
        return _size[index];
    }

    int MemoryLayout::GetExtent(size_t index) const
    {
        BoundsCheckDimensionIndex(index);
        return _extent[index];
    }

    int MemoryLayout::GetOffset(size_t index) const
    {
        BoundsCheckDimensionIndex(index);
        return _offset[index];
    }

    int MemoryLayout::GetLogicalDimensionActiveSize(size_t index) const
    {
        BoundsCheckDimensionIndex(index);
        return GetLogicalDimensionActiveSize()[index];
    }

    int MemoryLayout::GetLogicalDimensionExtent(size_t index) const
    {
        BoundsCheckDimensionIndex(index);
        return GetLogicalDimensionExtent()[index];
    }

    int MemoryLayout::GetLogicalDimensionOffset(size_t index) const
    {
        BoundsCheckDimensionIndex(index);
        return GetLogicalDimensionOffset()[index];
    }

    size_t MemoryLayout::GetCumulativeIncrement(size_t index) const
    {
        BoundsCheckDimensionIndex(index);
        return _increment[index];
    }

    size_t MemoryLayout::NumElements() const
    {
        return static_cast<size_t>(_size.NumElements());
    }

    size_t MemoryLayout::GetMemorySize() const
    {
        return static_cast<size_t>(_extent.NumElements());
    }

    size_t MemoryLayout::GetDataOffset() const
    {
        size_t result = 0;
        auto increment = GetCumulativeIncrement();
        const int numDimensions = NumDimensions();
        for (int index = 0; index < numDimensions; ++index)
        {
            result += _offset[index] * increment[index];
        }
        return result;
    }

    size_t MemoryLayout::GetEntryOffset(const MemoryCoordinates& physicalCoordinates) const
    {
        const auto& offset = GetOffset();
        const auto& increment = GetCumulativeIncrement();
        const auto numDimensions = NumDimensions();
        size_t result = 0;
        for (int index = 0; index < numDimensions; ++index)
        {
            result += increment[index] * (physicalCoordinates[index] + offset[index]);
        }
        return result;
    }

    size_t MemoryLayout::GetFirstEntryOffset() const
    {
        const auto numDimensions = NumDimensions();
        MemoryCoordinates firstEntry(std::vector<int>(numDimensions, 0));
        return GetEntryOffset(firstEntry);
    }

    MemoryCoordinates MemoryLayout::GetPhysicalCoordinatesFromOffset(size_t index) const
    {
        const auto numDim = NumDimensions();
        std::vector<int> result(numDim);
        int offset = static_cast<int>(index);
        for (int d = 0; d < numDim; ++d)
        {
            const int thisExtent = static_cast<int>(GetCumulativeIncrement(d));
            const int x = offset / thisExtent;
            result[d] = x - GetOffset(d);
            offset = (offset % thisExtent);
        }
        return result;
    }

    MemoryCoordinates MemoryLayout::GetLogicalCoordinates(const MemoryCoordinates& physicalCoordinates) const
    {
        if (NumDimensions() != physicalCoordinates.NumDimensions())
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Must have the same number of dimensions.");
        }

        return ReversePermute(physicalCoordinates, _dimensionOrder);
    }

    MemoryShape MemoryLayout::GetLogicalDimensionActiveSize() const
    {
        return ReversePermute(_size, _dimensionOrder);
    }

    MemoryShape MemoryLayout::GetLogicalDimensionExtent() const
    {
        return ReversePermute(_extent, _dimensionOrder);
    }

    MemoryShape MemoryLayout::GetLogicalDimensionOffset() const
    {
        return ReversePermute(_offset, _dimensionOrder);
    }

    MemoryShape MemoryLayout::GetLogicalDimensionIncrement() const
    {
        return ReversePermute(_increment, _dimensionOrder);
    }

    size_t MemoryLayout::GetLogicalDimensionIncrement(size_t index) const
    {
        return ReversePermute(_increment, _dimensionOrder)[index];
    }

    size_t MemoryLayout::GetLogicalEntryOffset(const MemoryCoordinates& logicalCoordinates) const
    {
        return GetEntryOffset(GetPhysicalCoordinates(logicalCoordinates));
    }

    MemoryCoordinates MemoryLayout::GetLogicalCoordinatesFromOffset(size_t index) const
    {
        return GetLogicalCoordinates(GetPhysicalCoordinatesFromOffset(index));
    }

    MemoryCoordinates MemoryLayout::GetPhysicalCoordinates(const MemoryCoordinates& logicalCoordinates) const
    {
        if (NumDimensions() != logicalCoordinates.NumDimensions())
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Must have the same number of dimensions.");
        }

        return Permute(logicalCoordinates, _dimensionOrder);
    }

    int MemoryLayout::GetPhysicalDimension(int logicalDimension) const
    {
        if (logicalDimension < 0 || logicalDimension >= _dimensionOrder.NumDimensions())
        {
            throw InputException(InputExceptionErrors::indexOutOfRange);
        }

        if (auto it = std::find(_dimensionOrder.begin(), _dimensionOrder.end(), logicalDimension);
            it != _dimensionOrder.end())
        {
            return static_cast<int>(std::distance(_dimensionOrder.begin(), it));
        }
        else
        {
            throw InputException(InputExceptionErrors::indexOutOfRange);
        }
    }

    int MemoryLayout::GetLogicalDimension(int physicalDimension) const
    {
        if (physicalDimension < 0 || physicalDimension >= _dimensionOrder.NumDimensions())
        {
            throw InputException(InputExceptionErrors::indexOutOfRange);
        }

        return _dimensionOrder[physicalDimension];
    }

    bool MemoryLayout::IsOutOfBounds(const MemoryCoordinates& physicalCoordinates) const
    {
        const int numDimensions = NumDimensions();
        for (int index = 0; index < numDimensions; ++index)
        {
            if (physicalCoordinates[index] + _offset[index] < 0 ||
                physicalCoordinates[index] - _offset[index] >= _extent[index])
            {
                return true;
            }
        }
        return false;
    }

    bool MemoryLayout::IsContiguous() const
    {
        return _size == _extent && IsCanonicalOrder();
    }

    bool MemoryLayout::IsCanonicalOrder() const
    {
        return _dimensionOrder.IsCanonicalOrder();
    }

    MemoryLayout MemoryLayout::ReorderedCopy(const DimensionOrder& newOrder) const
    {
        MemoryLayout result{ Permute(GetLogicalDimensionActiveSize(), newOrder),
                             Permute(GetLogicalDimensionExtent(), newOrder),
                             Permute(GetLogicalDimensionOffset(), newOrder),
                             newOrder };
        return result;
    }

    MemoryLayout MemoryLayout::GetSliceLayout(int physicalDimension) const
    {
        if (physicalDimension >= NumDimensions())
        {
            throw InputException(InputExceptionErrors::indexOutOfRange,
                                 "Can't slice along a dimension greater than the number of dimensions");
        }

        auto size = _size.ToVector();
        auto extent = _extent.ToVector();
        auto offset = _offset.ToVector();
        auto increment = _increment.ToVector();
        auto order = _dimensionOrder.ToVector();

        if (physicalDimension > 0)
        {
            extent[physicalDimension - 1] *= extent[physicalDimension];
        }

        for (auto v : { &size, &extent, &offset, &increment, &order })
        {
            v->erase(v->begin() + physicalDimension);
        }

        // If the chosen physical dimension maps to a logical dimension that's anything
        // other than the innermost (logical) dimension, decrease the remaining dimensions by 1
        // There is expected to be, at the most, one dimension that ends up at 0. Either a dimension
        // 1 that gets decreased to 0, or a dimension 0 that goes to -1 and then gets clamped to 0.
        auto dimensionRemoved = _dimensionOrder[physicalDimension];
        for (auto& i : order)
        {
            if (i > dimensionRemoved)
            {
                --i;
            }
        }

        return { MemoryShape{ size },
                 MemoryShape{ extent },
                 MemoryShape{ offset },
                 MemoryShape{ increment },
                 DimensionOrder{ order } };
    }

    MemoryLayout MemoryLayout::CopyWithExtraDimensions(int addedDimensions) const
    {
        if (addedDimensions < 0)
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Number of dimensions to add must be non-negative.");
        }

        // Create prefixes of new layout properties
        std::vector<int> size(addedDimensions, 1);
        std::vector<int> extent(addedDimensions, 1);
        std::vector<int> offset(addedDimensions, 0);
        std::vector<int> increment(addedDimensions, static_cast<int>(GetMemorySize()));
        std::vector<int> order(addedDimensions);
        std::iota(order.begin(), order.end(), 0);

        // Append existing layout properties
        size.insert(size.end(), _size.begin(), _size.end());
        extent.insert(extent.end(), _extent.begin(), _extent.end());
        offset.insert(offset.end(), _offset.begin(), _offset.end());
        increment.insert(increment.end(), _increment.begin(), _increment.end());
        order.insert(order.end(), _dimensionOrder.begin(), _dimensionOrder.end());

        // Fix up order
        auto start = order.begin() + addedDimensions;
        std::transform(start, order.end(), start, [addedDimensions](auto x) {
            return x + addedDimensions;
        });

        return { MemoryShape{ size },
                 MemoryShape{ extent },
                 MemoryShape{ offset },
                 MemoryShape{ increment },
                 DimensionOrder{ order } };
    }

    void MemoryLayout::WriteToArchive(utilities::Archiver& archiver) const
    {
        archiver["size"] << _size.ToVector();
        archiver["extent"] << _extent.ToVector();
        archiver["offset"] << _offset.ToVector();
        archiver["order"] << _dimensionOrder.ToVector();
        archiver["increment"] << _increment.ToVector();
    }

    void MemoryLayout::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        std::vector<int> temp;
        archiver["size"] >> temp;
        _size = { temp };

        // Try to read the extent field. If successful, check and make sure the old stride field doesn't exist.
        // If the extent field doesn't exist, then the stride field does need to exist.
        temp.resize(_size.NumDimensions());
        temp.clear();
        archiver.OptionalProperty("extent") >> temp;
        if (!temp.empty())
        {
            _extent = { temp };
            temp.clear();
            archiver.OptionalProperty("stride") >> temp;
            if (!temp.empty())
            {
                throw InputException(InputExceptionErrors::badData, "MemoryLayout cannot contain 'extent' and 'stride' information");
            }
        }
        else
        {
            archiver.OptionalProperty("stride") >> temp;
            if (!temp.empty())
            {
                _extent = { temp };
            }
        }

        archiver["offset"] >> temp;
        _offset = { temp };

        std::iota(temp.begin(), temp.end(), 0);
        archiver.OptionalProperty("order") >> temp;
        _dimensionOrder = { temp };

        temp.clear();
        archiver.OptionalProperty("increment", ContiguousCumulativeIncrement(_extent).ToVector()) >> temp;
        _increment = { temp };
    }

    void MemoryLayout::BoundsCheckDimensionIndex(size_t index) const
    {
        if (static_cast<int>(index) >= NumDimensions())
        {
            throw InputException(InputExceptionErrors::indexOutOfRange, "Dimension index out-of-bounds.");
        }
    }

    std::string MemoryLayout::ToString() const
    {
        std::stringstream ss;
        ss << *this;
        return ss.str();
    }

    bool Equal(const DimensionVector& shape1, const DimensionVector& shape2)
    {
        auto size = shape1.NumDimensions();
        if (size != shape2.NumDimensions())
        {
            return false;
        }

        for (int index = 0; index < size; ++index)
        {
            if (shape1[index] != shape2[index])
            {
                return false;
            }
        }
        return true;
    }

    bool operator==(const DimensionOrder& order1, const DimensionOrder& order2)
    {
        return Equal(order1, order2);
    }

    bool operator!=(const DimensionOrder& order1, const DimensionOrder& order2)
    {
        return !Equal(order1, order2);
    }

    bool operator==(const MemoryShape& shape1, const MemoryShape& shape2)
    {
        return Equal(shape1, shape2);
    }

    bool operator!=(const MemoryShape& shape1, const MemoryShape& shape2)
    {
        return !Equal(shape1, shape2);
    }

    bool operator==(const MemoryCoordinates& shape1, const MemoryCoordinates& shape2)
    {
        return Equal(shape1, shape2);
    }

    bool operator!=(const MemoryCoordinates& shape1, const MemoryCoordinates& shape2)
    {
        return !Equal(shape1, shape2);
    }

    bool MemoryLayoutsEqual(const MemoryLayout& layout1, const MemoryLayout& layout2)
    {
        return (layout1.GetExtent() == layout2.GetExtent()) && (layout1.GetActiveSize() == layout2.GetActiveSize()) &&
               (layout1.GetOffset() == layout2.GetOffset() && layout1.GetLogicalDimensionOrder() == layout2.GetLogicalDimensionOrder());
    }

    bool operator==(const MemoryLayout& layout1, const MemoryLayout& layout2)
    {
        return MemoryLayoutsEqual(layout1, layout2);
    }

    bool operator!=(const MemoryLayout& layout1, const MemoryLayout& layout2)
    {
        return !MemoryLayoutsEqual(layout1, layout2);
    }

    std::ostream& operator<<(std::ostream& out, const utilities::DimensionOrder& order)
    {
        if (order.NumDimensions() == 0)
        {
            out << "{}";
        }
        else
        {
            out << order[0];
            auto numDimensions = order.NumDimensions();
            for (int index = 1; index < numDimensions; ++index)
            {
                out << ", " << order[index];
            }
        }
        return out;
    }

    std::ostream& operator<<(std::ostream& out, const utilities::MemoryCoordinates& coords)
    {
        if (coords.NumDimensions() == 0)
        {
            out << "{}";
        }
        else
        {
            out << coords[0];
            auto numDimensions = coords.NumDimensions();
            for (int index = 1; index < numDimensions; ++index)
            {
                out << ", " << coords[index];
            }
        }
        return out;
    }

    std::ostream& operator<<(std::ostream& out, const utilities::MemoryShape& shape)
    {
        if (shape.NumDimensions() == 0)
        {
            out << "{}";
        }
        else
        {
            out << shape[0];
            auto numDimensions = shape.NumDimensions();
            for (int index = 1; index < numDimensions; ++index)
            {
                out << " x " << shape[index];
            }
        }
        return out;
    }

    std::ostream& operator<<(std::ostream& out, const utilities::MemoryLayout& layout)
    {
        if (layout == ScalarLayout)
        {
            out << "scalar layout";
        }
        else
        {
            out << "active size (physical): " << layout.GetActiveSize();
            out << " memory size (physical): " << layout.GetExtent();
            out << " memory strides (physical): " << layout.GetCumulativeIncrement();
        }
        return out;
    }

    MemoryLayout MemoryLayout::Flatten() const
    {
        if (IsContiguous())
        {
            return MemoryLayout{ { _extent.NumElements() } };
        }
        else
        {
            throw InputException(InputExceptionErrors::invalidArgument,
                                 "Cannot flatten a discontiguous MemoryLayout.");
        }
    }

    /*extern*/ MemoryLayout ScalarLayout{};

} // namespace utilities
} // namespace ell

size_t std::hash<::ell::utilities::DimensionVector>::operator()(const ::ell::utilities::DimensionVector& v) const noexcept
{
    return ::ell::utilities::HashValue(v.ToVector());
}

std::size_t std::hash<::ell::utilities::MemoryLayout>::operator()(const ell::utilities::MemoryLayout& arg) const noexcept
{
    using ::ell::utilities::HashCombine;

    size_t hash = 0;
    HashCombine(hash, arg.GetActiveSize().ToVector());
    HashCombine(hash, arg.GetExtent().ToVector());
    HashCombine(hash, arg.GetOffset().ToVector());
    HashCombine(hash, arg.GetCumulativeIncrement().ToVector());
    HashCombine(hash, arg.GetLogicalDimensionOrder().ToVector());
    return hash;
}
