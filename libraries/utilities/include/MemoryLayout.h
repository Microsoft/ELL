////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     MemoryLayout.h (utilities)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "IArchivable.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <ostream>
#include <string>
#include <vector>

namespace ell
{
namespace utilities
{
    /// <summary> An abstract base class for DimensionOrder,  MemoryShape, and MemoryCoordinates. </summary>
    class DimensionVector
    {
    public:
        /// <summary> Convert to a std::vector of integers. </summary>
        ///
        /// <returns> The elements as a std::vector. </returns>
        const std::vector<int>& ToVector() const { return _data; }

        /// <summary> Element access operator. </summary>
        int operator[](int index) const { return _data[index]; }
        int operator[](size_t index) const { return _data[index]; }

        /// <summary> Element access operator. </summary>
        int& operator[](int index) { return _data[index]; }
        int& operator[](size_t index) { return _data[index]; }

        /// <summary> Get the number of dimensions. </summary>
        int NumDimensions() const { return static_cast<int>(_data.size()); }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "DimensionVector"; }

        /// <summary> std::begin customization point </summary>
        auto begin() { return _data.begin(); }
        auto begin() const { return _data.cbegin(); }

        /// <summary> std::end customization point </summary>
        auto end() { return _data.end(); }
        auto end() const { return _data.cend(); }

    protected:
        DimensionVector() = default;
        DimensionVector(const std::vector<int>& elements) :
            _data(elements.begin(), elements.end()) {}
        DimensionVector(const std::initializer_list<int>& elements) :
            _data(elements.begin(), elements.end()) {}
        DimensionVector(const std::vector<size_t>& elements);
        DimensionVector(const std::initializer_list<size_t>& elements);

        std::vector<int> _data;
    };

    /// <summary> A vector of dimension indices representing the ordering of the logical dimensions (e.g., 'row', 'column') in memory. </summary>
    class DimensionOrder : public DimensionVector
    {
    public:
        DimensionOrder() = default;

        /// <summary> Constructor for the canonical order with a given number of dimensions. </summary>
        ///
        /// <param name="numDimensions"> The number of dimensions. </param>
        DimensionOrder(int numDimensions);

        /// <summary> Constructor from a vector of integers. </summary>
        ///
        /// <param name="order"> The ordering of the logical dimensions in memory (e.g., [0, 1] for
        ///     the canonical row-major ordering of 2D arrays, and [1, 0] for column-major). </param>
        DimensionOrder(const std::vector<int>& order);

        /// <summary> Constructor from a list of integers </summary>
        ///
        /// <param name="order"> The ordering of the logical dimensions in memory (e.g., [0, 1] for
        ///     the canonical row-major ordering of 2D arrays, and [1, 0] for column-major. </param>
        DimensionOrder(const std::initializer_list<int>& order);

        /// <summary> Constructor from an array of integers. </summary>
        ///
        /// <param name="order"> The ordering of the logical dimensions in memory (e.g., [0, 1] for
        ///     the canonical row-major ordering of 2D arrays, and [1, 0] for column-major). </param>
        template <size_t N>
        DimensionOrder(const std::array<int, N>& order) :
            DimensionOrder({ order.begin(), order.end() })
        {}

        /// <summary> Indicates if this object represents the canonical memory order (0, 1, 2, ...) </summary>
        bool IsCanonicalOrder() const;

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "DimensionOrder"; }

        /// <summary> Element access operator. </summary>
        int operator[](int index) const;

        std::string ToString() const;

    private:
        int& operator[](int index);
    };

    /// <summary> A vector of numbers representing shape (extent) information of a multidimensional array. </summary>
    class MemoryShape : public DimensionVector
    {
    public:
        MemoryShape() = default;

        /// <summary> Constructor from a vector of integers </summary>
        ///
        /// <param name="shape"> The size per dimension of the shape </param>
        MemoryShape(const std::vector<int>& shape) :
            DimensionVector(shape) {}

        /// <summary> Constructor from a list of integers </summary>
        ///
        /// <param name="shape"> The size per dimension of the shape </param>
        MemoryShape(const std::initializer_list<int>& shape) :
            DimensionVector(shape) {}

        /// <summary> Constructor from a vector of unsigned integers </summary>
        ///
        /// <param name="shape"> The size per dimension of the shape </param>
        MemoryShape(const std::vector<size_t>& shape) :
            DimensionVector(shape) {}

        /// <summary> Constructor from a list of unsigned integers </summary>
        ///
        /// <param name="shape"> The size per dimension of the shape </param>
        MemoryShape(const std::initializer_list<size_t>& shape) :
            DimensionVector(shape) {}

        /// <summary> Get the total number of elements. </summary>
        int NumElements() const;

        /// <summary>
        /// Resize to a different number of dimensions.
        /// If the new dimensionality is greater than the existing dimensionality, '1' will be appended to the front.
        /// For instance, resizing the shape (3, 4) to have 4 dimensions will result in the shape (1, 1, 3, 4).
        /// If the new dimensionality is less than the existing dimensionality, the leading dimensions will be squashed
        /// together. For instance, resizing the shape (1, 2, 3, 4) to 2 dimensions will result in the shape (6, 4)
        /// </summary>
        void Resize(int numDimensions);

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "MemoryShape"; }

        std::string ToString() const;
    };

    /// <summary> A vector of numbers representing an index into a multidimensional array. </summary>
    class MemoryCoordinates : public DimensionVector
    {
    public:
        MemoryCoordinates() = default;

        /// <summary> Constructor from a vector of integers </summary>
        ///
        /// <param name="coordinates"> The coordinates </param>
        MemoryCoordinates(const std::vector<int>& coordinates) :
            DimensionVector(coordinates) {}

        /// <summary> Constructor from a list of integers </summary>
        ///
        /// <param name="coordinates"> The coordinates </param>
        MemoryCoordinates(const std::initializer_list<int>& coordinates) :
            DimensionVector(coordinates) {}

        /// <summary> Constructor from a vector of unsigned integers </summary>
        ///
        /// <param name="coordinates"> The coordinates </param>
        MemoryCoordinates(const std::vector<size_t>& coordinates) :
            DimensionVector(coordinates) {}

        /// <summary> Constructor from a list of unsigned integers </summary>
        ///
        /// <param name="coordinates"> The coordinates </param>
        MemoryCoordinates(const std::initializer_list<size_t>& coordinates) :
            DimensionVector(coordinates) {}

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "MemoryCoordinates"; }

        std::string ToString() const;
    };

    /// <summary> A class representing layout of a block of data in memory where the block can also
    /// contain padding such that a certain offset is required to access the "active" memory inside the
    /// padded block.  Coordinates that can only see the "active" region are called "logical" coordinates, and
    /// coordinates that include the padding are called "physical" coordinates. </summary>
    class MemoryLayout : public utilities::IArchivable
    {
    public:
        MemoryLayout() = default;

        //
        // Constructors using the canonical logical dimension order
        //

        /// <summary> Constructor from size only (no padding). </summary>
        ///
        /// <param name="physicalDimensionSize"> The extent of the active area of the memory region, expressed in physical dimensions
        ///   (so, the first element is the size of the slowest-changing dimension, and the last element is the size of the
        ///   fastest-changing dimension). </param>
        MemoryLayout(const MemoryShape& physicalDimensionSize);

        /// <summary> Constructor from size and padding. </summary>
        ///
        /// <param name="physicalDimensionSize"> The extent of the active area of the memory region, expressed in physical dimensions
        ///   (so, the first element is the size of the slowest-changing dimension, and the last element is the size of the
        ///   fastest-changing dimension). </param>
        /// <param name="physicalDimensionPadding"> The amount of padding to apply to the beginning and end of each dimension </param>
        MemoryLayout(const MemoryShape& physicalDimensionSize, const MemoryShape& physicalDimensionPadding);

        /// <summary> General constructor. </summary>
        ///
        /// <param name="physicalDimensionSize"> The extent of the active area of the memory region, expressed in physical dimensions
        ///   (so, the first element is the size of the slowest-changing dimension, and the last element is the size of the
        ///   fastest-changing dimension). </param>
        /// <param name="physicalDimensionExtent"> The extent of the allocated memory of the memory region. </param>
        /// <param name="physicalDimensionOffset"> The offset into memory to the active area of the memory region. </param>
        MemoryLayout(const MemoryShape& physicalDimensionSize, const MemoryShape& physicalDimensionExtent, const MemoryShape& physicalDimensionOffset);

        /// <summary> General constructor. </summary>
        ///
        /// <param name="physicalDimensionSize"> The extent of the active area of the memory region, expressed in physical dimensions
        ///   (so, the first element is the size of the slowest-changing dimension, and the last element is the size of the
        ///   fastest-changing dimension). </param>
        /// <param name="physicalDimensionExtent"> The extent of the allocated memory of the memory region. </param>
        /// <param name="physicalDimensionOffset"> The offset into memory to the active area of the memory region. </param>
        /// <param name="physicalDimensionIncrement"> The cumulative increments for each dimension. This is the
        /// distance in memory between two entries that are adjacent in that dimension. </param>
        MemoryLayout(const MemoryShape& physicalDimensionSize, const MemoryShape& physicalDimensionExtent, const MemoryShape& physicalDimensionOffset, const MemoryShape& physicalDimensionIncrement);

        //
        // Constructors with a user-supplied logical dimension ordering
        //

        /// <summary> Constructor from size only (no padding). </summary>
        ///
        /// <param name="physicalDimensionSize"> The extent of the active area of the memory region, expressed in physical dimensions
        ///   (so, the first element is the size of the slowest-changing dimension, and the last element is the size of the
        ///   fastest-changing dimension). </param>
        /// <param name="order"> The ordering of the logical dimensions in memory (e.g., [0, 1] for
        ///     the canonical row-major ordering of 2D arrays, and [1, 0] for column-major. </param>
        MemoryLayout(const MemoryShape& physicalDimensionSize, const DimensionOrder& order);

        /// <summary> Constructor from size and padding. </summary>
        ///
        /// <param name="physicalDimensionSize"> The extent of the active area of the memory region, expressed in physical dimensions
        ///   (so, the first element is the size of the slowest-changing dimension, and the last element is the size of the
        ///   fastest-changing dimension). </param>
        /// <param name="physicalDimensionPadding"> The amount of padding to apply to the beginning and end of each dimension </param>
        /// <param name="order"> The ordering of the logical dimensions in memory (e.g., [0, 1] for
        ///     the canonical row-major ordering of 2D arrays, and [1, 0] for column-major. </param>
        MemoryLayout(const MemoryShape& physicalDimensionSize, const MemoryShape& physicalDimensionPadding, const DimensionOrder& order);

        /// <summary> General constructor. </summary>
        ///
        /// <param name="physicalDimensionSize"> The extent of the active area of the memory region, expressed in physical dimensions
        ///   (so, the first element is the size of the slowest-changing dimension, and the last element is the size of the
        ///   fastest-changing dimension). </param>
        /// <param name="physicalDimensionExtent"> The extent of the allocated memory of the memory region. </param>
        /// <param name="physicalDimensionOffset"> The offset into memory to the active area of the memory region. </param>
        /// <param name="order"> The ordering of the logical dimensions in memory (e.g., [0, 1] for
        ///     the canonical row-major ordering of 2D arrays, and [1, 0] for column-major. </param>
        MemoryLayout(const MemoryShape& physicalDimensionSize, const MemoryShape& physicalDimensionExtent, const MemoryShape& physicalDimensionOffset, const DimensionOrder& order);

        /// <summary> General constructor. </summary>
        ///
        /// <param name="physicalDimensionSize"> The extent of the active area of the memory region, expressed in physical dimensions
        ///   (so, the first element is the size of the slowest-changing dimension, and the last element is the size of the
        ///   fastest-changing dimension). </param>
        /// <param name="physicalDimensionExtent"> The extent of the allocated memory of the memory region. </param>
        /// <param name="physicalDimensionOffset"> The offset into memory to the active area of the memory region. </param>
        /// <param name="physicalDimensionIncrement"> The cumulative increments for each dimension. This is the
        /// distance in memory between two entries that are adjacent in that dimension. </param>
        /// <param name="order"> The ordering of the logical dimensions in memory (e.g., [0, 1] for
        ///     the canonical row-major ordering of 2D arrays, and [1, 0] for column-major. </param>
        MemoryLayout(const MemoryShape& physicalDimensionSize, const MemoryShape& physicalDimensionExtent, const MemoryShape& physicalDimensionOffset, const MemoryShape& physicalDimensionIncrement, const DimensionOrder& order);

        /// <summary> Returns the number of dimensions in this memory layout </summary>
        ///
        /// <returns> The number of dimensions </summary>
        int NumDimensions() const { return _size.NumDimensions(); }

        /// <summary>
        /// Returns the size of the "active" memory area (not counting any padding).
        /// </summary>
        ///
        /// <returns> A `MemoryShape` object containing the size of the memory area. </returns>
        const MemoryShape& GetActiveSize() const { return _size; }

        /// <summary>
        /// Indicates if this layout has any extra padding
        /// </summary>
        ///
        /// <returns> Returns `true` if there is any extra padding around the active area. </returns>
        bool HasPadding() const;

        /// <summary>
        /// Returns the allocated size of the memory (including padding).
        /// </summary>
        ///
        /// <returns> A `MemoryShape` object containing the allocated size in each dimension </returns>
        const MemoryShape& GetExtent() const { return _extent; }

        /// <summary>
        /// Returns the offsets to the "active" area of memory.
        /// </summary>
        ///
        /// <returns> A `MemoryShape` object containing the offset to the active part of memory for each dimension. </returns>
        const MemoryShape& GetOffset() const { return _offset; }

        /// <summary>
        /// Returns the size of the "active" memory area for the given dimension (not counting any padding).
        /// </summary>
        ///
        /// <param name="index"> The dimension. </param>
        ///
        /// <returns> The size of the memory area for the given dimension. </returns>
        int GetActiveSize(size_t index) const;

        /// <summary>
        /// Returns the allocated size of the memory for the given dimension (including padding).
        /// </summary>
        ///
        /// <param name="index"> The dimension. </param>
        ///
        /// <returns> The allocated size in each dimension </returns>
        int GetExtent(size_t index) const;

        /// <summary>
        /// Returns the cumulative increments for each dimension. This is the distance in memory
        /// between two entries that are adjacent in that dimension.
        /// </summary>
        ///
        /// <returns> A `MemoryShape` object containing the cumulative increment in each dimension </returns>
        const MemoryShape& GetCumulativeIncrement() const { return _increment; }

        /// <summary>
        /// Returns the offset to the "active" area of memory for the given physical dimension.
        /// </summary>
        ///
        /// <param name="index"> The dimension. </param>
        ///
        /// <returns> The offset to the active part of memory for the given dimension. </returns>
        int GetOffset(size_t index) const;

        /// <summary>
        /// Returns the cumulative increment for the requested physical dimension. This is the distance in memory
        /// between two entries that are adjacent in that dimension.
        /// </summary>
        ///
        /// <param name="index"> The dimension. </param>
        ///
        /// <returns> The cumulative increment for the given physical dimension. </returns>
        size_t GetCumulativeIncrement(size_t index) const;

        /// <summary> Returns the number of active elements in this memory layout </summary>
        ///
        /// <returns> The number of elements </returns>
        size_t NumElements() const;

        /// <summary> Returns the number of total (active plus extra extent) elements in this memory layout </summary>
        ///
        /// <returns> The number of elements </returns>
        size_t GetMemorySize() const;

        /// <summary> Gets the offset into memory for an entry </summary>
        ///
        /// <param name="physicalCoordinates"> The coordinates of the entry </param>
        /// <returns> The offset to the entry (from the beginning of memory) </returns>
        size_t GetEntryOffset(const MemoryCoordinates& physicalCoordinates) const;

        /// <summary> Gets the offset into memory for the first active entry </summary>
        ///
        /// <returns> The offset to the first active entry (from the beginning of memory) </returns>
        size_t GetFirstEntryOffset() const;

        /// <summary> Gets the physical coordinates that point to the element at a given offset from the beginning of memory. </summary>
        /// This is the inverse operation from GetEntryOffset.
        /// This function doesn't do any bounds-checking on the output. If the layout includes padding, the coordinates may be negative
        /// or outside the active memory bounds.
        MemoryCoordinates GetPhysicalCoordinatesFromOffset(size_t index) const;

        /// <summary> Transforms the given logic coordinates into a physical set of indices for the current layout. </summary>
        MemoryCoordinates GetPhysicalCoordinates(const MemoryCoordinates& logicalCoordinates) const;

        //
        // Getting information about logical layout
        //

        /// <summary>
        /// Returns the size of the "active" memory area (not counting any padding), in the logical coordinates for this layout.
        /// </summary>
        ///
        /// <returns> A `MemoryShape` object containing the size of the memory area. </returns>
        MemoryShape GetLogicalDimensionActiveSize() const;

        /// <summary>
        /// Returns the size of the "active" memory area (not counting any padding) for the given logical dimension.
        /// </summary>
        ///
        /// <param name="index"> The dimension. </param>
        ///
        /// <returns> A `MemoryShape` object containing the size of the memory area. </returns>
        int GetLogicalDimensionActiveSize(size_t index) const;

        /// <summary>
        /// Returns the allocated size of the memory (including padding), in the logical coordinates for this layout.
        /// </summary>
        ///
        /// <returns> A `MemoryShape` object containing the allocated size in each dimension </returns>
        MemoryShape GetLogicalDimensionExtent() const;

        /// <summary>
        /// Returns the allocated size of the memory (including padding) for the given logical dimension.
        /// </summary>
        ///
        /// <param name="index"> The dimension. </param>
        ///
        /// <returns> A `MemoryShape` object containing the allocated size in each dimension </returns>
        int GetLogicalDimensionExtent(size_t index) const;

        /// <summary>
        /// Returns the offsets to the "active" area of memory, in the logical coordinates for this layout.
        /// </summary>
        ///
        /// <returns> A `MemoryShape` object containing the offset to the active part of memory for that dimension. </returns>
        MemoryShape GetLogicalDimensionOffset() const;

        /// <summary>
        /// Returns the offsets to the "active" area of memory for the given logical dimension.
        /// </summary>
        ///
        /// <param name="index"> The dimension. </param>
        ///
        /// <returns> A `MemoryShape` object containing the offset to the active part of memory for that dimension. </returns>
        int GetLogicalDimensionOffset(size_t index) const;

        /// <summary> Gets the physical coordinates that point to the element at a given offset from the beginning of memory. </summary>
        /// This is the inverse operation from GetEntryOffset.
        /// This function doesn't do any bounds-checking on the output. If the layout includes padding, the coordinates may be negative
        /// or outside the active memory bounds.
        MemoryCoordinates GetLogicalCoordinatesFromOffset(size_t index) const;

        /// <summary>
        /// Returns the cumulative increments in the logical coordinates for this layout. This is the distance in memory
        /// between two entries that are adjacent in that dimension.
        /// </summary>
        ///
        /// <returns> A `MemoryShape` object containing the increments for each dimension. </returns>
        MemoryShape GetLogicalDimensionIncrement() const;

        /// <summary>
        /// Returns the cumulative increment for the requested logical dimension. This is the distance in memory
        /// between two entries that are adjacent in that dimension.
        /// </summary>
        ///
        /// <param name="index"> The dimension. </param>
        ///
        /// <returns> The cumulative increment for the given physical dimension. </returns>
        size_t GetLogicalDimensionIncrement(size_t index) const;

        /// <summary> Gets the offset into memory for an entry, given logical coordinates </summary>
        ///
        /// <param name="logicalCoordinates"> The logical coordinates of the entry </param>
        /// <returns> The offset to the entry (from the beginning of memory) </returns>
        size_t GetLogicalEntryOffset(const MemoryCoordinates& logicalCoordinates) const;

        /// <summary> Returns the ordering of the logical dimensions in memory (e.g., [0, 1] for
        ///     the canonical row-major ordering of 2D arrays, and [1, 0] for column-major. </summary>
        const DimensionOrder& GetLogicalDimensionOrder() const { return _dimensionOrder; }

        /// <summary> Transforms the given physical coordinates into a logical set of indices for the current layout. </summary>
        MemoryCoordinates GetLogicalCoordinates(const MemoryCoordinates& physicalCoordinates) const;

        //
        // Converting between logical and physical dimensions
        //

        /// <summary> Returns the corresponding physical dimension for the given logical dimension. </summary>
        int GetPhysicalDimension(int logicalDimension) const;

        /// <summary> Returns the corresponding logical dimension for the given physical dimension. </summary>
        int GetLogicalDimension(int physicalDimension) const;

        /// <summary> Checks if a location is outside of the stored memory extent in any dimension </summary>
        ///
        /// <param name="physicalCoordinates"> The coordinates of the entry </param>
        /// <returns> `true` if the location is out of bounds </returns>
        bool IsOutOfBounds(const MemoryCoordinates& physicalCoordinates) const;

        /// <summary> Checks if the memory defined by this layout is contiguous </summary>
        bool IsContiguous() const;

        /// <summary> Checks if the memory defined by this layout is in the canonical memory order (0, 1, 2, ...) </summary>
        bool IsCanonicalOrder() const;

        /// <summary> Creates a new MemoryLayout with the same memory layout, but with a new order for the dimensions </summary>
        ///
        /// <param name="newOrder"> The new order for the dimensions </param>
        /// <returns> A new MemoryLayout instance with the order switched </returns>
        MemoryLayout ReorderedCopy(const DimensionOrder& newOrder) const;

        /// <summary> Creates a new MemoryLayout with the same memory layout, but with the specified dimension sliced out </summary>
        ///
        /// <param name="physicalDimension"> The dimension to slice out </param>
        /// <returns> A new memory layout that matches this one, except all information at the specified dimension has been removed.
        /// The dimension order of the new memory layout has also been adjusted accordingly. </returns>
        /// <remarks> If there's a tensor that needs to be sliced into a row-channel matrix, the
        /// MemoryLayout that represents that matrix can be expressed by doing `layout.GetSliceLayout(layout.GetPhysicalDimension(1))`, where
        /// `layout` is the MemoryLayout that describes the original tensor. This preserves logical order
        /// of the physical dimensions, and provides the correctly adjusted extents and increments for the
        /// sliced layout. </remarks>
        MemoryLayout GetSliceLayout(int physicalDimension) const;

        /// <summary> Creates a new MemoryLayout by adding empty dimensions. </summary>
        ///
        /// <param name="addedDimensions"> The number of dimensions to add. </param>
        /// <returns> A new memory layout that matches this one, but with extra leading "empty" dimensions. </returns>
        /// <remarks>
        /// The new layout is created by appending dimensions of size (and extent) `1` and offset `0`.
        /// For instance, calling `CopyWithExtraDimensions(2)` on a layout of size `{2,3,4}` would result
        /// in a layout of size `{1,1,2,3,4}`.
        MemoryLayout CopyWithExtraDimensions(int addedDimensions) const;

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "MemoryLayout"; }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

        /// <summary> If the layout is contiguous, return a new layout that interprets this block as
        /// a simple one dimensional vector, otherwise throws an exception. </summary>
        MemoryLayout Flatten() const;

        std::string ToString() const;

    protected:
        size_t GetDataOffset() const; // offset for physical entry {0,0,0...}
        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;

    private:
        void BoundsCheckDimensionIndex(size_t index) const;

        MemoryShape _size; // The "active" area of the memory
        MemoryShape _extent; // The allocated size along each dimension
        MemoryShape _offset; // The offset to the active area for each dimension
        MemoryShape _increment; // The distance in memory between adjacent elements for each dimension

        // The memory order of the logical dimensions, encoded as a permutation of the physical order.
        // So, [0, 1, 2] means the physical and logical order are the same. An order of [2, 0, 1]
        // means that physical dimension 2 is logically first. In other words, logical element (r, c, d)
        // would map to physical element (d, r, c)
        DimensionOrder _dimensionOrder;
    };

    /// <summary> Helper value to denote a scalar (degree 0) memory layout </summary>
    extern MemoryLayout ScalarLayout;

    /// <summary> Checks if two dimension-order vectors are equal. </summary>
    ///
    /// <param name="order1"> The first order vector. </param>
    /// <param name="order2"> The other order vector. </param>
    bool operator==(const DimensionOrder& order1, const DimensionOrder& order2);
    bool operator!=(const DimensionOrder& order1, const DimensionOrder& order2);

    /// <summary> Checks if two shapes are equal. </summary>
    ///
    /// <param name="shape1"> The first shape. </param>
    /// <param name="shape2"> The other shape. </param>
    bool operator==(const MemoryShape& shape1, const MemoryShape& shape2);
    bool operator!=(const MemoryShape& shape1, const MemoryShape& shape2);

    /// <summary> Checks if two coordinates are equal. </summary>
    ///
    /// <param name="shape1"> The first shape. </param>
    /// <param name="shape2"> The other shape. </param>
    bool operator==(const MemoryCoordinates& shape1, const MemoryCoordinates& shape2);
    bool operator!=(const MemoryCoordinates& shape1, const MemoryCoordinates& shape2);

    /// <summary> Checks if two memory layouts are equal. </summary>
    ///
    /// <param name="layout1"> The first memory layout. </param>
    /// <param name="layout2"> The other memory layout. </param>
    bool MemoryLayoutsEqual(const MemoryLayout& layout1, const MemoryLayout& layout2);
    bool operator==(const MemoryLayout& shape1, const MemoryLayout& shape2);
    bool operator!=(const MemoryLayout& shape1, const MemoryLayout& shape2);

    /// <summary> Represents row-major matrix order </summary>
    constexpr std::array<int, 2> RowMajorMatrixOrder({ 0, 1 });

    /// <summary> Represents column-major matrix order </summary>
    constexpr std::array<int, 2> ColumnMajorMatrixOrder{ 1, 0 };

    /// <summary> Represents row-major 3D tensor order </summary>
    constexpr std::array<int, 3> RowMajorTensorOrder{ 0, 1, 2 };

    /// <summary> Represents channel-major 3D tensor order </summary>
    constexpr std::array<int, 3> ChannelMajorTensorOrder{ 2, 0, 1 };

    /// <summary> Writes a `DimensionOrder` to an output stream </summary>
    std::ostream& operator<<(std::ostream& out, const utilities::DimensionOrder& order);

    /// <summary> Writes a `MemoryCoordinates` to an output stream </summary>
    std::ostream& operator<<(std::ostream& out, const utilities::MemoryCoordinates& order);

    /// <summary> Writes a `MemoryShape`'s dimensions to an output stream </summary>
    std::ostream& operator<<(std::ostream& out, const utilities::MemoryShape& shape);

    /// <summary> Writes a `MemoryLayout` to an output stream </summary>
    std::ostream& operator<<(std::ostream& out, const utilities::MemoryLayout& layout);
} // namespace utilities
} // namespace ell

namespace std
{

template <>
struct hash<::ell::utilities::DimensionVector>
{
    using Type = ::ell::utilities::DimensionVector;

    /// <summary> Computes a hash of the input value. </summary>
    ///
    /// <returns> A hash value for the given input. </returns>
    [[nodiscard]] size_t operator()(const Type& value) const noexcept;
};

template <>
struct hash<::ell::utilities::MemoryLayout>
{
    using Type = ::ell::utilities::MemoryLayout;

    /// <summary> Computes a hash of the input value. </summary>
    ///
    /// <returns> A hash value for the given input. </returns>
    [[nodiscard]] size_t operator()(const Type& value) const noexcept;
};

} // namespace std
