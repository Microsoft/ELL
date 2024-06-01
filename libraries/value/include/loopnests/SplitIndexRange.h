////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     SplitIndexRange.h (value)
//  Authors:  Chuck Jacobs, Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Index.h"
#include "IndexRange.h"

#include <iosfwd>
#include <map>
#include <unordered_map>
#include <vector>

namespace ell
{
namespace value
{
    namespace loopnests
    {
        /// <summary>
        /// A hierarchically-decomposed dimension range, used to represent the sizes for the different loop levels of a tiled
        /// loop
        /// </summary>
        class SplitIndexRange
        {
        public:
            SplitIndexRange() = default;
            SplitIndexRange(const IndexRange& range);

            const Index& GetDimensionIndex() const; // The dimension index (e.g., `i`), not a concrete loop index
            Range GetDimensionRange() const; // Returns the full range over the dimension

            int NumSplits() const;
            int GetBegin() const;
            int GetSize() const;
            int GetIncrement() const;

            int GetSplitSize(int level) const; // note: returns '1' for the last level
            Index GetSplitIndex(int level) const;
            Range GetIndexRange(const Index& index) const;

            bool Contains(const Index& index) const; // returns `true` if the given index belongs to this domain

            bool IsLoopIndex(const Index& index) const; // a leaf node in the index tree
            bool IsComputedIndex(const Index& index) const; // an interior node in the index tree
            bool IsDimension(const Index& index) const; // the index corresponding to the original range

            bool IsParentOf(const Index& parent, const Index& child) const; // is 'parent' the (immediate) parent of 'child'?
            bool IsChildOf(const Index& child, const Index& parent) const; // is 'child' a(n) (immediate) child of 'parent'?
            bool DependsOn(const Index& index1, const Index& index2) const; // does 'index1' depend on 'index2'? (after a split, the parent depends on the new (leaf) indices)

            const std::vector<Index>& GetIndices() const;
            std::vector<Index> GetLoopIndices() const;
            std::vector<Index> GetComputedIndices() const;
            std::vector<Index> GetDependentIndices(const Index& index, bool includeSelf = false) const;
            std::vector<Index> GetDependentLoopIndices(const Index& index, bool includeSelf = false) const;

            bool HasParentIndex(const Index& parent) const;

            /// <summary> Get the index that was split in order to create the given index </summary>
            Index GetParentIndex(const Index& parent) const;

            bool IsOuterSplitIndex(const Index& index) const;
            bool IsInnerSplitIndex(const Index& index) const;
            Index GetOuterSplitIndex(const Index& parent) const;
            Index GetInnerSplitIndex(const Index& parent) const;

            std::vector<Index> GetAllParentIndices(const Index& index) const;
            std::vector<Index> GetChildIndices(const Index& index) const;

            void Print(std::ostream& os) const;

        private:
            friend class SplitIterationDomain;

            SplitIndex Split(int size); // add a split --- must be smaller than last split
            SplitIndex Split(Index index, int size); // split the given index
            SplitIndex SplitNode(int node, int size); // split the given index

            int GetNode(const Index& index) const; // returns the offset (index into a vector) for the given index
            int GetParent(int node) const;
            int GetLeftChild(int node) const;
            int GetRightChild(int node) const;
            int GetNthLeaf(int n) const;
            int GetSmallestLeaf(int node) const; // returns the "smallest" leaf descendent of index. If index is itself a leaf, return it, else return GetSmallestLeaf(index.rightChild)
            bool IsLeaf(int node) const;
            bool IsInteriorNode(int node) const;

            // The indices and their properties are stored in a binary tree
            // The root is the dimension index (e.g., 'i')
            // Leaves are concrete loop indices
            // Interior nodes are computed indices
            //
            // Initially, the tree has 1 node: the dimension index.
            // Splitting a leaf turns it into an interior node with 2 children (the loop indices of the 2 new loops).
            // Splitting an interior node is illegal. If an interior node is specified, the rightmost leaf node
            // is chosen (arbitrarily).

            std::unordered_map<Index, int> _indexOffset; // A map from Index -> location in the below vectors for info about that index
            std::vector<Index> _indices; // _indices[0] is i
            std::vector<int> _parentOffset; // parent[0] is null (-1)
            std::vector<int> _leftChildOffset; // offset to entry of the first (left) child. The right child is adjacent, so it's at (this value)+1
            std::vector<Range> _ranges;
        };
    } // namespace loopnests
} // namespace value
} // namespace ell
