////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     MemoryLayout_test.cpp (utilities)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MemoryLayout_test.h"

#include <utilities/include/MemoryLayout.h>

#include <testing/include/testing.h>

#include <algorithm>
#include <numeric>

namespace ell
{
using namespace utilities;

void TestMemoryLayoutCtors()
{
    auto test = [](int rows, int columns, int rowPadding, int columnPadding) {
        bool ok = true;
        {
            MemoryLayout layout({ rows, columns });
            ok = testing::IsTrue(
                layout.NumDimensions() == 2 && static_cast<int>(layout.NumElements()) == rows * columns &&
                layout.GetActiveSize() == MemoryShape({ rows, columns }) &&
                layout.GetExtent() == MemoryShape({ rows, columns }) && layout.GetOffset() == MemoryShape({ 0, 0 }) &&
                layout.GetCumulativeIncrement() == MemoryShape({ columns, 1 }));
        }
        if (!ok)
        {
            return ok;
        }
        {
            MemoryLayout layout({ rows, columns }, MemoryShape{ rowPadding, columnPadding });
            ok = testing::IsTrue(
                layout.NumDimensions() == 2 && static_cast<int>(layout.NumElements()) == rows * columns &&
                layout.GetActiveSize() == MemoryShape({ rows, columns }) &&
                layout.GetExtent() == MemoryShape({ rows + (2 * rowPadding), columns + (2 * columnPadding) }) &&
                layout.GetOffset() == MemoryShape({ rowPadding, columnPadding }) &&
                layout.GetCumulativeIncrement() == MemoryShape({ columns + (2 * columnPadding), 1 }));
        }
        if (!ok)
        {
            return ok;
        }

        try
        {
            if (rowPadding > 0 || columnPadding > 0)
            {
                MemoryLayout layout({ rows, columns }, { 0, 0 }, MemoryShape{ rowPadding, columnPadding });
                ok = false;
            }
        }
        catch (InputException&)
        {}

        if (!ok)
        {
            return ok;
        }
        {
            MemoryLayout layout({ rows, columns },
                                { rows + (2 * rowPadding), columns + (2 * columnPadding) },
                                MemoryShape{ rowPadding, columnPadding });
            ok = testing::IsTrue(
                layout.NumDimensions() == 2 && static_cast<int>(layout.NumElements()) == rows * columns &&
                layout.GetActiveSize() == MemoryShape({ rows, columns }) &&
                layout.GetExtent() == MemoryShape({ rows + (2 * rowPadding), columns + (2 * columnPadding) }) &&
                layout.GetOffset() == MemoryShape({ rowPadding, columnPadding }) &&
                layout.GetCumulativeIncrement() == MemoryShape({ columns + (2 * columnPadding), 1 }));
        }
        return ok;
    };

    bool ok = true;
    for (int rows = 0; rows < 3 && ok; ++rows)
    {
        for (int columns = 0; columns < 3 && ok; ++columns)
        {
            for (int rowPadding = 0; rowPadding < 2 && ok; ++rowPadding)
            {
                for (int columnPadding = 0; columnPadding < 2 && ok; ++columnPadding)
                {
                    ok &= test(rows, columns, rowPadding, columnPadding);
                }
            }
        }
    }
    testing::ProcessTest("Basic MemoryLayout ctor test", ok);
}

void TestMemoryLayoutSlice()
{
    auto testSlices = [](MemoryLayout layout) -> bool {
        bool ok = true;

        for (int sliceDimension = 0; sliceDimension < layout.NumDimensions() && ok; ++sliceDimension)
        {
            auto sliced = layout.GetSliceLayout(sliceDimension);

            ok = sliced.NumDimensions() == layout.NumDimensions() - 1;
            if (!ok)
            {
                break;
            }

            if (sliceDimension == 0)
            {
                ok = static_cast<int>(sliced.GetMemorySize()) ==
                     static_cast<int>(layout.GetMemorySize()) / layout.GetExtent(0);
            }
            else
            {
                ok = sliced.GetMemorySize() == layout.GetMemorySize();
            }

            for (int dimension = 0; dimension < sliceDimension && ok; ++dimension)
            {
                ok = sliced.GetActiveSize(dimension) == layout.GetActiveSize(dimension);
                ok &= sliced.GetCumulativeIncrement(dimension) == layout.GetCumulativeIncrement(dimension);
                if (dimension == sliceDimension - 1)
                {
                    ok &= sliced.GetExtent(dimension) == layout.GetExtent(dimension) * layout.GetExtent(dimension + 1);
                }
                else
                {
                    ok &= sliced.GetExtent(dimension) == layout.GetExtent(dimension);
                }
            }
            for (int dimension = sliceDimension; dimension < sliced.NumDimensions() && ok; ++dimension)
            {
                ok = sliced.GetActiveSize(dimension) == layout.GetActiveSize(dimension + 1);
                ok &= sliced.GetCumulativeIncrement(dimension) == layout.GetCumulativeIncrement(dimension + 1);
                ok &= sliced.GetExtent(dimension) == layout.GetExtent(dimension + 1);
            }
        }

        return ok;
    };

    constexpr int rows = 3, columns = 5, channels = 7, outerExtent = 4;

    bool ok = true;
    std::vector<int> physicalSize;
    for (auto i : { rows, columns, channels, outerExtent })
    {
        physicalSize.push_back(i);
        std::vector<int> order(physicalSize.size());
        std::iota(order.begin(), order.end(), 0);

        do
        {
            MemoryLayout layout(physicalSize, DimensionOrder(order));
            ok = testSlices(layout);
        } while (std::next_permutation(order.begin(), order.end()) && ok);
        if (!ok)
        {
            break;
        }
    }
    testing::ProcessTest("MemoryLayout::Slice test", ok);
}

void TestDimensionOrder()
{
    MemoryLayout layout({ 7, 5, 3 }, ChannelMajorTensorOrder);

    bool ok = true;
    ok &= testing::IsEqual(layout.GetPhysicalDimension(2), 0);
    ok &= testing::IsEqual(layout.GetPhysicalDimension(0), 1);
    ok &= testing::IsEqual(layout.GetPhysicalDimension(1), 2);
    testing::ProcessTest("MemoryLayout::GetPhysicalDimension", ok);

    ok = true;
    ok &= testing::IsEqual(layout.GetLogicalDimension(0), 2);
    ok &= testing::IsEqual(layout.GetLogicalDimension(1), 0);
    ok &= testing::IsEqual(layout.GetLogicalDimension(2), 1);
    testing::ProcessTest("MemoryLayout::GetLogicalDimension", ok);
}

void TestMemoryLayoutHash()
{
    MemoryLayout layout1({ 7, 5, 3 }, ChannelMajorTensorOrder);
    MemoryLayout layout2({ 7, 5, 3 }, ChannelMajorTensorOrder);
    MemoryLayout layout3({ 7, 5, 3 }, RowMajorTensorOrder);
    MemoryLayout layout4({ 7, 5, 3 }, { 1, 1, 0 }, ChannelMajorTensorOrder);

    bool ok = true;
    std::hash<MemoryLayout> layoutHash;
    ok &= testing::IsEqual(layoutHash(layout1), layoutHash(layout2));
    ok &= !testing::IsEqual(layoutHash(layout1), layoutHash(layout3));
    ok &= !testing::IsEqual(layoutHash(layout1), layoutHash(layout4));
    testing::ProcessTest("std::hash<MemoryLayout>", ok);

    DimensionOrder order1({ 0, 1, 2, 3 });
    DimensionOrder order2({ 0, 1, 2 });
    MemoryCoordinates coords1({ 1, 2, 3 });
    MemoryCoordinates coords2({ 1, 2 });
    MemoryShape shape1({ 1, 2, 3 });
    MemoryShape shape2({ 1, 2 });

    ok = true;
    std::hash<DimensionVector> dimHash;
    ok &= testing::IsEqual(dimHash(order1), dimHash(order1));
    ok &= !testing::IsEqual(dimHash(order1), dimHash(order2));
    testing::ProcessTest("std::hash<DimensionVector> 1", ok);

    ok = true;
    ok &= testing::IsEqual(dimHash(coords1), dimHash(coords1));
    ok &= !testing::IsEqual(dimHash(coords1), dimHash(coords2));
    testing::ProcessTest("std::hash<DimensionVector> 2", ok);

    ok = true;
    ok &= testing::IsEqual(dimHash(shape1), dimHash(shape1));
    ok &= !testing::IsEqual(dimHash(shape1), dimHash(shape2));
    testing::ProcessTest("std::hash<DimensionVector> 3", ok);
}

void TestScalarLayout()
{
    bool ok = true;
    ok &= ScalarLayout.GetMemorySize() == 1u;
    ok &= ScalarLayout.HasPadding() == false;
    ok &= ScalarLayout.IsCanonicalOrder() == true;
    ok &= ScalarLayout.IsContiguous() == true;
    ok &= ScalarLayout.NumDimensions() == 0;
    ok &= ScalarLayout.NumElements() == 1u;

    testing::ProcessTest("ScalarLayout test", ok);
}

void TestInflateMemoryLayout()
{
    // Test with a 3-dimensional input layout
    MemoryLayout layout({ 7, 5, 3 }, ChannelMajorTensorOrder);
    MemoryLayout layout2 = layout.CopyWithExtraDimensions(0); // should be the same
    MemoryLayout layout3 = layout.CopyWithExtraDimensions(2); // should be the same

    bool ok = true;
    ok &= testing::IsEqual(layout.GetActiveSize(), layout2.GetActiveSize());
    ok &= testing::IsEqual(layout.GetExtent(), layout2.GetExtent());
    ok &= testing::IsEqual(layout.GetOffset(), layout2.GetOffset());
    ok &= testing::IsEqual(layout.GetCumulativeIncrement(), layout2.GetCumulativeIncrement());
    ok &= testing::IsEqual(layout.GetLogicalDimensionOrder(), layout2.GetLogicalDimensionOrder());
    testing::ProcessTest("MemoryLayout::CopyWithExtraDimensions(0)", ok);

    ok = true;
    ok &= testing::IsNotEqual(layout.GetActiveSize(), layout3.GetActiveSize());
    ok &= testing::IsNotEqual(layout.GetExtent(), layout3.GetExtent());
    ok &= testing::IsNotEqual(layout.GetOffset(), layout3.GetOffset());
    ok &= testing::IsNotEqual(layout.GetCumulativeIncrement(), layout3.GetCumulativeIncrement());
    ok &= testing::IsNotEqual(layout.GetLogicalDimensionOrder(), layout3.GetLogicalDimensionOrder());
    ok &= testing::IsEqual(layout.NumElements(), layout3.NumElements());
    ok &= testing::IsEqual(layout.GetMemorySize(), layout3.GetMemorySize());
    ok &= testing::IsEqual(layout.NumDimensions(), layout3.NumDimensions() - 2);
    ok &= testing::IsEqual(layout.GetEntryOffset({ 1, 2, 3 }), layout3.GetEntryOffset({ 0, 0, 1, 2, 3 }));
    ok &= testing::IsEqual(layout.GetEntryOffset({ 3, 2, 1 }), layout3.GetEntryOffset({ 0, 0, 3, 2, 1 }));
    testing::ProcessTest("MemoryLayout::CopyWithExtraDimensions(2)", ok);
}

void TestInflateNullMemoryLayout()
{
    // Test with a 3-dimensional input layout
    MemoryLayout layout{};
    MemoryLayout layout2 = layout.CopyWithExtraDimensions(0); // should be the same ({})
    MemoryLayout layout3 = layout.CopyWithExtraDimensions(2); // should not be the same ({1, 1})

    bool ok = true;
    ok &= testing::IsEqual(layout.GetActiveSize(), layout2.GetActiveSize());
    ok &= testing::IsEqual(layout.GetExtent(), layout2.GetExtent());
    ok &= testing::IsEqual(layout.GetOffset(), layout2.GetOffset());
    ok &= testing::IsEqual(layout.GetCumulativeIncrement(), layout2.GetCumulativeIncrement());
    ok &= testing::IsEqual(layout.GetLogicalDimensionOrder(), layout2.GetLogicalDimensionOrder());
    testing::ProcessTest("Null MemoryLayout::CopyWithExtraDimensions(0)", ok);

    ok = true;
    ok &= testing::IsNotEqual(layout.GetActiveSize(), layout3.GetActiveSize());
    ok &= testing::IsNotEqual(layout.GetExtent(), layout3.GetExtent());
    ok &= testing::IsNotEqual(layout.GetOffset(), layout3.GetOffset());
    ok &= testing::IsNotEqual(layout.GetCumulativeIncrement(), layout3.GetCumulativeIncrement());
    ok &= testing::IsNotEqual(layout.GetLogicalDimensionOrder(), layout3.GetLogicalDimensionOrder());
    ok &= testing::IsEqual(layout.NumElements(), layout3.NumElements());
    ok &= testing::IsEqual(layout.GetMemorySize(), layout3.GetMemorySize());
    ok &= testing::IsEqual(layout.NumDimensions(), layout3.NumDimensions() - 2);
    ok &= testing::IsEqual(layout3.GetActiveSize(), MemoryShape{ 1, 1 });
    ok &= testing::IsEqual(layout3.GetExtent(), MemoryShape{ 1, 1 });
    ok &= testing::IsEqual(layout3.GetOffset(), MemoryShape{ 0, 0 });
    ok &= testing::IsEqual(layout3.GetCumulativeIncrement(), MemoryShape{ 1, 1 });
    ok &= testing::IsEqual(layout3.GetLogicalDimensionOrder(), DimensionOrder{ 0, 1 });
    testing::ProcessTest("Null MemoryLayout::CopyWithExtraDimensions(2)", ok);
}
} // namespace ell
