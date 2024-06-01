
////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     RingBuffer_test.cpp (utilities)
//  Authors:  Chris Lovett
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "RingBuffer_test.h"

#include <utilities/include/RingBuffer.h>

#include <testing/include/testing.h>

namespace ell
{
using namespace utilities;

template <typename ValueType>
std::vector<ValueType> ToArray(RingBuffer<ValueType> buffer)
{
    std::vector<ValueType> result;
    for (size_t i = 0, n = buffer.Size(); i < n; i++)
    {
        result.push_back(buffer[i]);
    }
    return result;
}

void TestRingBuffer()
{
    RingBuffer<float> buffer(5);

    testing::ProcessTest("TestRingBuffer is empty", testing::IsEqual(ToArray(buffer), std::vector<float>(5)));

    buffer.Append(1);

    testing::ProcessTest("TestRingBuffer is empty", testing::IsEqual(ToArray(buffer), std::vector<float>({ 1, 0, 0, 0, 0 })));

    buffer.Append(2);

    testing::ProcessTest("TestRingBuffer is empty", testing::IsEqual(ToArray(buffer), std::vector<float>({ 2, 1, 0, 0, 0 })));

    buffer.Append(3);
    buffer.Append(4);
    buffer.Append(5);
    buffer.Append(6); // overflows the ring buffer

    testing::ProcessTest("TestRingBuffer is empty", testing::IsEqual(ToArray(buffer), std::vector<float>({ 6, 5, 4, 3, 2 })));
}
} // namespace ell
