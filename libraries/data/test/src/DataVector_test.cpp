////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     DataVector_test.cpp (data_test)
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "DataVector_test.h"

#include <data/include/AutoDataVector.h>
#include <data/include/DataVector.h>
#include <data/include/DataVectorOperations.h>
#include <data/include/DenseDataVector.h>
#include <data/include/SparseBinaryDataVector.h>
#include <data/include/SparseDataVector.h>

#include <math/include/Vector.h>

#include <testing/include/testing.h>

#include <algorithm> // for std::transform
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>

namespace ell
{
template <typename DataVectorType>
void IDataVectorTest()
{
    DataVectorType u{ { 0, 2 }, { 3, -7 }, { 4, 1 } };
    testing::ProcessTest("Testing " + std::string(typeid(DataVectorType).name()) + "::Norm2Squared()", testing::IsEqual(u.Norm2Squared(), 2.0 * 2.0 + 7.0 * 7.0 + 1.0 * 1.0));

    math::RowVector<double> w{ 1, 1, 1, 0, -1, 0 };
    testing::ProcessTest("Testing " + std::string(typeid(DataVectorType).name()) + "::Dot()", testing::IsEqual(u.Dot(w), 1.0));

    u.AddTo(w);
    math::RowVector<double> r0{ 3, 1, 1, -7, 0, 0 };
    testing::ProcessTest("Testing " + std::string(typeid(DataVectorType).name()) + "::AddTo()", testing::IsEqual(w.ToArray(), r0.ToArray()));

    data::AddTransformedTo<DataVectorType, data::IterationPolicy::skipZeros>(u, w, [](data::IndexValue x) { return -2 * x.value; });
    math::RowVector<double> r1{ -1, 1, 1, 7, -2, 0 };
    testing::ProcessTest("Testing " + std::string(typeid(DataVectorType).name()) + "::AddTransformedTo<skipZeros>()", testing::IsEqual(w.ToArray(), r1.ToArray()));

    data::AddTransformedTo<DataVectorType, data::IterationPolicy::all>(u, w, [](data::IndexValue x) { return x.value + 1; });
    math::RowVector<double> r2{ 2, 2, 2, 1, 0, 1 };
    testing::ProcessTest("Testing " + std::string(typeid(DataVectorType).name()) + "::AddTransformedTo<all>()", testing::IsEqual(w.ToArray(), r2.ToArray()));

    std::stringstream ss;
    u.Print(ss);
    auto sss = ss.str();
    testing::ProcessTest("Testing " + std::string(typeid(DataVectorType).name()) + "::Print()", sss == "0:2\t3:-7\t4:1");
}

template <typename DataVectorType>
void IDataVectorBinaryTest()
{
    DataVectorType u{ { 0, 1 }, { 3, 1 }, { 4, 1 } };
    testing::ProcessTest("Testing " + std::string(typeid(DataVectorType).name()) + "::Norm2Squared()", testing::IsEqual(u.Norm2Squared(), 3.0));

    math::RowVector<double> w{ 1, 2, 3, 4, 5, 6 };
    testing::ProcessTest("Testing " + std::string(typeid(DataVectorType).name()) + "::Dot()", testing::IsEqual(u.Dot(w), 1.0 + 4.0 + 5.0));

    u.AddTo(w);
    math::RowVector<double> r0{ 2, 2, 3, 5, 6, 6 };
    testing::ProcessTest("Testing " + std::string(typeid(DataVectorType).name()) + "::AddTo()", testing::IsEqual(w.ToArray(), r0.ToArray()));

    data::AddTransformedTo<DataVectorType, data::IterationPolicy::skipZeros>(u, w, [](data::IndexValue x) { return -2 * x.value; });
    math::RowVector<double> r1{ 0, 2, 3, 3, 4, 6 };
    testing::ProcessTest("Testing " + std::string(typeid(DataVectorType).name()) + "::AddTransformedTo<skipZeros>()", testing::IsEqual(w.ToArray(), r1.ToArray()));

    data::AddTransformedTo<DataVectorType, data::IterationPolicy::all>(u, w, [](data::IndexValue x) { return x.value + 1; });
    math::RowVector<double> r2{ 2, 3, 4, 5, 6, 7 };
    testing::ProcessTest("Testing " + std::string(typeid(DataVectorType).name()) + "::AddTransformedTo<all>()", testing::IsEqual(w.ToArray(), r2.ToArray()));

    std::stringstream ss;
    u.Print(ss);
    auto sss = ss.str();
    testing::ProcessTest("Testing " + std::string(typeid(DataVectorType).name()) + "::Print()", sss == "0:1\t3:1\t4:1");
}

void IDataVectorTests()
{
    IDataVectorTest<data::DoubleDataVector>();
    IDataVectorTest<data::FloatDataVector>();
    IDataVectorTest<data::ShortDataVector>();
    IDataVectorTest<data::ByteDataVector>();
    IDataVectorTest<data::SparseDoubleDataVector>();
    IDataVectorTest<data::SparseFloatDataVector>();
    IDataVectorTest<data::SparseShortDataVector>();
    IDataVectorTest<data::SparseByteDataVector>();
    IDataVectorTest<data::AutoDataVector>();

    IDataVectorBinaryTest<data::DoubleDataVector>();
    IDataVectorBinaryTest<data::FloatDataVector>();
    IDataVectorBinaryTest<data::ShortDataVector>();
    IDataVectorBinaryTest<data::ByteDataVector>();
    IDataVectorBinaryTest<data::SparseDoubleDataVector>();
    IDataVectorBinaryTest<data::SparseFloatDataVector>();
    IDataVectorBinaryTest<data::SparseShortDataVector>();
    IDataVectorBinaryTest<data::SparseByteDataVector>();
    IDataVectorBinaryTest<data::AutoDataVector>();
    IDataVectorBinaryTest<data::SparseBinaryDataVector>();
}

template <typename DataVectorType1, typename DataVectorType2>
void DataVectorCopyAsTest(std::initializer_list<double> list, bool testDense = true)
{
    const DataVectorType1 a(list);
    auto b = data::CopyAs<DataVectorType1, DataVectorType2>(a);
    auto av = a.ToArray();
    auto bv = b.ToArray();

    std::string name1 = typeid(DataVectorType1).name();
    std::string name2 = typeid(DataVectorType2).name();

    testing::ProcessTest(name1 + "::CopyAs<" + name2 + ">", testing::IsEqual(av, bv, 1.0e-6));

    if (testDense)
    {
        auto d = data::TransformAs<DataVectorType1, data::IterationPolicy::all, DataVectorType2>(
            a, [](data::IndexValue x) { return x.value + 3; }, 3);
        auto dv = d.ToArray();
        std::vector<double> r{ av[0] + 3, av[1] + 3, av[2] + 3 };
        testing::ProcessTest(name1 + "::TransformAs<all," + name2 + ">", testing::IsEqual(r, dv, 1.0e-6));
    }

    auto c = data::TransformAs<DataVectorType1, data::IterationPolicy::skipZeros, DataVectorType2>(a, [](data::IndexValue x) { return x.value * x.value; });
    auto cv = c.ToArray();
    std::transform(av.begin(), av.end(), av.begin(), [](double x) { return x * x; });
    testing::ProcessTest(name1 + "::TransformAs<skipZeros," + name2 + ">", testing::IsEqual(av, cv, 1.0e-6));
}

enum class InitType
{
    fractional,
    integral,
    binary
};

template <typename DataVectorType>
void DataVectorCopyAsTestDispatch(InitType maxType)
{
    std::initializer_list<double> binaryInit = { 1, 0, 1, 0, 0, 0, 0, 1, 0, 1 };
    std::initializer_list<double> integeralInit = { 1, 0, 1, 0, 0, 0, 0, 3, 0, 4 };
    std::initializer_list<double> fractionalInit = { 1, 0, 1.1, 0, 0, 0, 0, 3, 0, 4.2 };
    if (maxType == InitType::binary) integeralInit = binaryInit;
    if (maxType != InitType::fractional) fractionalInit = integeralInit;

    DataVectorCopyAsTest<DataVectorType, data::AutoDataVector>(fractionalInit);
    DataVectorCopyAsTest<DataVectorType, data::DoubleDataVector>(fractionalInit);
    DataVectorCopyAsTest<DataVectorType, data::FloatDataVector>(fractionalInit);
    DataVectorCopyAsTest<DataVectorType, data::ShortDataVector>(integeralInit);
    DataVectorCopyAsTest<DataVectorType, data::ByteDataVector>(integeralInit);
    DataVectorCopyAsTest<DataVectorType, data::SparseDoubleDataVector>(fractionalInit);
    DataVectorCopyAsTest<DataVectorType, data::SparseFloatDataVector>(fractionalInit);
    DataVectorCopyAsTest<DataVectorType, data::SparseShortDataVector>(integeralInit);
    DataVectorCopyAsTest<DataVectorType, data::SparseByteDataVector>(integeralInit);
    DataVectorCopyAsTest<DataVectorType, data::SparseBinaryDataVector>(binaryInit, false);
}

void DataVectorCopyAsTests()
{
    DataVectorCopyAsTestDispatch<data::AutoDataVector>(InitType::fractional);
    DataVectorCopyAsTestDispatch<data::DoubleDataVector>(InitType::fractional);
    DataVectorCopyAsTestDispatch<data::FloatDataVector>(InitType::fractional);
    DataVectorCopyAsTestDispatch<data::ShortDataVector>(InitType::integral);
    DataVectorCopyAsTestDispatch<data::ByteDataVector>(InitType::integral);
    DataVectorCopyAsTestDispatch<data::SparseDoubleDataVector>(InitType::fractional);
    DataVectorCopyAsTestDispatch<data::SparseFloatDataVector>(InitType::fractional);
    DataVectorCopyAsTestDispatch<data::SparseShortDataVector>(InitType::integral);
    DataVectorCopyAsTestDispatch<data::SparseByteDataVector>(InitType::integral);
    DataVectorCopyAsTestDispatch<data::SparseBinaryDataVector>(InitType::binary);
}

void AutoDataVectorTest()
{
    data::AutoDataVector v1{ 0.123456789, 1.12345678901, 2.3456789012, 3.4567890123 };
    testing::ProcessTest("AutoDataVector ctor", v1.GetInternalType() == data::IDataVector::Type::DoubleDataVector);

    data::AutoDataVector v2{ 0.1f, 1.2f, 2.3f, 3.4f, 4.5f, 5.6f };
    testing::ProcessTest("AutoDataVector ctor", v2.GetInternalType() == data::IDataVector::Type::FloatDataVector);

    data::AutoDataVector v3{ 1234, 2345, 3456, 4567, 5678, 6789 };
    testing::ProcessTest("AutoDataVector ctor", v3.GetInternalType() == data::IDataVector::Type::ShortDataVector);

    data::AutoDataVector v4{ 10, 20, 30, 40, 50, 60, 70 };
    testing::ProcessTest("AutoDataVector ctor", v4.GetInternalType() == data::IDataVector::Type::ByteDataVector);

    data::AutoDataVector v5{ 0, 0, 0, 0, 0, 1.2345678901, 0, 0, 0 };
    testing::ProcessTest("AutoDataVector ctor", v5.GetInternalType() == data::IDataVector::Type::SparseDoubleDataVector);

    data::AutoDataVector v6{ 0, 0, 0, 0, 0, 1.2f, 0, 0, 0 };
    testing::ProcessTest("AutoDataVector ctor", v6.GetInternalType() == data::IDataVector::Type::SparseFloatDataVector);

    data::AutoDataVector v7{ 0, 0, 0, 0, 0, 1234, 0, 0, 0 };
    testing::ProcessTest("AutoDataVector ctor", v7.GetInternalType() == data::IDataVector::Type::SparseShortDataVector);

    data::AutoDataVector v8{ 0, 0, 0, 0, 0, 10, 0, 0, 0 };
    testing::ProcessTest("AutoDataVector ctor", v8.GetInternalType() == data::IDataVector::Type::SparseByteDataVector);

    data::AutoDataVector v9{ 0, 0, 0, 0, 0, 1, 0, 0, 0 };
    testing::ProcessTest("AutoDataVector ctor", v9.GetInternalType() == data::IDataVector::Type::SparseBinaryDataVector);
}

void TransformedDataVectorTest()
{
    math::RowVector<double> v{ 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    data::AutoDataVector u{ 0, 0, 2, 0, -1, 0, 0, 0, 0, 0, 0, 0 };

    v += u * 3.0;
    math::RowVector<double> r0{ 0, 0, 8, 0, -3, 0, 0, 0, 0, 0, 0, 0 };
    testing::ProcessTest("TransformedDataVectorTest: operator *", v == r0);

    v += Abs(u);
    math::RowVector<double> r1{ 0, 0, 10, 0, -2, 0, 0, 0, 0, 0, 0, 0 };
    testing::ProcessTest("TransformedDataVectorTest: Abs()", v == r1);

    v += Square(u);
    math::RowVector<double> r2{ 0, 0, 14, 0, -1, 0, 0, 0, 0, 0, 0, 0 };
    testing::ProcessTest("TransformedDataVectorTest: Square()", v == r2);

    v += ZeroIndicator(u);
    math::RowVector<double> r3{ 1, 1, 14, 1, -1, 1, 1, 1, 1, 1, 1, 1 };
    testing::ProcessTest("TransformedDataVectorTest: ZeroIndicator()", v == r3);

    v += Sqrt(u);
}

template <typename DataVectorType>
void IteratorTest()
{
    DataVectorType u{ 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0 };
    auto a = u.ToArray(13);

    std::vector<double> b(13);
    auto denseIter = data::GetIterator<DataVectorType, data::IterationPolicy::all>(u, 13);
    while (denseIter.IsValid())
    {
        b[denseIter.Get().index] = denseIter.Get().value;
        denseIter.Next();
    }
    std::string name = typeid(DataVectorType).name();
    testing::ProcessTest("IteratorTest<" + name + ">", std::equal(a.begin(), a.end(), b.begin()));

    std::vector<double> c(13);
    auto sparseIter = data::GetIterator<DataVectorType, data::IterationPolicy::skipZeros>(u);
    while (sparseIter.IsValid())
    {
        c[sparseIter.Get().index] = sparseIter.Get().value;
        sparseIter.Next();
    }
    testing::ProcessTest("IteratorTest<" + name + ">", std::equal(b.begin(), b.end(), c.begin()));
}

void IteratorTests()
{
    IteratorTest<data::DoubleDataVector>();
    IteratorTest<data::FloatDataVector>();
    IteratorTest<data::ShortDataVector>();
    IteratorTest<data::ByteDataVector>();
    IteratorTest<data::SparseDoubleDataVector>();
    IteratorTest<data::SparseFloatDataVector>();
    IteratorTest<data::SparseShortDataVector>();
    IteratorTest<data::SparseByteDataVector>();
    IteratorTest<data::SparseBinaryDataVector>();
}
} // namespace ell