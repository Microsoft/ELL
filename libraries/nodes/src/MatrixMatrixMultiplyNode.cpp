////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     MatrixMatrixMultiplyNode.cpp (nodes)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "MatrixMatrixMultiplyNode.h"

#include <math/include/Matrix.h>
#include <math/include/MatrixOperations.h>

namespace ell
{
namespace nodes
{
    namespace
    {
        //
        // Relevant archive format versions
        //
        constexpr utilities::ArchiveVersion currentArchiveVersion = { utilities::ArchiveVersionNumbers::v2 };

        template <typename ValueType>
        void MatrixMatrixMultiply(bool transposeA, bool transposeB, bool transposeC, int m, int n, int k, const std::vector<ValueType>& matrixAValues, const std::vector<ValueType>& matrixBValues, std::vector<ValueType>& matrixCValues)
        {
            if (transposeC)
            {
                // C' = B' * A'
                math::RowMatrixReference<ValueType> Ct(matrixCValues.data(), n, m);

                if (transposeA)
                {
                    math::ConstRowMatrixReference<ValueType> At(matrixAValues.data(), k, m);
                    if (transposeB)
                    {
                        math::ConstRowMatrixReference<ValueType> Bt(matrixBValues.data(), n, k);
                        math::MultiplyScaleAddUpdate(static_cast<ValueType>(1.0), Bt, At, static_cast<ValueType>(0.0), Ct);
                    }
                    else
                    {
                        math::ConstColumnMatrixReference<ValueType> Bt(matrixBValues.data(), n, k);
                        math::MultiplyScaleAddUpdate(static_cast<ValueType>(1.0), Bt, At, static_cast<ValueType>(0.0), Ct);
                    }
                }
                else
                {
                    math::ConstColumnMatrixReference<ValueType> At(matrixAValues.data(), k, m);
                    if (transposeB)
                    {
                        math::ConstRowMatrixReference<ValueType> Bt(matrixBValues.data(), n, k);
                        math::MultiplyScaleAddUpdate(static_cast<ValueType>(1.0), Bt, At, static_cast<ValueType>(0.0), Ct);
                    }
                    else
                    {
                        math::ConstColumnMatrixReference<ValueType> Bt(matrixBValues.data(), n, k);
                        math::MultiplyScaleAddUpdate(static_cast<ValueType>(1.0), Bt, At, static_cast<ValueType>(0.0), Ct);
                    }
                }
            }
            else // not transposeC
            {
                math::RowMatrixReference<ValueType> C(matrixCValues.data(), m, n);
                if (transposeA)
                {
                    math::ConstColumnMatrixReference<ValueType> A(matrixAValues.data(), m, k);
                    if (transposeB)
                    {
                        math::ConstColumnMatrixReference<ValueType> B(matrixBValues.data(), k, n);
                        math::MultiplyScaleAddUpdate(static_cast<ValueType>(1.0), A, B, static_cast<ValueType>(0.0), C);
                    }
                    else // not transposeB
                    {
                        math::ConstRowMatrixReference<ValueType> B(matrixBValues.data(), k, n);
                        math::MultiplyScaleAddUpdate(static_cast<ValueType>(1.0), A, B, static_cast<ValueType>(0.0), C);
                    }
                }
                else // not transposeA
                {
                    math::ConstRowMatrixReference<ValueType> A(matrixAValues.data(), m, k);
                    if (transposeB)
                    {
                        math::ConstColumnMatrixReference<ValueType> B(matrixBValues.data(), k, n);
                        math::MultiplyScaleAddUpdate(static_cast<ValueType>(1.0), A, B, static_cast<ValueType>(0.0), C);
                    }
                    else // not transposeB
                    {
                        math::ConstRowMatrixReference<ValueType> B(matrixBValues.data(), k, n);
                        math::MultiplyScaleAddUpdate(static_cast<ValueType>(1.0), A, B, static_cast<ValueType>(0.0), C);
                    }
                }
            }
        }
    } // namespace

    template <typename ValueType>
    MatrixMatrixMultiplyNode<ValueType>::MatrixMatrixMultiplyNode() :
        CompilableNode({ &_input1, &_input2 }, { &_output }),
        _input1(this, {}, defaultInput1PortName),
        _input2(this, {}, defaultInput2PortName),
        _output(this, defaultOutputPortName, 0)
    {
    }

    template <typename ValueType>
    MatrixMatrixMultiplyNode<ValueType>::MatrixMatrixMultiplyNode(const model::OutputPort<ValueType>& input1, const model::OutputPort<ValueType>& input2) :
        MatrixMatrixMultiplyNode<ValueType>(input1, input2, model::PortMemoryLayout({ input1.GetMemoryLayout().GetActiveSize(0), input2.GetMemoryLayout().GetActiveSize(1) }))
    {
    }

    template <typename ValueType>
    MatrixMatrixMultiplyNode<ValueType>::MatrixMatrixMultiplyNode(const model::OutputPort<ValueType>& input1, const model::OutputPort<ValueType>& input2, const model::PortMemoryLayout& outputLayout) :
        CompilableNode({ &_input1, &_input2 }, { &_output }),
        _input1(this, input1, defaultInput1PortName),
        _input2(this, input2, defaultInput2PortName),
        _output(this, defaultOutputPortName, outputLayout)
    {
        auto input1Layout = _input1.GetMemoryLayout();
        auto input2Layout = _input2.GetMemoryLayout();

        if (input1Layout.NumDimensions() != 2 || input2Layout.NumDimensions() != 2)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input matrices must have a memory layout with 2 dimensions");
        }

        _m = input1Layout.GetLogicalDimensionActiveSize(0);
        _k = input1Layout.GetLogicalDimensionActiveSize(1);
        _n = input2Layout.GetLogicalDimensionActiveSize(1);
        _lda = input1Layout.GetExtent(1);
        _ldb = input2Layout.GetExtent(1);
        _ldc = outputLayout.GetExtent(1);
        _transpose1 = !input1Layout.IsCanonicalOrder();
        _transpose2 = !input2Layout.IsCanonicalOrder();
        _transposeOutput = !outputLayout.IsCanonicalOrder();

        if (_k != input2Layout.GetLogicalDimensionActiveSize(0))
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input matrices incompatible");
        }
    }

    template <typename ValueType>
    MatrixMatrixMultiplyNode<ValueType>::MatrixMatrixMultiplyNode(const model::OutputPort<ValueType>& input1, int m, int n, int k, int matrix1Stride, const model::OutputPort<ValueType>& input2, int matrix2Stride, int outputMatrixStride) :
        CompilableNode({ &_input1, &_input2 }, { &_output }),
        _input1(this, input1, defaultInput1PortName),
        _input2(this, input2, defaultInput2PortName),
        _output(this, defaultOutputPortName, m * n),
        _m(m),
        _n(n),
        _k(k),
        _lda(matrix1Stride),
        _ldb(matrix2Stride),
        _ldc(outputMatrixStride),
        _transpose1(false),
        _transpose2(false)
    {
        if (static_cast<int>(input1.Size()) != m * k)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input matrix 1 size incorrect");
        }

        if (static_cast<int>(input2.Size()) != k * n)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input matrix 2 size incorrect");
        }
    }

    template <typename ValueType>
    MatrixMatrixMultiplyNode<ValueType>::MatrixMatrixMultiplyNode(const model::OutputPort<ValueType>& input1, int m, int n, int k, int matrix1Stride, bool transpose1, const model::OutputPort<ValueType>& input2, int matrix2Stride, bool transpose2, int outputMatrixStride) :
        MatrixMatrixMultiplyNode<ValueType>(input1, m, n, k, matrix1Stride, transpose1, input2, matrix2Stride, transpose2, outputMatrixStride, false)
    {
    }

    template <typename ValueType>
    MatrixMatrixMultiplyNode<ValueType>::MatrixMatrixMultiplyNode(const model::OutputPort<ValueType>& input1, int m, int n, int k, int matrix1Stride, bool transpose1, const model::OutputPort<ValueType>& input2, int matrix2Stride, bool transpose2, int outputMatrixStride, bool transposeOutput) :
        CompilableNode({ &_input1, &_input2 }, { &_output }),
        _input1(this, input1, defaultInput1PortName),
        _input2(this, input2, defaultInput2PortName),
        _output(this, defaultOutputPortName, m * n),
        _m(m),
        _n(n),
        _k(k),
        _lda(matrix1Stride),
        _ldb(matrix2Stride),
        _ldc(outputMatrixStride),
        _transpose1(transpose1),
        _transpose2(transpose2),
        _transposeOutput(transposeOutput)
    {
        // TODO: reset output layout (incl. transpose info)
        if (static_cast<int>(input1.Size()) != m * k)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input matrix 1 size incorrect");
        }

        if (static_cast<int>(input2.Size()) != k * n)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input matrix 2 size incorrect");
        }
    }

    template <typename ValueType>
    void MatrixMatrixMultiplyNode<ValueType>::Compute() const
    {
        assert(static_cast<int>(input1.Size()) == _m * _k);
        assert(static_cast<int>(input2.Size()) == _k * _n);
        auto inputMatrix1Values = input1.GetValue();
        auto inputMatrix2Values = input2.GetValue();
        std::vector<ValueType> outputMatrixValues(_m * _n);

        // op(A): mxk
        // op(B): kxn
        // C: mxn

        math::RowMatrixReference<ValueType> inputMatrix1Ref(inputMatrix1Values.data(), _m, _k);
        math::RowMatrixReference<ValueType> inputMatrix2Ref(inputMatrix2Values.data(), _k, _n);
        math::RowMatrixReference<ValueType> outputMatrixRef(outputMatrixValues.data(), _m, _n);

        MatrixMatrixMultiply(_transpose1, _transpose2, _transposeOutput, (int)_m, (int)_n, (int)_k, inputMatrix1Values, inputMatrix2Values, outputMatrixValues);

        _output.SetOutput(outputMatrixValues);
    };

    template <typename ValueType>
    void MatrixMatrixMultiplyNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInput1 = transformer.GetCorrespondingInputs(_input1);
        const auto& newInput2 = transformer.GetCorrespondingInputs(_input2);
        auto newNode = transformer.AddNode<MatrixMatrixMultiplyNode<ValueType>>(newInput1, _m, _n, _k, _lda, _transpose1, newInput2, _ldb, _transpose2, _ldc, _transposeOutput);
        transformer.MapNodeOutput(output, newNode->output);
    }

    template <typename ValueType>
    void MatrixMatrixMultiplyNode<ValueType>::Compile(model::IRMapCompiler& compiler, emitters::IRFunctionEmitter& function)
    {
        emitters::LLVMValue pInput1 = compiler.EnsurePortEmitted(input1);
        emitters::LLVMValue pInput2 = compiler.EnsurePortEmitted(input2);
        emitters::LLVMValue pOutput = compiler.EnsurePortEmitted(output);

        if (_transposeOutput)
        {
            function.CallGEMM<ValueType>(!_transpose2, !_transpose1, (int)_n, (int)_m, (int)_k, pInput2, (int)_ldb, pInput1, (int)_lda, pOutput, (int)_ldc);
        }
        else
        {
            function.CallGEMM<ValueType>(_transpose1, _transpose2, (int)_m, (int)_n, (int)_k, pInput1, (int)_lda, pInput2, (int)_ldb, pOutput, (int)_ldc);
        }
    }

    template <typename ValueType>
    ell::utilities::ArchiveVersion MatrixMatrixMultiplyNode<ValueType>::GetArchiveVersion() const
    {
        return std::max(currentArchiveVersion, CompilableNode::GetArchiveVersion());
    }

    template <typename ValueType>
    bool MatrixMatrixMultiplyNode<ValueType>::CanReadArchiveVersion(const utilities::ArchiveVersion& version) const
    {
        return CompilableNode::CanReadArchiveVersion(version);
    }

    template <typename ValueType>
    void MatrixMatrixMultiplyNode<ValueType>::WriteToArchive(utilities::Archiver& archiver) const
    {
        Node::WriteToArchive(archiver);
        archiver[defaultInput1PortName] << _input1;
        archiver[defaultInput2PortName] << _input2;
        archiver[defaultOutputPortName] << _output;
        archiver["m"] << _m;
        archiver["n"] << _n;
        archiver["k"] << _k;
        archiver["lda"] << _lda;
        archiver["ldb"] << _ldb;
        archiver["ldc"] << _ldc;
        archiver["transpose1"] << _transpose1;
        archiver["transpose2"] << _transpose2;
        archiver["transposeOutput"] << _transposeOutput;
    }

    template <typename ValueType>
    void MatrixMatrixMultiplyNode<ValueType>::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        // TODO: check version number and read this format if in back-compat mode
        Node::ReadFromArchive(archiver);
        archiver[defaultInput1PortName] >> _input1;
        archiver[defaultInput2PortName] >> _input2;
        archiver[defaultOutputPortName] >> _output;
        archiver["m"] >> _m;
        archiver["n"] >> _n;
        archiver["k"] >> _k;
        archiver["lda"] >> _lda;
        archiver["ldb"] >> _ldb;
        archiver["ldc"] >> _ldc;
        archiver["transpose1"] >> _transpose1;
        archiver["transpose2"] >> _transpose2;
        archiver.OptionalProperty("transposeOutput", false) >> _transposeOutput;
    }

    template <typename ValueType>
    const model::OutputPort<ValueType>& MatrixMatrixMultiply(const model::OutputPort<ValueType>& input1, const model::OutputPort<ValueType>& input2)
    {
        model::Model* model = input1.GetNode()->GetModel();
        if (model == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input not part of a model");
        }
        if (*model != *(input2.GetNode()->GetModel()))
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Inputs not part of the same model");
        }

        auto node = model->AddNode<MatrixMatrixMultiplyNode<ValueType>>(input1, input2);
        return node->output;
    }

    template <typename ValueType>
    const model::OutputPort<ValueType>& MatrixMatrixMultiply(const model::OutputPort<ValueType>& input1, const model::OutputPort<ValueType>& input2, const model::PortMemoryLayout& outputMemoryLayout)
    {
        model::Model* model = input1.GetNode()->GetModel();
        if (model == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input not part of a model");
        }
        if (*model != *(input2.GetNode()->GetModel()))
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Inputs not part of the same model");
        }

        auto node = model->AddNode<MatrixMatrixMultiplyNode<ValueType>>(input1, input2, outputMemoryLayout);
        return node->output;
    }

    // Explicitly instantiate versions
    template class MatrixMatrixMultiplyNode<float>;
    template const model::OutputPort<float>& MatrixMatrixMultiply(const model::OutputPort<float>& input1, const model::OutputPort<float>& input2);
    template const model::OutputPort<float>& MatrixMatrixMultiply(const model::OutputPort<float>& input1, const model::OutputPort<float>& input2, const model::PortMemoryLayout& outputMemoryLayout);

    template class MatrixMatrixMultiplyNode<double>;
    template const model::OutputPort<double>& MatrixMatrixMultiply(const model::OutputPort<double>& input1, const model::OutputPort<double>& input2);
    template const model::OutputPort<double>& MatrixMatrixMultiply(const model::OutputPort<double>& input1, const model::OutputPort<double>& input2, const model::PortMemoryLayout& outputMemoryLayout);

} // namespace nodes
} // namespace ell
