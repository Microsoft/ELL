////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     MatrixMatrixMultiplyCodeNode.h (nodes)
//  Authors:  Mason Remy, Denny Sun
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include <model/include/ModelTransformer.h>

#include <nodes/include/MatrixMatrixMultiplyCodeNode.h>

#include <utilities/include/ArchiveVersion.h>
#include <utilities/include/Exception.h>
#include <utilities/include/IArchivable.h>

#include <value/include/FunctionDeclaration.h>
#include <value/include/Matrix.h>
#include <value/include/MatrixOperations.h>
#include <value/include/Scalar.h>
#include <value/include/ScalarOperations.h>

#include <value/include/loopnests/CodeGenerator.h>
#include <value/include/loopnests/Kernel.h>
#include <value/include/loopnests/LoopNest.h>

#include <llvm/Analysis/TargetTransformInfo.h>
#include <value/include/CachingStrategies.h>
#include <value/include/LLVMContext.h>

//using namespace ell::utilities;
using namespace ell::value;

namespace ell
{
namespace nodes
{
    template <typename ValueType>
    MatrixMatrixMultiplyCodeNode<ValueType>::MatrixMatrixMultiplyCodeNode() :
        CompilableCodeNode("MatrixMatrixMultiplyCodeNode", { &_input1, &_input2 }, { &_output }),
        _input1(this, {}, defaultInput1PortName),
        _input2(this, {}, defaultInput2PortName),
        _output(this, defaultOutputPortName, 0),
        _impl(MatrixMatrixMultiplyImplementation::DEFAULT)
    {
    }

    template <typename ValueType>
    MatrixMatrixMultiplyCodeNode<ValueType>::MatrixMatrixMultiplyCodeNode(const model::OutputPort<ValueType>& input1, const model::OutputPort<ValueType>& input2, const MatrixMatrixMultiplyImplementation& gemmImpl) :
        MatrixMatrixMultiplyCodeNode<ValueType>(input1, input2, _defaultPanelM, _defaultPanelN, _defaultPanelK, _defaultKernelM, _defaultKernelN, _defaultKernelK, gemmImpl)
    {
    }

    template <typename ValueType>
    MatrixMatrixMultiplyCodeNode<ValueType>::MatrixMatrixMultiplyCodeNode(const model::OutputPort<ValueType>& input1, const model::OutputPort<ValueType>& input2, int panelM, int panelN, int panelK, int kernelM, int kernelN, int kernelK, const MatrixMatrixMultiplyImplementation& gemmImpl) :
        MatrixMatrixMultiplyCodeNode<ValueType>(input1, input2, model::PortMemoryLayout({ input1.GetMemoryLayout().GetActiveSize(0), input2.GetMemoryLayout().GetActiveSize(1) }), panelM, panelN, panelK, kernelM, kernelN, kernelK, gemmImpl)
    {
    }

    template <typename ValueType>
    MatrixMatrixMultiplyCodeNode<ValueType>::MatrixMatrixMultiplyCodeNode(const model::OutputPort<ValueType>& input1, const model::OutputPort<ValueType>& input2, const model::PortMemoryLayout& outputLayout, const MatrixMatrixMultiplyImplementation& gemmImpl) :
        MatrixMatrixMultiplyCodeNode<ValueType>(input1, input2, outputLayout, _defaultPanelM, _defaultPanelN, _defaultPanelK, _defaultKernelM, _defaultKernelN, _defaultKernelK, gemmImpl)
    {
    }

    template <typename ValueType>
    MatrixMatrixMultiplyCodeNode<ValueType>::MatrixMatrixMultiplyCodeNode(const model::OutputPort<ValueType>& input1, const model::OutputPort<ValueType>& input2, const model::PortMemoryLayout& outputLayout, int panelM, int panelN, int panelK, int kernelM, int kernelN, int kernelK, const MatrixMatrixMultiplyImplementation& gemmImpl) :
        MatrixMatrixMultiplyCodeNode<ValueType>::MatrixMatrixMultiplyCodeNode(input1,
                                                                              input1.GetMemoryLayout().GetLogicalDimensionActiveSize(0),
                                                                              input2.GetMemoryLayout().GetLogicalDimensionActiveSize(1),
                                                                              input1.GetMemoryLayout().GetLogicalDimensionActiveSize(1),
                                                                              input1.GetMemoryLayout().GetExtent(1),
                                                                              !input1.GetMemoryLayout().IsCanonicalOrder(),
                                                                              input2,
                                                                              input2.GetMemoryLayout().GetExtent(1),
                                                                              !input2.GetMemoryLayout().IsCanonicalOrder(),
                                                                              outputLayout.GetExtent(1),
                                                                              !outputLayout.IsCanonicalOrder(),
                                                                              panelM,
                                                                              panelN,
                                                                              panelK,
                                                                              kernelM,
                                                                              kernelN,
                                                                              kernelK,
                                                                              gemmImpl)
    {
        if (input1.GetMemoryLayout().NumDimensions() != 2 || input2.GetMemoryLayout().NumDimensions() != 2)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input matrices must have a memory layout with 2 dimensions");
        }
        if (_k != input2.GetMemoryLayout().GetLogicalDimensionActiveSize(0))
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input matrices incompatible");
        }
    }

    template <typename ValueType>
    MatrixMatrixMultiplyCodeNode<ValueType>::MatrixMatrixMultiplyCodeNode(const model::OutputPort<ValueType>& input1, int m, int n, int k, int matrix1Stride, const model::OutputPort<ValueType>& input2, int matrix2Stride, int outputMatrixStride, const MatrixMatrixMultiplyImplementation& gemmImpl) :
        MatrixMatrixMultiplyCodeNode<ValueType>(input1, m, n, k, matrix1Stride, input2, matrix2Stride, outputMatrixStride, _defaultPanelM, _defaultPanelN, _defaultPanelK, _defaultKernelM, _defaultKernelN, _defaultKernelK, gemmImpl)
    {
    }

    template <typename ValueType>
    MatrixMatrixMultiplyCodeNode<ValueType>::MatrixMatrixMultiplyCodeNode(const model::OutputPort<ValueType>& input1, int m, int n, int k, int matrix1Stride, const model::OutputPort<ValueType>& input2, int matrix2Stride, int outputMatrixStride, int panelM, int panelN, int panelK, int kernelM, int kernelN, int kernelK, const MatrixMatrixMultiplyImplementation& gemmImpl) :
        MatrixMatrixMultiplyCodeNode<ValueType>(input1, m, n, k, matrix1Stride, false, input2, matrix2Stride, false, outputMatrixStride, false, panelM, panelN, panelK, kernelM, kernelN, kernelK, gemmImpl)
    {
    }

    template <typename ValueType>
    MatrixMatrixMultiplyCodeNode<ValueType>::MatrixMatrixMultiplyCodeNode(const model::OutputPort<ValueType>& input1, int m, int n, int k, int matrix1Stride, bool transpose1, const model::OutputPort<ValueType>& input2, int matrix2Stride, bool transpose2, int outputMatrixStride, const MatrixMatrixMultiplyImplementation& gemmImpl) :
        MatrixMatrixMultiplyCodeNode<ValueType>(input1, m, n, k, matrix1Stride, transpose1, input2, matrix2Stride, transpose2, outputMatrixStride, _defaultPanelM, _defaultPanelN, _defaultPanelK, _defaultKernelM, _defaultKernelN, _defaultKernelK, gemmImpl)
    {
    }

    template <typename ValueType>
    MatrixMatrixMultiplyCodeNode<ValueType>::MatrixMatrixMultiplyCodeNode(const model::OutputPort<ValueType>& input1, int m, int n, int k, int matrix1Stride, bool transpose1, const model::OutputPort<ValueType>& input2, int matrix2Stride, bool transpose2, int outputMatrixStride, int panelM, int panelN, int panelK, int kernelM, int kernelN, int kernelK, const MatrixMatrixMultiplyImplementation& gemmImpl) :
        MatrixMatrixMultiplyCodeNode<ValueType>(input1, m, n, k, matrix1Stride, transpose1, input2, matrix2Stride, transpose2, outputMatrixStride, false, panelM, panelN, panelK, kernelM, kernelN, kernelK, gemmImpl)
    {
    }

    template <typename ValueType>
    MatrixMatrixMultiplyCodeNode<ValueType>::MatrixMatrixMultiplyCodeNode(const model::OutputPort<ValueType>& input1, int m, int n, int k, int matrix1Stride, bool transpose1, const model::OutputPort<ValueType>& input2, int matrix2Stride, bool transpose2, int outputMatrixStride, bool transposeOutput, const MatrixMatrixMultiplyImplementation& gemmImpl) :
        MatrixMatrixMultiplyCodeNode<ValueType>(input1, m, n, k, matrix1Stride, transpose1, input2, matrix2Stride, transpose2, outputMatrixStride, transposeOutput, _defaultPanelM, _defaultPanelN, _defaultPanelK, _defaultKernelM, _defaultKernelN, _defaultKernelK, gemmImpl)
    {
    }

    template <typename ValueType>
    MatrixMatrixMultiplyCodeNode<ValueType>::MatrixMatrixMultiplyCodeNode(const model::OutputPort<ValueType>& input1, int m, int n, int k, int matrix1Stride, bool transpose1, const model::OutputPort<ValueType>& input2, int matrix2Stride, bool transpose2, int outputMatrixStride, bool transposeOutput, int panelM, int panelN, int panelK, int kernelM, int kernelN, int kernelK, const MatrixMatrixMultiplyImplementation& gemmImpl) :
        CompilableCodeNode("MatrixMatrixMultiplyCodeNode", { &_input1, &_input2 }, { &_output }),
        _input1(this, input1, defaultInput1PortName),
        _input2(this, input2, defaultInput2PortName),
        _output(this, defaultOutputPortName, utilities::MemoryShape{ m, n }),
        _m(m),
        _n(n),
        _k(k),
        _lda(matrix1Stride),
        _ldb(matrix2Stride),
        _ldc(outputMatrixStride),
        _transpose1(transpose1),
        _transpose2(transpose2),
        _transposeOutput(transposeOutput),
        _panelM(panelM),
        _panelN(panelN),
        _panelK(panelK),
        _kernelM(kernelM),
        _kernelN(kernelN),
        _kernelK(kernelK),
        _impl(gemmImpl)
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
    void MatrixMatrixMultiplyCodeNode<ValueType>::ZeroMatrix(value::Matrix matrix) const
    {
        namespace loopnests = ell::value::loopnests;
        int M = (int)(matrix.Rows());
        int N = (int)(matrix.Columns());
        loopnests::Index m("m"), n("n");
        loopnests::LoopNest zeroingLoop({ { m, { 0, M } },
                                          { n, { 0, N } } });

        auto [mKernelOuter, mKernelInner] = zeroingLoop.Split(m, _kernelM);
        auto [nKernelOuter, nKernelInner] = zeroingLoop.Split(n, _kernelN);
        auto zeroingKernel = loopnests::Kernel("Zero_output")
                                 .Inputs(matrix.GetValue())
                                 .Indices(m, n)
                                 .Define([&](value::Matrix C, value::Scalar row, value::Scalar col) {
                                     C(row, col) = static_cast<ValueType>(0);
                                 });
        zeroingLoop.AddKernel(zeroingKernel);
        zeroingLoop.Unroll(mKernelInner);
        zeroingLoop.Unroll(nKernelInner);
        loopnests::CodeGenerator zeroingGenerator;
        zeroingGenerator.Run(zeroingLoop);
    }

    template <typename ValueType>
    void MatrixMatrixMultiplyCodeNode<ValueType>::ForLoopGEMM(const value::Matrix matA, const value::Matrix matB, value::Matrix matC)
    {
        namespace loopnests = ell::value::loopnests;
        // Currently treat beta as 0
        ZeroMatrix(matC);

        int M = (int)(matA.Rows());
        int N = (int)(matB.Columns());
        int K = (int)(matA.Columns());

        loopnests::Index m("m"), n("n"), k("k");
        loopnests::LoopNest loop({ { m, { 0, M } },
                                   { n, { 0, N } },
                                   { k, { 0, K } } });

        // innermost GEMM kernel
        auto kernel = loopnests::Kernel("GEMMKernel")
                          .Inputs(matA.GetValue(), matB.GetValue(), matC.GetValue())
                          .Indices(m, n, k)
                          .Define([](value::Matrix A, value::Matrix B, value::Matrix C, value::Scalar m, value::Scalar n, value::Scalar k) {
                              C(m, n) += A(m, k) * B(k, n);
                          });

        loop.AddKernel(kernel, loopnests::LoopFragmentType::body);
        loop.SetLoopOrder({ m, k, n });
        auto outputC = matC.GetValue();
        outputC.SetLayout({ { (int)matC.Size() } });
        //  ell::DebugPrintVector(outputC);
        loopnests::CodeGenerator generator;
        generator.Run(loop);
    }

    template <typename ValueType>
    void MatrixMatrixMultiplyCodeNode<ValueType>::Gemm(value::Matrix A, value::Matrix B, value::Matrix C)
    {
        using namespace value;

        int vectorSize = 4;
        int NumRowsInKernel = 2;

        InvokeForContext<LLVMContext>([&](LLVMContext& context) {
            auto targetMachine = context.GetModuleEmitter().GetTargetMachine();
            auto fn = context.GetFunctionEmitter().GetFunction();
            auto info = targetMachine->getTargetTransformInfo(*fn);
            // See https://llvm.org/doxygen/classllvm_1_1TargetTransformInfo.html for the big list of amazing things you can get from this TargetMachineInfo object
            vectorSize = static_cast<int>(info.getRegisterBitWidth(true)) / (8 * sizeof(float));
            if (vectorSize > 8)
            {
                // The vector width is 16 floats instead of 8 (e.g. in AVX-512), so double the sizes as needed
                vectorSize = 16;
                NumRowsInKernel = 12;
            }
        });

        int NumColumnsInKernel = 2 * vectorSize;
        if (NumColumnsInKernel > (int)B.Columns())
        {
            NumColumnsInKernel = vectorSize;
            NumRowsInKernel *= 2;
        }

        // Declare and/or calculate constants
        const int OutputRows = (int)(A.Rows());
        const int OutputColumns = (int)(B.Columns());
        const int InnerDimension = (int)(A.Columns());
        const int kUnroll = 4;
        int columnBlock = std::min(64, OutputColumns);
        int innerDimensionBlock = std::min(256, InnerDimension);

        // Declare indexes
        loopnests::Index i("i"), j("j"), k("k");
        // Define LoopNest
        auto nest = Using({ A, B }, ArgumentType::Input)
                        .Using({ C }, ArgumentType::Output)
                        .ForAll(i, 0, OutputRows)
                        .ForAll(j, 0, OutputColumns)
                        .ForAll(k, 0, InnerDimension)
                        .Do([](Matrix A_, Matrix B_, Matrix C_, Scalar i_, Scalar j_, Scalar k_) {
                            C_(i_, j_) += B_(k_, j_) * A_(i_, k_);
                        });
        auto& schedule = nest.GetSchedule();

        auto topLevelJ = j;
        auto topLevelK = k;

        // Declare splits
        auto jCache = schedule.Split(j, columnBlock);
        auto kCache = schedule.Split(k, innerDimensionBlock);
        auto kBlock = schedule.Split(k, kUnroll);
        auto jKernelOuter2 = schedule.Split(j, NumColumnsInKernel);
        auto jKernelOuter = schedule.Split(j, vectorSize);
        auto iKernelOuter = schedule.Split(i, NumRowsInKernel);

        // Set the order
        schedule.SetOrder({ jCache, kCache, iKernelOuter, jKernelOuter2, kBlock, k, i, jKernelOuter, j });

        // Set up caching
        if ((OutputColumns > NumColumnsInKernel) && ((OutputColumns % NumColumnsInKernel) == 0))
        {
            auto extraCacheBParams = std::make_tuple(NumColumnsInKernel, jKernelOuter2, BoundaryConditionHandling::ZeroPadding);
            schedule.template Cache<BLASTCopy>(B,
                                               { topLevelK, topLevelJ },
                                               { innerDimensionBlock, columnBlock },
                                               { kCache, jCache },
                                               std::nullopt, // Order isn't used by BLASTCopy
                                               extraCacheBParams);
        }
        auto extraZeroInputReduceOutputParams = std::make_tuple(vectorSize);
        schedule.template Cache<ZeroInputReduceOutput>(C,
                                                       { iKernelOuter, jKernelOuter2 },
                                                       { NumRowsInKernel, NumColumnsInKernel },
                                                       { iKernelOuter, jKernelOuter2 },
                                                       utilities::RowMajorMatrixOrder,
                                                       extraZeroInputReduceOutputParams);

        // Set unrolling
        schedule.Unroll(jKernelOuter);
        schedule.Unroll(i);
        schedule.Unroll(k);

        // Run the generator
        nest.Run();
    }

    template <typename ValueType>
    void MatrixMatrixMultiplyCodeNode<ValueType>::GemmFn(value::Matrix A, value::Matrix B, value::Matrix C, int thread_num)
    {
        value::DeclareFunction("InnerMatMul" + std::to_string(thread_num))
            .Parameters(A, B, C)
            .Define([this](value::Matrix A, value::Matrix B, value::Matrix C) {
                Gemm(A, B, C);
            })(A, B, C);
    }

    template <typename ValueType>
    void MatrixMatrixMultiplyCodeNode<ValueType>::ParallelizeGemmCol(Matrix A, Matrix B, Matrix C, int numThreads)
    {
        const int columns = B.Columns() / numThreads;
        const int col_spill = B.Columns() % numThreads;

        Parallelize(
            numThreads,
            std::tuple{ A, B, C },
            [=](value::Scalar id, value::Matrix A, value::Matrix B, value::Matrix C) {
                value::Scalar colStart = id * value::Scalar{ columns };
                int thread_seq = 0;

                EmitterContext::IfContext IfCxt = If(id == thread_seq,
                                                     [&] {
                                                         GemmFn(
                                                             A,
                                                             B.SubMatrix(value::Scalar{ 0 }, colStart, (int)B.Rows(), columns),
                                                             C.SubMatrix(value::Scalar{ 0 }, colStart, (int)C.Rows(), columns),
                                                             thread_seq);
                                                     });

                thread_seq++;

                for (int i = thread_seq; i < numThreads; i++)
                {
                    IfCxt.ElseIf(id == i,
                                 [&] {
                                     int actualColumns = i == (numThreads - 1) ? columns + col_spill : columns;

                                     GemmFn(
                                         A,
                                         B.SubMatrix(value::Scalar{ 0 }, colStart, (int)B.Rows(), actualColumns),
                                         C.SubMatrix(value::Scalar{ 0 }, colStart, (int)C.Rows(), actualColumns),
                                         i);
                                 });
                }
            });
    }

    template <typename ValueType>
    void MatrixMatrixMultiplyCodeNode<ValueType>::ParallelizeGemmRow(Matrix A, Matrix B, Matrix C, int numThreads)
    {
        const int rows = A.Rows() / numThreads;
        const int row_spill = A.Rows() % numThreads;

        Parallelize(
            numThreads,
            std::tuple{ A, B, C },
            [=](value::Scalar id, value::Matrix A, value::Matrix B, value::Matrix C) {
                value::Scalar rowStart = id * value::Scalar{ rows };
                int thread_seq = 0;

                EmitterContext::IfContext IfCxt = If(id == thread_seq,
                                                     [&] {
                                                         GemmFn(
                                                             A.SubMatrix(rowStart, value::Scalar{ 0 }, rows, (int)A.Columns()),
                                                             B,
                                                             C.SubMatrix(rowStart, value::Scalar{ 0 }, rows, (int)C.Columns()),
                                                             thread_seq);
                                                     });

                thread_seq++;

                for (int i = thread_seq; i < numThreads; i++)
                {
                    IfCxt.ElseIf(id == i,
                                 [&] {
                                     int actualRows = i == (numThreads - 1) ? rows + row_spill : rows;
                                     GemmFn(
                                         A.SubMatrix(rowStart, value::Scalar{ 0 }, actualRows, (int)A.Columns()),
                                         B,
                                         C.SubMatrix(rowStart, value::Scalar{ 0 }, actualRows, (int)C.Columns()),
                                         i);
                                 });
                }
            });
    }

    template <typename ValueType>
    void MatrixMatrixMultiplyCodeNode<ValueType>::ELLCodeGEMM(const value::Matrix matA, const value::Matrix matB, value::Matrix matC)
    {
        double computationSize = double(matC.Rows() * matC.Columns() * matA.Columns());

        size_t minThreadLoad = 112 * 1024;
        const size_t maxThreads = 4;
        size_t numThreads = maxThreads;

        if (computationSize < double(minThreadLoad * maxThreads))
        {
            numThreads = std::min(int(computationSize / double(minThreadLoad)) + 1, int(maxThreads));
        }
        if (numThreads > 1)
        {
            if (matC.Rows() > matC.Columns())
            {
                ParallelizeGemmRow(matA, matB, matC, int(numThreads));
            }
            else
            {
                ParallelizeGemmCol(matA, matB, matC, int(numThreads));
            }
        }
        else
        {
            Gemm(matA, matB, matC);
        }
    }

    template <typename ValueType>
    void MatrixMatrixMultiplyCodeNode<ValueType>::Define(value::FunctionDeclaration& fn)
    {
        (void)fn.Define([this](const value::Value valueA, const value::Value valueB, value::Value valueC) {
            auto tempA = valueA;
            tempA.SetLayout(utilities::MemoryLayout({ _m, _k }));
            auto tempB = valueB;
            tempB.SetLayout(utilities::MemoryLayout({ _k, _n }));
            auto tempC = valueC;
            if (_transposeOutput)
            {
                tempC.SetLayout(utilities::MemoryLayout({ _n, _m }, utilities::DimensionOrder{ 1, 0 }));
            }
            else
            {
                tempC.SetLayout(utilities::MemoryLayout({ _m, _n }));
            }

            auto matA = value::Matrix(tempA);
            auto matB = value::Matrix(tempB);
            auto matC = value::Matrix(tempC);

            switch (_impl)
            {
            case (MatrixMatrixMultiplyImplementation::SimpleForLoops):
                ForLoopGEMM(matA, matB, matC);
                break;
            case (MatrixMatrixMultiplyImplementation::Mlas_Loopnest_Value):
                ELLCodeGEMM(matA, matB, matC);
                break;
            case (MatrixMatrixMultiplyImplementation::LAST):
                throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "MatrixMatrixMultiplyImplementation::LAST is not a valid impl value");
                break;
            }
        });
    }

    template <typename ValueType>
    void MatrixMatrixMultiplyCodeNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInput1 = transformer.GetCorrespondingInputs(_input1);
        const auto& newInput2 = transformer.GetCorrespondingInputs(_input2);
        auto newNode = transformer.AddNode<MatrixMatrixMultiplyCodeNode<ValueType>>(newInput1, _m, _n, _k, _lda, _transpose1, newInput2, _ldb, _transpose2, _ldc, _transposeOutput, _panelM, _panelN, _panelK, _kernelM, _kernelN, _kernelK, _impl);
        transformer.MapNodeOutput(output, newNode->output);
    }

    template <typename ValueType>
    utilities::ArchiveVersion MatrixMatrixMultiplyCodeNode<ValueType>::GetArchiveVersion() const
    {
        constexpr utilities::ArchiveVersion currentArchiveVersion = { utilities::ArchiveVersionNumbers::v2 };
        return std::max(currentArchiveVersion, CompilableCodeNode::GetArchiveVersion());
    }

    template <typename ValueType>
    bool MatrixMatrixMultiplyCodeNode<ValueType>::CanReadArchiveVersion(const utilities::ArchiveVersion& version) const
    {
        return CompilableCodeNode::CanReadArchiveVersion(version);
    }

    template <typename ValueType>
    void MatrixMatrixMultiplyCodeNode<ValueType>::WriteToArchive(utilities::Archiver& archiver) const
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
        archiver["panelM"] << _panelM;
        archiver["panelN"] << _panelN;
        archiver["panelK"] << _panelK;
        archiver["kernelM"] << _kernelM;
        archiver["kernelN"] << _kernelN;
        archiver["kernelK"] << _kernelK;
        archiver["gemmImpl"] << static_cast<int>(_impl);
    }

    template <typename ValueType>
    void MatrixMatrixMultiplyCodeNode<ValueType>::ReadFromArchive(utilities::Unarchiver& archiver)
    {
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
        archiver["panelM"] >> _panelM;
        archiver["panelN"] >> _panelN;
        archiver["panelK"] >> _panelK;
        archiver["kernelM"] >> _kernelM;
        archiver["kernelN"] >> _kernelN;
        archiver["kernelK"] >> _kernelK;
        int gemmImpl = 0;
        archiver["gemmImpl"] >> gemmImpl;
        _impl = static_cast<MatrixMatrixMultiplyImplementation>(gemmImpl);
    }

    //
    // Explicit instantiation definitions
    //
    template class MatrixMatrixMultiplyCodeNode<float>;
    template class MatrixMatrixMultiplyCodeNode<double>;
} // namespace nodes
} // namespace ell
