////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRLoopEmitter.cpp (emitters)
//  Authors:  Umesh Madan, Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRLoopEmitter.h"
#include "IRFunctionEmitter.h"

namespace ell
{
namespace emitters
{
    const std::string LoopInitBlockName = "loop.init";
    const std::string LoopConditionBlockName = "loop.cond";
    const std::string LoopBodyBlockName = "loop.body";
    const std::string LoopIncBlockName = "loop.inc";
    const std::string LoopAfterBlockName = "loop.after";

    namespace
    {
        std::array<llvm::Metadata*, 2> GenerateUnrollMetadata(llvm::LLVMContext& context)
        {
            return { llvm::MDString::get(context, "llvm.loop.unroll.enable"),
                     llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                         llvm::Type::getInt1Ty(context), true)) };
        }

        std::array<llvm::Metadata*, 2> GenerateVectorizeMetadata(llvm::LLVMContext& context)
        {
            return { llvm::MDString::get(context, "llvm.loop.vectorize.enable"),
                     llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                         llvm::Type::getInt1Ty(context), true)) };
        }

        std::array<llvm::Metadata*, 2> GenerateVectorizeFollowupMetadata(llvm::LLVMContext& context)
        {
            return { llvm::MDString::get(context, "llvm.loop.vectorize.followup_vectorized"), llvm::MDNode::get(context, GenerateUnrollMetadata(context)) };
        }

        std::array<llvm::Metadata*, 1> GenerateDisableNonforcedMetadata(llvm::LLVMContext& context)
        {
            return { llvm::MDString::get(context, "llvm.loop.disable_nonforced") };
        }
    } // namespace

    IRLoopEmitter::IRLoopEmitter(IRFunctionEmitter& functionEmitter) :
        _functionEmitter(functionEmitter)
    {}

    void IRLoopEmitter::AddLoopMetadata(llvm::BranchInst* branch, bool unroll, bool vectorize)
    {
        // Add metadata
        auto& context = _functionEmitter.GetEmitter().GetContext();
        std::vector<llvm::Metadata*> metadataElements;

        // Reserve operand 0 for loop id self reference.
        auto tempNode = llvm::MDNode::getTemporary(context, {});
        metadataElements.push_back(tempNode.get());

        if (vectorize)
        {
            metadataElements.push_back(llvm::MDNode::get(context, GenerateVectorizeMetadata(context)));
            if (unroll)
            {
                metadataElements.push_back(llvm::MDNode::get(context, GenerateVectorizeFollowupMetadata(context)));
            }
        }
        else if (unroll)
        {
            metadataElements.push_back(llvm::MDNode::get(context, GenerateUnrollMetadata(context)));
        }

        if (unroll || vectorize)
        {
            metadataElements.push_back(llvm::MDNode::get(context, GenerateDisableNonforcedMetadata(context)));
        }

        auto loopID = llvm::MDNode::get(context, metadataElements);
        loopID->replaceOperandWith(0, loopID);
        branch->setMetadata("llvm.loop", loopID);
    }

    // Blocks used in a for loop:
    //
    // _pInitializationBlock -- setup
    // _pConditionBlock -- termination check
    // _pBodyBlock -- body of the loop
    // _pIncrementBlock -- increment iteration variable and branch back to condition block
    // _pAfterBlock -- branch to this block when done
    //

    IRForLoopEmitter::IRForLoopEmitter(IRFunctionEmitter& functionEmitter, const std::string& tag) :
        IRLoopEmitter(functionEmitter),
        _tag(tag)
    {}

    void IRForLoopEmitter::CreateBlocks()
    {
        _pInitializationBlock = _functionEmitter.Block(_tag + LoopInitBlockName);
        _pConditionBlock = _functionEmitter.Block(_tag + LoopConditionBlockName);
        _pBodyBlock = _functionEmitter.Block(_tag + LoopBodyBlockName);
        _pIncrementBlock = _functionEmitter.Block(_tag + LoopIncBlockName);
        _pAfterBlock = _functionEmitter.Block(_tag + LoopAfterBlockName);
    }

    llvm::BasicBlock* IRForLoopEmitter::Begin(int repeatCount)
    {
        return Begin(0, repeatCount, 1);
    }

    llvm::BasicBlock* IRForLoopEmitter::Begin(int iStartAt, int iMaxValue, int stepSize)
    {
        return Begin(_functionEmitter.Literal(iStartAt), _functionEmitter.Literal(iMaxValue), _functionEmitter.Literal(stepSize));
    }

    llvm::BasicBlock* IRForLoopEmitter::Begin(LLVMValue iStartAt, LLVMValue iMaxValue, LLVMValue stepSize)
    {
        CreateBlocks();
        EmitIterationVariable(iStartAt);
        EmitCondition(GetComparison(iMaxValue->getType(), BinaryPredicateType::less), iMaxValue);
        EmitIncrement(stepSize);
        return PrepareBody();
    }

    llvm::BasicBlock* IRForLoopEmitter::Begin(LLVMValue pRepeatCount)
    {
        if (pRepeatCount == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Repeat count must not be null");
        }

        return Begin(_functionEmitter.Literal(0), pRepeatCount, _functionEmitter.Literal(1));
    }

    LLVMValue IRForLoopEmitter::LoadIterationVariable()
    {
        return _functionEmitter.Load(_pIterationVariable);
    }

    void IRForLoopEmitter::EmitIterationVariable(LLVMValue pStartValue)
    {
        _functionEmitter.Branch(_pInitializationBlock);
        _functionEmitter.SetCurrentBlock(_pInitializationBlock);
        {
            _pIterationVariable = _functionEmitter.Variable(pStartValue->getType());
            _functionEmitter.Store(_pIterationVariable, pStartValue);
        }
    }

    void IRForLoopEmitter::EmitCondition(TypedComparison comparison, LLVMValue pTestValue)
    {
        _functionEmitter.Branch(_pConditionBlock);
        _functionEmitter.SetCurrentBlock(_pConditionBlock);
        auto branchInst = _functionEmitter.Branch(comparison, _functionEmitter.Load(_pIterationVariable), pTestValue, _pBodyBlock, _pAfterBlock);

        bool unroll = false;
        bool vectorize = true;
        AddLoopMetadata(branchInst, unroll, vectorize);
    }

    void IRForLoopEmitter::EmitIncrement(LLVMValue pIncrementValue)
    {
        auto isFp = pIncrementValue->getType()->isFloatingPointTy();
        _functionEmitter.SetCurrentBlock(_pIncrementBlock);
        _functionEmitter.OperationAndUpdate(_pIterationVariable,
                                            isFp ? TypedOperator::addFloat : TypedOperator::add,
                                            pIncrementValue);
        _functionEmitter.Branch(_pConditionBlock);
    }

    llvm::BasicBlock* IRForLoopEmitter::PrepareBody()
    {
        _functionEmitter.SetCurrentBlock(_pBodyBlock);
        return _pBodyBlock;
    }

    void IRForLoopEmitter::End()
    {
        if (_pIncrementBlock == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Got null increment block --- End() was probably called without first calling Begin()");
        }

        // Caller is done generating the body. Add a branch from the Body block to the increment block
        _functionEmitter.Branch(_pIncrementBlock);
        _functionEmitter.SetCurrentBlock(_pAfterBlock);
    }

    // Blocks used in a while loop:
    //
    // _pInitializationBlock -- setup
    // _pConditionBlock -- termination check
    // _pBodyBlock -- body of the loop
    // _pAfterBlock -- branch to this block when done
    //

    //
    // IRWhileLoopEmitter
    //
    IRWhileLoopEmitter::IRWhileLoopEmitter(IRFunctionEmitter& functionEmitter, const std::string& tag) :
        IRLoopEmitter(functionEmitter),
        _tag(tag)
    {}

    void IRWhileLoopEmitter::CreateBlocks()
    {
        _pInitializationBlock = _functionEmitter.Block(_tag + LoopInitBlockName);
        _pConditionBlock = _functionEmitter.Block(_tag + LoopConditionBlockName);
        _pBodyBlock = _functionEmitter.Block(_tag + LoopBodyBlockName);
        _pAfterBlock = _functionEmitter.Block(_tag + LoopAfterBlockName);
    }

    llvm::BasicBlock* IRWhileLoopEmitter::Begin(LLVMValue pTestValuePointer)
    {
        if (pTestValuePointer == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Test value must not be null");
        }

        CreateBlocks();
        EmitInitialization();
        EmitCondition(pTestValuePointer);
        return PrepareBody();
    }

    llvm::BasicBlock* IRWhileLoopEmitter::Begin(std::function<LLVMValue(IRFunctionEmitter&)> condition)
    {
        CreateBlocks();
        EmitInitialization();
        EmitCondition(condition);
        return PrepareBody();
    }

    void IRWhileLoopEmitter::EmitInitialization()
    {
        _functionEmitter.Branch(_pInitializationBlock);
        _functionEmitter.SetCurrentBlock(_pInitializationBlock);
        {
            // Currently, no initialization needs to be done in a while loop
        }
    }

    void IRWhileLoopEmitter::EmitCondition(LLVMValue pTestValuePointer)
    {
        _functionEmitter.Branch(_pConditionBlock);
        _functionEmitter.SetCurrentBlock(_pConditionBlock);
        {
            _functionEmitter.Branch(_functionEmitter.Load(_functionEmitter.PointerOffset(pTestValuePointer, _functionEmitter.Literal(0))),
                                    _pBodyBlock,
                                    _pAfterBlock);
        }
    }

    void IRWhileLoopEmitter::EmitCondition(std::function<LLVMValue(IRFunctionEmitter&)> condition)
    {
        _functionEmitter.Branch(_pConditionBlock);
        _functionEmitter.SetCurrentBlock(_pConditionBlock);
        auto conditionValue = condition(_functionEmitter);
        auto branchInst = _functionEmitter.Branch(conditionValue,
                                                  _pBodyBlock,
                                                  _pAfterBlock);
        bool unroll = false;
        bool vectorize = false;
        AddLoopMetadata(branchInst, unroll, vectorize);
    }

    llvm::BasicBlock* IRWhileLoopEmitter::PrepareBody()
    {
        _functionEmitter.SetCurrentBlock(_pBodyBlock);
        return _pBodyBlock;
    }

    void IRWhileLoopEmitter::End()
    {
        // Caller is done generating the body. Add a branch from the Body block to the condition block
        _functionEmitter.Branch(_pConditionBlock);
        _functionEmitter.SetCurrentBlock(_pAfterBlock);
    }
} // namespace emitters
} // namespace ell
