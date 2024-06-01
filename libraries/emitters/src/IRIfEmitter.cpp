////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRIfEmitter.cpp (emitters)
//  Authors:  Umesh Madan
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IRIfEmitter.h"
#include "IRBlockRegion.h"
#include "IRFunctionEmitter.h"

namespace ell
{
namespace emitters
{
    namespace
    {
        bool HasTerminatingBranch(llvm::BasicBlock* block)
        {
            auto term = block->getTerminator();
            if (term == nullptr)
                return false;

            auto branch = llvm::dyn_cast<llvm::BranchInst>(term);
            if (branch == nullptr)
                return false;

            return true;
        }
    } // namespace
    const std::string IfCondBlockName = "if.cond";
    const std::string IfThenBlockName = "if.then";
    const std::string IfElseBlockName = "if.else";
    const std::string IfEndBlockName = "if.end";
    const std::string IfAfterBlockName = "if.after";

    IRIfEmitter::IRIfEmitter(IRFunctionEmitter& functionEmitter, llvm::BasicBlock* pPrevBlock) :
        _functionEmitter(&functionEmitter),
        _pEndBlock(pPrevBlock)
    {
        assert(_functionEmitter);

        _pAfterBlock = _functionEmitter->BlockAfter(GetParentBlock(), IfAfterBlockName);
    }

    IRIfEmitter::IRIfEmitter(IRFunctionEmitter& functionEmitter, TypedComparison comparison, LLVMValue pValue, LLVMValue pTestValue) :
        IRIfEmitter(functionEmitter)
    {
        If(comparison, pValue, pTestValue);
    }

    // Move ctor and assignment op are needed to explicitly swap out values,
    // since the default behavior for "moving" fundamental types (aka, pointers) is to do a bitwise-copy
    IRIfEmitter::IRIfEmitter(IRIfEmitter&& other) noexcept
    {
        *this = std::move(other);
    }

    IRIfEmitter& IRIfEmitter::operator=(IRIfEmitter&& other) noexcept
    {
        if (this != &other)
        {
            std::swap(this->_functionEmitter, other._functionEmitter);
            std::swap(this->_pConditionBlock, other._pConditionBlock);
            std::swap(this->_pThenBlock, other._pThenBlock);
            std::swap(this->_pEndBlock, other._pEndBlock);
            std::swap(this->_pAfterBlock, other._pAfterBlock);
            std::swap(this->_isContinuation, other._isContinuation);
            std::swap(this->_finished, other._finished);
            // Turn off the 'other' object destructor auto block termination, since we are handing that responsibility off to this new copy.
            other._finished = true;
        }

        return *this;
    }

    IRIfEmitter::~IRIfEmitter()
    {
        End();
    }

    IRIfEmitter& IRIfEmitter::If(LLVMValue pValue, IRIfEmitter::IfElseBodyFunction body)
    {
        If(pValue);
        {
            body(*_functionEmitter);
        }
        EndPrev();
        _functionEmitter->SetCurrentBlock(GetParentBlock());
        _isContinuation = true;
        return *this;
    }

    IRIfEmitter& IRIfEmitter::If(TypedComparison comparison, LLVMValue pValue, LLVMValue pTestValue, IRIfEmitter::IfElseBodyFunction body)
    {
        If(comparison, pValue, pTestValue);
        {
            body(*_functionEmitter);
        }
        EndPrev();
        _functionEmitter->SetCurrentBlock(GetParentBlock());
        _isContinuation = true;
        return *this;
    }

    IRIfEmitter& IRIfEmitter::ElseIf(LLVMValue pValue, IRIfEmitter::IfElseBodyFunction body)
    {
        auto elseBlock = If(pValue);

        elseBlock->setName(IfElseBlockName);
        body(*_functionEmitter);

        EndPrev();
        _functionEmitter->SetCurrentBlock(GetParentBlock());
        _isContinuation = true;
        return *this;
    }

    IRIfEmitter& IRIfEmitter::Else(IRIfEmitter::IfElseBodyFunction body)
    {
        Else();
        {
            body(*_functionEmitter);
        }
        End();
        return *this;
    }

    llvm::BasicBlock* IRIfEmitter::If(LLVMValue pValue)
    {
        EndPrev();
        PrepareBlocks();
        _functionEmitter->SetCurrentBlock(_pConditionBlock);
        _functionEmitter->Branch(pValue, _pThenBlock, _pEndBlock);
        _functionEmitter->SetCurrentBlock(_pThenBlock);
        return _pThenBlock;
    }

    llvm::BasicBlock* IRIfEmitter::If(TypedComparison comparison, LLVMValue pValue, LLVMValue pTestValue)
    {
        EndPrev();
        PrepareBlocks();
        _functionEmitter->SetCurrentBlock(_pConditionBlock);
        _functionEmitter->Branch(comparison, pValue, pTestValue, _pThenBlock, _pEndBlock);
        _functionEmitter->SetCurrentBlock(_pThenBlock);
        return _pThenBlock;
    }

    llvm::BasicBlock* IRIfEmitter::If(std::function<LLVMValue()> comparison)
    {
        EndPrev();
        PrepareBlocks();
        _functionEmitter->SetCurrentBlock(_pConditionBlock);
        _functionEmitter->Branch(comparison(), _pThenBlock, _pEndBlock);
        _functionEmitter->SetCurrentBlock(_pThenBlock);
        return _pThenBlock;
    }

    llvm::BasicBlock* IRIfEmitter::If(LLVMValue pValue, bool testValue)
    {
        EndPrev();
        PrepareBlocks();
        _functionEmitter->SetCurrentBlock(_pConditionBlock);
        _functionEmitter->Branch(pValue, testValue, _pThenBlock, _pEndBlock);
        _functionEmitter->SetCurrentBlock(_pThenBlock);
        return _pThenBlock;
    }

    llvm::BasicBlock* IRIfEmitter::Else()
    {
        if (_pEndBlock == nullptr) // Else was called twice, OR without a preceding If
        {
            throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState, "Else called twice");
        }

        EndPrev();
        _pThenBlock = _pEndBlock;
        _pThenBlock->setName(IfElseBlockName);
        _pEndBlock = nullptr;
        _functionEmitter->SetCurrentBlock(_pThenBlock);
        return _pThenBlock;
    }

    void IRIfEmitter::End()
    {
        if (_finished)
        {
            return;
        }

        EndPrev();
        if (_pEndBlock != nullptr)
        {
            _functionEmitter->SetCurrentBlock(_pEndBlock);
            BranchToAfterBlock();
        }
        _functionEmitter->SetCurrentBlock(_pAfterBlock);
        _finished = true;
    }

    void IRIfEmitter::EndPrev()
    {
        if (_pThenBlock != nullptr && !_isContinuation)
        {
            BranchToAfterBlock();
        }
        _isContinuation = false;
    }

    void IRIfEmitter::BranchToAfterBlock()
    {
        if (!HasTerminatingBranch(_functionEmitter->GetCurrentBlock()))
        {
            _functionEmitter->Branch(_pAfterBlock);
        }
    }

    void IRIfEmitter::PrepareBlocks()
    {
        _pConditionBlock = GetParentBlock();
        _pThenBlock = _functionEmitter->BlockAfter(_pConditionBlock, IfThenBlockName);
        _pEndBlock = _functionEmitter->BlockAfter(_pThenBlock, IfEndBlockName);
    }

    llvm::BasicBlock* IRIfEmitter::GetParentBlock()
    {
        return (_pEndBlock != nullptr) ? _pEndBlock : _functionEmitter->GetCurrentBlock();
    }

    void IREmitter::BindArgumentNames(LLVMFunction pFunction, const FunctionArgumentList& arguments)
    {
        size_t i = 0;
        for (auto& argument : pFunction->args())
        {
            argument.setName(arguments[i++].GetName());
        }
    }

    void IREmitter::BindArgumentNames(LLVMFunction pFunction, const NamedVariableTypeList& arguments)
    {
        size_t i = 0;
        for (auto& argument : pFunction->args())
        {
            argument.setName(arguments[i++].first);
        }
    }

    void IREmitter::BindArgumentNames(LLVMFunction pFunction, const NamedLLVMTypeList& arguments)
    {
        size_t i = 0;
        for (auto& argument : pFunction->args())
        {
            argument.setName(arguments[i++].first);
        }
    }

} // namespace emitters
} // namespace ell
