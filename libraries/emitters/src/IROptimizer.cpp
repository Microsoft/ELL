//////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IROptimizer.cpp (emitters)
//  Authors:  Umesh Madan, Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IROptimizer.h"
#include "IRModuleEmitter.h"
#include "LLVMInclude.h"

#include <llvm/Analysis/TargetLibraryInfo.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/CodeGen/TargetPassConfig.h>
#include <llvm/IR/IRPrintingPasses.h>
#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Transforms/IPO.h>
#include <llvm/Transforms/IPO/PassManagerBuilder.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/GVN.h>

#include <memory>

namespace ell
{
namespace emitters
{
    using namespace llvm;

    IROptimizer::IROptimizer(IRModuleEmitter& module) :
        _module(module),
        _functionPasses(module.GetLLVMModule())
    {
    }

    IROptimizer::~IROptimizer()
    {
        (void)_functionPasses.doFinalization();
    }

    void IROptimizer::AddStandardPasses()
    {
        auto targetMachine = _module.GetTargetMachine();
        if (!targetMachine)
        {
            throw EmitterException(EmitterError::unexpected, "Unable to allocate target machine");
        }

        auto& llvmTargetMachine = static_cast<LLVMTargetMachine&>(*targetMachine);
        auto config = static_cast<llvm::Pass*>(llvmTargetMachine.createPassConfig(_modulePasses));
        _modulePasses.add(config);

        llvm::TargetLibraryInfoImpl targetLibraryInfo(llvm::Triple(_module.GetLLVMModule()->getTargetTriple()));
        _modulePasses.add(new llvm::TargetLibraryInfoWrapperPass(targetLibraryInfo));

        // Add internal analysis passes from the target machine.
        _modulePasses.add(llvm::createTargetTransformInfoWrapperPass(targetMachine ? targetMachine->getTargetIRAnalysis()
                                                                                   : llvm::TargetIRAnalysis()));

        _functionPasses.add(llvm::createTargetTransformInfoWrapperPass(
            targetMachine ? targetMachine->getTargetIRAnalysis() : llvm::TargetIRAnalysis()));

        _functionPasses.add(llvm::createVerifierPass());

        llvm::PassManagerBuilder builder;
        builder.OptLevel = 3;
        builder.SizeLevel = 0;
        builder.Inliner = llvm::createFunctionInliningPass(builder.OptLevel, builder.SizeLevel, false);
        builder.LoopVectorize = true;
        builder.SLPVectorize = true;
        builder.DisableUnrollLoops = false;

        if (targetMachine)
        {
            targetMachine->adjustPassManager(builder);
        }

        builder.populateFunctionPassManager(_functionPasses);
        builder.populateModulePassManager(_modulePasses);
    }

    void IROptimizer::BeginOptimizeFunctions()
    {
        (void)_functionPasses.doInitialization();
    }

    void IROptimizer::OptimizeFunction(LLVMFunction pFunction)
    {
        assert(pFunction != nullptr);
        _functionPasses.run(*pFunction);
    }

    void IROptimizer::EndOptimizeFunctions()
    {
        (void)_functionPasses.doFinalization();
    }

    void IROptimizer::OptimizeModule(llvm::Module* pModule)
    {
        _modulePasses.run(*pModule);
    }
} // namespace emitters
} // namespace ell
