////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IRAssemblyWriter.h (emitters)
//  Authors:  Kirk Olynyk, Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "ModuleEmitter.h"
#include "TargetDevice.h"

#include <llvm/Support/CodeGen.h> // for CodeGenOpt::Level enum
#include <llvm/Target/TargetMachine.h> // for CodeGenFileType
#include <llvm/Target/TargetOptions.h> // for FloatABI::ABIType and FPOpFusion::FpOpFusionMode

namespace ell
{
namespace emitters
{
    class IRModuleEmitter;

    /// <summary> An enum containing the optimization level {None, Less, Default, Aggressive} </summary>
    using OptimizationLevel = llvm::CodeGenOpt::Level;

    /// <summary> An enum containing the float ABI type {Default, Soft, Hard} </summary>
    using FloatABIType = llvm::FloatABI::ABIType;

    /// <summary> An enum indicating how much floating-point operations can be fused {Fast, Standard, Strict} </summary>
    using FloatFusionMode = llvm::FPOpFusion::FPOpFusionMode;

    /// <summary> An enum containing the type of output to generate {CGFT_AssemblyFile, CGFT_ObjectFile, CGFT_Null} </summary>
    using MachineCodeType = llvm::TargetMachine::CodeGenFileType;

    /// <summary> An enum containing the relocation model of the LLVM machine code output {Static, PIC_, DynamicNoPIC, ROPI, RWPI, ROPI_RWPI} </summary>
    using OutputRelocationModel = llvm::Reloc::Model;

    /// <summary> Options for LLVM machine code output (assembly or object code) </summary>
    struct MachineCodeOutputOptions
    {
        bool verboseOutput = false;
        bool verifyModule = false;

        TargetDevice targetDevice;

        OptimizationLevel optimizationLevel = OptimizationLevel::Default;
        FloatABIType floatABI = FloatABIType::Default;

        FloatFusionMode floatFusionMode = FloatFusionMode::Fast;
        bool unsafeFPMath = true;
        bool noInfsFPMath = true;
        bool noNaNsFPMath = true;
        bool noSignedZerosFPMath = true;

        OutputRelocationModel relocModel = OutputRelocationModel::Static;
    };

    /// <summary> Indicates if the requested output type is a machine code type (vs. IR) </summary>
    bool IsMachineCodeFormat(ModuleOutputFormat format);

    /// <summary> Indicates if the requested output type is binary or text </summary>
    bool IsBinaryOutputType(MachineCodeType filetype);

    /// <summary> Compile the given module to the given stream </summary>
    void GenerateMachineCode(llvm::raw_ostream& os, IRModuleEmitter& module, ModuleOutputFormat format, const MachineCodeOutputOptions& options);
} // namespace emitters
} // namespace ell
