////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     IREmitterTest.cpp (emitters_test)
//  Authors:  Umesh Madan, Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "IREmitterTest.h"

#include <emitters/include/EmitterException.h>
#include <emitters/include/EmitterTypes.h>
#include <emitters/include/IRBlockRegion.h>
#include <emitters/include/IREmitter.h>
#include <emitters/include/IRExecutionEngine.h>
#include <emitters/include/IRFunctionEmitter.h>
#include <emitters/include/IRHeaderWriter.h>
#include <emitters/include/IRModuleEmitter.h>

#include <testing/include/testing.h>

#include <utilities/include/StringUtil.h>
#include <utilities/include/TypeAliases.h>
#include <utilities/include/Unused.h>

#include <llvm/Transforms/Utils/Cloning.h>

#include <functional>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ell;
using namespace emitters;
using namespace utilities;

// Utility code
std::string g_outputBasePath = "";
void SetOutputPathBase(std::string path)
{
    g_outputBasePath = std::move(path);
}

std::string OutputPath(const char* pRelPath)
{
    return g_outputBasePath + pRelPath;
}

std::vector<llvm::Instruction*> RemoveTerminators(LLVMFunction pfn)
{
    std::vector<llvm::Instruction*> terms;
    auto& blocks = pfn->getBasicBlockList();
    for (auto pBlock = blocks.begin(); pBlock != blocks.end(); ++pBlock)
    {
        auto& instructions = pBlock->getInstList();
        for (auto pInst = instructions.begin(); pInst != instructions.end(); ++pInst)
        {
            if (pInst->isTerminator())
            {
                terms.push_back(&(*pInst));
            }
        }
    }
    for (auto pInstr : terms)
    {
        pInstr->removeFromParent();
    }
    return terms;
}

void InsertTerminators(LLVMFunction pfn, std::vector<llvm::Instruction*>& terms)
{
    size_t i = 0;
    auto& blocks = pfn->getBasicBlockList();
    for (auto pBlock = blocks.begin(); pBlock != blocks.end(); ++pBlock)
    {
        auto& instructions = pBlock->getInstList();
        instructions.push_back(terms[i++]);
    }
}

static std::stringstream g_debugOutput;

extern "C" {
void DebugPrint(char* message)
{
    g_debugOutput << message;
}
}

LLVMFunction DeclareDebugPrint(IRModuleEmitter& module)
{
    auto& context = module.GetIREmitter().GetContext();
    auto type = llvm::FunctionType::get(
        llvm::Type::getInt32Ty(context),
        { llvm::Type::getInt8PtrTy(context) },
        false);
    return module.DeclareFunction("DebugPrint", type);
}

void CallDebugPrint(IRFunctionEmitter& function, LLVMFunction printFunction, std::string message)
{
    function.Call(printFunction, { function.Literal(message.c_str()) });
}

void DefineDebugPrint(LLVMFunction debugPrintFunction, IRExecutionEngine& jitter)
{
    g_debugOutput.clear();
    jitter.DefineFunction(debugPrintFunction, reinterpret_cast<UIntPtrT>(&DebugPrint));
}

std::string GetCapturedDebugOutput()
{
    return g_debugOutput.str();
}

std::string EmitStruct(const char* moduleName)
{
    auto module = MakeHostModuleEmitter(moduleName);
    const char* TensorShapeName = "TensorShape";
    auto int32Type = VariableType::Int32;
    emitters::NamedVariableTypeList namedFields = { { "rows", int32Type }, { "columns", int32Type }, { "channels", int32Type } };
    auto shapeType = module.GetOrCreateStruct(TensorShapeName, namedFields);
    module.IncludeTypeInHeader(shapeType->getName());

    const emitters::NamedVariableTypeList parameters = { { "index", emitters::GetVariableType<int>() } };
    auto function = module.BeginFunction("Dummy", shapeType, parameters);
    auto shapeVariable = function.Variable(shapeType, 1);
    auto shape = function.ValueAt(shapeVariable, 0);
    function.Return(shape);
    function.IncludeInHeader();
    module.EndFunction();

    std::ostringstream out;
    emitters::WriteModuleHeader(out, module);
    return out.str();
}

// Tests

// Just test that IREmitter doesn't crash
void TestIREmitter()
{
    {
        IRModuleEmitter module("Module1", CompilerOptions{});
        auto& emitter = module.GetIREmitter();
        emitter.DeclareFunction(module.GetLLVMModule(), "foobar");
    }

    {
        IRModuleEmitter module("Module2", CompilerOptions{});
        auto& emitter = module.GetIREmitter();
        emitter.DeclareFunction(module.GetLLVMModule(), "foobar");
    }
}

// Just another test that IREmitter doesn't crash
void TestEmitLLVM()
{
    auto module = MakeHostModuleEmitter("Looper");

    llvm::StructType* structType = module.GetOrCreateStruct("ShiftRegister", { { "size", VariableType::Int32 }, { "value", VariableType::Double } });

    std::vector<double> data({ 3.3, 4.4, 5.5, 6.6, 7.7 });
    llvm::GlobalVariable* pData = module.ConstantArray("g_weights", data);
    llvm::GlobalVariable* pOutput = module.GlobalArray(VariableType::Double, "g_output", data.size());
    llvm::GlobalVariable* pTotal = module.Global(VariableType::Double, "g_total");
    llvm::GlobalVariable* pRegisters = module.GlobalArray("g_registers", structType, data.size());

    auto fnMain = module.BeginMainFunction();

    IRForLoopEmitter testLoop(fnMain);
    testLoop.Begin(data.size());
    testLoop.End();

    fnMain.For(data.size(), [pData, pOutput, pTotal, pRegisters](IRFunctionEmitter& fnMain, auto i) {
        auto item = fnMain.ValueAt(pData, i);
        auto sum = fnMain.Operator(TypedOperator::addFloat, fnMain.Literal(0.3), item);
        fnMain.SetValueAt(pOutput, i, sum);
        fnMain.OperationAndUpdate(pTotal, TypedOperator::addFloat, sum);

        LLVMValue pRegisterSum = fnMain.PointerOffset(pRegisters, i, fnMain.Literal(1));
        fnMain.Store(pRegisterSum, sum);

        fnMain.If(TypedComparison::lessThanFloat, item, fnMain.Literal(5.7), [](IRFunctionEmitter& fnMain) {
            fnMain.Print("First IF!\n");
        });

        fnMain.If(TypedComparison::equalsFloat, item, fnMain.Literal(6.6), [](IRFunctionEmitter& fnMain) {
                  fnMain.Print("Second If!\n");
              })
            .Else([](IRFunctionEmitter& fnMain) {
                fnMain.Print("Else\n");
            });

        fnMain.Printf({ fnMain.Literal("%d, %f\n"), i, item });
    });

    fnMain.SetValueAt(pOutput, fnMain.Literal(3), fnMain.Literal(10.0));
    fnMain.SetValueAt(pOutput, fnMain.Literal(4), fnMain.Literal(20.0));

    auto pOtherTotal = module.Global(VariableType::Double, "g_total");
    IRForLoopEmitter forLoop2(fnMain);
    forLoop2.Begin(data.size());
    {
        auto ival = forLoop2.LoadIterationVariable();
        auto v = fnMain.ValueAt(pOutput, ival);

        LLVMValue pRegisterSum = fnMain.Load(fnMain.PointerOffset(pRegisters, ival, fnMain.Literal(1)));

        fnMain.OperationAndUpdate(pOtherTotal, TypedOperator::addFloat, v);
        fnMain.Printf({ fnMain.Literal("%f, %f\n"), v, pRegisterSum });
    }
    forLoop2.End();
    fnMain.Printf({ fnMain.Literal("Total = %f, OtherTotal= %f\n"), fnMain.Load(pTotal), fnMain.Load(pOtherTotal) });

    fnMain.Return();
    module.EndFunction();
}

void TestLLVMShiftRegister()
{
    auto module = MakeHostModuleEmitter("ShiftRegister");

    std::vector<double> data({ 1.1, 2.1, 3.1, 4.1, 5.1 });
    std::vector<double> newData1({ 1.2, 2.2 });
    std::vector<double> newData2({ 3.3, 4.3 });

    auto fn = module.BeginFunction("ShiftRegisterTest", VariableType::Void, NamedVariableTypeList{ { "values", VariableType::DoublePointer } });
    {
        auto arguments = fn.Arguments().begin();
        auto output = &(*arguments++);

        llvm::GlobalVariable* pRegister = module.GlobalArray("g_shiftRegister", data);
        LLVMValue c1 = module.ConstantArray("c_1", newData1);
        LLVMValue c2 = module.ConstantArray("c_2", newData2);

        fn.ShiftAndUpdate<double>(pRegister, data.size(), newData1.size(), c1);
        fn.ShiftAndUpdate<double>(pRegister, data.size(), newData2.size(), c2);

        fn.MemoryCopy<double>(pRegister, output, data.size());
        fn.Return();
    }
    module.EndFunction();
    fn.Verify();

    IRExecutionEngine jit(std::move(module));
    auto testFn = jit.GetFunction<void(double*)>("ShiftRegisterTest");
    const double sentinel = -17.0;
    std::vector<double> result(6, sentinel);
    std::vector<double> expected = { 5.1, 1.2, 2.2, 3.3, 4.3, sentinel };
    testFn(result.data());
    testing::ProcessTest("Testing shift register", testing::IsEqual(result, expected));
}

void TestHighLevelNestedIf()
{
    auto module = MakeHostModuleEmitter("HighLevelNestedIf");
    NamedVariableTypeList argTypes = { { "x", VariableType::Double },
                                       { "y", VariableType::Double },
                                       { "z", VariableType::Double } };
    auto fn = module.BeginFunction("HighLevelNestedIfTest", VariableType::Int32, argTypes);
    {
        auto arguments = fn.Arguments().begin();
        auto x = &(*arguments++);
        auto y = &(*arguments++);
        auto z = &(*arguments++);
        auto result = fn.Variable(VariableType::Int32);
        {
            fn.If(fn.Comparison(emitters::TypedComparison::greaterThanFloat, x, y), [&result, x, z](IRFunctionEmitter& fn) {
                  fn.If(fn.Comparison(emitters::TypedComparison::lessThanFloat, x, z), [&result](IRFunctionEmitter& fn) {
                      fn.Store(result, fn.Literal<int>(1));
                  });
              }).ElseIf(fn.Comparison(emitters::TypedComparison::lessThanFloat, x, z), [&result, x, y](IRFunctionEmitter& fn) {
                    fn.If(fn.Comparison(emitters::TypedComparison::lessThanFloat, x, y), [&result](IRFunctionEmitter& fn) {
                        fn.Store(result, fn.Literal<int>(2));
                    });
                })
                .Else([&result, y, z](IRFunctionEmitter& fn) {
                    fn.If(fn.Comparison(emitters::TypedComparison::greaterThanFloat, y, z), [&result](IRFunctionEmitter& fn) {
                        fn.Store(result, fn.Literal<int>(3));
                    });
                });
        }
        fn.Return(fn.Load(result));
    }
    module.EndFunction();

    fn.Verify();

    IRExecutionEngine jit(std::move(module));
    auto testFn = jit.GetFunction<int32_t(double, double, double)>("HighLevelNestedIfTest");

    // test all 3 branches of the if-then-else block.
    int result;
    result = testFn(10.0, 5.0, 20.0);
    testing::ProcessTest("Testing nested if/else", result == 1);
    result = testFn(10.0, 15.0, 20.0);
    testing::ProcessTest("Testing nested if/else", result == 2);
    result = testFn(10.0, 15.0, 5.0);
    testing::ProcessTest("Testing nested if/else", result == 3);
}

void TestMixedLevelNestedIf()
{
    auto module = MakeHostModuleEmitter("MixedLevelNestedIf");
    NamedVariableTypeList argTypes = { { "x", VariableType::Double },
                                       { "y", VariableType::Double },
                                       { "z", VariableType::Double } };
    auto fn = module.BeginFunction("MixedLevelNestedIfTest", VariableType::Int32, argTypes);
    {
        auto arguments = fn.Arguments().begin();
        auto x = &(*arguments++);
        auto y = &(*arguments++);
        auto z = &(*arguments++);
        auto result = fn.Variable(VariableType::Int32);
        {
            IRIfEmitter ifEmitter = fn.If(fn.Comparison(emitters::TypedComparison::greaterThanFloat, x, y), [&result, x, z](IRFunctionEmitter& fn) {
                fn.If(fn.Comparison(emitters::TypedComparison::lessThanFloat, x, z), [&result](IRFunctionEmitter& fn) {
                    fn.Store(result, fn.Literal<int>(1));
                });
            });
            ifEmitter.ElseIf(fn.Comparison(emitters::TypedComparison::lessThanFloat, x, z), [&result, x, y](IRFunctionEmitter& fn) {
                fn.If(fn.Comparison(emitters::TypedComparison::lessThanFloat, x, y), [&result](IRFunctionEmitter& fn) {
                    fn.Store(result, fn.Literal<int>(2));
                });
            });
            ifEmitter.Else([&result, y, z](IRFunctionEmitter& fn) {
                fn.If(fn.Comparison(emitters::TypedComparison::greaterThanFloat, y, z), [&result](IRFunctionEmitter& fn) {
                    fn.Store(result, fn.Literal<int>(3));
                });
            });
            ifEmitter.End();
        }
        fn.Return(fn.Load(result));
    }
    module.EndFunction();

    fn.Verify();

    IRExecutionEngine jit(std::move(module));
    auto testFn = jit.GetFunction<int32_t(double, double, double)>("MixedLevelNestedIfTest");

    // test all 3 branches of the if-then-else block.
    int result;
    result = testFn(10.0, 5.0, 20.0);
    testing::ProcessTest("Testing nested if/else", result == 1);
    result = testFn(10.0, 15.0, 20.0);
    testing::ProcessTest("Testing nested if/else", result == 2);
    result = testFn(10.0, 15.0, 5.0);
    testing::ProcessTest("Testing nested if/else", result == 3);
}

void TestLogicalAnd()
{
    auto module = MakeHostModuleEmitter("LogicalAnd");

    auto fn = module.BeginFunction("LogicalAndTest", VariableType::Int32, NamedVariableTypeList{ { "x1", VariableType::Int32 }, { "x2", VariableType::Int32 }, { "x3", VariableType::Int32 } });
    {
        auto args = fn.Arguments().begin();
        llvm::Argument& val1 = *args++;
        llvm::Argument& val2 = *args++;
        llvm::Argument& val3 = *args++;

        auto result = fn.LogicalAnd(fn.Comparison(TypedComparison::equals, &val1, &val2), fn.Comparison(TypedComparison::equals, &val2, &val3));
        fn.Return(fn.CastValue<int32_t>(result));
    }
    module.EndFunction();

    fn.Verify();
    IRExecutionEngine jit(std::move(module));
    auto testFn = jit.GetFunction<int32_t(int32_t, int32_t, int32_t)>("LogicalAndTest");
    auto referenceFn = [](int x1, int x2, int x3) {
        return (x1 == x2) && (x2 == x3);
    };

    bool success = true;
    for (auto args : std::vector<std::vector<int32_t>>{ { 1, 1, 1 }, { 1, 2, 2 }, { 1, 2, 3 }, { 1, 1, 2 } })
    {
        auto result = testFn(args[0], args[1], args[2]);
        auto expected = referenceFn(args[0], args[1], args[2]);
        success = success && (result == expected);
    }
    testing::ProcessTest("Testing logical AND", success);
}

void TestLogicalOr()
{
    auto module = MakeHostModuleEmitter("LogicalOr");

    auto fn = module.BeginFunction("LogicalOrTest", VariableType::Int32, NamedVariableTypeList{ { "x1", VariableType::Int32 }, { "x2", VariableType::Int32 }, { "x3", VariableType::Int32 } });
    {
        auto args = fn.Arguments().begin();
        llvm::Argument& val1 = *args++;
        llvm::Argument& val2 = *args++;
        llvm::Argument& val3 = *args++;

        auto result = fn.LogicalOr(fn.Comparison(TypedComparison::equals, &val1, &val2), fn.Comparison(TypedComparison::equals, &val2, &val3));
        fn.Return(fn.CastValue<int32_t>(result));
    }
    module.EndFunction();

    fn.Verify();
    IRExecutionEngine jit(std::move(module));
    auto testFn = jit.GetFunction<int32_t(int32_t, int32_t, int32_t)>("LogicalOrTest");
    auto referenceFn = [](int x1, int x2, int x3) {
        return (x1 == x2) || (x2 == x3);
    };

    bool success = true;
    for (auto args : std::vector<std::vector<int32_t>>{ { 1, 1, 1 }, { 1, 2, 2 }, { 1, 2, 3 }, { 1, 1, 2 } })
    {
        auto result = testFn(args[0], args[1], args[2]);
        auto expected = referenceFn(args[0], args[1], args[2]);
        success = success && (result == expected);
    }
    testing::ProcessTest("Testing logical OR", success);
}

void TestLogicalNot()
{
    auto module = MakeHostModuleEmitter("LogicalNot");

    auto fn = module.BeginFunction("LogicalNotTest", VariableType::Int32, NamedVariableTypeList{ { "x1", VariableType::Int32 }, { "x2", VariableType::Int32 } });
    {
        auto args = fn.Arguments().begin();
        llvm::Argument& val1 = *args++;
        llvm::Argument& val2 = *args++;

        auto result = fn.LogicalNot(fn.Comparison(TypedComparison::equals, &val1, &val2));
        fn.Return(fn.CastValue<int32_t>(result));
    }
    module.EndFunction();

    fn.Verify();
    IRExecutionEngine jit(std::move(module));
    auto testFn = jit.GetFunction<int32_t(int32_t, int32_t)>("LogicalNotTest");
    auto referenceFn = [](int x1, int x2) {
        return !(x1 == x2);
    };

    bool success = true;
    for (auto args : std::vector<std::vector<int32_t>>{ { 0, 1 }, { 0, 0 }, { 1, 1 }, { -1, 1 } })
    {
        auto result = testFn(args[0], args[1]);
        auto expected = referenceFn(args[0], args[1]);
        success = success && (result == expected);
    }
    testing::ProcessTest("Testing logical NOT", success);
}

template <typename ValueType>
void TestBinaryNumericOperations()
{
    std::string functionName = utilities::FormatString("BinaryOperations_%s", typeid(ValueType).name());
    // (((1 + 5) - 3) * 2) / 3 = 2
    ValueType input = 1;
    ValueType expectedResult = 2;
    ValueType inputs[] = { 5, 3, 2, 3 };
    BinaryOperatorType ops[] = { BinaryOperatorType::add, BinaryOperatorType::subtract, BinaryOperatorType::multiply, BinaryOperatorType::divide };

    auto module = MakeHostModuleEmitter("module_" + functionName);
    {
        auto varType = emitters::GetVariableType<ValueType>();
        auto fn = module.BeginFunction(functionName, varType, NamedVariableTypeList{ { "input", emitters::GetVariableType<ValueType>() } });

        auto arguments = fn.Arguments().begin();
        auto x = &(*arguments++);
        auto result = fn.Variable(varType);
        fn.Store(result, x);

        for (int i = 0, n = sizeof(inputs) / sizeof(ValueType); i < n; i++)
        {
            auto op = GetOperator<ValueType>(ops[i]);
            fn.Store(result, fn.Operator(op, fn.Load(result), fn.template Literal<ValueType>(inputs[i])));
        }

        fn.Return(fn.Load(result));
        module.EndFunction();
        fn.Verify();
    }

    IRExecutionEngine jit(std::move(module));
    auto jittedFunction = jit.GetFunction<ValueType(ValueType)>(functionName);
    ValueType result = jittedFunction(input);
    auto prompt = "Testing " + functionName;
    testing::ProcessTest(prompt, result == expectedResult);
}

void TestBinaryOperations()
{
    TestBinaryNumericOperations<int>();
    TestBinaryNumericOperations<float>();
    TestBinaryNumericOperations<double>();
}

void AddLogicalOperator(emitters::IRModuleEmitter& module, std::string name, emitters::BinaryOperatorType op)
{
    auto varType = emitters::GetVariableType<bool>();
    auto fn = module.BeginFunction(name, varType, NamedVariableTypeList{ { "a", emitters::GetVariableType<bool>() }, { "b", emitters::GetVariableType<bool>() } });
    auto arguments = fn.Arguments().begin();
    auto a = &(*arguments++);
    auto b = &(*arguments++);
    auto result = fn.Variable(varType);
    fn.Store(result, fn.Operator(GetOperator<bool>(op), a, b));
    fn.Return(fn.Load(result));
    module.EndFunction();
    fn.Verify();
}

void TestBinaryLogicalOperations()
{
    std::string functionName = "BinaryLogicalOperations";

    bool input1[] = { false, false, true, true };
    bool input2[] = { false, true, false, true };

    bool andTable[] = { false, false, false, true };
    bool orTable[] = { false, true, true, true };
    bool xorTable[] = { false, true, true, false };

    auto module = MakeHostModuleEmitter("module_" + functionName);
    AddLogicalOperator(module, "LogicalOr", emitters::BinaryOperatorType::logicalOr);
    AddLogicalOperator(module, "LogicalAnd", emitters::BinaryOperatorType::logicalAnd);
    AddLogicalOperator(module, "LogicalXor", emitters::BinaryOperatorType::logicalXor);

    IRExecutionEngine jit(std::move(module));
    auto jittedOr = jit.GetFunction<bool(bool, bool)>("LogicalOr");
    auto jittedAnd = jit.GetFunction<bool(bool, bool)>("LogicalAnd");
    auto jittedXor = jit.GetFunction<bool(bool, bool)>("LogicalXor");

    int errors = 0;
    for (int i = 0; i < 4; i++)
    {
        bool a = input1[i];
        bool b = input2[i];
        bool r = jittedOr(a, b);
        bool e = orTable[i];
        if (r != e)
        {
            std::cout << "Logical Or failed with inputs " << a << "," << b << ", we got " << r << " but expected " << e << std::endl;
            errors++;
        }

        r = jittedAnd(a, b);
        e = andTable[i];
        if (r != e)
        {
            std::cout << "Logical And failed with inputs " << a << "," << b << ", we got " << r << " but expected " << e << std::endl;
            errors++;
        }

        r = jittedXor(a, b);
        e = xorTable[i];
        if (r != e)
        {
            std::cout << "Logical Xor failed with inputs " << a << "," << b << ", we got " << r << " but expected " << e << std::endl;
            errors++;
        }
    }

    testing::ProcessTest("Testing Logical Operations", errors == 0);
}

void TestForLoop()
{
    auto module = MakeHostModuleEmitter("ForLoop");
    const int numIter = 10;

    auto add = GetOperator<int32_t>(BinaryOperatorType::add);
    auto varType = VariableType::Int32;

    auto fn = module.BeginFunction("TestForLoop", varType, NamedVariableTypeList{});
    auto sum = fn.Variable(varType);
    fn.Store(sum, fn.Literal<int32_t>(0));
    fn.For(numIter, [sum, add](IRFunctionEmitter& fn, LLVMValue i) {
        fn.Store(sum, fn.Operator(add, fn.Load(sum), i));
    });

    fn.Return(fn.Load(sum));
    module.EndFunction();

    IRExecutionEngine jit(std::move(module));
    auto jittedFunction = jit.GetFunction<int32_t()>("TestForLoop");
    auto result = jittedFunction();
    int32_t expectedResult = 0;
    for (int i = 0; i < numIter; ++i)
    {
        expectedResult += i;
    }
    testing::ProcessTest("Testing for loop", result == expectedResult);
}

void TestWhileLoopWithVariableCondition()
{
    auto module = MakeHostModuleEmitter("WhileLoop");
    auto int8Type = GetVariableType<char>();
    auto int32Type = GetVariableType<int32_t>();
    auto fn = module.BeginFunction("TestWhileLoop", int32Type);
    {
        auto conditionVar = fn.Variable(int8Type, "cond");
        auto i = fn.Variable(int32Type);
        fn.Store(i, fn.Literal<int>(5));
        fn.Store(conditionVar, fn.CastBoolToByte(fn.TrueBit()));
        fn.While(conditionVar, [conditionVar, i](IRFunctionEmitter& fn) {
            // i++
            fn.OperationAndUpdate(i, TypedOperator::add, fn.Literal<int>(1)); // i++

            // update conditionVar (i != 10)
            fn.Store(conditionVar, fn.CastBoolToByte(fn.Comparison(TypedComparison::notEquals, fn.Load(i), fn.Literal<int>(10))));
        });

        fn.Return(fn.Load(i));
    }
    module.EndFunction();

    IRExecutionEngine jit(std::move(module));
    auto jittedFunction = jit.GetFunction<int32_t()>("TestWhileLoop");
    auto result = jittedFunction();
    const int expectedResult = 10;
    testing::ProcessTest("Testing while loop with stored variable exit condition", result == expectedResult);
}

void TestWhileLoopWithFunctionCondition()
{
    auto module = MakeHostModuleEmitter("WhileLoop");
    auto int32Type = GetVariableType<int32_t>();
    auto fn = module.BeginFunction("TestWhileLoop", int32Type);
    {
        auto i = fn.Variable(int32Type);
        fn.Store(i, fn.Literal<int>(5));
        auto condition = [i](IRFunctionEmitter& fn) {
            return fn.LocalScalar(fn.Load(i)) != 10;
        };

        fn.While(condition, [i](IRFunctionEmitter& fn) {
            fn.OperationAndUpdate(i, TypedOperator::add, fn.Literal<int>(1)); // i++
        });

        fn.Return(fn.Load(i));
    }
    module.EndFunction();

    IRExecutionEngine jit(std::move(module));
    auto jittedFunction = jit.GetFunction<int32_t()>("TestWhileLoop");
    auto result = jittedFunction();
    const int expectedResult = 10;
    testing::ProcessTest("Testing while loop with function exit condition", result == expectedResult);
}

void TestWhileLoopWithInt32Condition()
{
    auto module = MakeHostModuleEmitter("WhileLoop");
    auto int32Type = VariableType::Int32;
    auto fn = module.BeginFunction("TestWhileLoop", int32Type);
    {
        auto i = fn.Variable(int32Type);
        fn.Store(i, fn.Literal<int>(5));
        fn.While(i, [i](IRFunctionEmitter& fn) {
            fn.OperationAndUpdate(i, TypedOperator::subtract, fn.Literal<int>(1)); // --i
        });

        fn.Return(fn.Load(i));
    }
    module.EndFunction();

    IRExecutionEngine jit(std::move(module));
    auto jittedFunction = jit.GetFunction<int32_t()>("TestWhileLoop");
    auto result = jittedFunction();
    const int expectedResult = 0;
    testing::ProcessTest("Testing while loop with int32 condition", result == expectedResult);
}

void TestMetadata()
{
    auto module = MakeHostModuleEmitter("Metadata");

    // Function-level metadata
    auto fn = module.BeginFunction("TestMetadata", VariableType::Void);
    fn.InsertMetadata("hello.fn");
    fn.InsertMetadata("hello.fn.content", "test content");
    fn.Return();
    module.EndFunction();
    fn.Verify();

    // Module-level metadata
    module.InsertMetadata("hello.world");
    module.InsertMetadata("hello.world.content", { "12345" });
    module.InsertMetadata("hello.world.content", { "67890" });

    auto fnMain = module.BeginMainFunction();
    fnMain.Call("TestMetadata");
    fnMain.Return();

    // Missing metadata
    testing::ProcessTest("Testing missing module metadata check", testing::IsEqual(module.HasMetadata("does.not.exist"), false));
    testing::ProcessTest("Testing missing function metadata check", testing::IsEqual(module.HasFunctionMetadata("TestMetadata", "fn.does.not.exist"), false));

    // Empty metadata
    auto actualModuleMetadata = module.GetMetadata("hello.world");
    std::vector<std::string> flattenedModuleMetadata;
    for (auto m : actualModuleMetadata)
    {
        flattenedModuleMetadata.push_back(m[0]);
    }
    std::vector<std::string> expected{ "" };
    testing::ProcessTest("Testing empty module metadata check", testing::IsEqual(module.HasMetadata("hello.world"), true));
    testing::ProcessTest("Testing empty module metadata get", testing::IsEqual(flattenedModuleMetadata, expected));
    std::vector<std::string> actualFunctionMetadata = module.GetFunctionMetadata("TestMetadata", "hello.fn");
    testing::ProcessTest("Testing empty function metadata check", testing::IsEqual(module.HasFunctionMetadata("TestMetadata", "hello.fn"), true));
    testing::ProcessTest("Testing empty function metadata get", testing::IsEqual(actualFunctionMetadata, expected));

    // Non-empty metadata
    actualModuleMetadata = module.GetMetadata("hello.world.content");
    flattenedModuleMetadata.clear();
    for (auto m : actualModuleMetadata)
    {
        flattenedModuleMetadata.push_back(m[0]);
    }
    expected.clear();
    expected.push_back("12345");
    expected.push_back("67890");
    testing::ProcessTest("Testing non-empty module metadata check", testing::IsEqual(module.HasMetadata("hello.world.content"), true));
    testing::ProcessTest("Testing non-empty module metadata get", testing::IsEqual(flattenedModuleMetadata, expected));
    actualFunctionMetadata = module.GetFunctionMetadata("TestMetadata", "hello.fn.content");
    expected.clear();
    expected.push_back("test content");
    testing::ProcessTest("Testing non-empty function metadata check", testing::IsEqual(module.HasFunctionMetadata("TestMetadata", "hello.fn.content"), true));
    testing::ProcessTest("Testing non-empty function metadata get", testing::IsEqual(actualFunctionMetadata, expected));

    // Just for fun - metadata should have no effect
    IRExecutionEngine jit(std::move(module));
    jit.RunMain();
}

void TestHeaderStruct()
{
    auto module = MakeHostModuleEmitter("Predictor");

    auto int32Type = VariableType::Int32;
    emitters::NamedVariableTypeList namedFields = { { "rows", int32Type }, { "columns", int32Type }, { "channels", int32Type } };
    auto shapeType = module.GetOrCreateStruct("Shape", namedFields);
    // test that this casues the type to show up in the module header.
    module.IncludeTypeInHeader(shapeType->getName());
    std::string functionName = "Test_GetInputShape";
    const emitters::NamedVariableTypeList parameters = { { "index", emitters::GetVariableType<int>() } };
    auto function = module.BeginFunction(functionName, shapeType, parameters);
    // test that this causes the function to show up in the module header
    function.IncludeInHeader();
    auto& emitter = module.GetIREmitter();
    auto& irBuilder = emitter.GetIRBuilder();
    llvm::AllocaInst* shapeVar = function.Variable(shapeType, "shape");
    auto rowsPtr = irBuilder.CreateInBoundsGEP(shapeType, shapeVar, { function.Literal(0), function.Literal(0) });
    auto columnsPtr = irBuilder.CreateInBoundsGEP(shapeType, shapeVar, { function.Literal(0), function.Literal(1) });
    auto channelsPtr = irBuilder.CreateInBoundsGEP(shapeType, shapeVar, { function.Literal(0), function.Literal(2) });
    function.Store(rowsPtr, function.Literal(224));
    function.Store(columnsPtr, function.Literal(224));
    function.Store(channelsPtr, function.Literal(3));
    function.Return(function.ValueAt(shapeVar));
    module.EndFunction();

    // add some function comments
    module.GetFunctionDeclaration(functionName).GetComments().push_back("This is a really fun function");

    std::ostringstream out;
    emitters::WriteModuleHeader(out, module);

    std::string result = out.str();
    auto structPos = result.find("typedef struct Shape");
    auto funcPos = result.find("Shape Test_GetInputShape(int");
    auto commentPos = result.find("This is a really fun function");
    testing::ProcessTest("Testing header generation with structs", structPos != std::string::npos && funcPos != std::string::npos && commentPos != std::string::npos);
}

void TestHeaderVoidChar()
{
    auto module = MakeHostModuleEmitter("Predictor");

    emitters::NamedVariableTypeList parameters = { { "context", VariableType::VoidPointer }, { "name", VariableType::Char8Pointer } };

    auto function = module.BeginFunction("Test_GetMetadata", VariableType::Char8Pointer, parameters);

    // test that this causes the function to show up in the module header with correct types for void and char*.
    function.IncludeInHeader();
    function.Return(function.Literal("test"));
    module.EndFunction();

    std::ostringstream out;
    emitters::WriteModuleHeader(out, module);

    std::string result = out.str();
    auto funcPos = result.find("char* Test_GetMetadata(void* context, char* name");
    testing::ProcessTest("Testing special header for void and char", funcPos != std::string::npos);
}

void TestHeader()
{
    TestHeaderStruct();
    TestHeaderVoidChar();
}

void TestTwoEmitsInOneSession()
{
    auto emit1 = EmitStruct("Mod1");
    auto emit2 = EmitStruct("Mod2");
    auto badpos1 = emit1.find("TensorShape.");
    auto badpos2 = emit2.find("TensorShape.");
    testing::ProcessTest("Testing two uses of module emitter",
                         badpos1 == std::string::npos && badpos2 == std::string::npos);
}

void TestStruct()
{
    auto module = MakeHostModuleEmitter("StructTest");
    auto& context = module.GetLLVMContext();
    auto int32Type = llvm::Type::getInt32Ty(context);
    auto int8PtrType = llvm::Type::getInt8PtrTy(context);
    auto doubleType = llvm::Type::getDoubleTy(context);

    llvm::StructType* structType = module.GetOrCreateStruct("MytStruct", { { "intField", int32Type }, { "ptrField", int8PtrType }, { "doubleField", doubleType } });

    auto function = module.BeginMainFunction();
    {
        auto structVar = function.Variable(structType, "s");
        function.Store(function.GetStructFieldPointer(structVar, 0), function.Literal<int>(1));
        function.Store(function.GetStructFieldPointer(structVar, 1), function.Literal("Hello"));
        function.Store(function.GetStructFieldPointer(structVar, 2), function.Literal<double>(3.14));
        function.Return();
    }
    module.EndFunction();

    module.WriteToFile("testStruct.ll");
    module.WriteToFile("testStruct.h");
}

void TestDuplicateStructs()
{
    auto module = MakeHostModuleEmitter("DuplicateStructTest");
    auto& context = module.GetLLVMContext();
    auto int32Type = llvm::Type::getInt32Ty(context);
    auto int8PtrType = llvm::Type::getInt8PtrTy(context);
    auto doubleType = llvm::Type::getDoubleTy(context);

    // These should be fine --- the second GetOrCreateStruct call should return the existing type
    llvm::StructType* struct1TypeA = module.GetOrCreateStruct("MyStruct1", { { "intField", int32Type }, { "ptrField", int8PtrType }, { "doubleField", doubleType } });
    llvm::StructType* struct1TypeB = module.GetOrCreateStruct("MyStruct1", { { "intField", int32Type }, { "ptrField", int8PtrType }, { "doubleField", doubleType } });
    testing::ProcessTest("Testing double-declaration of equivalent structs", struct1TypeA == struct1TypeB);

    bool gotException = false;
    try
    {
        llvm::StructType* struct2TypeA = module.GetOrCreateStruct("MyStruct2", { { "intField", int32Type }, { "ptrField", int8PtrType }, { "doubleField", doubleType } });
        llvm::StructType* struct2TypeB = module.GetOrCreateStruct("MyStruct2", { { "intField", int8PtrType }, { "ptrField", int8PtrType }, { "doubleField", doubleType } });
        UNUSED(struct2TypeA);
        UNUSED(struct2TypeB);
    }
    catch (EmitterException&)
    {
        gotException = true;
    }

    testing::ProcessTest("Testing double-declaration of non-equivalent structs", gotException);
}

void TestScopedIf()
{
    auto module = MakeHostModuleEmitter("If");

    // returns `1` if arg is < 10.0, otherwise returns `2`
    auto fn = module.BeginFunction("ScopedIfTest", VariableType::Int32, NamedVariableTypeList{ { "x", VariableType::Double } });
    {
        auto arguments = fn.Arguments().begin();
        auto x = &(*arguments++);
        auto result = fn.Variable(VariableType::Int32);
        fn.Store(result, fn.Literal<int>(1));
        auto cmp = fn.Comparison(TypedComparison::lessThanFloat, x, fn.Literal(10.0));
        fn.If(cmp, [&result](IRFunctionEmitter& fn) {
            fn.Store(result, fn.Literal<int>(2));
        });
        fn.Return(fn.Load(result));
    }
    module.EndFunction();

    IRExecutionEngine jit(std::move(module));
    auto testFn = jit.GetFunction<int32_t(double)>("ScopedIfTest");
    auto referenceFn = [](double x) {
        int result = 1;
        if (x < 10.0)
        {
            result = 2;
        }
        return result;
    };

    bool success = true;
    for (int i = 0; i < 20; ++i)
    {
        auto result = testFn(static_cast<double>(i));
        auto expected = referenceFn(static_cast<double>(i));
        success = success && (result == expected);
    }
    testing::ProcessTest("Testing scoped If", success);
}

void TestScopedIfElse()
{
    auto module = MakeHostModuleEmitter("IfElse");

    auto fn = module.BeginFunction("ScopedIfElseTest", VariableType::Int32, NamedVariableTypeList{ { "x", VariableType::Int32 } });
    {
        auto arguments = fn.Arguments().begin();
        auto x = &(*arguments++);
        auto result = fn.Variable(VariableType::Int32);
        fn.Store(result, fn.Literal<int>(0));

        auto cmp = fn.Comparison(TypedComparison::lessThan, x, fn.Literal<int>(10));

        fn.If(cmp, [&result](IRFunctionEmitter& fn) {
              fn.Store(result, fn.Literal<int>(1));
          }).Else([&result](IRFunctionEmitter& fn) {
            fn.Store(result, fn.Literal<int>(2));
        });
        fn.Return(fn.Load(result));
    }
    module.EndFunction();

    IRExecutionEngine jit(std::move(module));
    auto testFn = jit.GetFunction<int32_t(int32_t)>("ScopedIfElseTest");
    auto referenceFn = [](int x) {
        int result = 0;
        if (x < 10)
        {
            result = 1;
        }
        else
        {
            result = 2;
        }
        return result;
    };

    bool success = true;
    for (int32_t i = 0; i < 20; ++i)
    {
        auto result = testFn(i);
        auto expected = referenceFn(i);
        success = success && (result == expected);
    }
    testing::ProcessTest("Testing scoped IfElse", success);
}

void TestScopedIfElse2()
{
    auto module = MakeHostModuleEmitter("IfElse2");

    auto fn = module.BeginFunction("ScopedIfElse2Test", VariableType::Int32, NamedVariableTypeList{ { "x", VariableType::Int32 } });
    {
        auto arguments = fn.Arguments().begin();
        auto x = &(*arguments++);
        auto result = fn.Variable(VariableType::Int32);
        fn.Store(result, fn.Literal<int>(0));

        auto cmp1 = fn.Comparison(TypedComparison::lessThan, x, fn.Literal<int>(3));
        auto cmp2 = fn.Comparison(TypedComparison::greaterThan, x, fn.Literal<int>(6));
        fn.If(cmp1, [&result](IRFunctionEmitter& fn) {
              fn.Store(result, fn.Literal<int>(1));
          }).ElseIf(cmp2, [&result](IRFunctionEmitter& fn) {
                fn.Store(result, fn.Literal<int>(3));
            })
            .Else([&result](IRFunctionEmitter& fn) {
                fn.Store(result, fn.Literal<int>(2));
            });
        fn.Return(fn.Load(result));
    }
    module.EndFunction();

    IRExecutionEngine jit(std::move(module));
    auto testFn = jit.GetFunction<int32_t(int32_t)>("ScopedIfElse2Test");
    auto referenceFn = [](int x) {
        int result = 0;
        if (x < 3)
        {
            result = 1;
        }
        else if (x > 6)
        {
            result = 3;
        }
        else
        {
            result = 2;
        }
        return result;
    };

    bool success = true;
    for (int32_t i = 0; i < 20; ++i)
    {
        auto result = testFn(i);
        auto expected = referenceFn(i);
        success = success && (result == expected);
    }
    testing::ProcessTest("Testing scoped IfElse2", success);
}

void TestElseIfWithComputedCondition()
{
    auto module = MakeHostModuleEmitter("ElseIfComputedCondition");

    const auto returnType = emitters::GetVariableType<int>();
    const emitters::NamedVariableTypeList parameters = { { "a", VariableType::Int32 }, { "b", VariableType::Int32 } };

    auto fn = module.BeginFunction("ElseIfComputedConditionTest", returnType, parameters);
    {
        auto arguments = fn.Arguments().begin();
        auto a = fn.LocalScalar(&(*arguments++));
        auto b = fn.LocalScalar(&(*arguments++));

        auto result = fn.Variable(returnType, "result");
        fn.Store(result, fn.Literal(0));

        fn.If(a == 1 && b == 1, [result](emitters::IRFunctionEmitter& fn) {
              fn.Store(result, fn.Literal(1));
          }).ElseIf(a == 2 || b == 2, [result](emitters::IRFunctionEmitter& fn) {
            fn.Store(result, fn.Literal(2));
        });

        fn.Return(fn.Load(result));
    }
    module.EndFunction();

    fn.Verify();

    IRExecutionEngine jit(std::move(module));
    auto testFn = jit.GetFunction<int32_t(int32_t, int32_t)>("ElseIfComputedConditionTest");
    auto referenceFn = [](int a, int b) {
        if (a == 1 && b == 1)
            return 1;
        else if (a == 2 || b == 2)
            return 2;
        else
            return 0;
    };

    bool success = true;
    auto trials = std::vector<std::vector<int32_t>>{ { 1, 1 }, { 1, 2 }, { 2, 2 }, { 3, 3 } };
    for (auto args : trials)
    {
        auto result = testFn(args[0], args[1]);
        auto expected = referenceFn(args[0], args[1]);
        success = success && (result == expected);
    }

    testing::ProcessTest("Testing elseif with inline condition", success);
}

template <typename InT, typename OutT>
void TestCastValue()
{
    auto module = MakeHostModuleEmitter("CastValue");
    VariableType inType = emitters::GetVariableType<InT>();
    VariableType outType = emitters::GetVariableType<OutT>();
    const emitters::NamedVariableTypeList parameters = { { "x", inType } };
    auto fn = module.BeginFunction("CastValue", outType, parameters);
    {
        auto arguments = fn.Arguments().begin();
        auto x = fn.LocalScalar(&(*arguments++));
        auto result = fn.CastValue(x, outType);
        fn.Return(result);
    }
    module.EndFunction();

    fn.Verify();

    IRExecutionEngine jit(std::move(module));
    auto testFn = jit.GetFunction<OutT(InT)>("CastValue");

    bool success = true;
    auto trials = std::vector<InT>{ 1, 2, 35, 4216 };
    for (auto val : trials)
    {
        auto result = testFn(val);
        auto expected = static_cast<OutT>(val);
        success = success && (result == expected);
    }

    testing::ProcessTest("Testing CastValue", success);
}

void TestCastValue()
{
    // short, int, int64_t, float, double
    TestCastValue<short, short>();
    TestCastValue<short, int>();
    TestCastValue<short, int64_t>();
    TestCastValue<short, float>();
    TestCastValue<short, double>();

    TestCastValue<int, short>();
    TestCastValue<int, int>();
    TestCastValue<int, int64_t>();
    TestCastValue<int, float>();
    TestCastValue<int, double>();

    TestCastValue<int64_t, short>();
    TestCastValue<int64_t, int>();
    TestCastValue<int64_t, int64_t>();
    TestCastValue<int64_t, float>();
    TestCastValue<int64_t, double>();

    TestCastValue<float, short>();
    TestCastValue<float, int>();
    TestCastValue<float, int64_t>();
    TestCastValue<float, float>();
    TestCastValue<float, double>();

    TestCastValue<double, short>();
    TestCastValue<double, int>();
    TestCastValue<double, int64_t>();
    TestCastValue<double, float>();
    TestCastValue<double, double>();
}

template <typename InT>
void TestCastToConditionalBool()
{
    auto module = MakeHostModuleEmitter("CastToConditionalBool");
    VariableType inType = emitters::GetVariableType<InT>();
    VariableType outType = emitters::GetVariableType<bool>();
    const emitters::NamedVariableTypeList parameters = { { "x", inType } };
    auto fn = module.BeginFunction("CastToConditionalBool", outType, parameters);
    {
        auto arguments = fn.Arguments().begin();
        auto x = fn.LocalScalar(&(*arguments++));
        auto result = fn.CastToConditionalBool(x);
        fn.Return(fn.CastBoolToByte(result));
    }
    module.EndFunction();

    fn.Verify();

    IRExecutionEngine jit(std::move(module));
    auto testFn = jit.GetFunction<bool(InT)>("CastToConditionalBool");

    bool success = true;
    auto trials = std::vector<InT>{ 1, 2, 35 };
    for (auto val : trials)
    {
        auto result = testFn(val);
        auto expected = val != 0;
        success = success && (result == expected);
    }

    testing::ProcessTest("Testing CastToConditionalBool", success);
}

void TestCastToConditionalBool()
{
    TestCastToConditionalBool<char>();
    TestCastToConditionalBool<short>();
    TestCastToConditionalBool<int>();
    TestCastToConditionalBool<int64_t>();
    TestCastToConditionalBool<float>();
    TestCastToConditionalBool<double>();
}

void TestInlineAssembly()
{
    VariableType inType = emitters::GetVariableType<int>();
    VariableType outType = emitters::GetVariableType<int>();

    auto module = MakeHostModuleEmitter("TestInlineAssembly");
    auto targetDevice = module.GetCompilerOptions().targetDevice;
    auto functionIdentifier = targetDevice.IsMacOS() ? "_square" : "square";
    auto functionName = "square";

    std::string asmStr;
    if (targetDevice.IsWindows())
    {
        asmStr = R"XX(
    .globl FUNCTION
FUNCTION:
    movl    %ecx, %eax
    imull   %ecx, %eax
    retq
)XX";
    }
    else
    {
        asmStr = R"XX(
    .globl     FUNCTION
FUNCTION:
    imull      %edi, %edi
    movl       %edi, %eax
    retq
)XX";
    }

    ReplaceAll(asmStr, "FUNCTION", functionIdentifier);

    module.GetLLVMModule()->appendModuleInlineAsm(asmStr);

    module.DeclareFunction(functionName, outType, { inType });
    module.DebugDump();

    const emitters::NamedVariableTypeList parameters = { { "x", inType } };
    auto fn = module.BeginFunction("InlineAssembly", outType, parameters);
    {
        auto arguments = fn.Arguments().begin();
        auto x = fn.LocalScalar(&(*arguments++));
        auto squareFn = module.GetFunction(functionName);
        squareFn->addFnAttr(llvm::Attribute::AttrKind::AlwaysInline);
        auto result = fn.Call(squareFn, { x });

        fn.Return(result);
    }
    module.EndFunction();

#if 0
    module.DebugDump();
    module.WriteToStream(std::cout, ModuleOutputFormat::assembly);
#endif
    fn.Verify();

    IRExecutionEngine jit(std::move(module));
    auto testFn = jit.GetFunction<int(int)>("InlineAssembly");

    bool success = true;
    auto trials = std::vector<int>{ 1, 2, 35 };
    for (auto val : trials)
    {
        auto result = testFn(val);
        auto expected = val * val;
        success = success && (result == expected);
    }

    testing::ProcessTest("Testing InlineAssembly", success);
}
