////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     FunctionDeclaration.cpp (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "FunctionDeclaration.h"

#include <utilities/include/Hash.h>

#include <cctype>

namespace ell
{
namespace value
{
    using namespace ell::utilities;

    FunctionDeclaration::FunctionDeclaration(std::string name) :
        _originalFunctionName(name),
        _isEmpty(false)
    {
        if (!std::isalpha(_originalFunctionName[0]) && _originalFunctionName[0] != '_')
        {
            throw InputException(InputExceptionErrors::invalidArgument, "Function names must begin with an _ or alphabetical character");
        }
    }

    FunctionDeclaration& FunctionDeclaration::DefineFromFile(std::string file)
    {
        CheckNonEmpty();

        _importedSource = file;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::Returns(ViewAdapter returnType)
    {
        CheckNonEmpty();

        _returnType = returnType;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::Parameters(std::vector<ViewAdapter> paramTypes)
    {
        CheckNonEmpty();

        _paramTypes.assign(paramTypes.begin(), paramTypes.end());
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::Decorated(bool shouldDecorate)
    {
        CheckNonEmpty();

        _isDecorated = shouldDecorate;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::Public(bool isPublic)
    {
        _isPublic = isPublic;
        return *this;
    }

    FunctionDeclaration& FunctionDeclaration::Inlined(FunctionInlining shouldInline)
    {
        CheckNonEmpty();

        _inlineState = shouldInline;
        return *this;
    }

    std::optional<Value> FunctionDeclaration::Call(std::vector<ViewAdapter> arguments) const
    {
        CheckNonEmpty();

        if (!_importedSource.empty() && !IsDefined())
        {
            GetContext().ImportCodeFile(_importedSource);
        }

        return GetContext().Call(*this, arguments);
    }

    const std::string& FunctionDeclaration::GetFunctionName() const
    {
        CheckNonEmpty();

        if (_isDecorated)
        {
            if (!_decoratedFunctionName)
            {
                size_t hash = 0;
                if (_returnType)
                {
                    HashCombine(hash, static_cast<int>(_returnType->GetBaseType()));
                    HashCombine(hash, _returnType->PointerLevel());
                    if (_returnType->IsConstrained())
                    {
                        HashCombine(hash, _returnType->GetLayout());
                    }
                }
                for (auto p : _paramTypes)
                {
                    HashCombine(hash, static_cast<int>(p.GetBaseType()));
                    HashCombine(hash, p.PointerLevel());
                    if (p.IsConstrained())
                    {
                        HashCombine(hash, p.GetLayout());
                    }
                }
                _decoratedFunctionName = _originalFunctionName + "_" + std::to_string(hash);
            }
            return *_decoratedFunctionName;
        }
        else
        {
            return _originalFunctionName;
        }
    }

    const std::vector<Value>& FunctionDeclaration::GetParameterTypes() const
    {
        CheckNonEmpty();

        return _paramTypes;
    }

    const std::optional<Value>& FunctionDeclaration::GetReturnType() const
    {
        CheckNonEmpty();

        return _returnType;
    }

    bool FunctionDeclaration::IsPublic() const
    {
        CheckNonEmpty();

        return _isPublic;
    }

    bool FunctionDeclaration::IsDefined() const
    {
        CheckNonEmpty();

        return GetContext().IsFunctionDefined(*this);
    }

    bool FunctionDeclaration::IsImported() const
    {
        CheckNonEmpty();

        return !_importedSource.empty();
    }

    bool FunctionDeclaration::IsEmpty() const
    {
        return _isEmpty;
    }

    FunctionInlining FunctionDeclaration::InlineState() const
    {
        CheckNonEmpty();
        return _inlineState;
    }

    void FunctionDeclaration::CheckNonEmpty() const
    {
        if (_isEmpty)
        {
            throw LogicException(LogicExceptionErrors::notInitialized, "FunctionDeclaration is empty");
        }
    }

    Scalar FunctionDeclaration::GetPointer() const
    {
        if (_pointer)
        {
            return *_pointer;
        }

        return GetContext().GetFunctionAddress(*this);
    }

    FunctionDeclaration DeclareFunction(std::string name)
    {
        return FunctionDeclaration(name);
    }

    /*extern*/ FunctionDeclaration AbsFunctionDeclaration = DeclareFunction("Abs").Decorated(false);
    /*extern*/ FunctionDeclaration CosFunctionDeclaration = DeclareFunction("Cos").Decorated(false);
    /*extern*/ FunctionDeclaration CopySignFunctionDeclaration = DeclareFunction("CopySign").Decorated(false);
    /*extern*/ FunctionDeclaration ExpFunctionDeclaration = DeclareFunction("Exp").Decorated(false);
    /*extern*/ FunctionDeclaration LogFunctionDeclaration = DeclareFunction("Log").Decorated(false);
    /*extern*/ FunctionDeclaration Log10FunctionDeclaration = DeclareFunction("Log10").Decorated(false);
    /*extern*/ FunctionDeclaration Log2FunctionDeclaration = DeclareFunction("Log2").Decorated(false);
    /*extern*/ FunctionDeclaration MaxNumFunctionDeclaration = DeclareFunction("MaxNum").Decorated(false);
    /*extern*/ FunctionDeclaration MinNumFunctionDeclaration = DeclareFunction("MinNum").Decorated(false);
    /*extern*/ FunctionDeclaration PowFunctionDeclaration = DeclareFunction("Pow").Decorated(false);
    /*extern*/ FunctionDeclaration SinFunctionDeclaration = DeclareFunction("Sin").Decorated(false);
    /*extern*/ FunctionDeclaration SqrtFunctionDeclaration = DeclareFunction("Sqrt").Decorated(false);
    /*extern*/ FunctionDeclaration TanhFunctionDeclaration = DeclareFunction("Tanh").Decorated(false);
    /*extern*/ FunctionDeclaration RoundFunctionDeclaration = DeclareFunction("Round").Decorated(false);
    /*extern*/ FunctionDeclaration FloorFunctionDeclaration = DeclareFunction("Floor").Decorated(false);
    /*extern*/ FunctionDeclaration CeilFunctionDeclaration = DeclareFunction("Ceil").Decorated(false);
    /*extern*/ FunctionDeclaration FmaFunctionDeclaration = DeclareFunction("Fma").Decorated(false);
    /*extern*/ FunctionDeclaration InitializeVectorFunctionDeclaration = DeclareFunction("InitVector").Decorated(false);
    /*extern*/ FunctionDeclaration MemCopyFunctionDeclaration = DeclareFunction("MemCpy").Decorated(false);
    /*extern*/ FunctionDeclaration MemMoveFunctionDeclaration = DeclareFunction("MemMove").Decorated(false);
    /*extern*/ FunctionDeclaration MemSetFunctionDeclaration = DeclareFunction("MemSet").Decorated(false);

} // namespace value
} // namespace ell

size_t std::hash<::ell::value::FunctionDeclaration>::operator()(const ::ell::value::FunctionDeclaration& value) const
{
    // The function name already encodes the return values and parameter names. And for undecorated functions, we
    // don't want to support overloading
    return std::hash<std::string>{}(value.GetFunctionName());
}
