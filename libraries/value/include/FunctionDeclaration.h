////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     FunctionDeclaration.h (value)
//  Authors:  Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "EmitterContext.h"
#include "Scalar.h"
#include "Value.h"

#include <utilities/include/FunctionUtils.h>
#include <utilities/include/StringUtil.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace ell
{
namespace value
{
    /// <summary> Helper enum used to specify whether a FunctionDeclaration should be inlined </summary>
    enum class FunctionInlining
    {
        defaultInline,
        always,
        never
    };

    /// <summary> Describes a function that can be acted upon by an EmitterContext instance </summary>
    class [[nodiscard]] FunctionDeclaration
    {
    public:
        /// <summary> Default constructor. Creates an empty function declaration </summary>
        FunctionDeclaration() = default;

        /// <summary> Constructor </summary>
        /// <param name="name"> The name of the function </param>
        explicit FunctionDeclaration(std::string name);

        /// <summary> Sets the return type for this function declaration </summary>
        /// <param name="returnType"> A Value instance describing type of the value that is expected and its memory layout to be returned by the function </param>
        /// <returns> A reference to this instance </returns>
        /// <remarks> If this function is not called, the instance defaults to a void return type </remarks>
        FunctionDeclaration& Returns(ViewAdapter returnType);

        /// <summary> Sets whether this function should be decorated (mangled) </summary>
        /// <param name="shouldDecorate"> An enum value specifying whether this function should be decorated </param>
        /// <returns> A reference to this instance </returns>
        /// <remarks> By default, a function is decorated, which means the name gets suffixed by an encoding of the function's parameter and return types.
        /// Functions that are declared externally should probably not be decorated </remarks>
        FunctionDeclaration& Decorated(bool shouldDecorate);

        /// <summary> If `public` is true, set the function to appear in the public header, otherwise the function is internal </summary>
        FunctionDeclaration& Public(bool isPublic);

        /// <summary> Sets whether this function should be inlined  </summary>
        /// <param name="shouldInline"> A FunctionInlining value specifying whether this function should be inlined or not </param>
        FunctionDeclaration& Inlined(FunctionInlining shouldInline = FunctionInlining::always);

        /// <summary> Specifies a function definition for this declaration </summary>
        /// <param name="fn"> A function object that takes zero or more Value library observer types and returns void or a Value library observer type.
        /// This function object defines this function. </param>
        /// <returns> A std::function function object that matches the signature of the function passed in </returns>
        /// <remarks> If this function or `Imported` is not called, this function declaration is treated as an external function. Not all contexts may support an external
        /// function </remarks>
        template <typename Fn>
        [[maybe_unused]] auto Define(Fn&& fn);

        /// <summary> Specifies the code file that is to be imported to define this function
        /// <remarks> The specified file is imported when this function declaration is used to emit a call. </remarks>
        FunctionDeclaration& DefineFromFile(std::string file);

        /// <summary> Sets the parameters this function requires </summary>
        /// <param name="paramTypes"> Zero or more Value instances or view types with a GetValue() member function describing
        /// the types of the arguments and their memory layout expected by the function </param>
        /// <returns> A reference to this instance </returns>
        /// <remarks> If this function is not called, the instance defaults to taking no arguments </remarks>
        template <typename... Types>
        FunctionDeclaration& Parameters(Types&&... paramTypes);

        /// <summary> Sets the parameters this function requires </summary>
        /// <param name="paramTypes"> Zero or more Value instances describing the types of the arguments and their memory layout expected by the function </param>
        /// <returns> A reference to this instance </returns>
        /// <remarks> If this function is not called, the instance defaults to taking no arguments </remarks>
        [[nodiscard]] FunctionDeclaration& Parameters(std::vector<ViewAdapter> parameters);

        /// <summary> Emits a call to the function declaration </summary>
        /// <param name="arguments"> A vector of Value instances that hold the arguments for the function call </param>
        /// <returns> A std::optional instance that holds a Value instance with the return value of the call, if it is expected, otherwise empty </returns>
        /// <remarks> If the function is not defined and the context is capable of it, this will emit a call to an external function </remarks>
        [[maybe_unused]] std::optional<Value> Call(std::vector<ViewAdapter> arguments) const;

        /// <summary> Emits a call to the function declaration </summary>
        /// <param name="arguments"> Zero or more Value instances or view types with a GetValue() member function describing
        /// the arguments for this function call </param>
        /// <returns> A std::optional instance that holds a Value instance with the return value of the call, if it is expected, otherwise empty </returns>
        /// <remarks> If the function is not defined and the context is capable of it, this will emit a call to an external function </remarks>
        template <typename... Types>
        [[maybe_unused]] std::optional<Value> Call(Types&&... arguments) const;

        /// <summary> Gets the final function name, including any decoration if so applicable </summary>
        const std::string& GetFunctionName() const;

        /// <summary> Gets the vector of Value instances describing the parameter types the function requires </summary>
        const std::vector<Value>& GetParameterTypes() const;

        /// <summary> Gets the return type, wrapped in a std::optional. If the function expects to return a value, its type is described.
        /// Otherwise, the std::optional instance is empty </summary>
        const std::optional<Value>& GetReturnType() const;

        /// <summary> Returns true if function is to appear in the public header, false otherwise </summary>
        [[nodiscard]] bool IsPublic() const;

        /// <summary> Returns true if function is defined for current context, false otherwise </summary>
        [[nodiscard]] bool IsDefined() const;

        /// <summary> Returns true if the instance is an empty function declaration </summary>
        [[nodiscard]] bool IsEmpty() const;

        /// <summary> Returns true if the instance represents an imported function </summary>
        [[nodiscard]] bool IsImported() const;

        /// <summary> Returns true if the instance is inlined </summary>
        [[nodiscard]] FunctionInlining InlineState() const;

        Scalar GetPointer() const;

        void SetPointer(Scalar pointer) { _pointer = pointer; }

        bool IsPointerSet() const { return _pointer.has_value(); }

    private:
        template <typename ReturnT, typename... Args>
        [[maybe_unused]] std::function<ReturnT(Args...)> DefineImpl(std::function<ReturnT(Args...)> fn);

        template <typename ReturnT, typename... Args>
        [[maybe_unused]] std::function<ReturnT(Args...)> DefineImpl(std::false_type, std::function<ReturnT(Args...)> fn);

        template <typename ReturnT, typename... Args>
        [[maybe_unused]] inline std::function<ReturnT(Args...)> DefineImpl(std::true_type, std::function<ReturnT(Args...)>);

        void CheckNonEmpty() const;

        std::string _importedSource;
        std::string _originalFunctionName;
        mutable std::optional<std::string> _decoratedFunctionName;
        std::optional<Value> _returnType;
        std::vector<Value> _paramTypes;
        std::optional<Scalar> _pointer;

        FunctionInlining _inlineState = FunctionInlining::defaultInline;
        bool _isDecorated = true;
        bool _isPublic = false;
        bool _isEmpty = true;
    };

    inline bool operator==(const FunctionDeclaration& decl1, const FunctionDeclaration& decl2)
    {
        return decl1.GetFunctionName() == decl2.GetFunctionName();
    }

    [[nodiscard]] FunctionDeclaration DeclareFunction(std::string name);

} // namespace value
} // namespace ell

namespace std
{
template <>
struct hash<::ell::value::FunctionDeclaration>
{
    using Type = ::ell::value::FunctionDeclaration;

    size_t operator()(const Type& value) const;
};
} // namespace std

#pragma region implementation

namespace ell
{
namespace value
{

    namespace detail
    {
        // Until MacOS's compiler has proper std::function deduction guides
#if defined(__APPLE__)
        template <typename>
        struct StdFunctionDeductionGuideHelper
        {};

        template <typename ReturnT, typename Class, bool IsNoExcept, typename... Args>
        struct StdFunctionDeductionGuideHelper<ReturnT (Class::*)(Args...) noexcept(IsNoExcept)>
        {
            using Type = ReturnT(Args...);
        };

        template <typename ReturnT, typename Class, bool IsNoExcept, typename... Args>
        struct StdFunctionDeductionGuideHelper<ReturnT (Class::*)(Args...) & noexcept(IsNoExcept)>
        {
            using Type = ReturnT(Args...);
        };

        template <typename ReturnT, typename Class, bool IsNoExcept, typename... Args>
        struct StdFunctionDeductionGuideHelper<ReturnT (Class::*)(Args...) const noexcept(IsNoExcept)>
        {
            using Type = ReturnT(Args...);
        };

        template <typename ReturnT, typename Class, bool IsNoExcept, typename... Args>
        struct StdFunctionDeductionGuideHelper<ReturnT (Class::*)(Args...) const & noexcept(IsNoExcept)>
        {
            using Type = ReturnT(Args...);
        };

        template <typename Fn>
        struct Function : public std::function<Fn>
        {
            Function(const std::function<Fn>& fn) :
                std::function<Fn>(fn) {}
            Function(std::function<Fn>&& fn) :
                std::function<Fn>(std::move(fn)) {}
            using std::function<Fn>::function;
        };

        // Function pointer
        template <typename ReturnT, typename... Args>
        Function(ReturnT (*)(Args...)) -> Function<ReturnT(Args...)>;

        // Functor
        template <typename Functor,
                  typename Signature = typename StdFunctionDeductionGuideHelper<decltype(&Functor::operator())>::Type>
        Function(Functor) -> Function<Signature>;
#endif // defined(__APPLE__)

    } // namespace detail

#if defined(__APPLE__)
#define FUNCTION_TYPE detail::Function
#else
#define FUNCTION_TYPE std::function
#endif // defined(__APPLE__)

    template <typename Fn>
    [[maybe_unused]] auto FunctionDeclaration::Define(Fn&& fn)
    {
        return DefineImpl(FUNCTION_TYPE(std::forward<Fn>(fn)));
    }
#undef FUNCTION_TYPE

    template <typename ReturnT, typename... Args>
    inline std::function<ReturnT(Args...)> FunctionDeclaration::DefineImpl(std::function<ReturnT(Args...)> fn)
    {
        if constexpr (sizeof...(Args) == 1 && utilities::AllSame<utilities::RemoveCVRefT<Args>..., std::vector<Value>>)
        {
            return DefineImpl(std::true_type{}, fn);
        }
        else
        {
            return DefineImpl(std::false_type{}, fn);
        }
    }

    template <typename ReturnT, typename... Args>
    inline std::function<ReturnT(Args...)> FunctionDeclaration::DefineImpl(std::true_type, std::function<ReturnT(Args...)> fn)
    {
        auto createdFn = GetContext().CreateFunction(*this, [fn = std::move(fn)](std::vector<Value> args) -> std::optional<Value> {
            if constexpr (std::is_same_v<ReturnT, void>)
            {
                fn(args);
                return std::nullopt;
            }
            else
            {
                return fn(args);
            }
        });

        return [createdFn = std::move(createdFn)](Args... args) -> ReturnT {
            if constexpr (std::is_same_v<ReturnT, void>)
            {
                createdFn(args...);
            }
            else
            {
                return *createdFn(args...);
            }
        };
    }

    template <typename ReturnT, typename... Args>
    [[maybe_unused]] std::function<ReturnT(Args...)> FunctionDeclaration::DefineImpl(std::false_type, std::function<ReturnT(Args...)> fn)
    {
        if constexpr (std::is_same_v<ReturnT, void>)
        {
            if (_returnType.has_value())
            {
                throw utilities::InputException(utilities::InputExceptionErrors::typeMismatch, utilities::FormatString("[%s] Defining function has a return value, but declaration does not", GetFunctionName().c_str()));
            }
        }
        else
        {
            if (!_returnType.has_value())
            {
                throw utilities::InputException(utilities::InputExceptionErrors::typeMismatch, utilities::FormatString("[%s] Defining function returns void, but declaration does not", GetFunctionName().c_str()));
            }

            // Try to instantiate an instance of the return type (R) with the Value instance that represents the return type (_returnType)
            // If this throws, the return value of the defining function is not compatible with the Value instance specified in the declaration
            ReturnT returnType = *_returnType;
        }

        if (sizeof...(Args) != _paramTypes.size())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::typeMismatch, utilities::FormatString("[%s] Defining function takes %zu parameters, but declaration was specific to have %zu.", GetFunctionName().c_str(), sizeof...(Args), _paramTypes.size()));
        }

        if constexpr (sizeof...(Args) > 0)
        {
            // Same as the return type checking, instantiate the arguments of the function with the Value instances in _paramTypes to ensure everything
            // is correct. If this throws, there's a mismatch between the defining function's parameters and the declaration.
            auto paramTypes = utilities::VectorToTuple<Args...>(_paramTypes);
        }

        auto createdFn = GetContext().CreateFunction(*this, [fn = std::move(fn)](std::vector<Value> args) -> std::optional<Value> {
            std::tuple<Args...> tupleArgs = utilities::VectorToTuple<Args...>(args);
            if constexpr (std::is_same_v<ReturnT, void>)
            {
                std::apply(fn, tupleArgs);
                return std::nullopt;
            }
            else
            {
                ReturnT r = std::apply(fn, tupleArgs);
                return detail::GetValue(r);
            }
        });

        return [createdFn = std::move(createdFn), name = GetFunctionName()](Args&&... args) -> ReturnT {
            constexpr auto argSize = sizeof...(Args);
            std::vector<Value> argValues;
            argValues.reserve(argSize);
            (argValues.push_back(detail::GetValue(args)), ...);

            auto fnReturn = createdFn(argValues);
            if constexpr (std::is_same_v<void, ReturnT>)
            {
                if (fnReturn)
                {
                    throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState,
                                                    utilities::FormatString("[%s] Function is supposed to return void, but a value was returned from the defining function", name.c_str()));
                }
            }
            else
            {
                return ReturnT(*fnReturn);
            }
        };
    }

    template <typename... Types>
    FunctionDeclaration& FunctionDeclaration::Parameters(Types&&... paramTypes)
    {
        return Parameters(std::vector<ViewAdapter>{ std::forward<Types>(paramTypes)... });
    }

    template <typename... Types>
    std::optional<Value> FunctionDeclaration::Call(Types&&... arguments) const
    {
        return Call(std::vector<ViewAdapter>{ std::forward<Types>(arguments)... });
    }

} // namespace value
} // namespace ell

#pragma endregion implementation
