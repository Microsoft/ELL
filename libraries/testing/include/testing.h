////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     testing.h (testing)
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ell
{
/// <summary> testing namespace </summary>
namespace testing
{
    /// <summary> Exception due to test failure. </summary>
    class TestFailureException : public std::runtime_error
    {
    public:
        /// <summary> Constructor. </summary>
        ///
        /// <param name="testDescription"> A description of the test failure that caused the exception. </param>
        TestFailureException(const std::string& testDescription);
    };

    /// <summary> Exception due to test not implemented. </summary>
    class TestNotImplementedException : public std::runtime_error
    {
    public:
        /// <summary> Constructor. </summary>
        ///
        /// <param name="testDescription"> A description of the test that wasn't implemented. </param>
        TestNotImplementedException(const std::string& testDescription);
    };

    /// <summary> Checks if a value is true </summary>
    ///
    /// <param name="a"> The value to check </param>
    ///
    /// <returns> The truth value of the value </returns>
    inline bool IsTrue(bool a)
    {
        return a;
    }

    /// <summary> Checks if a value is false </summary>
    ///
    /// <param name="a"> The value to check </param>
    ///
    /// <returns> The negated truth value of the value </returns>
    inline bool IsFalse(bool a)
    {
        return !a;
    }

    /// <summary> Checks if two values are exactly equal. </summary>
    ///
    /// <param name="a"> The first value. </param>
    /// <param name="b"> The second value. </param>
    ///
    /// <returns> true if equal, false if not. </returns>
    template <typename T1, typename T2>
    inline std::enable_if_t<!std::is_floating_point<T1>::value && !std::is_floating_point<T2>::value, bool>
    IsEqual(T1 a, T2 b)
    {
        return a == b;
    }

    /// <summary> Checks if two floats are equal, up to a small numerical error. </summary>
    ///
    /// <param name="a"> The first number. </param>
    /// <param name="b"> The second number. </param>
    /// <param name="tolerance"> The tolerance. </param>
    ///
    /// <returns> true if equal, false if not. </returns>
    template <
        typename T1,
        typename T2,
        typename T3 = std::conditional_t<sizeof(T1) >= sizeof(T2), T1, T2>>
    inline std::enable_if_t<std::is_floating_point<T1>::value && std::is_floating_point<T2>::value, bool>
    IsEqual(T1 a, T2 b, T3 tolerance = std::is_same_v<float, T3> ? 1.0e-6f : 1.0e-8)
    {
        if (std::isnan(a))
        {
            return std::isnan(b);
        }
        if (std::isinf(a))
        {
            return std::isinf(b);
        }
        return (a - b < tolerance && b - a < tolerance);
    }

    /// <summary>
    /// Checks if two vectors are equal.
    /// </summary>
    ///
    /// <param name="a"> The first vector. </param>
    /// <param name="b"> The second vector. </param>
    ///
    /// <returns> true if equal, false if not. </returns>
    bool IsEqual(const std::vector<bool>& a, const std::vector<bool>& b);

    /// <summary>
    /// Checks if two vectors are equal.
    /// </summary>
    ///
    /// <param name="a"> The first vector. </param>
    /// <param name="b"> The second vector. </param>
    ///
    /// <returns> true if equal, false if not. </returns>
    bool IsEqual(const std::vector<int>& a, const std::vector<int>& b);
    bool IsEqual(const std::vector<int64_t>& a, const std::vector<int64_t>& b);

    /// <summary>
    /// Checks if two vectors are equal, up to a small numerical error in each coordinate.
    /// </summary>
    ///
    /// <param name="a"> The first vector. </param>
    /// <param name="b"> The second vector. </param>
    /// <param name="tolerance"> The tolerance. </param>
    ///
    /// <returns> true if equal, false if not. </returns>
    bool IsEqual(const std::vector<float>& a, const std::vector<float>& b, float tolerance = 1.0e-8);
    bool IsEqual(const std::vector<double>& a, const std::vector<double>& b, double tolerance = 1.0e-8);

    template <typename ValueType1, typename ValueType2>
    bool IsEqual(const std::vector<std::vector<ValueType1>>& a, const std::vector<std::vector<ValueType2>>& b, double tolerance = 1.0e-8);

    /// <summary>
    /// Checks if two bool vectors are equal.
    /// </summary>
    ///
    /// <param name="a"> The first vector. </param>
    /// <param name="b"> The second vector. </param>
    ///
    /// <returns> true if equal, false if not. </returns>
    bool IsEqual(const std::vector<bool>& a, const std::vector<bool>& b);

    /// <summary>
    /// Checks if two string vectors are equal.
    /// </summary>
    ///
    /// <param name="a"> The first vector. </param>
    /// <param name="b"> The second vector. </param>
    ///
    /// <returns> true if equal, false if not. </returns>
    bool IsEqual(const std::vector<std::string>& a, const std::vector<std::string>& b);

    /// <summary> Checks if two values are not equal (using the default tolerance for floating-point values). </summary>
    ///
    /// <param name="a"> The first value. </param>
    /// <param name="b"> The second value. </param>
    ///
    /// <returns> true if not equal, true if equal. </returns>
    template <typename T1, typename T2>
    inline bool IsNotEqual(T1 t1, T2 t2)
    {
        return !IsEqual(t1, t2);
    }

    /// <summary> Process the test. </summary>
    ///
    /// <param name="testDescription"> Information describing the test. </param>
    /// <param name="success"> true if the test was a success, false if it failed. </param>
    /// <returns> The success value, for convenience. </returns>
    bool ProcessTest(const std::string& testDescription, bool success);

    /// <summary> Process a test, quietly. </summary>
    /// Identical to `ProcessTest`, except that it doesn't emit anything on success.
    ///
    /// <param name="testDescription"> Information describing the test. </param>
    /// <param name="success"> true if the test was a success, false if it failed. </param>
    /// <returns> The success value, for convenience. </returns>
    bool ProcessQuietTest(const std::string& testDescription, bool success);

    /// <summary> Process a test, throwing an exception on failure. </summary>
    ///
    /// <param name="testDescription"> Information describing the test. </param>
    /// <param name="success"> true if the test was a success, false if it failed. </param>
    void ProcessCriticalTest(const std::string& testDescription, bool success);

    /// <summary> Note a test failure. </summary>
    ///
    /// <param name="message"> A message to print. </param>
    void TestFailed(const std::string& message);

    /// <summary> Note a test success. </summary>
    ///
    /// <param name="message"> A message to print. </param>
    void TestSucceeded(const std::string& message);

    /// <summary> Note a test warning. </summary>
    ///
    /// <param name="message"> A message to print. </param>
    void TestWarning(const std::string& message);

    /// <summary> Checks if one of the tests failed. </summary>
    ///
    /// <returns> true if one of the tests failed. </returns>
    bool DidTestFail();

    /// <summary> Get the exit code to return from main. </summary>
    ///
    /// <returns> 1 if any of the tests failed, otherwise 0. </returns>
    int GetExitCode();

    /// <summary> Call a function, registering a test failure if an exception is thrown, and continue execution. </summary>
    ///
    /// <param name="function"> The test function to call. </param>
    /// <param name="args"> Aguments to pass to the test function. </param>
    template <typename FunctionType, typename... Args>
    bool FailOnException(FunctionType&& function, Args&&... args)
    {
        using namespace std::literals::string_literals;
        try
        {
            function(std::forward<Args>(args)...);
            return true;
        }
        catch (const std::exception& exception)
        {
            TestFailed("Got exception in test: "s + exception.what());
        }
        catch (...)
        {
            TestFailed("Got unknown exception type in test");
        }
        return false;
    }

    /// <summary> Call a test function, but register success if a "TestNotImplemented" exception is thrown. </summary>
    /// <param name="function"> The test function to call. </param>
    /// <param name="args"> Aguments to pass to the test function. </param>
    template <typename FunctionType, typename... Args>
    bool NoFailOnUnimplemented(FunctionType&& function, Args&&... args)
    {
        using namespace std::literals::string_literals;
        try
        {
            function(std::forward<Args>(args)...);
            return true;
        }
        catch (const TestNotImplementedException& exception)
        {
            TestWarning("Skipping unimplemented test: "s + exception.what());
            return true;
        }
        catch (...)
        {
            throw;
        }
        return false;
    }

    /// <summary> Get the exit code to return from main. </summary>
    ///
    /// <returns> 1 if any of the tests failed, otherwise 0. </returns>
    int GetExitCode();

    /// <summary> RAII helper to turn on logging for a specific test/scope
    ///
    /// Example:
    /// ```
    /// EnableLoggingHelper enableLogging;
    /// ```
    /// </summary>
    struct EnableLoggingHelper
    {
        EnableLoggingHelper();
        ~EnableLoggingHelper();

        EnableLoggingHelper(const EnableLoggingHelper&) = delete;
        EnableLoggingHelper(EnableLoggingHelper&&) = delete;
        EnableLoggingHelper& operator=(const EnableLoggingHelper&) = delete;
        EnableLoggingHelper& operator=(EnableLoggingHelper&&) = delete;
    };
} // namespace testing
} // namespace ell

// Forces a symbol to be defined so that LLVM jit can find it
#if defined(_WIN32)
#if defined(_WIN64)
#define TESTING_FORCE_DEFINE_SYMBOL(x, returnType, ...) __pragma(comment(linker, "/export:" #x))
#else
#define TESTING_FORCE_DEFINE_SYMBOL(x, returnType, ...) __pragma(comment(linker, "/export:_" #x))
#endif
#else
#define TESTING_FORCE_DEFINE_SYMBOL(x, returnType, ...) returnType (*__##x##_fp)(__VA_ARGS__) __attribute__((used)) = &x;
#endif
