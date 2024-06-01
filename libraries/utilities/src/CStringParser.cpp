////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CStringParser.cpp (utilities)
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CStringParser.h"

#include <cctype>
#include <cerrno>
#include <cstdlib>

namespace ell
{
namespace utilities
{
    void TrimLeadingWhitespace(const char*& pStr)
    {
        while (std::isspace(*pStr))
        {
            ++pStr;
        }
    }

    bool IsEndOfString(char c)
    {
        return c == '\0';
    }

    bool IsWhitespace(char c)
    {
        return std::isspace(c) != 0;
    }

    bool IsDigit(char c)
    {
        return std::isdigit(c) != 0;
    }

    template <typename ValueType, typename ParseFunctionType>
    ParseResult ParseFloat(const char* pStr, char*& pEnd, ParseFunctionType parse, ValueType& value)
    {
        if (IsWhitespace(*pStr))
        {
            return ParseResult::badFormat;
        }

        auto tmp = errno;
        errno = 0;

        value = parse(pStr, &pEnd);

        if (pStr == pEnd)
        {
            return ParseResult::badFormat;
        }
        if (errno == ERANGE)
        {
            return ParseResult::outOfRange;
        }

        errno = tmp;
        return ParseResult::success;
    }

    template <typename ValueType, typename ParseFunctionType>
    ParseResult ParseInt(const char* pStr, char*& pEnd, ParseFunctionType parse, ValueType& value)
    {
        if (!IsDigit(*pStr))
        {
            return ParseResult::badFormat;
        }

        auto tmp = errno;
        errno = 0;

        auto x = parse(pStr, &pEnd, 0);
        if (x != static_cast<ValueType>(x))
        {
            return ParseResult::outOfRange;
        }

        value = static_cast<ValueType>(x);

        if (pStr == pEnd)
        {
            return ParseResult::badFormat;
        }
        if (errno == ERANGE)
        {
            return ParseResult::outOfRange;
        }

        errno = tmp;
        return ParseResult::success;
    }

    // wrapper for std::strtof
    inline ParseResult cParse(const char* pStr, char*& pEnd, float& value)
    {
        return ParseFloat(pStr, pEnd, std::strtof, value);
    }

    // wrapper for std::strtod
    inline ParseResult cParse(const char* pStr, char*& pEnd, double& value)
    {
        return ParseFloat(pStr, pEnd, std::strtod, value);
    }

    // wrapper for strtoul
    inline ParseResult cParse(const char* pStr, char*& pEnd, unsigned int& value)
    {
        return ParseInt(pStr, pEnd, std::strtoul, value);
    }

    // wrapper for strtoull
    inline ParseResult cParse(const char* pStr, char*& pEnd, uint64_t& value)
    {
        return ParseInt(pStr, pEnd, std::strtoull, value);
    }

    // wrapper for strtol
    inline ParseResult cParse(const char* pStr, char*& pEnd, int& value)
    {
        return ParseInt(pStr, pEnd, std::strtol, value);
    }

    // wrapper for strtol
    inline ParseResult cParse(const char* pStr, char*& pEnd, long& value)
    {
        return ParseInt(pStr, pEnd, std::strtol, value);
    }

    // wrapper for strtoul
    inline ParseResult cParse(const char* pStr, char*& pEnd, unsigned short& value)
    {
        return ParseInt(pStr, pEnd, std::strtoul, value);
    }

    // wrapper for strtol
    inline ParseResult cParse(const char* pStr, char*& pEnd, short& value)
    {
        return ParseInt(pStr, pEnd, std::strtol, value);
    }

    // parse a single char from the input string.
    inline ParseResult cParse(const char* pStr, char*& pEnd, char& value)
    {
        value = *pStr;
        pEnd = const_cast<char*>(++pStr);
        return ParseResult::success;
    }

    // parser for std:string, scans until finding the a character other than alphanumeric or '_'
    inline ParseResult cParse(const char* pStr, char*& pEnd, std::string& value)
    {
        const char* iter = pStr;
        while (std::isalnum(*iter) || *iter == '_')
        {
            ++iter;
        }
        value = std::string(pStr, iter);
        pEnd = const_cast<char*>(iter);

        return ParseResult::success;
    }

    // wrapper for strtoul
    template <typename std::enable_if_t<!std::is_same<unsigned long, unsigned int>::value, int> = 0>
    inline ParseResult cParse(const char* pStr, char*& pEnd, unsigned long& value)
    {
        return ParseInt(pStr, pEnd, std::strtoul, value);
    }

    // wrapper for strtol
    template <typename std::enable_if_t<!std::is_same<long, int>::value, int> = 0>
    inline ParseResult cParse(const char* pStr, char*& pEnd, long& value)
    {
        return ParseInt(pStr, pEnd, std::strtoul, value);
    }

    template <typename ValueType>
    ParseResult Parse(const char*& pStr, ValueType& value)
    {
        // check for eof
        if (IsEndOfString(*pStr))
        {
            return ParseResult::endOfString;
        }

        // check for "//" comment indicator
        if (*pStr == '/')
        {
            if (*(pStr + 1) == '/')
            {
                return ParseResult::beginComment;
            }
        }

        // check for "#" comment indicator
        if (*pStr == '#')
        {
            return ParseResult::beginComment;
        }

        char* pEnd = nullptr;
        auto parseResult = cParse(pStr, pEnd, value);
        pStr = pEnd;

        return parseResult;
    }

    // explicit instantiation

    template ParseResult Parse(const char*& pStr, float& value);
    template ParseResult Parse(const char*& pStr, double& value);
    template ParseResult Parse(const char*& pStr, unsigned int& value);
    template ParseResult Parse(const char*& pStr, int& value);
    template ParseResult Parse(const char*& pStr, uint64_t& value);
    template ParseResult Parse(const char*& pStr, long& value);
    template ParseResult Parse(const char*& pStr, unsigned short& value);
    template ParseResult Parse(const char*& pStr, short& value);
    template ParseResult Parse(const char*& pStr, char& value);
    template ParseResult Parse(const char*& pStr, std::string& value);

// On most platforms, uint64_t is an alias to `unsigned long`. However, on macOS, uint64_t is an alias of `unsigned long long`,
// so we need to add 'unsigned long' when compiling on the Mac to make sure it is archivable.
#if defined(__APPLE__)
    template ParseResult Parse(const char*& pStr, unsigned long& value);
#endif
} // namespace utilities
} // namespace ell