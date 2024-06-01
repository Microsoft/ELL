////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CompilerOptions.cpp (emitters)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CompilerOptions.h"

#include <utilities/include/Exception.h>

#include <map>

#define ADD_TO_STRING_ENTRY(NAMESPACE, ENTRY) \
    case NAMESPACE::ENTRY:                    \
        return #ENTRY;
#define BEGIN_FROM_STRING if (false)
#define ADD_FROM_STRING_ENTRY(NAMESPACE, ENTRY) else if (s == #ENTRY) return NAMESPACE::ENTRY

namespace ell
{
namespace emitters
{
    std::string ToString(BlasType t)
    {
        switch (t)
        {
        case BlasType::unknown:
            return "unknown";
        case BlasType::openBLAS:
            return "openBLAS";
        case BlasType::atlas:
            return "atlas";
        default:
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument);
        }
    }

    /// <summary> Constructor from a property bag </summary>
    CompilerOptions::CompilerOptions(const utilities::PropertyBag& properties)
    {
        AddOptions(properties);
    }

    CompilerOptions CompilerOptions::AppendOptions(const utilities::PropertyBag& properties) const
    {
        CompilerOptions result = *this;
        result.AddOptions(properties);
        return result;
    }

    void CompilerOptions::AddOptions(const utilities::PropertyBag& properties)
    {
        blasType = properties.GetOrParseEntry<BlasType>("blasType", blasType);

        if (properties.HasEntry("positionIndependentCode"))
        {
            positionIndependentCode = properties.GetOrParseEntry<bool>("positionIndependentCode");
        }

        optimize = properties.GetOrParseEntry("optimize", optimize);
        unrollLoops = properties.GetOrParseEntry("unrollLoops", unrollLoops);
        inlineOperators = properties.GetOrParseEntry<bool>("inlineOperators", inlineOperators);
        allowVectorInstructions = properties.GetOrParseEntry<bool>("allowVectorInstructions", allowVectorInstructions);
        vectorWidth = properties.GetOrParseEntry<int>("vectorWidth", vectorWidth);
        useBlas = properties.GetOrParseEntry<bool>("useBlas", useBlas);
        profile = properties.GetOrParseEntry<bool>("profile", profile);
        includeDiagnosticInfo = properties.GetOrParseEntry<bool>("includeDiagnosticInfo", includeDiagnosticInfo);
        parallelize = properties.GetOrParseEntry<bool>("parallelize", parallelize);
        useThreadPool = properties.GetOrParseEntry<bool>("useThreadPool", useThreadPool);
        maxThreads = properties.GetOrParseEntry<int>("maxThreads", maxThreads);
        useFastMath = properties.GetOrParseEntry<bool>("useFastMath", useFastMath);
        debug = properties.GetOrParseEntry<bool>("debug", debug);
        globalValueAlignment = properties.GetOrParseEntry<int>("globalValueAlignment", globalValueAlignment);
        skip_ellcode = properties.GetOrParseEntry<bool>("skip_ellcode", skip_ellcode);

        if (properties.HasEntry("deviceName"))
        {
            targetDevice = GetTargetDevice(properties.GetEntry<std::string>("deviceName"));
        }
    }

} // namespace emitters

namespace utilities
{
    template <>
    emitters::BlasType FromString<emitters::BlasType>(const std::string& s)
    {
        static std::map<std::string, emitters::BlasType> nameMap = { { "unknown", emitters::BlasType::unknown },
                                                                     { "openBLAS", emitters::BlasType::openBLAS },
                                                                     { "atlas", emitters::BlasType::atlas } };
        auto it = nameMap.find(s);
        if (it == nameMap.end())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::indexOutOfRange, "Unknown BlasType");
        }

        return it->second;
    }
} // namespace utilities
} // namespace ell
