////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     BroadcastOperationNodes.cpp (nodes)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "BroadcastOperationNodes.h"

#include <optional>

namespace ell
{
namespace nodes
{
    // Returns a MemoryLayout that's broadcast-compatible with each of given arguments. That is, for each dimension, it has
    // the same size as the argument, or '1'. If there is no layout that's compatible with all the arguments (i.e., at least
    // one pair of arguments have differing non-1 sizes for some dimension), then return std::nullopt.
    std::optional<utilities::MemoryLayout> GetBroadcastCompatibleLayout(const std::vector<utilities::MemoryLayout>& arguments)
    {
        if (arguments.empty())
        {
            return std::nullopt;
        }

        std::vector<int> resultSize(arguments[0].NumDimensions(), 1);
        for (const auto& layout : arguments)
        {
            // inflatedLayout will hold the layout for this argument, expanded to the same number of
            // dimensions as the current result (if necessary).
            auto inflatedLayout = layout;
            int extraDimensions = layout.NumDimensions() - inflatedLayout.NumDimensions();
            if (extraDimensions > 0)
            {
                // if the argument layout has more dimensions than the current result, append extra '1'
                // values to the beginning (slow-moving) dimensions of the result.
                resultSize.insert(resultSize.begin(), static_cast<size_t>(extraDimensions), 1);
            }
            else if (extraDimensions < 0)
            {
                // if the argument layout has fewer dimensions than the current result, inflate the argument
                // layout to be the same size as the current result.
                inflatedLayout = layout.CopyWithExtraDimensions(-extraDimensions);
            }

            // Now inflatedLayout and the current result have the same number of dimensions.
            // Loop through them and check for compatibility. They're compatible if they're
            // both the same, or if at least one of them is a '1'.
            for (int d = 0; d < inflatedLayout.NumDimensions(); ++d)
            {
                auto x1 = resultSize[d];
                auto x2 = inflatedLayout.GetLogicalDimensionActiveSize(d);
                auto minmax = std::minmax(x1, x2);
                if (minmax.first != 1 && minmax.first != minmax.second)
                {
                    return std::nullopt;
                }

                resultSize[d] = minmax.second;
            }
        }

        utilities::MemoryLayout result(resultSize);
        return result;
    }

    utilities::MemoryLayout ComputeBroadcastedLayout(const std::vector<utilities::MemoryLayout>& arguments)
    {
        auto resultLayout = GetBroadcastCompatibleLayout(arguments);

        // Get broadcast-compatible memory layout for all arguments
        if (!resultLayout)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Error: arguments have incompatible shapes for broadcasting");
        }

        return resultLayout.value();
    }

    utilities::MemoryLayout ComputeBroadcastedLayout(const std::vector<const model::OutputPortBase*>& arguments)
    {
        if (arguments.empty())
        {
            return utilities::ScalarLayout;
        }

        std::vector<utilities::MemoryLayout> layouts;
        for (const auto& arg : arguments)
        {
            layouts.push_back(arg->GetMemoryLayout());
        }
        return ComputeBroadcastedLayout(layouts);
    }

    void VerifyLayoutsCompatible(const std::vector<utilities::MemoryLayout>& inputs)
    {
        auto resultLayout = GetBroadcastCompatibleLayout(inputs);

        // Get broadcast-compatible memory layout for all arguments
        if (!resultLayout)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Error: arguments have incompatible shapes for broadcasting");
        }
    }
} // namespace nodes
} // namespace ell
