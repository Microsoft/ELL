////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Kernel.h (value)
//  Authors:  Chuck Jacobs, Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "Index.h"

#include "../FunctionDeclaration.h"
#include "../Scalar.h"
#include "../Value.h"

#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

namespace ell
{
namespace value
{
    namespace loopnests
    {
        /// <summary>
        /// Code that runs inside the loop nest. This is the code that actually implements the operation. The loops and
        /// scheduling are all for the purpose of calling this code.
        /// </summary>
        class Kernel
        {
        public:
            using Id = std::string;

            explicit Kernel(std::string name);
            Kernel(std::string name, Id id);

            template <typename... Types>
            Kernel& Inputs(Types... inputs);

            Kernel& Inputs(const std::vector<Value>& inputs);

            Kernel& Indices(std::vector<Index> indices);

            template <typename... Types>
            Kernel& Indices(Types... indices);

            template <typename Fn>
            Kernel& Define(Fn&& fn);

            // TODO : make this a template specialization of Define(), currently lambdas and std::functions aren't
            // getting matched correctly
            Kernel& DefineEx(std::function<void(std::vector<Value>, std::vector<Scalar>)>&& fn);

            void Call(std::vector<Value> arguments, std::vector<Value> indices) const;

            const std::string& GetName() const;
            const Id& GetId() const;
            const std::vector<Value>& GetArgs() const;
            const std::vector<Index>& GetIndices() const;

        private:
            Id _id;
            std::string _kernelName;
            std::vector<Value> _inputs;
            std::vector<Index> _indices;
            std::function<void(std::vector<Value> arguments, std::vector<Value> indices)> _kernel;
        };

        inline bool operator==(const Kernel& i1, const Kernel& i2)
        {
            return i1.GetId() == i2.GetId();
        }
        inline bool operator!=(const Kernel& i1, const Kernel& i2)
        {
            return !(i1 == i2);
        }
    } // namespace loopnests
} // namespace value
} // namespace ell

namespace std
{
/// <summary> Implements a hash function for the Kernel class, so that it can be used with associative containers (maps, sets, and the like). </summary>
template <>
struct hash<::ell::value::loopnests::Kernel>
{
    using argument_type = ::ell::value::loopnests::Kernel;
    using result_type = std::size_t;
    result_type operator()(const argument_type& kernel) const;
};
} // namespace std

#pragma region implementation

namespace ell
{
namespace value
{
    namespace loopnests
    {
        template <typename... Types>
        Kernel& Kernel::Inputs(Types... inputs)
        {
            static_assert(std::conjunction_v<std::is_convertible<Types, ViewAdapter>...>);
            return Inputs(std::vector<Value>{ ViewAdapter{ std::forward<Types>(inputs) }... });
        }

        template <typename... Types>
        Kernel& Kernel::Indices(Types... indices)
        {
            static_assert(utilities::AllSame<Index, std::decay_t<Types>...>);
            return Indices(std::vector<Index>{ std::forward<Types>(indices)... });
        }

        template <typename Fn>
        Kernel& Kernel::Define(Fn&& fn)
        {
            _kernel = [numOriginalIndices = _indices.size(),
                       originalInputs = _inputs,
                       kernelName = UniqueName(_kernelName + "KernelFn"),
                       fnDefinition = std::move(fn)](std::vector<Value> arguments, std::vector<Value> indices) {
                using namespace utilities;

                if (arguments.size() != originalInputs.size())
                {
                    throw InputException(InputExceptionErrors::invalidArgument, "Number of arguments does not match number of expected inputs");
                }
                if (indices.size() != numOriginalIndices)
                {
                    throw InputException(InputExceptionErrors::invalidArgument, "Number of indices does not match number of expected indices");
                }

                std::vector<ViewAdapter> fnInputs(arguments.begin(), arguments.end());
                fnInputs.insert(fnInputs.end(), indices.begin(), indices.end());

                std::vector<ViewAdapter> fnParameters(originalInputs.begin(), originalInputs.end());
                fnParameters.insert(fnParameters.end(), indices.begin(), indices.end());
                for (auto i = 0u; i < originalInputs.size(); ++i)
                {
                    Value& param = fnParameters[i];
                    const Value& input = fnInputs[i];

                    if (!input.IsConstrained())
                    {
                        param.ClearLayout();
                    }
                    else
                    {
                        param.SetLayout(input.GetLayout());
                    }
                }

                auto fn = DeclareFunction(kernelName).Parameters(fnParameters);
                fn.Inlined(FunctionInlining::always);
                if (!fn.IsDefined())
                {
                    fn.Define(fnDefinition);
                }

                fn.Call(fnInputs);
            };

            return *this;
        }

    } // namespace loopnests
} // namespace value
} // namespace ell
#pragma endregion
