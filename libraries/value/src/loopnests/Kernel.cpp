////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Kernel.cpp (value)
//  Authors:  Chuck Jacobs, Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "loopnests/Kernel.h"

namespace ell
{
namespace value
{
    namespace loopnests
    {
        Kernel::Kernel(std::string name) :
            _id(name),
            _kernelName(name)
        {}

        Kernel::Kernel(std::string name, Id id) :
            _id(id.empty() ? name : id),
            _kernelName(name)
        {}

        Kernel& Kernel::Inputs(const std::vector<Value>& inputs)
        {
            _inputs = inputs;
            return *this;
        }

        Kernel& Kernel::Indices(std::vector<Index> indices)
        {
            _indices = indices;
            return *this;
        }

        const std::string& Kernel::GetName() const
        {
            return _kernelName;
        }

        const Kernel::Id& Kernel::GetId() const
        {
            return _id;
        }

        const std::vector<Value>& Kernel::GetArgs() const
        {
            return _inputs;
        }

        const std::vector<Index>& Kernel::GetIndices() const
        {
            return _indices;
        }

        void Kernel::Call(std::vector<Value> inputs, std::vector<Value> indices) const
        {
            assert(_kernel);
            _kernel(inputs, indices);
        }

        // TODO : make this a template specialization of Define(), currently lambdas and std::functions aren't
        // getting matched correctly
        Kernel& Kernel::DefineEx(std::function<void(std::vector<Value>, std::vector<Scalar>)>&& fn)
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

                // Flatten the vectors of parameters into a single vector in order to define the emitted function
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
                    // Add a function layer to coalesce the vectors of parameters inside the function call
                    fn.Define([originalArgCount = arguments.size(),
                               originalIndexCount = indices.size(),
                               innerFn = std::move(fnDefinition)](std::vector<Value> args) {
                        if (args.size() != originalArgCount + originalIndexCount)
                        {
                            throw InputException(InputExceptionErrors::invalidArgument, "Number of arguments + indices does not match number of expected inputs");
                        }

                        std::vector<Value> inputs;
                        inputs.reserve(originalArgCount);
                        for (unsigned idx = 0; idx < originalArgCount; ++idx)
                        {
                            inputs.push_back(args[idx]);
                        }

                        std::vector<Scalar> parameters;
                        parameters.reserve(originalIndexCount);
                        for (unsigned idx = originalArgCount; idx < (originalArgCount + originalIndexCount); ++idx)
                        {
                            parameters.push_back(args[idx]);
                        }

                        innerFn(std::move(inputs), std::move(parameters));
                    });
                }

                fn.Call(fnInputs);
            };

            return *this;
        }
    } // namespace loopnests
} // namespace value
} // namespace ell

using namespace ell::value::loopnests;

std::hash<Kernel>::result_type std::hash<Kernel>::operator()(const argument_type& kernel) const
{
    return static_cast<size_t>(std::hash<Kernel::Id>()(kernel.GetId()));
}
