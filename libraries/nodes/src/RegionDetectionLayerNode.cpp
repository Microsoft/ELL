///////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     RegionDetectionLayer.cpp (neural)
//  Authors:  Kern Handa
//
///////////////////////////////////////////////////////////////////////////////

#include "RegionDetectionLayerNode.h"
#include "ActivationFunctions.h"
#include "ActivationLayerNode.h"

#include <emitters/include/IRMath.h>

#include "ActivationFunctions.h"

namespace ell
{
namespace nodes
{
    template <typename ValueType>
    RegionDetectionLayerNode<ValueType>::RegionDetectionLayerNode(const model::OutputPort<ValueType>& input, const predictors::neural::RegionDetectionLayer<ValueType>& layer) :
        BaseType(input, layer)
    {
    }

    template <typename ValueType>
    bool RegionDetectionLayerNode<ValueType>::Refine(model::ModelTransformer& transformer) const
    {
        const auto& newInput = transformer.GetCorrespondingInputs(this->_input);

        const auto& detectionParams = this->_layer.GetDetectionParameters();

        auto detectionNode = transformer.AddNode<RegionDetectionNode<ValueType>>(newInput,
                                                                                 detectionParams,
                                                                                 this->GetInputMemoryLayout(),
                                                                                 this->GetOutputMemoryLayout());

        transformer.MapNodeOutput(this->output, detectionNode->output);

        return false;
    }

    template <typename ValueType>
    void RegionDetectionLayerNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInputs = transformer.GetCorrespondingInputs(this->_input);
        auto newNode = transformer.AddNode<RegionDetectionLayerNode<ValueType>>(newInputs, this->_layer);
        transformer.MapNodeOutput(this->_output, newNode->output);
    }

    //
    // RegionDetectionNode
    //
    template <typename ValueType>
    RegionDetectionNode<ValueType>::RegionDetectionNode() :
        RegionDetectionNode({}, {}, {}, {})
    {
    }

    template <typename ValueType>
    RegionDetectionNode<ValueType>::RegionDetectionNode(const model::OutputPort<ValueType>& input,
                                                        predictors::neural::RegionDetectionParameters params,
                                                        const model::PortMemoryLayout& inputMemoryLayout,
                                                        const model::PortMemoryLayout& outputMemoryLayout) :
        CompilableNode({ &_input }, { &_output }),
        _input(this, input, defaultInputPortName),
        _params(std::move(params)),
        _output(this, defaultOutputPortName, outputMemoryLayout),
        _inputMemoryLayout(inputMemoryLayout)
    {
    }

    template <typename ValueType>
    void RegionDetectionNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newInput = transformer.GetCorrespondingInputs(_input);
        auto newNode = transformer.AddNode<RegionDetectionNode>(newInput, _params, GetInputMemoryLayout(), GetOutputMemoryLayout());
        transformer.MapNodeOutput(output, newNode->output);
    }

    template <typename ValueType>
    void RegionDetectionNode<ValueType>::Compile(model::IRMapCompiler& compiler, emitters::IRFunctionEmitter& function)
    {
        // the stride between each box in a cell (numAnchors + 1 + numClasses)
        // the stride of each cell (numBoxesPerCell * boxStride)

        std::vector<int> expectedStride{ _params.width, _params.height, _params.numBoxesPerCell * (_params.numAnchors + 1 + _params.numClasses) };
        if (GetInputMemoryLayout().GetExtent() != expectedStride)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input to region detection layer has incorrect shape");
        }

        if (GetOutputMemoryLayout().GetExtent() != expectedStride)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Output of region detection layer has incorrect shape");
        }

        auto input = function.LocalTensor(compiler.EnsurePortEmitted(this->input), GetInputMemoryLayout().GetExtent().ToVector(), emitters::RowMajorTensorLayout);
        auto output = function.LocalTensor(compiler.EnsurePortEmitted(this->output), GetOutputMemoryLayout().GetExtent().ToVector(), emitters::RowMajorTensorLayout);

        function.For(_params.width, [input, output, params = _params](auto& fn, auto i) {
            fn.For(params.height, [input, output, params, i](auto& fn, auto j) {
                fn.For(params.numBoxesPerCell, [input, output, params, i, j](auto& fn, auto k) {
                    auto numClasses = params.numClasses;
                    auto numAnchors = params.numAnchors;

                    auto boxStride = params.numAnchors + 1 + params.numClasses;
                    auto boxOffset = k * boxStride;
                    auto confidenceOffset = boxOffset + numAnchors;
                    auto classProbabilityOffset = confidenceOffset + 1;

                    for (int anchorIndex = 0; anchorIndex < numAnchors; ++anchorIndex)
                    {
                        output({ i, j, boxOffset + anchorIndex }) = input({ i, j, boxOffset + anchorIndex });
                    }

                    SigmoidActivationFunction<ValueType> sigmoid;
                    emitters::IRLocalScalar e = input({ i, j, confidenceOffset });
                    output({ i, j, confidenceOffset }) = sigmoid.Compile(fn, e);

                    // output[classProbabilityOffset : classProbabilityOffset + numClasses] = softmax(input[classProbabilityOffset : classProbabilityOffset + numClasses])
                    if (params.applySoftmax)
                    {
                        // set the initial max value to the first element
                        auto classProbMax = fn.LocalScalar(fn.Variable(emitters::GetVariableType<ValueType>()));
                        fn.Store(classProbMax, static_cast<emitters::IRLocalScalar>(input({ i, j, classProbabilityOffset })));

                        // find the max value in the rest of the probabilities
                        fn.For(classProbabilityOffset + 1, classProbabilityOffset + numClasses, [classProbMax, input, i, j](auto& fn, auto c) {
                            emitters::IRLocalScalar val = input({ i, j, c });
                            fn.If(val > fn.Load(classProbMax), [classProbMax, val](auto& fn) {
                                fn.Store(classProbMax, val);
                            });
                        });

                        // Calculate the sum of the all the euler values, which are calculated using the max above
                        auto sum = fn.Variable(emitters::GetVariableType<ValueType>());
                        fn.Store(sum, fn.Literal(ValueType{ 0 }));
                        fn.For(classProbabilityOffset, classProbabilityOffset + numClasses, [sum, classProbMax, input, output, i, j](auto& fn, auto c) {
                            emitters::IRLocalScalar inputVal = input({ i, j, c });
                            auto eulerVal = Exp(inputVal - fn.Load(classProbMax));
                            fn.Store(sum, eulerVal + fn.Load(sum));
                            output({ i, j, c }) = eulerVal;
                        });

                        // Divide each element by the sum
                        fn.For(classProbabilityOffset, classProbabilityOffset + numClasses, [sum, output, i, j](auto& fn, auto c) {
                            output({ i, j, c }) = (output({ i, j, c }) / fn.Load(sum));
                        });
                    }
                    else
                    {
                        fn.template MemoryCopy<ValueType>(input.PointerTo({ i, j, classProbabilityOffset }), output.PointerTo({ i, j, classProbabilityOffset }), fn.LocalScalar(numClasses));
                    }
                });
            });
        });
    }

    // Explicit instantiations
    template class RegionDetectionLayerNode<float>;
    template class RegionDetectionLayerNode<double>;
    template class RegionDetectionNode<float>;
    template class RegionDetectionNode<double>;
} // namespace nodes
} // namespace ell
