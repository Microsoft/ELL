////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     LinearPredictorNode.h (nodes)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "BinaryOperationNode.h"
#include "ConstantNode.h"
#include "DotProductNode.h"
#include "LinearPredictorNode.h"

#include <data/include/DenseDataVector.h>

#include <model/include/Model.h>
#include <model/include/ModelTransformer.h>
#include <model/include/Node.h>
#include <model/include/PortElements.h>

#include <predictors/include/LinearPredictor.h>

#include <utilities/include/Exception.h>

#include <string>
#include <vector>

namespace ell
{
namespace nodes
{
    /// <summary> A node that represents a linear predictor. </summary>
    template <typename ElementType>
    class LinearPredictorNode : public model::Node
    {
    public:
        /// @name Input and Output Ports
        /// @{
        static constexpr const char* weightedElementsPortName = "weightedElements";
        const model::InputPort<ElementType>& input = _input;
        const model::OutputPort<ElementType>& output = _output;
        const model::OutputPort<ElementType>& weightedElements = _weightedElements;
        /// @}

        using LinearPredictorType = typename predictors::LinearPredictor<ElementType>;

        /// <summary> Default Constructor </summary>
        LinearPredictorNode();

        /// <summary> Constructor </summary>
        ///
        /// <param name="input"> The signal to predict from </param>
        /// <param name="predictor"> The linear predictor to use when making the prediction. </param>
        LinearPredictorNode(const model::OutputPort<ElementType>& input, const LinearPredictorType& predictor);

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return utilities::GetCompositeTypeName<ElementType>("LinearPredictorNode"); }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

        /// <summary> Refines this node in the model being constructed by the transformer </summary>
        bool Refine(model::ModelTransformer& transformer) const override;

    protected:
        void Compute() const override;
        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;

    private:
        void Copy(model::ModelTransformer& transformer) const override;

        // Inputs
        model::InputPort<ElementType> _input;

        // Output
        model::OutputPort<ElementType> _output;
        model::OutputPort<ElementType> _weightedElements;

        // Linear predictor
        LinearPredictorType _predictor;
    };

    /// <summary> Convenience function to add a linear predictor node. </summary>
    ///
    /// <typeparam name="ElementType"> The fundamental type used by this predictor. </typeparam>
    /// <param name="input"> The input to the predictor. </param>
    /// <param name="predictor"> The linear predictor. </param>
    ///
    /// <returns> The output of the new node. </returns>
    template <typename ElementType>
    const model::OutputPort<ElementType>& LinearPredictor(const model::OutputPort<ElementType>& input,
                                                          const predictors::LinearPredictor<ElementType>& predictor);

    /// <summary> Adds a linear predictor node to a model transformer. </summary>
    ///
    /// <typeparam name="ElementType"> The fundamental type used by this predictor. </typeparam>
    /// <param name="input"> The input to the predictor. </param>
    /// <param name="predictor"> The linear predictor. </param>
    /// <param name="transformer"> [in,out] The model transformer. </param>
    ///
    /// <returns> The node added to the model. </returns>
    template <typename ElementType>
    LinearPredictorNode<ElementType>* AddNodeToModelTransformer(const model::PortElements<ElementType>& input, const predictors::LinearPredictor<ElementType>& predictor, model::ModelTransformer& transformer);
} // namespace nodes
} // namespace ell

#pragma region implementation

namespace ell
{
namespace nodes
{
    template <typename ElementType>
    LinearPredictorNode<ElementType>::LinearPredictorNode() :
        Node({ &_input }, { &_output, &_weightedElements }),
        _input(this, {}, defaultInputPortName),
        _output(this, defaultOutputPortName, 1),
        _weightedElements(this, weightedElementsPortName, 0)
    {
    }

    template <typename ElementType>
    LinearPredictorNode<ElementType>::LinearPredictorNode(const model::OutputPort<ElementType>& input, const predictors::LinearPredictor<ElementType>& predictor) :
        Node({ &_input }, { &_output, &_weightedElements }),
        _input(this, input, defaultInputPortName),
        _output(this, defaultOutputPortName, 1),
        _weightedElements(this, weightedElementsPortName, input.Size()),
        _predictor(predictor)
    {
        if (input.Size() != predictor.Size())
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "LinearPredictorNode: input size must match the predictor size");
        }
    }

    template <typename ElementType>
    void LinearPredictorNode<ElementType>::WriteToArchive(utilities::Archiver& archiver) const
    {
        Node::WriteToArchive(archiver);
        archiver[defaultInputPortName] << _input;
        archiver["weightedElements"] << _weightedElements;
        archiver["predictor"] << _predictor;
    }

    template <typename ElementType>
    void LinearPredictorNode<ElementType>::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        Node::ReadFromArchive(archiver);
        archiver[defaultInputPortName] >> _input;
        archiver["weightedElements"] >> _weightedElements;
        archiver["predictor"] >> _predictor;
    }

    template <typename ElementType>
    void LinearPredictorNode<ElementType>::Copy(model::ModelTransformer& transformer) const
    {
        const auto& newPortElements = transformer.GetCorrespondingInputs(_input);
        auto newNode = transformer.AddNode<LinearPredictorNode>(newPortElements, _predictor);
        transformer.MapNodeOutput(output, newNode->output);
        transformer.MapNodeOutput(weightedElements, newNode->weightedElements);
    }

    template <typename ElementType>
    bool LinearPredictorNode<ElementType>::Refine(model::ModelTransformer& transformer) const
    {
        const auto& newPortElements = transformer.GetCorrespondingInputs(_input);

        const auto& weights = Constant(transformer, _predictor.GetWeights().ToArray());
        const auto& scaledInput = Multiply(weights, newPortElements);
        const auto& dotProduct = DotProduct(weights, newPortElements);
        const auto& bias = Constant(transformer, _predictor.GetBias());
        const auto& sum = Add(dotProduct, bias);

        transformer.MapNodeOutput(output, sum);
        transformer.MapNodeOutput(weightedElements, scaledInput);
        return true;
    }

    template <typename ElementType>
    void LinearPredictorNode<ElementType>::Compute() const
    {
        using DataVectorType = typename LinearPredictorType::DataVectorType;
        auto inputDataVector = DataVectorType(_input.GetValue());
        _output.SetOutput({ _predictor.Predict(inputDataVector) });
        auto weightedData = _predictor.GetWeightedElements(inputDataVector).ToArray();
        if constexpr (std::is_same_v<ElementType, double>)
        {
            _weightedElements.SetOutput(weightedData);
        }
        else if constexpr (std::is_same_v<ElementType, float>)
        {
            std::vector<float> castedData;
            castedData.reserve(weightedData.size());
            std::transform(weightedData.begin(), weightedData.end(), std::back_inserter(castedData), [](double d) { return static_cast<float>(d); });
            _weightedElements.SetOutput(castedData);
        }
        else
        {
            throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented);
        }
    }

    template <typename ElementType>
    const model::OutputPort<ElementType>& LinearPredictor(const model::OutputPort<ElementType>& input,
                                                          const predictors::LinearPredictor<ElementType>& predictor)
    {
        model::Model* model = input.GetNode()->GetModel();
        if (model == nullptr)
        {
            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Input not part of a model");
        }

        auto node = model->AddNode<LinearPredictorNode<ElementType>>(input, predictor);
        return node->output;
    }

    template <typename ElementType>
    LinearPredictorNode<ElementType>* AddNodeToModelTransformer(const model::PortElements<ElementType>& input, const predictors::LinearPredictor<ElementType>& predictor, model::ModelTransformer& transformer)
    {
        return transformer.AddNode<LinearPredictorNode<ElementType>>(input, predictor);
    }
} // namespace nodes
} // namespace ell

#pragma endregion implementation
