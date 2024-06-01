////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ConstantNode.h (nodes)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <model/include/CompilableNode.h>
#include <model/include/CompilableNodeUtilities.h>
#include <model/include/IRMapCompiler.h>
#include <model/include/MapCompiler.h>
#include <model/include/ModelTransformer.h>
#include <model/include/OutputPort.h>
#include <model/include/PortElements.h>

#include <emitters/include/IRFunctionEmitter.h>

#include <predictors/include/ConstantPredictor.h>

#include <utilities/include/IArchivable.h>
#include <utilities/include/TypeName.h>
#include <utilities/include/TypeTraits.h>

#include <memory>
#include <vector>

namespace ell
{
/// <summary> nodes namespace </summary>
namespace nodes
{
    /// <summary> A node that contains a constant value. Has no inputs. </summary>
    template <typename ValueType>
    class ConstantNode : public model::CompilableNode
    {
    public:
        /// @name Input and Output Ports
        /// @{
        const model::OutputPort<ValueType>& output = _output;
        /// @}

        /// <summary> Default Constructor </summary>
        ConstantNode();

        /// <summary> Constructor for a scalar constant </summary>
        ///
        /// <param name="value"> The scalar value </param>
        ConstantNode(ValueType value);

        /// Constructor for a vector constant
        ///
        /// <param name="value"> The vector value </param>
        ConstantNode(const std::vector<ValueType>& value);

        /// Constructor for an arbitrary-shaped array constant
        ///
        /// <param name="value"> The vector value </param>
        /// <param name="shape"> The shape of the output data </param>
        ConstantNode(const std::vector<ValueType>& value, const model::MemoryShape& shape);

        /// Constructor for an arbitrary-shaped array constant
        ///
        /// <param name="value"> The vector value </param>
        /// <param name="layout"> The memory layout of the output data </param>
        ConstantNode(const std::vector<ValueType>& value, const model::PortMemoryLayout& layout);

        /// <summary> Gets the values contained in this node </summary>
        ///
        /// <returns> The values contained in this node </returns>
        const std::vector<ValueType>& GetValues() const { return _values; }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return utilities::GetCompositeTypeName<ValueType>("ConstantNode"); }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

    protected:
        void Compute() const override;
        void Compile(model::IRMapCompiler& compiler, emitters::IRFunctionEmitter& function) override;
        bool HasState() const override { return true; }
        bool ShouldCompileInline() const override { return true; }
        utilities::ArchiveVersion GetArchiveVersion() const override;
        bool CanReadArchiveVersion(const utilities::ArchiveVersion& version) const override;
        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;

    private:
        void Copy(model::ModelTransformer& transformer) const override;

        // Output
        model::OutputPort<ValueType> _output;

        // Constant value
        std::vector<ValueType> _values;
    };

    /// <summary> Convenience function for adding a ConstantNode to a model. </summary>
    ///
    /// <param name="model"> The Model or ModelTransformer to add the node to. </param>
    /// <param name="value"> The scalar value </param>
    ///
    /// <returns> The output of the new node. </returns>
    template <typename ValueType, typename ModelLikeType>
    const model::OutputPort<ValueType>& Constant(ModelLikeType& model, ValueType value);

    /// <summary> Convenience function for adding a ConstantNode to a model. </summary>
    ///
    /// <param name="model"> The Model or ModelTransformer to add the node to. </param>
    /// <param name="value"> The vector value </param>
    ///
    /// <returns> The output of the new node. </returns>
    template <typename ValueType, typename ModelLikeType>
    const model::OutputPort<ValueType>& Constant(ModelLikeType& model, const std::vector<ValueType>& value);

    /// <summary> Convenience function for adding a ConstantNode to a model. </summary>
    ///
    /// <param name="model"> The Model or ModelTransformer to add the node to. </param>
    /// <param name="value"> The vector value </param>
    /// <param name="shape"> The shape of the output data </param>
    ///
    /// <returns> The output of the new node. </returns>
    template <typename ValueType, typename ModelLikeType>
    const model::OutputPort<ValueType>& Constant(ModelLikeType& model, const std::vector<ValueType>& value, const model::MemoryShape& shape);

    /// <summary> Convenience function for adding a ConstantNode to a model. </summary>
    ///
    /// <param name="model"> The Model or ModelTransformer to add the node to. </param>
    /// <param name="value"> The vector value </param>
    /// <param name="layout"> The memory layout of the output data </param>
    ///
    /// <returns> The output of the new node. </returns>
    template <typename ValueType, typename ModelLikeType>
    const model::OutputPort<ValueType>& Constant(ModelLikeType& model, const std::vector<ValueType>& value, const model::PortMemoryLayout& layout);

    /// <summary> Adds a constant node (which represents a constant predictor) to a model transformer. </summary>
    ///
    /// <param name="input"> The input to the predictor, which is ignored. </param>
    /// <param name="predictor"> The constant predictor. </param>
    /// <param name="transformer"> [in,out] The model transformer. </param>
    ///
    /// <returns> The node added to the model. </returns>
    ConstantNode<double>* AddNodeToModelTransformer(const model::PortElements<double>& input, const predictors::ConstantPredictor& predictor, model::ModelTransformer& transformer);
} // namespace nodes
} // namespace ell

#pragma region implementation

namespace ell
{
namespace nodes
{
    // superclass (Node) constructor takes two array arguments: inputs and outputs. These are pointers to our local InputPort and OutputPort storage.

    // Default constructor
    template <typename ValueType>
    ConstantNode<ValueType>::ConstantNode() :
        CompilableNode({}, { &_output }),
        _output(this, defaultOutputPortName, 0){};

    // Constructor for a scalar constant
    template <typename ValueType>
    ConstantNode<ValueType>::ConstantNode(ValueType value) :
        CompilableNode({}, { &_output }),
        _output(this, defaultOutputPortName, 1),
        _values({ value }){};

    // Constructor for a vector constant
    template <typename ValueType>
    ConstantNode<ValueType>::ConstantNode(const std::vector<ValueType>& values) :
        CompilableNode({}, { &_output }),
        _output(this, defaultOutputPortName, values.size()),
        _values(values){};

    template <typename ValueType>
    ConstantNode<ValueType>::ConstantNode(const std::vector<ValueType>& values, const model::MemoryShape& shape) :
        CompilableNode({}, { &_output }),
        _output(this, defaultOutputPortName, shape),
        _values(values){};

    template <typename ValueType>
    ConstantNode<ValueType>::ConstantNode(const std::vector<ValueType>& values, const model::PortMemoryLayout& layout) :
        CompilableNode({}, { &_output }),
        _output(this, defaultOutputPortName, layout),
        _values(values){};

    template <typename ValueType>
    void ConstantNode<ValueType>::Compute() const
    {
        _output.SetOutput(_values);
    }

    template <typename ValueType>
    void ConstantNode<ValueType>::Copy(model::ModelTransformer& transformer) const
    {
        auto newNode = transformer.AddNode<ConstantNode<ValueType>>(_values, _output.GetMemoryLayout());
        transformer.MapNodeOutput(output, newNode->output);
    }

    template <typename ValueType>
    void ConstantNode<ValueType>::Compile(model::IRMapCompiler& compiler, emitters::IRFunctionEmitter& function)
    {
        auto values = this->GetValues();
        emitters::Variable* pVar = nullptr;
        pVar = function.GetModule().Variables().AddVariable<emitters::LiteralVectorVariable<ValueType>>(values);
        compiler.SetVariableForPort(output, pVar); // Just set the variable corresponding to the output port to be the global variable we created
    }

    template <typename ValueType>
    utilities::ArchiveVersion ConstantNode<ValueType>::GetArchiveVersion() const
    {
        return utilities::ArchiveVersionNumbers::v8_port_memory_layout;
    }

    template <typename ValueType>
    bool ConstantNode<ValueType>::CanReadArchiveVersion(const utilities::ArchiveVersion& version) const
    {
        return version <= utilities::ArchiveVersionNumbers::v8_port_memory_layout;
    }

    template <typename ValueType>
    void ConstantNode<ValueType>::WriteToArchive(utilities::Archiver& archiver) const
    {
        Node::WriteToArchive(archiver);
        archiver["values"] << _values;
        archiver["layout"] << _output.GetMemoryLayout();
    }

    template <typename ValueType>
    void ConstantNode<ValueType>::ReadFromArchive(utilities::Unarchiver& archiver)
    {
        Node::ReadFromArchive(archiver);
        archiver["values"] >> _values;
        model::PortMemoryLayout layout;
        archiver["layout"] >> layout;
        _output.SetMemoryLayout(layout);
    }

    template <typename ValueType, typename ModelLikeType>
    const model::OutputPort<ValueType>& Constant(ModelLikeType& model, ValueType value)
    {
        static_assert(utilities::IsOneOf<ModelLikeType, model::Model, model::ModelTransformer>, "'model' parameter must be a model::Model or model::ModelTransformer");
        auto node = model.template AddNode<ConstantNode<ValueType>>(value);
        return node->output;
    }

    template <typename ValueType, typename ModelLikeType>
    const model::OutputPort<ValueType>& Constant(ModelLikeType& model, const std::vector<ValueType>& values)
    {
        static_assert(utilities::IsOneOf<ModelLikeType, model::Model, model::ModelTransformer>, "'model' parameter must be a model::Model or model::ModelTransformer");
        auto node = model.template AddNode<ConstantNode<ValueType>>(values);
        return node->output;
    }

    template <typename ValueType, typename ModelLikeType>
    const model::OutputPort<ValueType>& Constant(ModelLikeType& model, const std::vector<ValueType>& values, const ell::model::MemoryShape& shape)
    {
        static_assert(utilities::IsOneOf<ModelLikeType, model::Model, model::ModelTransformer>, "'model' parameter must be a model::Model or model::ModelTransformer");
        auto node = model.template AddNode<ConstantNode<ValueType>>(values, shape);
        return node->output;
    }

    template <typename ValueType, typename ModelLikeType>
    const model::OutputPort<ValueType>& Constant(ModelLikeType& model, const std::vector<ValueType>& values, const ell::model::PortMemoryLayout& layout)
    {
        static_assert(utilities::IsOneOf<ModelLikeType, model::Model, model::ModelTransformer>, "'model' parameter must be a model::Model or model::ModelTransformer");
        auto node = model.template AddNode<ConstantNode<ValueType>>(values, layout);
        return node->output;
    }
} // namespace nodes
} // namespace ell

#pragma endregion implementation
