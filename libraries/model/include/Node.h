////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Node.h (model)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <utilities/include/IArchivable.h>
#include <utilities/include/MemoryLayout.h>
#include <utilities/include/PropertyBag.h>
#include <utilities/include/UniqueId.h>

#include <ostream>
#include <string>
#include <vector>

namespace ell
{
/// <summary> model namespace </summary>
namespace model
{
    class InputPortBase;
    class MapCompiler;
    class Model;
    class ModelTransformer;
    class OutputPortBase;
    class Port;

    /// <summary> Superclass for all node types. </summary>
    class Node : public utilities::IArchivable
    {
    public:
        static constexpr const char* defaultOutputPortName = "output";
        static constexpr const char* defaultInputPortName = "input";
        static constexpr const char* defaultInput1PortName = "input1";
        static constexpr const char* defaultInput2PortName = "input2";
        static constexpr const char* defaultInput3PortName = "input3";

        Node() = default;

        /// <summary> Type to use for our node id </summary>
        using NodeId = utilities::UniqueId;

        /// <summary> Returns the unique ID for this node </summary>
        ///
        /// <returns> The unique ID for this node </returns>
        const NodeId GetId() const { return _id; }

        /// <summary> Returns the number of input ports for this node </summary>
        ///
        /// <returns> The number of input ports </returns>
        int NumInputPorts() const { return static_cast<int>(_inputs.size()); }

        /// <summary> Returns the number of output ports for this node </summary>
        ///
        /// <returns> The number of output ports </returns>
        int NumOutputPorts() const { return static_cast<int>(_outputs.size()); }

        /// <summary> Returns the input ports for this node </summary>
        ///
        /// <returns> The input "ports" for this node </returns>
        const std::vector<InputPortBase*>& GetInputPorts() const { return _inputs; }

        /// <summary> Returns the named input port </summary>
        ///
        /// <param name="portName"> The name of the port </param>
        /// <returns> A pointer to the port </returns>
        InputPortBase* GetInputPort(const std::string& portName);

        /// <summary> Returns the named input port </summary>
        ///
        /// <param name="portName"> The name of the port </param>
        /// <returns> A pointer to the port </returns>
        const InputPortBase* GetInputPort(const std::string& portName) const;

        /// <summary> Returns an input port by index </summary>
        ///
        /// <param name="portIndex"> The index of the port </param>
        /// <returns> A pointer to the port </returns>
        InputPortBase* GetInputPort(size_t portIndex);

        /// <summary> Returns an input port by index </summary>
        ///
        /// <param name="portIndex"> The index of the port </param>
        /// <returns> A pointer to the port </returns>
        const InputPortBase* GetInputPort(size_t portIndex) const;

        /// <summary> Returns true if the node can accept input with this memory layout order, else false </summary>
        ///
        /// <param name="order"> The memory layout order for all the input ports </summary>
        /// <returns> If the node can accept the input memory layout order, true, else false </returns>
        virtual bool CanAcceptInputLayout(const utilities::DimensionOrder& order) const;

        /// <summary> Returns the output "ports" for this node </summary>
        ///
        /// <returns> The output "ports" for this node </returns>
        const std::vector<OutputPortBase*>& GetOutputPorts() const { return _outputs; }

        /// <summary> Returns the named output port </summary>
        ///
        /// <param name="portName"> The name of the port </param>
        /// <returns> A pointer to the port </returns>
        OutputPortBase* GetOutputPort(const std::string& portName);

        /// <summary> Returns the named output port </summary>
        ///
        /// <param name="portName"> The name of the port </param>
        /// <returns> A pointer to the port </returns>
        const OutputPortBase* GetOutputPort(const std::string& portName) const;

        /// <summary> Returns an output port by index </summary>
        ///
        /// <param name="portIndex"> The index of the port </param>
        /// <returns> A pointer to the port </returns>
        OutputPortBase* GetOutputPort(size_t portIndex);

        /// <summary> Returns an output port by index </summary>
        ///
        /// <param name="portIndex"> The index of the port </param>
        /// <returns> A pointer to the port </returns>
        const OutputPortBase* GetOutputPort(size_t portIndex) const;

        /// <summary> Attempts to set the memory layout order of all the output ports </summary>
        ///
        /// <param name="order"> The memory layout order to be applied to all the output ports </summary>
        /// <returns> If the node supports the output memory layout order, true, else false </returns>
        virtual bool TrySetOutputLayout(const utilities::DimensionOrder& order);

        /// <summary> Returns the named port </summary>
        ///
        /// <param name="portName"> The name of the port </param>
        /// <returns> A pointer to the port </returns>
        Port* GetPort(const std::string& portName);

        /// <summary> Returns the named port </summary>
        ///
        /// <param name="portName"> The name of the port </param>
        /// <returns> A const pointer to the port </returns>
        const Port* GetPort(const std::string& portName) const;

        /// <summary> Get all nodes that this nodes uses for input (and therefore depends on) </summary>
        ///
        /// <returns> a vector of all the nodes used for input </summary>
        std::vector<const Node*> GetParentNodes() const;

        /// <summary> Get all nodes that depend (directly) on us </summary>
        ///
        /// <returns> a vector of all the nodes that depend on this node </summary>
        std::vector<const Node*> GetDependentNodes() const;

        /// <summary> Returns the model this node is part of </summary>
        Model* GetModel() const { return _model.get(); }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "Node"; }

        /// <summary> Indicates if this node is able to compile itself to code. </summary>
        virtual bool IsCompilable(const MapCompiler* compiler) const
        {
            UNUSED(compiler);
            return false;
        }

        /// <summary> Print a human-readable representation of the Node. </summary>
        ///
        /// <param name="os"> The stream to write data to. </param>
        virtual void Print(std::ostream& os) const;

        /// <summary> Computes the output of this node and stores it in the output ports </summary>
        virtual void Compute() const = 0;

        /// <summary> Resets any state on the node, if any </summary>
        virtual void Reset() {}

        /// <summary> Get this object's metadata object. </summary>
        ///
        /// <returns> A reference to the PropertyBag containing the metadata for this object. </returns>
        utilities::PropertyBag& GetMetadata() { return _metadata; }

        /// <summary> Get this object's metadata object. </summary>
        ///
        /// <returns> A const reference to the PropertyBag containing the metadata for this object. </returns>
        const utilities::PropertyBag& GetMetadata() const { return _metadata; }

        /// <summary> Get a friendly name for this node from the metadata key "name". </summary>
        ///
        /// <returns> A friendly name or empty string if none was defined. </returns>
        std::string GetFriendlyName() const;

    protected:
        Node(const std::vector<InputPortBase*>& inputs, const std::vector<OutputPortBase*>& outputs);

        virtual bool HasState() const { return true; }

        void AddInputPort(InputPortBase* input);
        void AddOutputPort(OutputPortBase* output);

        utilities::ArchiveVersion GetArchiveVersion() const override;
        bool CanReadArchiveVersion(const utilities::ArchiveVersion& version) const override;
        // We're supplying a base implementation from WriteToArchive and ReadFromArchive, but also
        // declaring them as abstract so that subclasses need to implement this themselves.
        void WriteToArchive(utilities::Archiver& archiver) const override = 0;
        void ReadFromArchive(utilities::Unarchiver& archiver) override = 0;

    private:
        friend class Model;
        friend class ModelEditor;
        friend class ModelTransformer;

        virtual void Copy(ModelTransformer& transformer) const = 0;
        virtual bool Refine(ModelTransformer& transformer) const;

        void SetId(Node::NodeId id);
        void SetModel(Model* model);
        void UpdateInputPorts();

        std::unique_ptr<Model> _model;
        NodeId _id;
        std::vector<InputPortBase*> _inputs;
        std::vector<OutputPortBase*> _outputs;

        utilities::PropertyBag _metadata;
    };
} // namespace model
} // namespace ell
