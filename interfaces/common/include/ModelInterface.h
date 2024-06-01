////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     ModelInterface.h (interfaces)
//  Authors:  Chuck Jacobs, Kirk Olynyk, Chris Lovett
//
////////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#ifndef SWIG

#include "CallbackInterface.h"
#include "MathInterface.h"
#include "NeuralNetworkPredictorInterface.h"
#include "Ports.h"

#include <emitters/include/EmitterTypes.h>

#include <math/include/Tensor.h>

#include <model/include/IRCompiledMap.h>
#include <model/include/InputNode.h>
#include <model/include/InputPort.h>
#include <model/include/Model.h>
#include <model/include/Node.h>
#include <model/include/OutputNode.h>
#include <model/include/OutputPort.h>
#include <model/include/Port.h>
#include <model/include/PortElements.h>
#include <model/include/PortMemoryLayout.h>

#include <nodes/include/BinaryOperationNode.h>
#include <nodes/include/SinkNode.h>
#include <nodes/include/SourceNode.h>
#include <nodes/include/UnaryOperationNode.h>

#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#endif

namespace ELL_API
{

//
// Forward declarations
//
class AutoDataVector;
class CompiledMap;
class Map;
class InputPort;
class Model;
class Node;
class NodeIterator;
class OutputPort;
class PortMemoryLayout;
struct MapCompilerOptions;
struct ModelOptimizerOptions;

//
// Operation types for unary operation nodes
//
enum class UnaryOperationType
{
    none = (int)ell::nodes::UnaryOperationType::none,
    abs = (int)ell::nodes::UnaryOperationType::abs,
    exp = (int)ell::nodes::UnaryOperationType::exp,
    hardSigmoid = (int)ell::nodes::UnaryOperationType::hardSigmoid,
    hardTanh = (int)ell::nodes::UnaryOperationType::hardTanh,
    log = (int)ell::nodes::UnaryOperationType::log,
    log10 = (int)ell::nodes::UnaryOperationType::log10,
    logicalNot = (int)ell::nodes::UnaryOperationType::logicalNot,
    sin = (int)ell::nodes::UnaryOperationType::sin,
    sigmoid = (int)ell::nodes::UnaryOperationType::sigmoid,
    sign = (int)ell::nodes::UnaryOperationType::sign,
    softmax = (int)ell::nodes::UnaryOperationType::softmax,
    square = (int)ell::nodes::UnaryOperationType::square,
    cos = (int)ell::nodes::UnaryOperationType::cos,
    sqrt = (int)ell::nodes::UnaryOperationType::sqrt,
    tanh = (int)ell::nodes::UnaryOperationType::tanh,
    min = (int)ell::nodes::UnaryOperationType::min,
    max = (int)ell::nodes::UnaryOperationType::max,
};

//
// Operation types for binary operation nodes
//
enum class BinaryOperationType
{
    none = (int)ell::nodes::BinaryOperationType::none,
    add = (int)ell::nodes::BinaryOperationType::add,
    subtract = (int)ell::nodes::BinaryOperationType::subtract,
    multiply = (int)ell::nodes::BinaryOperationType::multiply, // elementwise only
    divide = (int)ell::nodes::BinaryOperationType::divide,
    logicalAnd = (int)ell::nodes::BinaryOperationType::logicalAnd,
    logicalOr = (int)ell::nodes::BinaryOperationType::logicalOr,
    logicalXor = (int)ell::nodes::BinaryOperationType::logicalXor,
    maximum = (int)ell::nodes::BinaryOperationType::maximum,
    minimum = (int)ell::nodes::BinaryOperationType::minimum
};

//
// NodeIterator
//
class NodeIterator
{
public:
    NodeIterator() = default;
    bool IsValid();
    void Next();
    Node Get();
#ifndef SWIG
    NodeIterator(const std::vector<const ell::model::Node*>& nodes, std::shared_ptr<ell::model::Model> model);
    NodeIterator(ell::model::ForwardNodeIterator& other, std::shared_ptr<ell::model::Model> model);
    std::shared_ptr<ell::model::Model> GetModel() const { return _model; }
#endif
private:
    size_t _i = 0;
    bool _isVector = false;
    std::vector<const ell::model::Node*> _nodes;
    ell::model::ForwardNodeIterator _iterator;
    std::shared_ptr<ell::model::Model> _model;
};

//
// Node
//

class Node
{
public:
    Node() = default;
    Node(const Node&) = default;
    std::string GetId();
    Model GetModel();
    NodeIterator GetParents();
    NodeIterator GetDependents();
    OutputPort GetOutputPort(const std::string& portName);
    InputPort GetInputPort(const std::string& portName);
    Port GetPort(const std::string& portName);
    OutputPortIterator GetOutputPorts();
    InputPortIterator GetInputPorts();
    std::string GetRuntimeTypeName();
    std::string GetMetadataValue(const std::string& key);
    void SetMetadataValue(const std::string& key, const std::string& value);
    void CopyMetadataFrom(const Node& other);
    void ResetInput(PortElements newInput, std::string input_port_name = "input");
    void ResetInput(Node newInput, std::string input_port_name = "input");
#ifndef SWIG
    Node(const ell::model::Node* other, std::shared_ptr<ell::model::Model> model);
    const ell::model::Node* GetNode() const { return _node; }
#endif
private:
    const ell::model::Node* _node = nullptr;
    std::shared_ptr<ell::model::Model> _model;
};

//
// InputNode
//

class InputNode : public Node
{
public:
    InputNode(const InputNode& node, std::shared_ptr<ell::model::Model> model);
    InputNode(Node node);
    using Node::GetInputPort;
    using Node::GetOutputPort;
#ifndef SWIG
    InputNode() = default;
    InputNode(const ell::model::InputNodeBase* other, std::shared_ptr<ell::model::Model> model);
    const ell::model::InputNodeBase* GetInputNode() const;
#endif
};

//
// OutputNode
//

class OutputNode : public Node
{
public:
    OutputNode(const OutputNode& node, std::shared_ptr<ell::model::Model> model);
    OutputNode(Node node);
    using Node::GetInputPort;
    using Node::GetOutputPort;
#ifndef SWIG
    OutputNode() = default;
    OutputNode(const ell::model::OutputNodeBase* other, std::shared_ptr<ell::model::Model> model);
    const ell::model::OutputNodeBase* GetOutputNode() const;
#endif
};

//
// SourceNode
//

class SourceNode : public Node
{
public:
    SourceNode(const SourceNode& node, std::shared_ptr<ell::model::Model> model);

    using Node::GetInputPort;
    using Node::GetOutputPort;

    template <typename ElementType>
    void RegisterCallback(ell::api::CallbackBase<ElementType>& callback);
#ifndef SWIG
    SourceNode() = default;
    SourceNode(ell::model::SourceNodeBase* other, std::shared_ptr<ell::model::Model> model);
    ell::model::SourceNodeBase* GetSourceNode() const;

private:
    ell::model::SourceNodeBase* _sourceNode;
#endif
};

//
// SinkNode
//

class SinkNode : public Node
{
public:
    SinkNode(const SinkNode& node, std::shared_ptr<ell::model::Model> model);

    using Node::GetInputPort;
    using Node::GetOutputPort;

    template <typename ElementType>
    void RegisterCallback(ell::api::CallbackBase<ElementType>& callback);
#ifndef SWIG
    SinkNode() = default;
    SinkNode(ell::model::SinkNodeBase* other, std::shared_ptr<ell::model::Model> model);
    ell::model::SinkNodeBase* GetSinkNode() const;

private:
    ell::model::SinkNodeBase* _sinkNode;
#endif
};

class CallbackInfo
{
public:
    CallbackInfo() = default;
    CallbackInfo(std::string name, PortType dataType) :
        _name(name),
        _type(dataType)
    {
    }
    std::string GetName() const { return _name; }
    PortType GetDataType() const { return _type; }

private:
    std::string _name;
    PortType _type;
};

//
// Model
//
class Model
{
public:
    Model();
    Model(const std::string& filename);
    void Load(const std::string& filename);
    void LoadFromString(const std::string& str);
    void Save(const std::string& filename);
    size_t Size();
    NodeIterator GetNodes();
    std::string GetJson() const;

    // This provides a simpler version of ModelBuilder that hides Ports
    Node AddBinaryOperation(Node input1, Node input2, BinaryOperationType operation);
    Node AddBuffer(Node input, int windowSize);
    Node AddClock(Node input, double interval, double lagThreshold, const std::string& lagNotificationName);
    Node AddConcatenation(const PortMemoryLayout& outputMemoryLayout, const std::vector<Node>& inputs);
    Node AddConstant(std::vector<double> values, const PortMemoryLayout& outputMemoryLayout, PortType type);
    Node AddConstant(std::vector<double> values, PortType type);
    Node AddDCT(Node input, int numFilters);
    Node AddDotProduct(Node input1, Node input2);
    Node AddDTW(std::vector<std::vector<double>> prototype, Node input);
    Node AddFastGRNN(Node input, Node reset, size_t hiddenUnits, size_t wRank, size_t uRank, Node W1, Node W2, Node U1, Node U2, Node biasGate, Node biasUpdate, Node zeta, Node nu, ell::api::predictors::neural::ActivationType gateNonlinearity, ell::api::predictors::neural::ActivationType updateNonlinearity);
    Node AddFFT(Node input, int nfft = 0);
    Node AddGRU(Node input, Node reset, size_t hiddenUnits, Node inputWeights, Node hiddenWeights, Node inputBias, Node hiddenBias, ell::api::predictors::neural::ActivationType activation, ell::api::predictors::neural::ActivationType recurrentActivation);
    Node AddHammingWindow(Node input);
    Node AddIIRFilter(Node input, std::vector<double> bCoeffs, std::vector<double> aCoeffs);
    InputNode AddInput(const PortMemoryLayout& memoryLayout, PortType type);
    Node AddLinearFilterBank(Node input, double sampleRate, int numFilters, int numFiltersToUse, double offset = 0);
    Node AddLSTM(Node input, Node reset, size_t hiddenUnits, Node inputWeights, Node hiddenWeights, Node inputBias, Node hiddenBias, ell::api::predictors::neural::ActivationType activation, ell::api::predictors::neural::ActivationType recurrentActivation);
    Node AddMatrixMultiply(Node input1, Node input2);
    Node AddMelFilterBank(Node input, double sampleRate, int fftSize, int numFilters, int numFiltersToUse, double offset = 0);
    Node AddNeuralNetworkPredictor(Node input, ell::api::predictors::NeuralNetworkPredictor predictor);
    OutputNode AddOutput(const PortMemoryLayout& memoryLayout, Node input);
    OutputNode AddOutput(Node input);
    Node AddReinterpretLayout(Node input, PortMemoryLayout outputMemoryLayout);
    Node AddReorderData(Node input, std::vector<int> order);
    Node AddReorderData(Node input, PortMemoryLayout inputMemoryLayout, PortMemoryLayout outputMemoryLayout, std::vector<int> order = {}, double outputPaddingValue = 0.0);
    Node AddScalingNode(Node input, double scale);
    Node AddRNN(Node input, Node reset, size_t hiddenUnits, Node inputWeights, Node hiddenWeights, Node inputBias, Node hiddenBias, ell::api::predictors::neural::ActivationType activation);
    SinkNode AddSink(Node input, const PortMemoryLayout& memoryLayout, const std::string& sinkFunctionName, Node trigger = Node());
    SourceNode AddSource(Node input, PortType outputType, const PortMemoryLayout& memoryLayout, const std::string& sourceFunctionName);
    Node AddSplice(const std::vector<Node>& inputs);
    Node AddTypeCast(Node input, PortType outputType);
    Node AddUnaryOperation(Node input, UnaryOperationType operation);
    Node AddVoiceActivityDetector(Node input, double sampleRate, double frameDuration, double tauUp, double tauDown, double largeInput, double gainAtt, double thresholdUp, double thresholdDown, double levelThreshold);

    std::string GetMetadataValue(const std::string& key);
    void SetMetadataValue(const std::string& key, const std::string& value);
#ifndef SWIG
    Model(ell::model::Model&& other);
    Model(std::shared_ptr<ell::model::Model> underlyingModel);
    std::shared_ptr<ell::model::Model> GetModel() const;
#endif

private:
    std::shared_ptr<ell::model::Model> _model;
};

//
// Map
//
class Map
{
public:
    Map();
    Map(const Map& other) = default;
    Map(Model model, InputNode inputNode, PortElements output);
    Map(Model model, InputNode inputNode, OutputNode output);
    Map(Model model, const std::vector<InputNode*> inputNodes, const std::vector<PortElements*> outputs);
    Map(Model model, const std::vector<InputNode*> inputNodes, const std::vector<OutputNode*> outputs);
    Map(const std::string& filename);
#ifndef SWIG
    Map(std::shared_ptr<ell::model::Map>& map);
#endif

    void Save(const std::string& filename) const;
    void Load(const std::string& filename);

    int NumInputs() const;
    ell::api::math::TensorShape GetInputShape(int index = 0) const;
    PortMemoryLayout GetInputLayout(int index = 0) const;
    PortType GetInputType(int index = 0) const;

    int NumOutputs() const;
    ell::api::math::TensorShape GetOutputShape(int index = 0) const;
    PortMemoryLayout GetOutputLayout(int index = 0) const;
    PortType GetOutputType(int index = 0) const;

    Model GetModel() const;
    void Refine(int iterations);

    CompiledMap Compile(const std::string& targetDevice, const std::string& moduleName, const std::string& functionName, const MapCompilerOptions& compilerSettings, const ModelOptimizerOptions& optimizerSettings) const;

    // Return true if the model contains a SourceNode.  In this case you need
    // to register the callbacks via RegisterCallback method on the SourceNode.
    bool HasSourceNodes();

    std::string GetMetadataValue(const std::string& key);
    void SetMetadataValue(const std::string& key, const std::string& value);

    std::vector<CallbackInfo> GetSinkCallbackInfo();
    std::vector<CallbackInfo> GetSourceCallbackInfo();
    std::vector<CallbackInfo> GetLagCallbackInfo();

    template <typename ElementType>
    void Step(ell::api::TimeTickType timestamp = 0.0);

    void Reset();

    // Simple non callback API assumes you only have one input and one output.
    // Each language can decide to add more, for example, model_python_post.i adds a ComputeMultiple metehod
    // that supports multiple heterogeneously typed inputs and outputs.
    std::vector<double> ComputeDouble(const AutoDataVector& inputData);
    std::vector<double> ComputeDouble(const std::vector<double>& inputData);
    std::vector<float> ComputeFloat(const std::vector<float>& inputData);
    std::vector<int> ComputeInt(const std::vector<int>& inputData);
    std::vector<int64_t> ComputeInt64(const std::vector<int64_t>& inputData);

#ifndef SWIG
    std::shared_ptr<ell::model::Map> GetInnerMap() const
    {
        return _map;
    }
#endif

private:
    std::shared_ptr<ell::model::Map> _map;
    enum class TriState
    {
        Uninitialized,
        No,
        Yes
    };
    TriState _sourceNodeState = TriState::Uninitialized;
};

//
// CompiledMap
//
class CompiledMap
{
public:
    ~CompiledMap();

    void WriteIR(const std::string& filePath);
    void WriteBitcode(const std::string& filePath);
    void WriteSwigInterface(const std::string& filePath);
    std::string GetCodeString();

    template <typename ElementType>
    void Step(ell::api::TimeTickType timestamp = 0.0);

#ifndef SWIG
    CompiledMap() = default;
    CompiledMap(
        std::shared_ptr<ell::model::IRMapCompiler> compiler,
        std::shared_ptr<ell::model::IRCompiledMap> compiledMap,
        std::shared_ptr<ell::model::Map> map);

    std::shared_ptr<ell::model::Map> GetInnerMap() const { return _map; }
    std::shared_ptr<ell::model::IRCompiledMap> GetInnerCompiledMap() const { return _compiledMap; }
#endif

    // Return true if the model contains a SourceNode.  In this case you need
    // to register the callback via RegisterCallback method on the SourceNode.
    bool HasSourceNodes();

    int NumInputs() const;
    ell::api::math::TensorShape GetInputShape(int index = 0) const;
    PortMemoryLayout GetInputLayout(int index = 0) const;
    PortType GetInputType(int index = 0) const;

    int NumOutputs() const;
    ell::api::math::TensorShape GetOutputShape(int index = 0) const;
    PortMemoryLayout GetOutputLayout(int index = 0) const;
    PortType GetOutputType(int index = 0) const;

    // Older non callback based API, only makes sense when model has single input/output nodes and no source/sink nodes.
    std::vector<double> ComputeDouble(const std::vector<double>& inputData);
    std::vector<float> ComputeFloat(const std::vector<float>& inputData);
    std::vector<int> ComputeInt(const std::vector<int>& inputData);
    std::vector<int64_t> ComputeInt64(const std::vector<int64_t>& inputData);

    void Reset();

private:
    std::shared_ptr<ell::model::IRMapCompiler> _compiler;
    std::shared_ptr<ell::model::IRCompiledMap> _compiledMap;
    std::shared_ptr<ell::model::Map> _map;

    enum class TriState
    {
        Uninitialized,
        No,
        Yes
    };
    TriState _sourceNodeState = TriState::Uninitialized;
};

//
// Compiler options
//
struct MapCompilerOptions
{
    /// <summary> Optimize output code using LLVM. </summary>
    bool optimize = true;

    /// <summary> Emit profiling code, </summary>
    bool profile = false;

    /// <summary> Enable ELL's parallelization. </summary>
    bool parallelize = false;

    /// <summary> Use thread pool for parallelization (if parallelization enabled). </summary>
    bool useThreadPool = true;

    /// <summary> Maximum num of parallel threads. </summary>
    int maxThreads = 4;

    /// <summary> Allow emitting more efficient code that isn't necessarily IEEE-754 compatible. </summary>
    bool useFastMath = true;

    /// <summary> Allow printing of diagnostic messages from the compiled model. </summary>
    bool includeDiagnosticInfo = false;

    // Options that can be changed during code generation (e.g., per function)
    /// <summary> Emit code that calls an external BLAS library. </summary>
    bool useBlas = true;

    /// <summary> Explicitly unroll loops in certain cases. </summary>
    bool unrollLoops = false;

    /// <summary> Emit inline code for common operations. </summary>
    bool inlineOperators = true;

    /// <summary> Enable ELL's vectorization </summary>
    bool allowVectorInstructions = false;

    /// <summary> Size of vector units. </summary>
    int vectorWidth = 4;

    /// <summary> Emit debug code. </summary>
    bool debug = false;

    /// <summary> Skip ELLCode optimization. </summary>
    bool skip_ellcode = false;
};

//
// Optimizer settings
//
struct ModelOptimizerOptions
{
    bool fuseLinearFunctionNodes = true;
};

} // namespace ELL_API

#pragma region implementation

#ifndef SWIG

#include <model/include/IRMapCompiler.h>

#include <nodes/include/SinkNode.h>
#include <nodes/include/SourceNode.h>

namespace ELL_API
{

//
// Map
//

template <typename ElementType>
void SourceNode::RegisterCallback(ell::api::CallbackBase<ElementType>& callback)
{
    ell::nodes::SourceNode<ElementType>* node = dynamic_cast<ell::nodes::SourceNode<ElementType>*>(GetSourceNode());
    if (node != nullptr)
    {
        std::function<bool(std::vector<ElementType>&)> function([&callback](auto& input) {
            // Note: the caller is responsible for keeping the CallbackBase object alive until this point.
            return callback.Run(input);
        });
        node->SetSourceFunction(function);
    }
}

template <typename ElementType>
void SinkNode::RegisterCallback(ell::api::CallbackBase<ElementType>& callback)
{
    ell::nodes::SinkNode<ElementType>* node = dynamic_cast<ell::nodes::SinkNode<ElementType>*>(GetSinkNode());
    if (node != nullptr)
    {
        std::function<void(const std::vector<ElementType>&)> function([&callback](const std::vector<ElementType>& output) {
            // Reason for the const_cast:
            // SWIG warns that the const overload gets shadowed, so CallbackBase only
            // provides a non-const vector overload for Run.
            callback.Run(const_cast<std::vector<ElementType>&>(output));
        });
        node->SetSinkFunction(function);
    }
}

template <typename ElementType>
void Map::Step(ell::api::TimeTickType timestamp)
{
    std::vector<ElementType> input = { static_cast<ElementType>(timestamp) };
    _map->Compute<ElementType>(input);
}

//
// CompiledMap
//
template <typename ElementType>
void CompiledMap::Step(ell::api::TimeTickType timestamp)
{
    std::vector<ElementType> input = { static_cast<ElementType>(timestamp) };
    _compiledMap->Compute<ElementType>(input);
}

} // namespace ELL_API

#endif // SWIG

#pragma endregion implementation
