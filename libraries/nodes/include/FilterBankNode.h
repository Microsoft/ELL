////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     FilterBankNode.h (nodes)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <model/include/CompilableNode.h>
#include <model/include/IRMapCompiler.h>
#include <model/include/InputPort.h>
#include <model/include/MapCompiler.h>
#include <model/include/ModelTransformer.h>
#include <model/include/Node.h>
#include <model/include/OutputPort.h>

#include <dsp/include/FilterBank.h>

#include <utilities/include/TypeName.h>
#include <utilities/include/TypeTraits.h>

#include <cmath>
#include <string>
#include <vector>

namespace ell
{
namespace nodes
{
    /// <summary>
    /// Base class for nodes that perform elementwise multiply between a set of triangular filters and the input frequency response.
    /// This can be useful as a way to sample different frequency bands in an FFT output to form a type of spectrogram.
    /// Each value in the FilterBankNode output is the result of convolving the FFT output with a triangular filter, with some width,
    /// centered at some location on the FFT output.  As an example, imagine we have a 10-element input, and a filter of width 6 centered
    /// over the 6th input:
    ///
    /// |                   ^               |
    /// |                  /|\              |
    /// |                 / | \             |
    /// |                /  |  \            |
    /// |               /   |   \           |
    /// |              /    |    \          |
    /// |             /     |     \         |
    /// |            /      |      \        |
    /// |           /       |       \       |
    /// |          /        |        \      |
    /// |         /         |         \     |
    /// |        /          |          \    |
    /// |---|---|-.-|-.-|-.-|-.-|-.-|-.-|---|
    /// 0   1   2   3   4   5   6   7   8   9
    ///
    /// then the result from this one filter would be the following (where I is the input vector):
    ///    (I[2] * 0) + (I[3] * 0.333...) + (I[4] * 0.666) + (I[5] * 1) + (I[6] * 0.666) + (I[7] * 0.333) + (I[8] * 0)
    ///
    /// the idea then is the filters can overlap to create smooth samples of each band in the input, and the output then is sized to
    /// the number of filters.  The implementation is optimized on the assumption that each triangle is a relatively small slice of
    /// the input such that it is faster to compute each triangle than to do a dot products for each filter against the entire input.
    /// </summary>
    template <typename ValueType>
    class FilterBankNode : public model::CompilableNode
    {
    public:
        /// @name Input and Output Ports
        /// @{
        const model::InputPort<ValueType>& input = _input;
        const model::OutputPort<ValueType>& output = _output;
        /// @}

    protected:
        /// <summary> Construct a FilterBankeNode from the given filters </summary>
        FilterBankNode(const dsp::TriangleFilterBank& filters);

        /// <summary> Construct a FilterBankeNode from an output port, the given filters</summary>
        FilterBankNode(const model::OutputPort<ValueType>& input, const dsp::TriangleFilterBank& filters);

        void Compute() const override;
        void Compile(model::IRMapCompiler& compiler, emitters::IRFunctionEmitter& function) override;
        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;
        bool HasState() const override { return true; } // Stored state: filters

        // Inputs
        model::InputPort<ValueType> _input;

        // Output
        model::OutputPort<ValueType> _output;

    private:
        // FilterBank
        const dsp::TriangleFilterBank& _filters;
        ValueType _offset;
    };

    /// <summary>
    /// A node that applies a linearly-spaced filter bank to an FFT output
    /// </summary>
    template <typename ValueType>
    class LinearFilterBankNode : public FilterBankNode<ValueType>
    {
    public:
        /// @name Input and Output Ports
        /// @{
        using FilterBankNode<ValueType>::input;
        using FilterBankNode<ValueType>::output;
        /// @}

        /// <summary> Default Constructor </summary>
        LinearFilterBankNode();

        /// <summary> Constructor </summary>
        ///
        /// <param name="input"> The signal to process. </param>
        LinearFilterBankNode(const model::OutputPort<ValueType>& input, const dsp::LinearFilterBank& filters);

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return utilities::GetCompositeTypeName<ValueType>("LinearFilterBankNode"); }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

    protected:
        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;

    private:
        using FilterBankNode<ValueType>::_input;
        using FilterBankNode<ValueType>::_output;

        void Copy(model::ModelTransformer& transformer) const override;

        // FilterBank
        dsp::LinearFilterBank _linearFilters;
    };

    /// <summary>
    /// A node that applies a mel filter bank to an FFT output
    /// </summary>
    template <typename ValueType>
    class MelFilterBankNode : public FilterBankNode<ValueType>
    {
    public:
        /// @name Input and Output Ports
        /// @{
        using FilterBankNode<ValueType>::input;
        using FilterBankNode<ValueType>::output;
        /// @}

        /// <summary> Default Constructor </summary>
        MelFilterBankNode();

        /// <summary> Constructor </summary>
        ///
        /// <param name="input"> The signal to process. </param>
        MelFilterBankNode(const model::OutputPort<ValueType>& input, const dsp::MelFilterBank& filters);

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return utilities::GetCompositeTypeName<ValueType>("MelFilterBankNode"); }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

    protected:
        void WriteToArchive(utilities::Archiver& archiver) const override;
        void ReadFromArchive(utilities::Unarchiver& archiver) override;

    private:
        using FilterBankNode<ValueType>::_input;
        using FilterBankNode<ValueType>::_output;

        void Copy(model::ModelTransformer& transformer) const override;

        // FilterBank
        dsp::MelFilterBank _melFilters;
    };
} // namespace nodes
} // namespace ell
