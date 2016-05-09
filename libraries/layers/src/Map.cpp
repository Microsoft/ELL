////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Machine Learning Library (EMLL)
//  File:     Map.cpp (layers)
//  Authors:  Ofer Dekel, Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Map.h"
#include "Model.h"

// utility
#include "StlIndexValueIterator.h"

// stl
#include <stdexcept>
#include <vector>
#include <memory>
#include <algorithm>
#include <utility>

namespace layers
{
    //
    // Map::OutputIterator implementation
    //
    bool Map::OutputIterator::IsValid() const
    {
        return _index < _outputs.size();
    }

    void Map::OutputIterator::Next()
    {
        if (IsValid())
        {
            ++_index;
        }
    }

    linear::IndexValue Map::OutputIterator::Get() const
    {
        return{ _index, _outputs[_index] };
    }

    Map::OutputIterator::OutputIterator(std::vector<double> outputs) : _outputs(std::move(outputs)), _index(0)
    {}

    //
    // Map class implementataion
    //
    Map::Map(const Model& model, const CoordinateList& outputCoordinateList) : _model(model), _outputCoordinateList(outputCoordinateList)
    {
        uint64_t inputLayerSize = std::max(model.GetRequiredLayerSize(0), outputCoordinateList.GetRequiredLayerSize(0));

        // allocate reusable memory needed to compute the map
        _layerOutputs.resize(_model.NumLayers());
        _layerOutputs[0].resize(inputLayerSize);

        for (uint64_t layerIndex = 1; layerIndex < _model.NumLayers(); ++layerIndex)
        {
            auto layerSize = _model.GetLayer(layerIndex).GetOutputDimension();
            _layerOutputs[layerIndex].resize(layerSize);
        }
    }
    
    std::vector<double> Map::Compute(const std::vector<double>& inputs) const
    {
        auto inputIterator = utilities::MakeStlIndexValueIterator(inputs);
        InitializeLayerOutputs(inputIterator, _layerOutputs[0]);
        
        // compute layers 1,2,... in order
        for (uint64_t i = 1; i < _model.NumLayers(); ++i)
        {
            std::cout << "About to init layer " << i << " of type " << _model.GetLayer(i).GetRuntimeTypeName() << std::endl;
            _model.GetLayer(i).Compute(_layerOutputs, _layerOutputs[i]);
            std::cout << "...done" << std::endl;
        }
        std::cout << "Done initializing inputs" << std::endl;

        // copy the outputs to a vector
        auto outputSize = _outputCoordinateList.Size();
        std::vector<double> outputs(outputSize);
        for(uint64_t index = 0; index < outputSize; ++index)
        {
            auto coordinate = _outputCoordinateList[index];
            auto layerIndex = coordinate.GetLayerIndex();
            auto elementIndex = coordinate.GetElementIndex();
            outputs[index] = _layerOutputs[layerIndex][elementIndex];
        }        

        return outputs;
    }

    const CoordinateList& Map::GetOutputCoordinateList() const
    {
        return _outputCoordinateList;
    }

    const Model& Map::GetModel() const 
    {
        return _model;
    }
}
