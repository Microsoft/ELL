////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     TransformData.h (finetune)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "DataUtils.h"
#include "ModelOutputDataCache.h"

#include <model/include/Model.h>
#include <model/include/OutputPort.h>
#include <model/include/Submodel.h>

namespace ell
{
UnlabeledDataContainer TransformDataWithSubmodel(const UnlabeledDataContainer& dataset, ell::model::Submodel& submodel);
UnlabeledDataContainer TransformDataWithSubmodel(const UnlabeledDataContainer& dataset, ell::model::Submodel& submodel, ModelOutputDataCache& dataCache, bool cacheResult = true);

// TODO: make these be TransformDataWithSubmodel
template <typename ElementType>
UnlabeledDataContainer TransformDataWithModel(const UnlabeledDataContainer& dataset, const ell::model::OutputPort<ElementType>& output);

template <typename ElementType>
BinaryLabelDataContainer TransformDataInputsWithModel(const BinaryLabelDataContainer& dataset, const ell::model::OutputPort<ElementType>& output);

template <typename ElementType>
MultiClassDataContainer TransformDataInputsWithModel(const MultiClassDataContainer& dataset, const ell::model::OutputPort<ElementType>& output);

template <typename ElementType>
VectorLabelDataContainer TransformDataInputsWithModel(const VectorLabelDataContainer& dataset, const ell::model::OutputPort<ElementType>& output);

double GetModelAccuracy(const ell::model::OutputPortBase& output, const BinaryLabelDataContainer& testDataset);
double GetModelAccuracy(const ell::model::OutputPortBase& output, const MultiClassDataContainer& testDataset);
double GetModelAccuracy(const ell::model::OutputPortBase& output, const VectorLabelDataContainer& testDataset);

template <typename DataVectorType>
DataVectorType RemovePadding(const DataVectorType& data, const ell::utilities::MemoryLayout& layout);
} // namespace ell