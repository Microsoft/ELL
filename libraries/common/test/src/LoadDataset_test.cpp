
//
// Loading map tests
//

#include "LoadMap_test.h"

// common
#include "DataLoaders.h"
#include "LoadModel.h"
#include "MapLoadArguments.h"
#include "DataLoadArguments.h"

// testing
#include "testing.h"

// stl
#include <iostream>

namespace emll
{

common::DataLoadArguments GetDataLoadArguments()
{
    common::DataLoadArguments args;
    args.inputDataFilename = "../../examples/data/testData.txt";
    return args;
}
void TestLoadDataset()
{
    auto dataLoadArguments = GetDataLoadArguments();
    auto dataset = common::GetDataset(dataLoadArguments);
}

void TestLoadMappedDataset()
{
    common::MapLoadArguments args;
    args.inputModelFile = "../../examples/data/model_1.json";
    args.modelInputsString = "";
    args.modelOutputsString = "1017.output";

    auto map = common::LoadMap(args);
    auto dataLoadArguments = GetDataLoadArguments();
    auto dataset = common::GetMappedDataset(dataLoadArguments, map);
}
}
