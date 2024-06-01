////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     main.cpp (makeExamples)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "GenerateModels.h"
#include "ModelGenerateArguments.h"

#include <common/include/LoadModel.h>

#include <utilities/include/CommandLineParser.h>
#include <utilities/include/Files.h>

#include <iostream>

using namespace ell;

void SaveModels(const std::string& ext, std::string outputPath)
{
    if (outputPath.empty())
    {
        outputPath = ".";
    }

    ell::utilities::EnsureDirectoryExists(outputPath);

    common::SaveModel(GenerateIdentityModel(3), outputPath + "/identity." + ext);
    common::SaveModel(GenerateTimesTwoModel(3), outputPath + "/times_two." + ext);
    common::SaveModel(GenerateIsEqualModel(), outputPath + "/is_equal." + ext);
    common::SaveModel(GenerateArgMaxModel(3), outputPath + "/arg_max." + ext);

    common::SaveModel(GenerateModel1(), outputPath + "/model_1." + ext);
    common::SaveModel(GenerateModel2(), outputPath + "/model_2." + ext);
    common::SaveModel(GenerateModel3(), outputPath + "/model_3." + ext);

    common::SaveModel(GenerateTreeModel(0), outputPath + "/tree_0." + ext);
    common::SaveModel(GenerateTreeModel(1), outputPath + "/tree_1." + ext);
    common::SaveModel(GenerateTreeModel(2), outputPath + "/tree_2." + ext);
    common::SaveModel(GenerateTreeModel(3), outputPath + "/tree_3." + ext);

    common::SaveModel(GenerateRefinedTreeModel(0), outputPath + "/refined_tree_0." + ext);
    common::SaveModel(GenerateRefinedTreeModel(1), outputPath + "/refined_tree_1." + ext);
    common::SaveModel(GenerateRefinedTreeModel(2), outputPath + "/refined_tree_2." + ext);
    common::SaveModel(GenerateRefinedTreeModel(3), outputPath + "/refined_tree_3." + ext);

    common::SaveModel(GenerateMultiOutModel(3), outputPath + "/multi_out." + ext);

    common::SaveModel(GenerateBroadcastTimesTwoModel<float>(256), outputPath + "/broadcast_times_two." + ext);
}

int main(int argc, char* argv[])
{
    try
    {
        ParsedModelGenerateArguments arguments;

        // create a command line parser
        utilities::CommandLineParser commandLineParser(argc, argv);

        // add arguments to the command line parser
        commandLineParser.AddOptionSet(arguments);

        // parse command line
        commandLineParser.Parse();

        // generate models
        SaveModels("model", arguments.outputPath);
    }
    catch (const utilities::CommandLineParserPrintHelpException& exception)
    {
        std::cout << exception.GetHelpText() << std::endl;
        return 0;
    }
    catch (const utilities::CommandLineParserErrorException& exception)
    {
        std::cerr << "Command line parse error:" << std::endl;
        for (const auto& error : exception.GetParseErrors())
        {
            std::cerr << error.GetMessage() << std::endl;
        }
        return 1;
    }
    catch (const utilities::Exception& exception)
    {
        std::cerr << "ERROR, got ELL exception. Message: " << exception.GetMessage() << std::endl;
        throw;
    }
    catch (const std::exception& exception)
    {
        std::cerr << "ERROR, got std exception. Message: " << exception.what() << std::endl;
        throw;
    }
    catch (...)
    {
        std::cerr << "ERROR, got unknown exception." << std::endl;
        throw;
    }
}
