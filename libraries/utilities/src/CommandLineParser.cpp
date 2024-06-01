////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CommandLineParser.cpp (utilities)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CommandLineParser.h"
#include "Logger.h"
#include "Unused.h"

#include <algorithm>
#include <sstream>

namespace ell
{
namespace utilities
{
    //
    // ParsedArgSet class
    //
    void ParsedArgSet::AddArgs(CommandLineParser& parser)
    {
        UNUSED(parser);
    }

    CommandLineParseResult ParsedArgSet::PostProcess(const CommandLineParser& parser)
    {
        UNUSED(parser);
        return CommandLineParseResult();
    }

    //
    // CommandLineParseResult class
    //
    CommandLineParseResult::CommandLineParseResult(bool ok) :
        _isOK(ok)
    {
    }

    CommandLineParseResult::CommandLineParseResult(const char* message) :
        _isOK(false)
    {
        _messages.emplace_back(message);
    }

    CommandLineParseResult::CommandLineParseResult(const std::string& message) :
        _isOK(false)
    {
        _messages.emplace_back(message);
    }

    CommandLineParseResult::CommandLineParseResult(const std::vector<std::string>& messages) :
        _messages(messages)
    {
        _isOK = _messages.size() == 0;
    }

    //
    // ParseError class
    //
    ParseError::ParseError(const std::string& message) :
        _message(message)
    {
    }

    //
    // OptionInfo internal class
    //
    CommandLineParser::OptionInfo::OptionInfo(std::string name, std::string shortName, std::string description, std::string defaultValue, std::string emptyValueString, std::function<bool(std::string)> setValueCallback) :
        name(name),
        shortName(shortName),
        description(description),
        defaultValueString(defaultValue),
        emptyValueString(emptyValueString),
        setValueCallbacks({ setValueCallback })
    {
    }

    std::string CommandLineParser::OptionInfo::optionNameString() const
    {
        if (shortName == "")
        {
            return name + " [" + defaultValueString + "]";
        }
        else
        {
            return name + " (-" + shortName + ") [" + defaultValueString + "]";
        }
    }

    size_t CommandLineParser::OptionInfo::optionNameHelpLength() const
    {
        size_t len = name.size() + 2;
        if (shortName != "")
        {
            len += (shortName.size() + 4);
        }

        len += defaultValueString.size() + 3; // 3 for " [" + "]" at begin/end

        const size_t maxNameLen = 32;
        return std::min(maxNameLen, len);
    }

    //
    // CommandLineParser class
    //
    CommandLineParser::CommandLineParser(int argc, char* argv[])
    {
        SetArgs(argc, const_cast<const char**>(argv));
    }

    CommandLineParser::CommandLineParser(int argc, const char* argv[])
    {
        SetArgs(argc, argv);
    }

    void CommandLineParser::SetArgs(int argc, const char* argv[])
    {
        _originalArgs.clear();
        _originalArgs.insert(_originalArgs.end(), &argv[0], &argv[argc]);

        std::string exe_path = _originalArgs[0];
        size_t slashPos = exe_path.find_last_of("/\\");
        if (slashPos == std::string::npos)
        {
            _exeName = exe_path;
        }
        else
        {
            _exeName = exe_path.substr(slashPos + 1);
        }
    }

    void CommandLineParser::Parse()
    {
        if (_originalArgs.size() == 0)
            return;

        bool hasHelpOption = HasOption("help");
        bool hasVerboseOption = HasOption("verbose");

        if (!hasHelpOption || !hasVerboseOption)
        {
            AddDocumentationString("");
            AddDocumentationString("General options");
        }

        bool printHelpAndExit = false;
        if (!hasHelpOption)
        {
            if (HasShortName("h"))
            {
                AddOption(printHelpAndExit, "help", "", "Print help and exit", false);
            }
            else
            {
                AddOption(printHelpAndExit, "help", "h", "Print help and exit", false);
            }
        }

        bool setVerbose = false;
        if (!hasVerboseOption)
        {
            if (HasShortName("v"))
            {
                AddOption(setVerbose, "verbose", "", "Verbose output", false);
            }
            else
            {
                AddOption(setVerbose, "verbose", "v", "Verbose output", false);
            }
        }

        // While we're parsing the arguments, we may add new conditional options. If we do so, we need
        // to reparse the inputs in case some earlier commandline option referred to one of these new
        // options. So we repeatedly parse the command line text until we haven't added any new conditional options.

        bool needsReparse = true;
        while (needsReparse)
        {
            std::set<std::string, case_insensitive_comparer> unset_args;
            for (const auto& opt : _options)
            {
                if (opt.second.enabled)
                {
                    unset_args.insert(opt.first);
                }
            }

            needsReparse = false;
            size_t argc = _originalArgs.size();
            for (size_t index = 1; index < argc; index++)
            {
                std::string arg = _originalArgs[index];
                if (arg == "--") // "--" is the special separator for passthrough args
                {
                    while (++index < argc)
                    {
                        arg = _originalArgs[index];
                        _passthroughArgs.push_back(arg);
                    }
                }
                else if (arg[0] == '-') // it's an option
                {
                    std::string option;
                    if (arg[1] == '-') // long name
                    {
                        option = std::string(arg.begin() + 2, arg.end());
                    }
                    else // short name
                    {
                        std::string shortName = std::string(arg.begin() + 1, arg.end());
                        if (!HasShortName(shortName))
                        {
                            throw CommandLineParserErrorException("Unknown option", { std::string("Error: unknown option ") + arg });
                        }
                        option = _shortToLongNameMap[shortName];
                    }

                    if (!HasOption(option) || !_options[option].enabled)
                    {
                        throw CommandLineParserErrorException("Unknown option", { std::string("Error: unknown option ") + arg });
                    }
                    else
                    {
                        unset_args.erase(option);
                        if (index < argc - 1)
                        {
                            std::string val = _originalArgs[index + 1];
                            if (val[0] == '-')
                            {
                                // next token in an option, so use the default unset-value string
                                needsReparse = SetOption(option) || needsReparse;
                            }
                            else
                            {
                                needsReparse = SetOption(option, val) || needsReparse;
                                index++;
                            }
                        }
                        else // this is the last thing on the line --- assume it's a shortcut for --option true
                        {
                            needsReparse = SetOption(option) || needsReparse;
                        }
                    }
                }
                else
                {
                    _positionalArgs.push_back(arg);
                }
            }

            // Need to set default args here, in case one of them enables a conditional argument set
            needsReparse = SetDefaultArgs(unset_args) || needsReparse;
        }

        // Finally, invoke the post-parse callbacks
        bool isValid = true;
        std::vector<ParseError> parseErrors;
        for (const auto& callback : _postParseCallbacks)
        {
            auto callbackResult = callback(*this);
            if (!callbackResult) // callbackResult is an error
            {
                isValid = false;
                for (auto message : callbackResult._messages)
                {
                    parseErrors.emplace_back(message);
                }
            }
        }

        if (printHelpAndExit)
        {
            throw CommandLineParserPrintHelpException(GetHelpString());
        }

        if (setVerbose)
        {
            logging::ShouldLog() = true;
        }

        if (!isValid)
        {
            throw CommandLineParserErrorException("Error in parse callback", parseErrors);
        }
    }

    CommandLineParser::OptionInfo* CommandLineParser::FindOption(const std::string& name)
    {
        // case insensitive matching.
        auto iter = _options.find(name);
        if (iter == _options.end())
        {
            return nullptr;
        }

        return &iter->second;
    }

    bool CommandLineParser::SetDefaultArgs(const std::set<std::string, case_insensitive_comparer>& unset_args)
    {
        bool needsReparse = false;
        for (std::string argumentName : unset_args)
        {
            // look up arg
            auto iter = _options.find(argumentName);
            if (iter != _options.end())
            {
                const OptionInfo& arg_info = iter->second;
                std::string default_val = arg_info.defaultValueString;
                needsReparse = SetOption(argumentName, default_val) || needsReparse;
            }
        }
        return needsReparse;
    }

    void CommandLineParser::AddOption(std::string& option, std::string name, std::string shortName, std::string description, std::initializer_list<std::string> enumValues, std::string defaultValue, std::string emptyValueString)
    {
        // transform initializer list into useful things that will stick around
        std::vector<std::string> valueNameStrings;
        std::vector<std::pair<std::string, std::string>> valueNamesTable;
        for (auto v : enumValues)
        {
            valueNameStrings.push_back(v);
            valueNamesTable.push_back({ v, v });
        }

        auto callback = [&option, this, name, valueNamesTable](std::string optionVal) {
            std::string optionString;
            bool didParse = ParseVal(optionVal, valueNamesTable, option, optionString);
            if (didParse)
            {
                _options[name].currentValueString = optionString;
                return true;
            }
            else
            {
                throw utilities::InputException(utilities::InputExceptionErrors::badStringFormat, "Could not parse value for option " + name);
                return false;
            }
        };

        OptionInfo info(name, shortName, description, defaultValue, emptyValueString, callback);
        info.enumValues = valueNameStrings;
        AddOption(info);
    }

    bool CommandLineParser::HasOption(std::string name)
    {
        // case insensitive matching.
        return _options.find(name) != _options.end();
    }

    bool CommandLineParser::HasShortName(std::string shortName)
    {
        return _shortToLongNameMap.find(shortName) != _shortToLongNameMap.end();
    }

    void CommandLineParser::AddOption(const OptionInfo& info)
    {
        if (HasOption(info.name))
        {
            auto messageStr = std::string{ "Error: adding same option more than once (" } + info.name + ")";
            throw CommandLineParserInvalidOptionsException(messageStr.c_str());
        }

        if (HasShortName(info.shortName))
        {
            auto messageStr = std::string{ "Error: adding same short name more than once (" } + info.shortName + ")";
            throw CommandLineParserInvalidOptionsException(messageStr.c_str());
        }

        _options[info.name] = info;
        _docEntries.emplace_back(DocumentationEntry::Type::option, info.name);

        if (info.shortName != "")
        {
            _shortToLongNameMap[info.shortName] = info.name;
        }
    }

    void CommandLineParser::AddPostParseCallback(const PostParseCallback& callback)
    {
        _postParseCallbacks.push_back(callback);
    }

    inline bool FindBestMatch(std::string str, const std::vector<std::string>& val_names, std::string& resultString)
    {
        bool didFindOne = false;
        for (const auto& valName : val_names)
        {
            bool matches = false;
            if (str == "")
            {
                // find always finds "", so we need to be more precise here
                matches = (valName == "");
            }
            else if (valName.find(str) != std::string::npos)
            {
                matches = true;
            }
            if (matches)
            {
                if (didFindOne)
                {
                    // duplicate match, so it is ambiguous.
                    return false;
                }
                resultString = valName;
                didFindOne = true;
            }
        }

        return didFindOne;
    }

    bool CommandLineParser::SetOption(std::string option_name)
    {
        auto value = _options[option_name].emptyValueString;
        return SetOption(option_name, value);
    }

    bool CommandLineParser::SetOption(std::string option_name, std::string option_val)
    {
        std::string oldValueString = _options[option_name].currentValueString;
        _options[option_name].currentValueString = option_val;

        bool ok = true;
        bool didEnableMoreParams = false;
        for (auto setValueCb : _options[option_name].setValueCallbacks)
        {
            ok = ok && setValueCb(option_val);
        }

        if (ok)
        {
            auto iter = _options[option_name].didSetValueCallbacks.begin();
            while (iter != _options[option_name].didSetValueCallbacks.end())
            {
                auto& didSetValueCb = *iter;
                bool did_set = didSetValueCb(_options[option_name].currentValueString);
                if (did_set)
                {
                    didEnableMoreParams = true;
                    // remove from list once it's called
                    iter = _options[option_name].didSetValueCallbacks.erase(iter);
                }
                else
                {
                    ++iter;
                }
            }
        }
        else
        {
            _options[option_name].currentValueString = oldValueString;
        }

        return didEnableMoreParams;
    }

    void CommandLineParser::AddOptionSet(ParsedArgSet& options)
    {
        AddPostParseCallback([&options](CommandLineParser& p) { return options.PostProcess(p); });
        options.AddArgs(*this);
    }

    void CommandLineParser::AddDocumentationString(std::string str)
    {
        _docEntries.emplace_back(DocumentationEntry::Type::str, str);
    }

    void CommandLineParser::DisableOption(std::string name)
    {
        auto option = FindOption(name);
        if (option != nullptr)
        {
            option->enabled = false;
        }
    }

    void CommandLineParser::EnableOption(std::string name)
    {
        auto option = FindOption(name);
        if (option != nullptr)
        {
            option->enabled = true;
        }
    }

    std::string CommandLineParser::GetHelpString()
    {
        std::stringstream out;
        // Find longest option name so we can align descriptions
        size_t longest_name = 0;
        for (const auto& iter : _options)
        {
            if (ToLowercase(iter.first) == ToLowercase(iter.second.name)) // wasn't a previously-undefined option
            {
                longest_name = std::max(longest_name, iter.second.optionNameHelpLength());
            }
        }

        out << "Usage: " << _exeName << " [options]" << std::endl;
        out << std::endl;

        for (const auto& entry : _docEntries)
        {
            switch (entry.EntryType)
            {
            case DocumentationEntry::Type::option: {
                const OptionInfo& info = _options[entry.EntryString];
                if (info.enabled)
                {
                    std::string option_name = info.optionNameString();
                    size_t thisOptionNameLen = info.optionNameHelpLength();
                    size_t pad_len = 2 + (longest_name - thisOptionNameLen);
                    std::string padding(pad_len, ' ');
                    out << "\t--" << option_name << padding << info.description;
                    if (info.enumValues.size() > 0)
                    {
                        out << "  {";
                        std::string sep = "";
                        out << info.enumValues[0];
                        for (size_t index = 1; index < info.enumValues.size(); ++index)
                        {
                            out << " | " << info.enumValues[index];
                        }
                        out << "}";
                    }
                    out << std::endl;
                }
            }
            break;

            case DocumentationEntry::Type::str:
                out << entry.EntryString << std::endl;
                break;
            }
        }

        return out.str();
    }

    std::string CommandLineParser::GetCurrentValuesString()
    {
        std::stringstream out;
        out << "Current parameters for " << _exeName << std::endl;

        std::set<std::string, case_insensitive_comparer> visited_options;
        for (auto& entry : _docEntries)
        {
            if (entry.EntryType == DocumentationEntry::Type::option)
            {
                const auto& opt = _options[entry.EntryString];
                visited_options.insert(opt.name);
                out << "\t--" << opt.name << ": ";
                if (opt.currentValueString != "")
                {
                    out << opt.currentValueString;
                    if (opt.currentValueString == opt.defaultValueString)
                    {
                        out << " (default)";
                    }
                    out << std::endl;
                }
                else
                {
                    out << "[" << opt.defaultValueString << "]" << std::endl;
                }
            }
        }

        bool did_print = false;
        for (auto& opt : _options)
        {
            if (visited_options.find(opt.first) == visited_options.end())
            {
                if (!did_print)
                {
                    out << std::endl;
                    out << "Unknown parameters" << std::endl;
                }
                out << "\t--" << opt.first << ": " << opt.second.currentValueString << std::endl;
                did_print = true;
            }
        }

        return out.str();
    }

    std::string CommandLineParser::GetCommandLine() const
    {
        std::stringstream out;
        out << _exeName;
        std::for_each(_originalArgs.begin() + 1, _originalArgs.end(), [&out](const std::string& s) { out << " " << s; });
        return out.str();
    }

    std::string CommandLineParser::GetOptionValue(const std::string& name)
    {
        auto it = _options.find(name);
        if (it != _options.end())
        {
            return it->second.currentValueString;
        }
        else
        {
            return "";
        }
    }
} // namespace utilities
} // namespace ell
