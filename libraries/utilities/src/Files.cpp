////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Files.cpp (utilities)
//  Authors:  Ofer Dekel
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Files.h"
#include "Exception.h"
#include "StringUtil.h"

#include <algorithm>
#include <cstdlib>
#include <ios>
#include <memory>
#ifndef WIN32
#include <sys/stat.h>
#include <unistd.h>
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif // WIN32

namespace ell
{
namespace utilities
{
    namespace
    {
        std::ifstream OpenIfstream(const std::string& filepath, std::ios_base::openmode mode)
        {
#ifdef WIN32
            auto path = fs::u8path(filepath);
#else
            const auto& path = filepath;
#endif
            // open file
            if (!FileExists(filepath))
            {
                throw utilities::InputException(InputExceptionErrors::invalidArgument, "file " + filepath + " doesn't exist");
            }

            auto stream = std::ifstream(path, mode);

            // check that it opened
            if (!stream)
            {
                throw utilities::InputException(InputExceptionErrors::invalidArgument, "error opening file " + filepath);
            }

            return stream;
        }

        std::ofstream OpenOfstream(const std::string& filepath, std::ios_base::openmode mode)
        {
#ifdef WIN32
            auto path = fs::u8path(filepath);
#else
            const auto& path = filepath;
#endif
            // open file
            std::ofstream stream(path, mode);

            // check that it opened
            if (!stream.is_open())
            {
                throw utilities::InputException(InputExceptionErrors::invalidArgument, "error opening file " + filepath);
            }

            return stream;
        }
    } // namespace

    std::ifstream OpenIfstream(const std::string& filepath)
    {
        return OpenIfstream(filepath, std::ios_base::in);
    }

    std::ifstream OpenBinaryIfstream(const std::string& filepath)
    {
        return OpenIfstream(filepath, std::ios_base::in | std::ios_base::binary);
    }

    std::ofstream OpenOfstream(const std::string& filepath)
    {
        return OpenOfstream(filepath, std::ios_base::out);
    }

    std::ofstream OpenBinaryOfstream(const std::string& filepath)
    {
        return OpenOfstream(filepath, std::ios_base::out | std::ios_base::binary);
    }

    bool IsFileReadable(const std::string& filepath)
    {
#ifdef WIN32
        auto path = fs::u8path(filepath);
#else
        const auto& path = filepath;
#endif
        // open file
        std::ifstream stream(path);

        // check that it opened
        if (stream.is_open())
        {
            return true;
        }

        return false;
    }

    bool IsFileWritable(const std::string& filepath)
    {
#ifdef WIN32
        auto path = fs::u8path(filepath);
#else
        const auto& path = filepath;
#endif
        // open file
        std::ofstream stream(path);

        // check that it opened
        if (stream.is_open())
        {
            return true;
        }

        return false;
    }

    bool FileExists(const std::string& filepath)
    {
#ifdef WIN32
        std::error_code ec;
        return fs::is_regular_file(fs::u8path(filepath), ec);
#else
        struct stat buf;
        int rc = stat(filepath.c_str(), &buf);

        if (rc != -1)
        {
            return (buf.st_mode & S_IFDIR) == 0;
        }
        return false;
#endif
    }

    bool DirectoryExists(const std::string& path)
    {
#ifdef WIN32
        std::error_code ec;
        return fs::is_directory(fs::u8path(path), ec);
#else
        struct stat buf;
        int rc = stat(path.c_str(), &buf);
        if (rc != -1)
        {
            return (buf.st_mode & S_IFDIR) == S_IFDIR;
        }
        return false;
#endif
    }

    std::string GetFileExtension(const std::string& filepath, bool toLowercase)
    {
        auto dotPos = filepath.find_last_of('.');
        if (dotPos == std::string::npos)
        {
            return "";
        }
        else
        {
            auto ext = filepath.substr(dotPos + 1);
            if (toLowercase)
            {
                return ext;
            }
            else
            {
                return ext;
            }
        }
    }

    std::string RemoveFileExtension(const std::string& filepath)
    {
        auto filename = filepath.substr(filepath.find_last_of("/\\") + 1);
        auto dotPos = filename.find_last_of('.');
        if (dotPos == std::string::npos)
        {
            return filename;
        }
        else
        {
            auto ext = filename.substr(0, dotPos);
            return ext;
        }
    }

    // PORTABILITY should be replaced by C++17 filesystem when available
#ifdef WIN32
    std::string path_separator = "\\";
#else
    std::string path_separator = "/";
#endif

    std::string GetFileName(const std::string& filepath)
    {
        // PORTABILITY should be replaced by C++17 filesystem when available
        size_t pos = filepath.find_last_of(path_separator);
        if (pos == std::string::npos)
        {
            pos = filepath.find_last_of('/');
            if (pos == std::string::npos)
            {
                return filepath;
            }
        }
        return filepath.substr(pos + 1);
    }

    std::string GetDirectoryPath(const std::string& filepath)
    {
        size_t pos = filepath.find_last_of(path_separator);
        if (pos == std::string::npos)
        {
            pos = filepath.find_last_of('/');
            if (pos == std::string::npos)
            {
                return "";
            }
        }
        auto path = filepath.substr(0, pos);
        return path;
    }

    std::string JoinPaths(const std::string& path1, const std::string& path2)
    {
        return JoinPaths(path1, { path2 });
    }

    std::string JoinPaths(const std::string& path, std::initializer_list<std::string> toAdd)
    {
        std::vector<std::string> list(toAdd.begin(), toAdd.end());
        return JoinPaths(path, list);
    }

    inline bool ends_with(std::string const& value, std::string const& ending)
    {
        if (ending.size() > value.size()) return false;
        return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
    }

    std::string JoinPaths(const std::string& path, std::vector<std::string> toAdd)
    {
        std::string result = path;
        for (const auto& current : toAdd)
        {
            if (current.length() == 0 && result.length() == 0)
            {
                // special case for paths that start with a slash.
                result = path_separator;
            }
            else if (current.length() > 0)
            {
                if (result.length() > 0 && !ends_with(result, path_separator))
                {
                    result += path_separator;
                }
                result += current;
            }
        }
        return result;
    }

    std::vector<std::string> SplitPath(const std::string& path)
    {
        std::vector<std::string> result;
        size_t start = 0;
        size_t pos = path.find_first_of(path_separator, start);
        while (pos != std::string::npos)
        {
            auto part = path.substr(start, pos - start);
            result.push_back(part);
            start = pos + path_separator.size();
            pos = path.find_first_of(path_separator, start);
        }
        if (start < path.length())
        {
            auto part = path.substr(start, path.length() - start);
            result.push_back(part);
        }
        return result;
    }

    void EnsureDirectoryExists(const std::string& path)
    {
        if (!DirectoryExists(path))
        {

            // mkdir can only do one directory level at a time, so here we
            // recursively walk up the path making sure all intermediate paths exist.
            std::string dirName = GetFileName(path);
            std::string parentPath = GetDirectoryPath(path);
            if (parentPath != "")
            {
                EnsureDirectoryExists(parentPath);
            }

            int rc = 0;
#ifdef WIN32
            std::error_code ec;
            if (!fs::create_directory(fs::u8path(path), ec))
            {
                rc = ec.value();
            }
#else
            // Linux can do unicode file names in utf-8.
            constexpr mode_t modeInOctal = 0777;
            rc = mkdir(path.c_str(), modeInOctal);
#endif
            if (rc != 0)
            {
                throw ell::utilities::Exception(ell::utilities::FormatString("mkdir failed with error code %d", errno));
            }
        }
    }

    std::string GetWorkingDirectory()
    {
        int rc = 0;
        std::string utf8wd;
#ifdef WIN32
        std::error_code ec;
        auto path = fs::current_path(ec);
        if (ec)
        {
            rc = ec.value();
        }
        else
        {
            utf8wd = path.generic_string();
        }
#else
        const int maxPath = 8192;
        char path[maxPath];
        if (getcwd(path, maxPath) == nullptr)
        {
            rc = errno;
        }
        utf8wd = std::string(path);
#endif
        if (rc != 0)
        {
            throw ell::utilities::Exception(ell::utilities::FormatString("error getting current working directory: %d", rc));
        }
        return utf8wd;
    }

    std::string FindExecutable(const std::string& name)
    {
#ifdef WIN32
#pragma warning(disable : 4996)
        std::string path = getenv("PATH");
        char separator = ';';
#else
        std::string path = getenv("PATH");
        char separator = ':';
#endif
        std::vector<std::string> paths = Split(path, separator);
        for (auto ptr = paths.begin(), end = paths.end(); ptr != end; ptr++)
        {
            std::string fullPath = JoinPaths(*ptr, name);
            if (FileExists(fullPath))
            {
                return *ptr;
            }
        }
        throw ell::utilities::Exception("Could not find '" + name + "' in your PATH environment");
    }
} // namespace utilities
} // namespace ell
