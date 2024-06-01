////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     CompiledMap.h (model)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <emitters/include/ModuleEmitter.h>
#include <utilities/include/CallbackRegistry.h>

#include "Map.h"
#include "MapCompilerOptions.h"

#include <ostream>
#include <string>

namespace ell
{
namespace model
{
    /// <summary> Abstract base class for a map that has been compiled </summary>
    class CompiledMap : public Map
    {
    public:
        CompiledMap(const CompiledMap& other) = delete;

        CompiledMap(CompiledMap&& other) = default;

        ~CompiledMap() override = default;

        /// <summary> Get the name of function this map compiles to </summary>
        ///
        /// <returns> The name of function this map compiles to </returns>
        std::string GetFunctionName() const { return _functionName; }

        /// <summary> Output the compiled model to the given file </summary>
        ///
        /// <param name="filePath"> The file to write to </param>
        virtual void WriteCode(const std::string& filePath) const = 0;

        /// <summary> Output the compiled model to the given file with the given format </summary>
        ///
        /// <param name="filePath"> The file to write to </param>
        /// <param name="format"> The format to write out (ir, bitcode, or assembly) </param>
        virtual void WriteCode(const std::string& filePath, emitters::ModuleOutputFormat format) const = 0;

        /// <summary> Output the compiled model to an output stream with the given format </summary>
        ///
        /// <param name="stream"> The stream to write to </param>
        /// <param name="format"> The format to write out (ir, bitcode, or assembly) </param>
        virtual void WriteCode(std::ostream& stream, emitters::ModuleOutputFormat format) const = 0;

        /// <summary> Output a 'C'-style function prototype for the compiled function </summary>
        ///
        /// <param name="filePath"> The path to the file to write </param>
        /// <param name="format"> The format to write out (cHeader or Swig interface) </param>
        virtual void WriteCodeHeader(const std::string& filePath, emitters::ModuleOutputFormat format) const = 0;

        /// <summary> Output a 'C'-style function prototype for the compiled function </summary>
        ///
        /// <param name="stream"> The stream to write the prototype to </param>
        /// <param name="format"> The format to write out (cHeader or Swig interface) </param>
        virtual void WriteCodeHeader(std::ostream& stream, emitters::ModuleOutputFormat format) const = 0;

        /// <summary> Output a 'C'-style function prototype for the compiled function </summary>
        ///
        /// <returns> A string with the function prototype </returns>
        virtual std::string GetCodeHeaderString() const = 0;

        /// <summary> Can this compiled map be used? </summary>
        ///
        /// <returns> true if active, false if not. </returns>
        virtual bool IsValid() const = 0;

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        static std::string GetTypeName() { return "CompiledMap"; }

        /// <summary> Gets the name of this type (for serialization). </summary>
        ///
        /// <returns> The name of this type. </returns>
        std::string GetRuntimeTypeName() const override { return GetTypeName(); }

        /// <summary> Reset any model state. </summary>
        virtual void Reset() = 0;

        /// <summary> Return the typed CallbackRegistry object that is used to manage any std::functions defined
        /// on any SourceNodes or SinkNodes in the graph. </summary>
        template <typename ElementType>
        ell::utilities::CallbackRegistry<ElementType>& GetCallbackRegistry() const;

        /// <summary> Returns true if the CallbackRegistry objects contain some functions. </summary>
        bool HasCallbackFunctions() const;

    protected:
        CompiledMap(Map map, std::string functionName, const MapCompilerOptions& options);
        MapCompilerOptions GetMapCompilerOptions() const { return _compilerOptions; }

        std::string _functionName;
        MapCompilerOptions _compilerOptions;

        ell::utilities::CallbackRegistry<float> _floatCallbacks;
        ell::utilities::CallbackRegistry<double> _doubleCallbacks;
        ell::utilities::CallbackRegistry<int> _intCallbacks;
        ell::utilities::CallbackRegistry<int64_t> _int64Callbacks;
        ell::utilities::CallbackRegistry<bool> _boolCallbacks;
    };
} // namespace model
} // namespace ell
