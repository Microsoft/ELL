////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     Archiver.h (utilities)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "ArchiveVersion.h"
#include "TypeFactory.h"
#include "TypeName.h"
#include "TypeTraits.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

namespace ell
{
namespace utilities
{
    class IArchivable;
    class ArchivedAsPrimitive;
    class Variant;

    /// <summary> Enabled if ValueType inherits from IArchivable. </summary>
    template <typename ValueType>
    using IsIArchivable = typename std::enable_if_t<std::is_base_of<IArchivable, typename std::decay<ValueType>::type>::value, int>;

    /// <summary> Enabled if ValueType inherits from IArchivable but not from ArchivedAsPrimitive. </summary>
    template <typename ValueType>
    using IsStandardArchivable = typename std::enable_if_t<std::is_base_of<IArchivable, typename std::decay<ValueType>::type>::value && !std::is_base_of<ArchivedAsPrimitive, typename std::decay<ValueType>::type>::value, int>;

    /// <summary> Enabled if ValueType inherits from IArchivable. </summary>
    template <typename ValueType>
    using IsArchivedAsPrimitive = typename std::enable_if_t<std::is_base_of<ArchivedAsPrimitive, typename std::decay<ValueType>::type>::value, int>;

    /// <summary> Enabled if ValueType does not inherit from ArchivedAsPrimitive. </summary>
    template <typename ValueType>
    using IsNotArchivedAsPrimitive = typename std::enable_if_t<!std::is_base_of<ArchivedAsPrimitive, typename std::decay<ValueType>::type>::value, int>;

    /// <summary> Enabled if ValueType does not inherit from IArchivable. </summary>
    template <typename ValueType>
    using IsNotArchivable = typename std::enable_if_t<(!std::is_base_of<IArchivable, typename std::decay<ValueType>::type>::value), int>;

    /// <summary> A registry of functions to enable Variant deserialization. </summary>
    class VariantTypeRegistry
    {
        void SetVariantType(Variant& variant, const std::string& typeName)
        {
            auto it = _functionMap.find(typeName);
            if (it != _functionMap.end())
            {
                (it->second)(variant);
            }
        }

        bool IsEmpty() const { return _functionMap.empty(); }

    private:
        friend class Variant;

        template <typename ValueType>
        void SetVariantTypeFunction(std::function<void(Variant&)> f)
        {
            auto typeName = ::ell::utilities::GetTypeName<ValueType>();
            _functionMap[typeName] = f;
        }
        std::unordered_map<std::string, std::function<void(Variant&)>> _functionMap;
    };

    /// <summary> A context object used during deserialization. Contains a GenericTypeFactory and a VariantTypeRegistry. </summary>
    class SerializationContext
    {
    public:
        SerializationContext() = default;
        SerializationContext(const SerializationContext&) = default;
        SerializationContext(SerializationContext&&) = default;

        virtual ~SerializationContext() = default;

        /// <summary> Gets the type factory associated with this context. </summary>
        ///
        /// <returns> The type factory associated with this context. </returns>
        GenericTypeFactory& GetTypeFactory();

        /// <summary> Gets the variant type registry associated with this context. </summary>
        /// Used by Variant to enable deserialization.
        ///
        /// <returns> The variant type registry associated with this context. </returns>
        VariantTypeRegistry& GetVariantTypeRegistry();

        /// <summary> Gets the previous context in a stack of contexts </summary>
        ///
        /// <returns> The SerializationContext from the previous stack frame. Returns nullptr if there is no previous context on the stack. </returns>
        SerializationContext* GetPreviousContext() { return _previousContext; }

    protected:
        struct c_tor
        {}; // tag struct for calling constructor that stores a previous context
        SerializationContext(SerializationContext& previousContext, c_tor) :
            _previousContext(&previousContext) {}

    private:
        friend class Variant;

        SerializationContext* _previousContext = nullptr;
        GenericTypeFactory _typeFactory;
        VariantTypeRegistry _variantTypeRegistry;
    };

    /// <summary> Info struct for archived objects </summary>
    struct ArchivedObjectInfo
    {
        std::string type;
        ArchiveVersion version;
    };

    bool operator==(const ArchivedObjectInfo& a, const ArchivedObjectInfo& b);
    bool operator!=(const ArchivedObjectInfo& a, const ArchivedObjectInfo& b);

/// <summary> Macros to make repetitive boilerplate code in archiver implementations easier to implement. </summary>

// On most platforms, uint64_t is an alias to `unsigned long`. However, on macOS, uint64_t is an alias of `unsigned long long`,
// so we need to add 'unsigned long' when compiling on the Mac to make sure it is archivable.
#if defined(__APPLE__)
#define ARCHIVABLE_TYPES_LIST      \
    ARCHIVE_TYPE_OP(bool)          \
    ARCHIVE_TYPE_OP(char)          \
    ARCHIVE_TYPE_OP(short)         \
    ARCHIVE_TYPE_OP(int)           \
    ARCHIVE_TYPE_OP(unsigned int)  \
    ARCHIVE_TYPE_OP(unsigned long) \
    ARCHIVE_TYPE_OP(int64_t)       \
    ARCHIVE_TYPE_OP(uint64_t)      \
    ARCHIVE_TYPE_OP(float)         \
    ARCHIVE_TYPE_OP(double)
#else
#define ARCHIVABLE_TYPES_LIST     \
    ARCHIVE_TYPE_OP(bool)         \
    ARCHIVE_TYPE_OP(char)         \
    ARCHIVE_TYPE_OP(short)        \
    ARCHIVE_TYPE_OP(int)          \
    ARCHIVE_TYPE_OP(unsigned int) \
    ARCHIVE_TYPE_OP(int64_t)      \
    ARCHIVE_TYPE_OP(uint64_t)     \
    ARCHIVE_TYPE_OP(float)        \
    ARCHIVE_TYPE_OP(double)
#endif

#define DECLARE_ARCHIVE_VALUE_BASE(type) virtual void ArchiveValue(const char* name, type value, IsFundamental<type> = true) = 0;
#define DECLARE_ARCHIVE_ARRAY_BASE(type) virtual void ArchiveArray(const char* name, const std::vector<type>& value, IsFundamental<type> = true) = 0;
#define DECLARE_ARCHIVE_VALUE_OVERRIDE(type) void ArchiveValue(const char* name, type value, IsFundamental<type> = true) override;
#define DECLARE_ARCHIVE_ARRAY_OVERRIDE(type) void ArchiveArray(const char* name, const std::vector<type>& value, IsFundamental<type> = true) override;

    /// <summary>
    /// The Archiver and Unarchiver abstract base classes facilitate serialization and
    /// deserialization of some fundamental types, `std::string`s, `std::vector`s, and
    /// classes that implement the necessary functions. Serializing a couple of
    /// variables to a string stream is as simple as
    ///
    ///     double x = 5.3;
    ///     size_t y = 12;
    ///     std::stringstream stream;
    ///     ArchiverType archiver(stream);
    ///     archiver.WriteToArchive(x);
    ///     archiver.WriteToArchive(y);
    ///
    /// Deserialization must occur in the same order.
    ///
    ///     UnarchiverType unarchiver(stream);
    ///     double xx;
    ///     size_t yy;
    ///     unarchiver.deserialize(xx);
    ///     unarchiver.deserialize(yy);
    ///     assert(x == xx &amp;&amp; y == yy);
    ///
    /// The Archiver subclasses support serialization of named variables, in which case the
    /// deserialization must specify the correct variable name. This is most often used when
    /// serializing named fields in objects.
    ///
    ///     x = 0.4;
    ///     archiver.WriteToArchive("x", x);
    ///     unarchiver.deserialize("x", xx);
    ///     assert(x == xx);
    ///
    /// Serialization of `std::string`s and `std::vector`s of fundamental types is similar.
    ///
    /// To make a class archivable, the IArchivable class must be inherited from, and a default constructor
    /// needs to be implemented. Additionally, the static method `GetTypeName` needs to be implemented.
    ///
    ///     class Bar: public IArchivable
    ///     {
    ///     public:
    ///         Bar();
    ///         void ReadFromArchive(utilities::Unarchiver&amp; unarchiver) const
    ///         void WriteToArchive(utilities::Archiver&amp; archiver);
    ///         virtual std::string GetRuntimeTypeName() const;
    ///
    ///         static std::string GetTypeName();
    ///     }
    ///
    /// A typical implementation of Serialize will include a sequence of
    /// calls to archiver.WriteToArchive(), in the same order. To serialize the class, call:
    ///
    ///     Bar z;
    ///     archiver.WriteToArchive("z", z);
    ///
    /// A typical implementation of ReadFromArchive() will include a similar sequence of calls to
    /// archiver.ReadFromArchive().
    ///
    /// Serialization and deserialization of pointers and `std::unique_pointer`s to archivable objects
    /// (that is, classes that derive from IArchivable, have a default constructor, and implement
    /// the static `GetTypeName` function) is supported as well.
    ///
    /// </summary>
    class Archiver
    {
    public:
        /// <summary> Represents an archiver that is scoped to a particular property. </summary>
        class PropertyArchiver
        {
        public:
            /// <summary> Write the property to the archiver. </summary>
            ///
            /// <typeparam name="ValueType"> The type of the property. </typeparam>
            /// <param name="value"> The property to serialize. </param>
            template <typename ValueType>
            void operator<<(ValueType&& value);

        private:
            friend class Archiver;
            PropertyArchiver(Archiver& archiver, const std::string& name);
            Archiver& _archiver;
            std::string _propertyName;
        };

        /// <summary> Destructor </summary>
        virtual ~Archiver() { EndArchiving(); }

        /// <summary> Add value to an archive. </summary>
        ///
        /// <param name="value"> The value to add to the archive. </param>
        template <typename ValueType>
        void Archive(ValueType&& value);

        /// <summary> Add value to an archive. </summary>
        ///
        /// <param name="value"> The value to add to the archive. </param>
        template <typename ValueType>
        void operator<<(ValueType&& value);

        /// <summary> Add named value to an archive. </summary>
        ///
        /// <param name="name"> The name to archive the value under. </param>
        /// <param name="value"> The value to add to the archive. </param>
        template <typename ValueType>
        void Archive(const char* name, ValueType&& value);

        /// <summary> Get an archiver scoped to a particular property name. </summary>
        ///
        /// <param name="name"> The name to archive the value under. </param>
        /// <returns> An archiver that will save an object with the given name </returns>
        PropertyArchiver operator[](const std::string& name);

    protected:
        // These are all the virtual function that need to be implemented by archivers

#define ARCHIVE_TYPE_OP(t) DECLARE_ARCHIVE_VALUE_BASE(t);
        ARCHIVABLE_TYPES_LIST
#undef ARCHIVE_TYPE_OP

        // archive a null value in a way that we can check for null on the unarchive side.
        virtual void ArchiveNull(const char* name) = 0;

        virtual void ArchiveValue(const char* name, const std::string& value) = 0;
        virtual void ArchiveValue(const char* name, const IArchivable& value);

#define ARCHIVE_TYPE_OP(t) DECLARE_ARCHIVE_ARRAY_BASE(t);
        ARCHIVABLE_TYPES_LIST
#undef ARCHIVE_TYPE_OP

        virtual void ArchiveArray(const char* name, const std::vector<std::string>& array) = 0;
        virtual void ArchiveArray(const char* name, const std::string& baseTypeName, const std::vector<const IArchivable*>& array) = 0;

        virtual void BeginArchiveObject(const char* name, const IArchivable& value);
        virtual void ArchiveObject(const char* name, const IArchivable& value);
        virtual void EndArchiveObject(const char* name, const IArchivable& value);

        virtual void EndArchiving() {}

        ArchiveVersion GetArchiveVersion(const IArchivable& value) const;

    private:
        // unique pointer to non-archivable object
        template <typename ValueType, IsNotArchivable<ValueType> concept = true>
        void ArchiveItem(const char* name, const std::unique_ptr<ValueType>& value);

        // unique pointer to standard archivable object
        template <typename ValueType, IsStandardArchivable<ValueType> concept = true>
        void ArchiveItem(const char* name, const std::unique_ptr<ValueType>& value);

        // unique pointer to archived-as-primitive object
        template <typename ValueType, IsArchivedAsPrimitive<ValueType> concept = true>
        void ArchiveItem(const char* name, const std::unique_ptr<ValueType>& value);

        template <typename ValueType, IsNotVector<ValueType> concept = true>
        void ArchiveItem(const char* name, ValueType&& value);

        template <typename ValueType>
        void ArchiveItem(const char* name, ValueType* value);

        template <typename ValueType, IsFundamental<ValueType> concept = true>
        void ArchiveItem(const char* name, const std::vector<ValueType>& value);

        void ArchiveItem(const char* name, const std::vector<std::string>& value);

        template <typename ValueType, IsIArchivable<ValueType> concept = true>
        void ArchiveItem(const char* name, const std::vector<ValueType>& value);

        template <typename ValueType, IsIArchivable<ValueType> concept = true>
        void ArchiveItem(const char* name, const std::vector<const ValueType*>& value);
    };

/// <summary> Macros to make repetitive boilerplate code in unarchiver implementations easier to implement. </summary>
#define DECLARE_UNARCHIVE_VALUE_BASE(type) virtual void UnarchiveValue(const char* name, type& value, IsFundamental<type> = true) = 0;
#define DECLARE_UNARCHIVE_ARRAY_BASE(type) virtual void UnarchiveArray(const char* name, std::vector<type>& value, IsFundamental<type> = true) = 0;
#define DECLARE_UNARCHIVE_VALUE_OVERRIDE(type) void UnarchiveValue(const char* name, type& value, IsFundamental<type> = true) override;
#define DECLARE_UNARCHIVE_ARRAY_OVERRIDE(type) void UnarchiveArray(const char* name, std::vector<type>& value, IsFundamental<type> = true) override;

    /// <summary> Unarchiver class </summary>
    class Unarchiver
    {
    public:
        /// <summary> Represents an unarchiver that is scoped to a particular property. </summary>
        class PropertyUnarchiver
        {
        public:
            /// <summary> Reads the property from the archive. </summary>
            ///
            /// <typeparam name="ValueType"> The type of the property. </typeparam>
            /// <param name="value"> The variable to read the property into. </param>
            template <typename ValueType>
            void operator>>(ValueType&& value);

        private:
            friend class Unarchiver;
            PropertyUnarchiver(Unarchiver& unarchiver, const std::string& name);
            Unarchiver& _unarchiver;
            std::string _propertyName;
        };

        /// <summary> Represents an unarchiver that is scoped to an optional property. </summary>
        template <typename DefaultValueType>
        class OptionalPropertyUnarchiver
        {
        public:
            /// <summary> Reads the property from the archive. </summary>
            ///
            /// <typeparam name="ValueType"> The type of the property. </typeparam>
            /// <param name="value"> The variable to read the property into. </param>
            template <typename ValueType>
            void operator>>(ValueType&& value);

        private:
            friend class Unarchiver;
            OptionalPropertyUnarchiver(Unarchiver& unarchiver, const std::string& name, const DefaultValueType& defaultValue);
            Unarchiver& _unarchiver;
            std::string _propertyName;
            DefaultValueType _defaultValue;
        };

        struct NoDefault
        {}; // tag type to denote that we don't have a default value

        /// <summary> Constructor </summary>
        ///
        /// <param name="context"> The initial `SerializationContext` to use </param>
        Unarchiver(SerializationContext context);

        /// <summary> Destructor </summary>
        virtual ~Unarchiver() { EndUnarchiving(); }

        /// <summary> Read value from an archive. </summary>
        ///
        /// <param name="value"> The value to read into. </param>
        template <typename ValueType>
        void Unarchive(ValueType&& value);

        /// <summary> Read value from an archive. </summary>
        ///
        /// <param name="value"> The value to read into. </param>
        template <typename ValueType>
        void operator>>(ValueType&& value);

        /// <summary> Read named value from an archive. </summary>
        ///
        /// <param name="name"> The name of the value to read. </param>
        /// <param name="value"> The value to read. </param>
        template <typename ValueType>
        void Unarchive(const char* name, ValueType&& value);

        /// <summary> Get an unarchiver scoped to a particular property name. </summary>
        ///
        /// <param name="name"> The name of the property </param>
        ///
        /// <returns>
        /// An unarchiver object that can unarchive a property with the given name.
        /// </returns>
        PropertyUnarchiver operator[](const std::string& name);

        /// <summary> Get an unarchiver for an optional property. </summary>
        ///
        /// <param name="name"> The name of the property </param>
        ///
        /// <returns>
        /// An unarchiver object that can unarchive an optional property with the given name, if it exists.
        /// If it is unable to unarchive a property with the given name, it is not an error.
        /// </returns>
        OptionalPropertyUnarchiver<NoDefault> OptionalProperty(const std::string& name);

        /// <summary> Get an unarchiver for an optional property with a default value. </summary>
        ///
        /// <typeparam name="DefaultValueType"> The type of the default value provided for the property </typeparam>
        /// <param name="name"> The name of the property </param>
        /// <param name="defaultValue"> The value to use for the property if it is not present </param>
        ///
        /// <returns>
        /// An unarchiver object that can unarchive an optional property with the given name, if it exists.
        /// If it is unable to unarchive a property with the given name, it is not an error.
        /// </returns>
        template <typename DefaultValueType>
        OptionalPropertyUnarchiver<DefaultValueType> OptionalProperty(const std::string& name, const DefaultValueType& defaultValue);

        /// <summary> Indicates if a property with the given name is available to be read next </summary>
        ///
        /// <param name="name"> The name of the property </param>
        ///
        /// <returns> true if a property with the given name can be read next </returns>
        virtual bool HasNextPropertyName(const std::string& name) = 0;

        /// <summary> Set a new serialization context to be current </summary>
        ///
        /// <param name="context"> The context </param>
        void PushContext(SerializationContext& context);

        /// <summary> Remove the current serialization context and use the previous one </summary>
        void PopContext() { _contexts.pop_back(); }

        /// <summary> Get the current serialization context </summary>
        ///
        /// <returns> The current serialization context </returns>
        SerializationContext& GetContext() { return _contexts.back(); }

        /// <summary> Get the information about the object currently being unarchived </summary>
        ///
        /// <returns> The ArchivedObjectInfo for the object currently being unarchived </returns>
        ArchivedObjectInfo GetCurrentObjectInfo() const;

    protected:
#define ARCHIVE_TYPE_OP(t) DECLARE_UNARCHIVE_VALUE_BASE(t);
        ARCHIVABLE_TYPES_LIST
#undef ARCHIVE_TYPE_OP

        virtual void UnarchiveValue(const char* name, std::string& value) = 0;
        virtual void UnarchiveValue(const char* name, IArchivable& value);

        // return true if the stream contains the result of ArchiveNull(name).
        virtual bool UnarchiveNull(const char* name) = 0;

#define ARCHIVE_TYPE_OP(t) DECLARE_UNARCHIVE_ARRAY_BASE(t);
        ARCHIVABLE_TYPES_LIST
#undef ARCHIVE_TYPE_OP

        virtual void UnarchiveArray(const char* name, std::vector<std::string>& array) = 0;

        // Extra functions needed for deserializing arrays.
        virtual void BeginUnarchiveArray(const char* name, const std::string& typeName);
        virtual bool BeginUnarchiveArrayItem(const std::string& typeName) = 0;
        virtual void EndUnarchiveArrayItem(const std::string& typeName) = 0;
        virtual void EndUnarchiveArray(const char* name, const std::string& typeName);

        // Extra functions needed for deserializing IArchivable objects.
        virtual ArchivedObjectInfo BeginUnarchiveObject(const char* name, const std::string& typeName);
        virtual void UnarchiveObject(const char* name, IArchivable& value);
        virtual void EndUnarchiveObject(const char* name, const std::string& typeName);
        virtual void UnarchiveObjectAsPrimitive(const char* name, IArchivable& value);

        virtual void EndUnarchiving() {}

    private:
        SerializationContext _baseContext;
        std::vector<std::reference_wrapper<SerializationContext>> _contexts;
        std::vector<ArchivedObjectInfo> _objectInfo;

        // non-vector standard thing
        template <typename ValueType, IsNotVector<ValueType> concept1 = true, IsNotArchivedAsPrimitive<ValueType> concept2 = true>
        void UnarchiveItem(const char* name, ValueType&& value);

        // non-vector archivable-as-fundamental
        template <typename ValueType, IsNotVector<ValueType> concept1 = true, IsArchivedAsPrimitive<ValueType> concept2 = true>
        void UnarchiveItem(const char* name, ValueType&& value);

        // unique pointer to non-archivable object
        template <typename ValueType, IsNotArchivable<ValueType> concept = true>
        void UnarchiveItem(const char* name, std::unique_ptr<ValueType>& value);

        // unique pointer to standard archivable object
        template <typename ValueType, IsStandardArchivable<ValueType> concept = true>
        void UnarchiveItem(const char* name, std::unique_ptr<ValueType>& value);

        // unique pointer to archived-as-primitive object
        template <typename ValueType, IsArchivedAsPrimitive<ValueType> concept = true>
        void UnarchiveItem(const char* name, std::unique_ptr<ValueType>& value);

        // vector of fundamental values
        template <typename ValueType, IsFundamental<ValueType> concept = true>
        void UnarchiveItem(const char* name, std::vector<ValueType>& value);

        // vector of strings
        void UnarchiveItem(const char* name, std::vector<std::string>& value);

        // vector of IArchivable values
        template <typename ValueType, IsIArchivable<ValueType> concept = true>
        void UnarchiveItem(const char* name, std::vector<ValueType>& value);

        // vector of unique pointers to IArchivable
        template <typename ValueType, IsIArchivable<ValueType> concept = true>
        void UnarchiveItem(const char* name, std::vector<std::unique_ptr<ValueType>>& value);

        // vector of pointers to IArchivable
        template <typename ValueType, IsIArchivable<ValueType> concept = true>
        void UnarchiveItem(const char* name, std::vector<const ValueType*>& value);
    };

    //
    // Utility classes useful to Archivers
    //

    /// <summary> A class that manages stream precision for
    /// floating point numbers, ensuring that maximum precision is
    /// set after the constructor, and reset to its previous values
    /// when it goes out of scope.
    /// </summary>
    template <typename ValueType>
    class EnsureMaxPrecision
    {
    public:
        EnsureMaxPrecision(std::ostream& out);
        ~EnsureMaxPrecision();

    private:
        std::ios::fmtflags _flags;
        std::streamsize _precision;
        std::ostream& _out;
    };

    //
    // Utility functions useful to archivers and unarchivers
    //

    /// <summary> Returns type name used for serialization </summary>
    ///
    /// <typeparam name="T"> The type to serialize </typeparam>
    /// <returns> The name to use when serializing type `T` </returns>
    template <typename T>
    std::string GetArchivedTypeName();

    /// <summary> Returns type name used for serialization of a (potentially-)polymorphic object </summary>
    ///
    /// <param name="value"> The value to serialize </param>
    /// <returns> The name to use when serializing the value </returns>
    template <typename T>
    std::string GetArchivedTypeName(const T& value);

/// <summary> Macros to make repetitive boilerplate code in archiver implementations easier to implement. </summary>
#define IMPLEMENT_ARCHIVE_VALUE(base, type)                                    \
    void base::ArchiveValue(const char* name, type value, IsFundamental<type>) \
    {                                                                          \
        WriteScalar(name, value);                                              \
    }
#define IMPLEMENT_ARCHIVE_ARRAY(base, type)                                                        \
    void base::ArchiveArray(const char* name, const std::vector<type>& value, IsFundamental<type>) \
    {                                                                                              \
        WriteArray(name, value);                                                                   \
    }

/// <summary> Macros to make repetitive boilerplate code in unarchiver implementations easier to implement. </summary>
#define IMPLEMENT_UNARCHIVE_VALUE(base, type)                                     \
    void base::UnarchiveValue(const char* name, type& value, IsFundamental<type>) \
    {                                                                             \
        ReadScalar(name, value);                                                  \
    }
#define IMPLEMENT_UNARCHIVE_ARRAY(base, type)                                                  \
    void base::UnarchiveArray(const char* name, std::vector<type>& value, IsFundamental<type>) \
    {                                                                                          \
        ReadArray(name, value);                                                                \
    }
} // namespace utilities
} // namespace ell

#pragma region implementation

namespace ell
{
namespace utilities
{
    //
    // PropertyArchiver class
    //

    template <typename ValueType>
    void Archiver::PropertyArchiver::operator<<(ValueType&& value)
    {
        _archiver.Archive(_propertyName.c_str(), value);
    }

    //
    // Archiver class
    //

    template <typename ValueType>
    void Archiver::Archive(ValueType&& value)
    {
        Archive("", std::forward<ValueType>(value));
    }

    template <typename ValueType>
    void Archiver::operator<<(ValueType&& value)
    {
        Archive(std::forward<ValueType>(value));
    }

    template <typename ValueType>
    void Archiver::Archive(const char* name, ValueType&& value)
    {
        ArchiveItem(name, value);
    }

    //
    // Implementations
    //

    // unique pointer to non-archivable object
    template <typename ValueType, IsNotArchivable<ValueType> concept>
    void Archiver::ArchiveItem(const char* name, const std::unique_ptr<ValueType>& value)
    {
        if (!value.get())
        {
            // write out a special value indicating null value.
            ArchiveNull(name);
        }
        else
        {
            // write out a special value indicating null value.
            ArchiveItem(name, *value.get());
        }
    }

    // unique pointer to standard archivable object
    template <typename ValueType, IsStandardArchivable<ValueType> concept>
    void Archiver::ArchiveItem(const char* name, const std::unique_ptr<ValueType>& value)
    {
        if (!value.get())
        {
            // write out a special value indicating null value.
            ArchiveNull(name);
        }
        else
        {
            // write out a special value indicating null value.
            ArchiveItem(name, *value.get());
        }
    }

    // unique pointer to archived-as-primitive object
    template <typename ValueType, IsArchivedAsPrimitive<ValueType> concept>
    void Archiver::ArchiveItem(const char* name, const std::unique_ptr<ValueType>& value)
    {
        if (!value.get())
        {
            // write out a special value indicating null value.
            ArchiveNull(name);
        }
        else
        {
            // write out a special value indicating null value.
            ArchiveItem(name, *value.get());
        }
    }

    // Non-vectors
    template <typename ValueType, IsNotVector<ValueType> concept>
    void Archiver::ArchiveItem(const char* name, ValueType&& value)
    {
        ArchiveValue(name, value);
    }

    // Pointers
    template <typename ValueType>
    void Archiver::ArchiveItem(const char* name, ValueType* value)
    {
        Archive(name, *value);
    }

    // Vector of fundamental types
    template <typename ValueType, IsFundamental<ValueType> concept>
    void Archiver::ArchiveItem(const char* name, const std::vector<ValueType>& array)
    {
        ArchiveArray(name, array);
    }

    // Vector of strings
    inline void Archiver::ArchiveItem(const char* name, const std::vector<std::string>& array)
    {
        ArchiveArray(name, array);
    }

    // Vector of serializable objects
    template <typename ValueType, IsIArchivable<ValueType> concept>
    void Archiver::ArchiveItem(const char* name, const std::vector<ValueType>& array)
    {
        auto baseTypeName = GetArchivedTypeName<ValueType>();
        std::vector<const utilities::IArchivable*> tmpArray;
        for (const auto& item : array)
        {
            tmpArray.push_back(&item);
        }
        ArchiveArray(name, baseTypeName, tmpArray);
    }

    // Vector of serializable pointers
    template <typename ValueType, IsIArchivable<ValueType> concept>
    void Archiver::ArchiveItem(const char* name, const std::vector<const ValueType*>& array)
    {
        auto baseTypeName = GetArchivedTypeName<ValueType>();
        std::vector<const utilities::IArchivable*> tmpArray;
        for (const auto& item : array)
        {
            tmpArray.push_back(item);
        }
        ArchiveArray(name, baseTypeName, tmpArray);
    }

    //
    // PropertyUnarchiver class
    //
    template <typename ValueType>
    void Unarchiver::PropertyUnarchiver::operator>>(ValueType&& value)
    {
        _unarchiver.Unarchive(_propertyName.c_str(), value);
    }

    //
    // OptionalPropertyUnarchiver class
    //
    template <typename DefaultValueType>
    Unarchiver::OptionalPropertyUnarchiver<DefaultValueType>::OptionalPropertyUnarchiver(Unarchiver& archiver, const std::string& name, const DefaultValueType& defaultValue) :
        _unarchiver(archiver),
        _propertyName(name),
        _defaultValue(defaultValue){};

    template <typename DefaultValueType>
    template <typename ValueType>
    void Unarchiver::OptionalPropertyUnarchiver<DefaultValueType>::operator>>(ValueType&& value)
    {
        if (_unarchiver.HasNextPropertyName(_propertyName))
        {
            _unarchiver.Unarchive(_propertyName.c_str(), value);
        }
        else
        {
            value = _defaultValue;
        }
    }

    template <>
    template <typename ValueType>
    void Unarchiver::OptionalPropertyUnarchiver<Unarchiver::NoDefault>::operator>>(ValueType&& value)
    {
        if (_unarchiver.HasNextPropertyName(_propertyName))
        {
            _unarchiver.Unarchive(_propertyName.c_str(), value);
        }
    }

    //
    // Unarchiver class
    //
    template <typename ValueType>
    void Unarchiver::Unarchive(ValueType&& value)
    {
        Unarchive("", value);
    }

    template <typename ValueType>
    void Unarchiver::operator>>(ValueType&& value)
    {
        Unarchive(std::forward<ValueType>(value));
    }

    template <typename ValueType>
    void Unarchiver::Unarchive(const char* name, ValueType&& value)
    {
        UnarchiveItem(name, value);
    }

    // STYLE: inline to keep next to its sibling overload
    inline Unarchiver::OptionalPropertyUnarchiver<Unarchiver::NoDefault> Unarchiver::OptionalProperty(const std::string& name)
    {
        return OptionalPropertyUnarchiver<Unarchiver::NoDefault>(*this, name, {});
    }

    template <typename DefaultValueType>
    Unarchiver::OptionalPropertyUnarchiver<DefaultValueType> Unarchiver::OptionalProperty(const std::string& name, const DefaultValueType& defaultValue)
    {
        return OptionalPropertyUnarchiver<DefaultValueType>(*this, name, defaultValue);
    }

    // types:
    // Fundamental
    // IArchivable (& ArchivedAsPrimitive)
    // Array
    template <typename ValueType, IsNotVector<ValueType> concept1, IsNotArchivedAsPrimitive<ValueType> concept2>
    void Unarchiver::UnarchiveItem(const char* name, ValueType&& value)
    {
        UnarchiveValue(name, value);
    }

    template <typename ValueType, IsNotVector<ValueType> concept1, IsArchivedAsPrimitive<ValueType> concept2>
    void Unarchiver::UnarchiveItem(const char* name, ValueType&& value)
    {
        UnarchiveObjectAsPrimitive(name, value);
    }

    // unique pointer to non-archivable type
    template <typename ValueType, IsNotArchivable<ValueType> concept>
    void Unarchiver::UnarchiveItem(const char* name, std::unique_ptr<ValueType>& value)
    {
        if (!UnarchiveNull(name))
        {
            auto ptr = std::make_unique<ValueType>();
            UnarchiveValue(name, *ptr);
            value = std::move(ptr);
        }
    }

    // unique pointer to standard archivable object
    template <typename ValueType, IsStandardArchivable<ValueType> concept>
    void Unarchiver::UnarchiveItem(const char* name, std::unique_ptr<ValueType>& value)
    {
        if (!UnarchiveNull(name))
        {
            auto baseTypeName = GetArchivedTypeName<ValueType>();
            auto objInfo = BeginUnarchiveObject(name, baseTypeName);
            _objectInfo.push_back(objInfo);
            auto encodedTypeName = objInfo.type;
            std::unique_ptr<ValueType> newPtr = GetContext().GetTypeFactory().Construct<ValueType>(encodedTypeName);
            UnarchiveObject(name, *newPtr);
            EndUnarchiveObject(name, encodedTypeName);
            // TODO: assert back of _objectInfo == objInfo
            _objectInfo.pop_back();
            value = std::move(newPtr);
        }
    }

    // pointer to serializable-as-primitive type
    template <typename ValueType, IsArchivedAsPrimitive<ValueType> concept>
    void Unarchiver::UnarchiveItem(const char* name, std::unique_ptr<ValueType>& value)
    {
        if (!UnarchiveNull(name))
        {
            auto baseTypeName = GetArchivedTypeName<ValueType>();
            std::unique_ptr<ValueType> newPtr = std::make_unique<ValueType>();
            UnarchiveObject(name, *newPtr);
            value = std::move(newPtr);
        }
    }

    // Vector of fundamental types
    template <typename ValueType, IsFundamental<ValueType> concept>
    void Unarchiver::UnarchiveItem(const char* name, std::vector<ValueType>& arr)
    {
        arr.clear();
        UnarchiveArray(name, arr);
    }

    // Vector of strings
    inline void Unarchiver::UnarchiveItem(const char* name, std::vector<std::string>& arr)
    {
        arr.clear();
        UnarchiveArray(name, arr);
    }

    // Vector of serializable objects
    template <typename ValueType, IsIArchivable<ValueType> concept>
    void Unarchiver::UnarchiveItem(const char* name, std::vector<ValueType>& arr)
    {
        arr.clear();
        auto typeName = GetArchivedTypeName<ValueType>();
        BeginUnarchiveArray(name, typeName);
        while (true)
        {
            auto good = BeginUnarchiveArrayItem(typeName);
            if (!good)
            {
                break;
            }
            ValueType value;
            Unarchive(value);
            arr.push_back(value);
            EndUnarchiveArrayItem(typeName);
        }
        EndUnarchiveArray(name, typeName);
    }

    // Vector of unique pointers to serializable objects
    template <typename ValueType, IsIArchivable<ValueType> concept>
    void Unarchiver::UnarchiveItem(const char* name, std::vector<std::unique_ptr<ValueType>>& arr)
    {
        arr.clear();
        auto typeName = GetArchivedTypeName<ValueType>();
        BeginUnarchiveArray(name, typeName);
        while (true)
        {
            auto good = BeginUnarchiveArrayItem(typeName);
            if (!good)
            {
                break;
            }
            std::unique_ptr<ValueType> newPtr;
            Unarchive(newPtr);
            arr.push_back(std::move(newPtr));
            EndUnarchiveArrayItem(typeName);
        }
        EndUnarchiveArray(name, typeName);
    }

    // Vector of raw pointers to serializable objects
    template <typename ValueType, IsIArchivable<ValueType> concept>
    void Unarchiver::UnarchiveItem(const char* name, std::vector<const ValueType*>& arr)
    {
        arr.clear();
        auto typeName = GetArchivedTypeName<ValueType>();
        BeginUnarchiveArray(name, typeName);
        while (true)
        {
            auto good = BeginUnarchiveArrayItem(typeName);
            if (!good)
            {
                break;
            }
            std::unique_ptr<ValueType> newPtr;
            Unarchive(newPtr);
            arr.push_back(newPtr.release());
            EndUnarchiveArrayItem(typeName);
        }
        EndUnarchiveArray(name, typeName);
    }

    //
    // Utility classes
    //

    template <typename ValueType>
    EnsureMaxPrecision<ValueType>::EnsureMaxPrecision(std::ostream& out) :
        _flags(out.flags()),
        _precision(out.precision()),
        _out(out)
    {
        _out.precision(std::numeric_limits<ValueType>::digits10 + 1);
    }

    template <typename ValueType>
    EnsureMaxPrecision<ValueType>::~EnsureMaxPrecision()
    {
        _out.flags(_flags);
        _out.precision(_precision);
    }

    //
    // Utility functions
    //
    namespace ArchiverImpl
    {
        template <typename T>
        static std::string GetTypeName(...)
        {
            return TypeName<T>::GetName();
        }

        template <typename T, IsIntegral<T> concept = true>
        static std::string GetTypeName(bool)
        {
            return "int";
        }

        template <typename T, IsFloatingPoint<T> concept = true>
        static std::string GetTypeName(bool)
        {
            return "float";
        }
    } // namespace ArchiverImpl

    template <typename T>
    std::string GetArchivedTypeName()
    {
        return ArchiverImpl::GetTypeName<T>(true);
    }

    template <typename T>
    std::string GetArchivedTypeName(const T& value)
    {
        return value.GetRuntimeTypeName();
    }
} // namespace utilities
} // namespace ell

#pragma endregion implementation
