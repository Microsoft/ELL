////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     LoopNest.h (value)
//  Authors:  Chuck Jacobs, Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "loopnests/LoopNest.h"
#include "loopnests/LoopNestPrinter.h"

#include <utilities/include/Exception.h>
#include <utilities/include/Hash.h>
#include <utilities/include/Logger.h>

#include <deque>
#include <iostream>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>

namespace ell
{
namespace value
{
    namespace loopnests
    {
        using logging::EOL;
        using logging::Log;

        //
        // LoopVisitSchedule
        //

        LoopVisitSchedule::LoopVisitSchedule(const LoopNest& nest, LoopVisitSchedule::StateQueue state) :
            LoopVisitSchedule(nest, 0, state)
        {}

        LoopVisitSchedule::LoopVisitSchedule(const LoopNest& nest, int level, StateQueue state) :
            _level(level),
            _state(std::move(state)),
            _nest(nest)
        {}

        LoopVisitSchedule::LoopVisitSchedule(const LoopVisitSchedule& other) :
            _level(other._level),
            _state(other._state),
            _nest(other._nest)
        {}

        LoopVisitSchedule& LoopVisitSchedule::operator=(const LoopVisitSchedule& other)
        {
            _level = other._level;
            _state = other._state;
            _nest = other._nest;
            return *this;
        }

        const LoopVisitSchedule::LoopInfo& LoopVisitSchedule::Front() const
        {
            return _state[_level];
        }

        const SplitIterationDomain& LoopVisitSchedule::GetDomain() const
        {
            return GetLoopNest().GetDomain();
        }

        int LoopVisitSchedule::CurrentNestLevel() const
        {
            return _level;
        };

        bool LoopVisitSchedule::IsDone() const
        {
            return _level == static_cast<int>(_state.size());
        }

        bool LoopVisitSchedule::IsInnermostLoop() const
        {
            return _level == static_cast<int>(_state.size()) - 1;
        }

        Index LoopVisitSchedule::CurrentDimension() const
        {
            return Front().dimension;
        }

        Range LoopVisitSchedule::LoopRange() const
        {
            // ### debugging
            assert(Front().indexRange.GetRange() == GetLoopNest().GetDomain().GetIndexRange(CurrentLoopIndex()));
            return Front().indexRange.GetRange();
        }

        int LoopVisitSchedule::LoopSize() const
        {
            return LoopRange().Size();
        }

        int LoopVisitSchedule::DimensionSize() const
        {
            return GetDomain().GetDimensionSize(Front().dimension);
        }

        int LoopVisitSchedule::NonBoundaryEnd() const
        {
            auto numFullLoopIterations = LoopSize() / LoopIncrement();
            auto nonBoundaryLoopSize = LoopIncrement() * numFullLoopIterations;
            return nonBoundaryLoopSize + LoopRange().Begin();
        }

        int LoopVisitSchedule::LoopIncrement() const
        {
            return LoopRange().Increment();
        }

        int LoopVisitSchedule::LoopIndexScale() const
        {
            return Front().scale;
        }

        bool LoopVisitSchedule::CurrentLoopHasFragment(std::vector<ScheduledKernel> activeKernels, LoopFragmentType fragmentType) const
        {
            auto currentIndex = CurrentLoopIndex();
            for (const auto& kernel : GetLoopNest().GetKernels())
            {
                const auto& where = kernel.constraints;
                if (where.GetPlacement() == fragmentType)
                {
                    // Boundary constraints: return `true` if this loop causes all the boundary indices to be defined
                    //   (which is to say, they're all fully-defined here but not in previous loop)
                    const auto& outsideIndices = where.GetBoundaryIndices();
                    if (outsideIndices.size() != 0)
                    {
                        bool allFullyDefined = std::all_of(outsideIndices.begin(), outsideIndices.end(), [&](auto index) {
                            return IsFullyDefined(index);
                        });
                        bool definedByThisLoop = std::any_of(outsideIndices.begin(), outsideIndices.end(), [&](auto index) {
                            return IsFullyDefinedByThisLoop(index);
                        });
                        if (allFullyDefined && definedByThisLoop)
                        {
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        bool LoopVisitSchedule::FragmentCanRunAlone(std::vector<ScheduledKernel> activeKernels, LoopFragmentType fragmentType) const
        {
            return true;
        }

        bool LoopVisitSchedule::FutureLoopHasFragmentForThisIndex(std::vector<ScheduledKernel> activeKernels, LoopFragmentType fragmentType) const
        {
            auto currentIndex = CurrentLoopIndex();
            for (const auto& kernel : GetLoopNest().GetKernels())
            {
                const auto& where = kernel.constraints;
                if (where.GetPlacement() == fragmentType)
                {
                    // Boundary constraints: return `true` if this loop causes all the boundary indices to be defined
                    //   (which is to say, they're all fully-defined here but not in previous loop)
                    const auto& outsideIndices = where.GetBoundaryIndices();
                    bool allFullyDefined = std::all_of(outsideIndices.begin(), outsideIndices.end(), [&](auto index) {
                        return IsFullyDefined(index);
                    });
                    bool thisIndexWasUsed = std::any_of(outsideIndices.begin(), outsideIndices.end(), [&](auto index) {
                        auto domain = GetLoopNest().GetDomain();
                        if (index == currentIndex || domain.DependsOn(index, currentIndex))
                        {
                            return true;
                        }
                        return false;
                    });

                    if (!allFullyDefined && thisIndexWasUsed)
                    {
                        return true;
                    }
                }
            }
            return false;
        }

        int LoopVisitSchedule::CurrentIndexEndBoundarySize() const
        {
            return Front().boundarySize;
        }

        Index LoopVisitSchedule::CurrentLoopIndex() const
        {
            return Front().indexRange.GetIndex();
        }

        LoopVisitSchedule LoopVisitSchedule::Next() const
        {
            if (IsDone())
            {
                throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState, "Error: calling Next() at end of schedule");
            }
            return { _nest.get(), _level + 1, _state };
        }

        LoopVisitSchedule LoopVisitSchedule::Prev() const
        {
            if (_level == 0)
            {
                throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState, "Error: calling Prev() on first loop level");
            }
            return { _nest.get(), _level - 1, _state };
        }

        bool LoopVisitSchedule::WillVisitIndex(const Index& index) const
        {
            auto dependentIndices = GetDomain().GetDependentIndices(index);

            // Check if a future loop has a loop index that the query index depends on
            for (auto it = _state.begin() + CurrentNestLevel(); it != _state.end(); ++it)
            {
                auto i = it->indexRange.GetIndex();
                if (std::find(dependentIndices.begin(), dependentIndices.end(), i) != dependentIndices.end())
                {
                    return true;
                }
            }
            return false;
        }

        bool LoopVisitSchedule::IsFullyDefined(const Index& index) const
        {
            if (index == CurrentLoopIndex())
            {
                return true;
            }

            for (const auto& i : GetDomain().GetDependentIndices(index))
            {
                if (GetDomain().IsLoopIndex(i))
                {
                    if (!WasIterationVariableDefined(i))
                    {
                        return false;
                    }
                }
            }
            return true;
        }

        bool LoopVisitSchedule::IsFullyDefinedByThisLoop(const Index& index) const
        {
            // return true if:
            //    1) the given index is this loop's index variable
            //    1) the given index is synthetic and one of its terms is this loop's index variable, and the rest
            //       of the terms have already been defined
            if (index == CurrentLoopIndex())
            {
                return true;
            }

            // look to see if this index has been defined
            if (IsFullyDefined(index))
            {
                if (CurrentNestLevel() == 0)
                {
                    return true;
                }
                return !Prev().IsFullyDefined(index);
            }
            return false;
        }

        bool LoopVisitSchedule::WasIterationVariableDefined(const Index& index) const
        {
            for (auto it = _state.begin(); it != _state.begin() + CurrentNestLevel() + 1; ++it)
            {
                auto iterVar = it->indexRange.GetIndex();
                if (iterVar == index)
                {
                    return true;
                }
            }
            return false;
        }

        KernelPredicate LoopVisitSchedule::GetKernelPredicate(const ScheduledKernel& kernel) const
        {
            // Convert constraints to predicate

            const auto& domain = GetDomain();

            // Get list of conditions in existing predicate
            std::set<FragmentTypePredicate> predicateConditions;
            std::set<Index> predicateIndices;
            std::set<Index> constrainedIndices;

            kernel.predicate.Visit([&](const auto& p) {
                if (auto fragmentPred = p.template As<FragmentTypePredicate>(); fragmentPred != nullptr)
                {
                    auto predicateIndex = fragmentPred->GetIndex();
                    auto fragment = fragmentPred->GetCondition();

                    // Convert computed indices to loop indices
                    for (auto loopIndex : domain.GetDependentLoopIndices(predicateIndex, true))
                    {
                        predicateConditions.insert(FragmentTypePredicate(loopIndex, fragment));
                        predicateIndices.insert(loopIndex);
                    }
                }
                else if (p.template Is<IndexDefinedPredicate>())
                {
                    throw utilities::LogicException(utilities::LogicExceptionErrors::notImplemented, "IsDefined predicate not implemented");
                }
            });

            // BEGIN CONVERT CONSTRAINTS
            // Convert CodePositionConstraints
            const bool convertConstraints = !kernel.newVersion;
            if (convertConstraints)
            {
                for (auto constraintIndex : kernel.constraints.GetRequiredIndices())
                {
                    // Convert computed indices to loop indices
                    for (auto loopIndex : domain.GetDependentLoopIndices(constraintIndex, true))
                    {
                        if (predicateIndices.count(loopIndex) != 0)
                        {
                            throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Constraint applied to index " + loopIndex.GetName() + ", which already has a predicate");
                        }

                        predicateConditions.insert({ loopIndex, Fragment::all });
                        predicateIndices.insert(loopIndex);
                        constrainedIndices.insert(loopIndex);
                    }
                }

                // Convert the kernel's predicate + constraints into equivalent predicate
                // All "body" constraints turn into "all" conditions
                // All "prologue" constraints turn into "first" conditions
                // All "epilogue" constraints turn into "last" conditions

                // Issue: empty "boundary indices" means everything not mentioned

                const auto placement = kernel.constraints.GetPlacement();
                auto constraintCondition = (placement == LoopFragmentType::prologue) || (placement == LoopFragmentType::body) ? Fragment::first : Fragment::last;
                auto boundaryIndices = kernel.constraints.GetBoundaryIndices();
                if (boundaryIndices.empty())
                {
                    // add all unmentioned loop indices --- all indices not dependent on any of the already-"all"'d indices
                    for (auto loopIndex : GetLoopNest().GetLoopSequence())
                    {
                        if (constrainedIndices.count(loopIndex) == 0 && predicateIndices.count(loopIndex) == 0)
                        {
                            boundaryIndices.push_back(loopIndex);
                        }
                    }
                }
                for (auto boundaryIndex : boundaryIndices)
                {
                    // Convert any boundary indices into concrete loop indices
                    for (auto loopIndex : domain.GetDependentLoopIndices(boundaryIndex, true))
                    {
                        constrainedIndices.insert(loopIndex);
                        predicateIndices.insert(loopIndex);
                        predicateConditions.insert({ loopIndex, constraintCondition });
                    }
                }

                // All unmentioned loop indices become "first" conditions
                for (auto loopIndex : GetLoopNest().GetLoopSequence())
                {
                    if (constrainedIndices.count(loopIndex) == 0 && predicateIndices.count(loopIndex) == 0)
                    {
                        predicateConditions.insert({ loopIndex, Fragment::first });
                    }
                }
            }
            // END CONVERT CONSTRAINTS

            // new predicate == conjunction of all conditions in conditions set
            if (predicateConditions.size() == 0)
            {
                return {}; // ?
            }

            auto begin = predicateConditions.begin();
            auto first = KernelPredicate(*begin);
            ++begin;
            auto fullPredicate = std::accumulate(begin, predicateConditions.end(), first, [](auto lhs, auto rhs) -> KernelPredicate { return { KernelPredicateConjunction(lhs, rhs) }; });
            auto result = fullPredicate.Simplify();
            return result;
        }

        //
        // LoopNest
        //
        LoopNest::LoopNest(IterationDomain domain) :
            _domain(domain)
        {
            InitLoopSequence();
        }

        void LoopNest::InitLoopSequence()
        {
            // For each dimension, get a queue of loop indices
            int numDimensions = _domain.NumDimensions();
            std::vector<std::queue<Index>> dimensionIndices(numDimensions);
            for (int d = 0; d < numDimensions; ++d)
            {
                const auto indices = _domain.GetLoopIndicesForDimension(_domain.GetBaseIndex(d));
                dimensionIndices[d] = std::queue<Index>({ indices.begin(), indices.end() });
            }

            for (;;)
            {
                bool done = true;

                // for each index
                for (int d = 0; d < numDimensions; ++d)
                {
                    if (!dimensionIndices[d].empty())
                    {
                        _loopSequence.push_back(dimensionIndices[d].front());
                        dimensionIndices[d].pop();
                        done = false;
                    }
                }
                if (done) break;
            }
        }

        void LoopNest::ConvertKernelConstraints()
        {
            for (auto& k : _kernels)
            {
                ConvertKernelConstraints(k);
            }
        }

        void LoopNest::ConvertKernelConstraints(ScheduledKernel& kernel)
        {
            // TODO: convert first/last into inequality check (<=, >=), so they can work with boundaries
        }

        void LoopNest::AddKernel(const Kernel& kernel, ConstraintType type)
        {
            if (type == ConstraintType::constraint)
            {
                AddKernel(kernel, LoopFragmentType::body);
            }
            else
            {
                CodePositionConstraints constraints{ LoopFragmentType::body, {}, {} }; // null constraints
                _kernels.push_back({ true, kernel, constraints, {}, {} });
            }
        }

        void LoopNest::AddKernel(const Kernel& kernel, LoopFragmentType where)
        {
            CodePositionConstraints constraints{ where, kernel.GetIndices(), {} };
            AddKernel(kernel, constraints);
        }

        void LoopNest::AddKernel(const Kernel& kernel, const CodePositionConstraints& where)
        {
            // old version
            _kernels.push_back({ false, kernel, where, {}, {} });
        }

        void LoopNest::AddKernel(const Kernel& kernel, const KernelPredicate& predicate)
        {
            AddKernel(kernel, predicate, {});
        }

        void LoopNest::AddKernel(const Kernel& kernel, const KernelPredicate& predicate, const KernelPredicate& placement)
        {
            // new version
            CodePositionConstraints constraints{ LoopFragmentType::body, {}, {} }; // null constraints
            _kernels.push_back({ true, kernel, constraints, predicate, placement });
        }

        void LoopNest::AddKernel(const Kernel& kernel, const CodePositionConstraints& where, const KernelPredicate& predicate, const KernelPredicate& placement)
        {
            // new version
            _kernels.push_back({ true, kernel, where, predicate, {} });
        }

        const std::vector<ScheduledKernel>& LoopNest::GetKernels() const
        {
            return _kernels;
        }

        std::vector<ScheduledKernelGroup> LoopNest::GetKernelGroups() const
        {
            std::vector<ScheduledKernelGroup> result;
            for (const auto& kernel : _kernels)
            {
                auto it = std::find_if(result.begin(), result.end(), [&](const ScheduledKernelGroup& g) {
                    return g.id == kernel.kernel.GetId();
                });
                if (it == result.end())
                {
                    result.push_back({ kernel.kernel.GetId(), { kernel } });
                }
                else
                {
                    it->kernels.push_back(kernel);
                }
            }
            return result;
        }

        int LoopNest::NumDimensions() const
        {
            return static_cast<int>(_domain.NumDimensions());
        }

        Range LoopNest::GetIndexRange(Index index) const
        {
            return _domain.GetIndexRange(index);
        }

        std::vector<IndexRange> LoopNest::GetLoopIndexRanges() const
        {
            std::vector<IndexRange> result;
            for (int d = 0; d < NumDimensions(); ++d)
            {
                const auto& dimRange = GetDimensionRange(d);
                for (const auto& index : dimRange.GetLoopIndices())
                {
                    result.emplace_back(index, dimRange.GetIndexRange(index));
                }
            }
            return result;
        }

        const SplitIndexRange& LoopNest::GetDimensionRange(int dimension) const
        {
            return _domain.GetDimensionRange(dimension);
        }

        const SplitIndexRange& LoopNest::GetDimensionRange(const Index& dimension) const
        {
            return _domain.GetDimensionRange(dimension);
        }

        int LoopNest::NumSplits(const Index& dimension) const
        {
            return GetDimensionRange(dimension).NumSplits();
        }

        const std::vector<Index>& LoopNest::GetLoopSequence() const
        {
            return _loopSequence;
        }

        LoopVisitSchedule LoopNest::GetLoopSchedule() const
        {
            LoopVisitSchedule::StateQueue queue;

            std::map<Index, std::set<Index>> availableLoopIndices;
            int numDimensions = _domain.NumDimensions();
            for (int i = 0; i < numDimensions; ++i)
            {
                auto dimension = _domain.GetBaseIndex(i);
                auto loopIndices = _domain.GetLoopIndicesForDimension(dimension);
                availableLoopIndices[dimension] = { loopIndices.begin(), loopIndices.end() };
            }

            for (auto loopIndex : GetLoopSequence())
            {
                auto range = _domain.GetIndexRange(loopIndex);
                auto dimensionSize = _domain.GetDimensionSize(loopIndex);
                int splitSize = range.Increment(); // need to keep track of the split size here, I think
                int boundarySize = dimensionSize % splitSize;

                auto loopIndexScale = GetLoopIndexScale(loopIndex);
                auto dimensionIndex = _domain.GetBaseIndex(loopIndex);
                queue.push_back(LoopVisitSchedule::LoopInfo{ dimensionIndex, IndexRange{ loopIndex, range }, boundarySize, loopIndexScale });
            }

            return { *this, queue };
        }

        SplitIndex LoopNest::Split(Index index, int size)
        {
            auto result = _domain.Split(index, size);

            // Need to recompute loopSequence here (by replacing the index that got split with result.outer)
            auto parent = _domain.GetParentIndex(result.outer); // this is the specific index that was split
            auto it = std::find(_loopSequence.begin(), _loopSequence.end(), parent);
            if (it == _loopSequence.end())
            {
                throw utilities::LogicException(utilities::LogicExceptionErrors::illegalState);
            }
            else
            {
                *it = result.outer;
            }

            _loopSequence.push_back(result.inner);
            return result;
        }

        void LoopNest::Parallelize(Index index)
        {
            _parallelizedIndices.push_back(index);
        }

        // TODO: Move this out to the API surface
        SplitIndex LoopNest::Parallelize(Index index, int factor)
        {
            auto result = Split(index, factor);
            Parallelize(result.outer);
            return result;
        }

        void LoopNest::Unroll(Index index)
        {
            _unrolledIndices.push_back(index);
        }

        // TODO: Move this out to the API surface
        SplitIndex LoopNest::Unroll(Index index, int factor)
        {
            auto result = Split(index, factor);
            Unroll(result.outer);
            return result;
        }

        void LoopNest::SetLoopOrder(const std::vector<Index>& order)
        {
            if (order.size() != _loopSequence.size())
            {
                throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "SetLoopOrder() --- new order wrong length");
            }

            std::map<Index, std::set<Index>> availableLoopIndices;
            int numDimensions = _domain.NumDimensions();
            for (int i = 0; i < numDimensions; ++i)
            {
                auto dimension = _domain.GetBaseIndex(i);
                auto loopIndices = _domain.GetLoopIndicesForDimension(dimension);
                availableLoopIndices[dimension] = { loopIndices.begin(), loopIndices.end() };
            }

            // Function to get the next available concrete loop index that's a child index of a given index.
            // Throws if there isn't one available.
            auto getNextAvailable = [this, &availableLoopIndices](const Index& specifiedIndex) {
                auto dimensionIndex = _domain.GetBaseIndex(specifiedIndex);
                auto possibleIndices = _domain.GetDependentLoopIndices(specifiedIndex);
                if (_domain.IsLoopIndex(specifiedIndex))
                {
                    possibleIndices.push_back(specifiedIndex);
                }
                for (auto i : possibleIndices)
                {
                    if (availableLoopIndices[dimensionIndex].count(i) != 0)
                    {
                        availableLoopIndices[dimensionIndex].erase(i);
                        return i;
                    }
                }
                throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "SetLoopOrder() --- new order wrong length");
            };

            std::vector<Index> newLoopSequence;
            for (auto specifiedIndex : order)
            {
                auto loopIndex = getNextAvailable(specifiedIndex);
                newLoopSequence.push_back(loopIndex);
            }

            _loopSequence = newLoopSequence;
        }

        void LoopNest::RenameVariable(ViewAdapter oldVariable, ViewAdapter newVariable, const std::vector<Index>& where, const std::vector<Kernel>& excludedKernels)
        {
            std::vector<Kernel::Id> kernelIds;
            std::transform(
                excludedKernels.begin(),
                excludedKernels.end(),
                std::back_inserter(kernelIds),
                [](const Kernel& kernel) { return kernel.GetId(); });

            _renameActions.push_back({ oldVariable, newVariable, where, kernelIds });
        }

        int LoopNest::GetLoopIndexScale(const Index& index) const
        {
            // TODO: later we may normalize the loops, in which case indexScale here will be the loop increment
            return 1;
        }

        Index LoopNest::GetLoopIndex(const Index& dimension, int level) const
        {
            const auto& dim = GetDimensionRange(dimension);
            return dim.GetSplitIndex(level);
        }

        bool LoopNest::IsUsed(const Index& index, const std::vector<ScheduledKernel>& activeKernels) const
        {
            for (auto k : activeKernels)
            {
                for (auto kernelIndex : k.kernel.GetIndices())
                {
                    if (kernelIndex == index || GetDomain().DependsOn(kernelIndex, index))
                    {
                        return true;
                    }
                }
            }

            return false;
        }

        bool LoopNest::IsParallelized(const Index& index) const
        {
            return std::find(_parallelizedIndices.begin(), _parallelizedIndices.end(), index) != _parallelizedIndices.end();
        }

        bool LoopNest::IsUnrolled(const Index& index) const
        {
            return std::find(_unrolledIndices.begin(), _unrolledIndices.end(), index) != _unrolledIndices.end();
        }

        const std::vector<RenameAction>& LoopNest::GetRenameActions() const
        {
            return _renameActions;
        }

        const SplitIterationDomain& LoopNest::GetDomain() const
        {
            return _domain;
        }

        Index LoopNest::GetBaseIndex(const Index& index) const
        {
            return _domain.GetBaseIndex(index);
        }

        bool LoopNest::IsLoopIndex(const Index& index) const
        {
            return _domain.IsLoopIndex(index);
        }

        bool LoopNest::IsComputedIndex(const Index& index) const
        {
            return _domain.IsComputedIndex(index);
        }

        IndexExpression LoopNest::GetIndexExpression(const Index& index) const
        {
            auto loopIndices = _domain.GetDependentLoopIndices(index);

            std::vector<ScaledIndex> result;
            for (auto loopIndex : loopIndices)
            {
                auto indexScale = GetLoopIndexScale(index);
                result.push_back({ indexScale, loopIndex });
            }

            auto begin = _domain.GetDimensionBegin(index);
            return { result, begin };
        }

        void LoopNest::DebugDump(std::string tag, std::ostream* stream) const
        {
            auto& targetStream = stream != nullptr ? *stream : std::cerr;

            GetDomain().Print(targetStream);

            targetStream << "Loop order: ";
            for (auto i : GetLoopSequence())
            {
                targetStream << i << " ";
            }
            targetStream << std::endl;

            LoopNestPrinter printer(targetStream);

            printer.Print(*this);

            if (!tag.empty())
            {
                targetStream << "[tag = " << tag << "]";
            }
            targetStream << '\n';
        }

        void DebugDump(const LoopNest& nest, std::string tag, std::ostream* stream)
        {
            nest.DebugDump(tag, stream);
        }

        bool operator==(const ScheduledKernel& i1, const ScheduledKernel& i2)
        {
            return (i1.kernel == i2.kernel) && (i1.constraints == i2.constraints);
        }

        bool operator!=(const ScheduledKernel& i1, const ScheduledKernel& i2)
        {
            return !(i1 == i2);
        }

        LoopNest Fuse(const LoopNest& nest1, const LoopNest& nest2)
        {
            return Fuse(nest1, nest2, {}, {});
        }

        LoopNest Fuse(const LoopNest& nest1, const LoopNest& nest2, const std::vector<Index>& dependentIndexVec1, const std::vector<Index>& dependentIndexVec2)
        {
            // Collect all the indices for nest1 and nest2
            std::map<Index, Range> nestIndices;

            auto makeSet = [&](const auto& container) {
                std::set<std::decay_t<decltype(*container.cbegin())>> result(container.begin(), container.end());
                return result; };

            std::set<Index> dependentIndices1 = makeSet(dependentIndexVec1);
            std::set<Index> dependentIndices2 = makeSet(dependentIndexVec2);
            std::set<Index> nest1Indices = makeSet(nest1.GetDomain().GetAllLoopIndices());
            std::set<Index> nest2Indices = makeSet(nest2.GetDomain().GetAllLoopIndices());

            // Collect vector of all IndexRanges, and indices in only one nest
            // add indices in nest2 but not nest1 as "first" predicates for the nest1 kernels
            // add indices in nest1 but not nest2 as "last" predicates for the nest2 kernels
            auto domain1 = nest1.GetDomain();
            std::vector<IndexRange> indexRanges;
            for (const auto& index : nest1Indices)
            {
                auto range = domain1.GetIndexRange(index);
                if (nest2Indices.count(index) == 0)
                {
                    dependentIndices2.insert(index);
                }
                indexRanges.emplace_back(index, range);
            }

            auto domain2 = nest2.GetDomain();
            for (const auto& index : nest2Indices)
            {
                auto range = domain2.GetIndexRange(index);
                if (nest1Indices.count(index) != 0)
                {
                    auto range1 = domain1.GetIndexRange(index);
                    if (range != range1)
                    {
                        throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Fusing loops with incompatible ranges for index " + index.GetName());
                    }
                }
                else
                {
                    dependentIndices1.insert(index);
                    indexRanges.emplace_back(index, range);
                }
            }

            // Make new "selector" index
            // Index s("sel");
            // indexRanges.emplace_back(s, Range{ 0, 2 });

            // Create new loop nest
            LoopNest result = { indexRanges };

            // Add kernels for nest1 with "first" selector
            for (const auto& kernel : nest1.GetKernels())
            {
                auto fullPredicate = std::accumulate(dependentIndices1.begin(), dependentIndices1.end(), kernel.predicate, [](auto lhs, auto rhs) -> KernelPredicate { return lhs && First(rhs); });
                // result.AddKernel(kernel.kernel, kernel.constraints, fullPredicate && First(s)); // if using selector
                result.AddKernel(kernel.kernel, kernel.constraints, fullPredicate, kernel.placement);
            }

            // Add kernels for nest2 with "last" selector, and using the dependent indices passed in
            for (const auto& kernel : nest2.GetKernels())
            {
                auto fullPredicate = std::accumulate(dependentIndices2.begin(), dependentIndices2.end(), kernel.predicate, [](auto lhs, auto rhs) -> KernelPredicate { return lhs && Last(rhs); });
                // result.AddKernel(kernel.kernel, kernel.constraints, fullPredicate && Last(s));
                result.AddKernel(kernel.kernel, kernel.constraints, fullPredicate, kernel.placement);
            }

            return result;
        }
    } // namespace loopnests
} // namespace value
} // namespace ell

using namespace ell::value::loopnests;

std::hash<ScheduledKernel>::result_type std::hash<ScheduledKernel>::operator()(const argument_type& kernel) const
{
    using ::ell::utilities::HashCombine;

    size_t hash = 0;
    HashCombine(hash, kernel.kernel);
    HashCombine(hash, kernel.constraints);

    return hash;
}
