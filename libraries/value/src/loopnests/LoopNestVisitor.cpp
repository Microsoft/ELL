////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     LoopNestVisitor.cpp (value)
//  Authors:  Chuck Jacobs, Kern Handa
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "LLVMContext.h"

#include "loopnests/KernelPredicate.h"
#include "loopnests/LoopNestPrinter.h"
#include "loopnests/LoopNestVisitor.h"

#include <utilities/include/Exception.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <set>
#include <stdexcept>

namespace ell
{
namespace value
{
    namespace loopnests
    {
        namespace
        {
            // computes ceil(a/b)
            int CeilDiv(int a, int b)
            {
                return (a - 1) / b + 1;
            }

            // check for a "placement" predicate without an index
            bool IsBodyPlacementPredicate(const KernelPredicate& predicate)
            {
                if (auto placementPred = predicate.As<PlacementPredicate>(); placementPred != nullptr)
                {
                    return !placementPred->HasIndex();
                }
                return false;
            }
        } // namespace

        //
        // LoopNestVisitor::RecursionState
        //
        LoopNestVisitor::RecursionState::RecursionState(const LoopNest& loopNest) :
            currentFragment(LoopFragmentFlags::All()),
            activeKernels(loopNest.GetKernels())
        {
        }

        //
        // LoopNestVisitor::RecursionStateNew
        //
        LoopNestVisitor::RecursionStateNew::RecursionStateNew(const LoopNest& loopNest)
        {
            kernelGroups.reserve(loopNest.GetKernelGroups().size());
            for (const auto& g : loopNest.GetKernelGroups())
            {
                kernelGroups.emplace_back(true, g);
            }
        }

        //
        // LoopNestVisitor
        //
        void LoopNestVisitor::Visit(const LoopNest& loopNest) const
        {
            auto schedule = loopNest.GetLoopSchedule();

            if (UseNewVersion(loopNest))
            {
                // 0) convert old-style constraints into new predicate model
                //    - have a "GetPredicate" function that appends constraint conditions to scheduled kernel's predicate
                // 1) generate simple structure representing perfectly-nested loops with predicates on kernels
                //    - should replace old `LoopNest::GetLoopSchedule()`
                // 2) unswitch conditions by splitting loops
                // 3) replace constant predicates with either a simple kernel invocation or a no-op
                // 4) replace single-iteration loops with simply setting the index value and evaluating the loop body
                // 5) identify loops / index variable statements to omit

                // We need to create a RecursionState object, because it's passed in as a mutable (in/out) parameter
                RecursionStateNew state = { loopNest };
                GenerateLoopsNew(state, schedule);
            }
            else
            {
                GenerateLoopsOld({ loopNest }, schedule);
            }
        }

        bool LoopNestVisitor::UseNewVersion(const LoopNest& loopNest) const
        {
            for (const auto& k : loopNest.GetKernels())
            {
                if (k.newVersion)
                {
                    return true;
                }
            }
            return false;
        }

        void LoopNestVisitor::GenerateLoopsNew(RecursionStateNew& state, const LoopVisitSchedule& schedule) const
        {
            if (schedule.IsDone())
            {
                return;
            }

            // We're descending into the heart of the loop

            // Find the active range for the current loop dimension and reduce our end amount if it exceeds the active range (boundary case)
            auto loopIndex = schedule.CurrentLoopIndex();

            bool hasValidKernels = false;
            for (const auto& k : state.kernelGroups)
            {
                if (k.first)
                {
                    hasValidKernels = true;
                }
            }

            // if state.kernelGroups is empty, just put all the remaining indices in the symbol table, marked "done"
            if (!hasValidKernels)
            {
                // get each inner index and add it to state.loopIndices
                auto s = schedule;
                while (!s.IsDone())
                {
                    auto innerLoopIndex = s.CurrentLoopIndex();
                    DefinePostLoopIndex(innerLoopIndex, state.loopIndices, s);
                    s = s.Next();
                }
                return;
            }

            // Alg:

            // 1) get splits/partitions
            // 2) copy partition per kernel (group)
            // 3) eval predicates and mark valid regions
            // 4) make representation that's a list of kernels to run for each partition (e.g., [1,2 | 2 | 2, 3])
            // 5) move adjacent fully-matching suffix on left into right partition (and expand)
            // 6) move adjacent fully-matching prefix on right into left partition (and expand)

            // ex, with S1: first(i), S2: all, S3: last(i):

            // step 1: partitions: (0..1), (1..N-1), (N-1..N)
            // step 2: partitions w/ kernels: (0..1: S1, S2, S3), (1..N-1: S1, S2, S3), (N-1..N: S1, S2, S3)
            // step 3: eval predicates and remove kernels: (0..1: S1, S2), (1..N-1: S2), (N-1..N: S2, S3)
            // step 4: ...
            // step 5: Suffix of first partition matches entirety of second: move
            //         --> (0..1: S1), (0..N-1: S2), (N1-..N: S2, S3)
            // step 6: prefix of last partition matches entirety of second: move
            //         --> (0..1: S1), (0..N: S2), (N1-..N: S3)

            auto loopRange = GetLoopRange(loopIndex, state.loopIndices, schedule);
            auto partitions = GetPartitions(loopIndex, loopRange, state.kernelGroups, state.loopIndices, schedule);
            std::vector<LoopRange> ranges;
            LoopFragmentFlags bodyFlags;
            bodyFlags.SetFlag(LoopFragmentType::boundary, false);
            for (const auto& p : partitions)
            {
                ranges.push_back({ p.range.Begin(), p.range.End(), p.range.Increment(), bodyFlags, LoopFragmentType::body });
            }

            for (auto r : ranges)
            {
#if 1
                std::function<void(Scalar)> codegenFn = GetCodegenFnNew(r, state, schedule);
                GenerateLoopRangeNew(r, state, schedule, codegenFn);
#else
                std::function<void(Scalar)> codegenFn = GetCodegenFnNew(r, state, schedule);
                const int startInt = r.start.Get<int>();
                const int stopInt = r.stop.Get<int>();
                const int stepInt = r.step.Get<int>();
                auto numIterations = CeilDiv(stopInt - startInt, stepInt);

                if (numIterations == 0)
                {
                    // throw?
                }
                else if (numIterations == 1)
                {
                    // TODO: set initial value of index variable (at least in loop-nest-printing case)
                    // SetLoopIndexValue(loopIndex, r.start);
                    codegenFn(r.start);
                }
                else
                {
                    GenerateLoopRangeNew(r, state, schedule, codegenFn);
                }
#endif
            }

            // set the loop index state to be "done"
            DefinePostLoopIndex(loopIndex, state.loopIndices, schedule);
        }

        void LoopNestVisitor::GenerateLoopsOld(const RecursionState& state, const LoopVisitSchedule& schedule) const
        {
            // Loop-unswitching / duplicating rules:
            //
            // Need to duplicate the outermost loop involving an index used to compute the constraint index
            // Only the innermost loop involving an index used to compute the constraint index needs to start from `1` for the body case
            // If all the loops with indices used to compute the constraint index are contiguous, and the kernel is run in the innermost of these loops,
            //   then we can omit the 'body' from the prologue (/epilogue) fragment, and allow the body loop to start from `0`
            //   (really, we can have the prologue (/epilogue) fragment contain only the constrained kernel)

            if (schedule.IsDone())
            {
                return;
            }

            // We're descending into the heart of the loop

            // If the index we're looping over in this loop has any prologue / epilogue kernels, we have to (potentially) break up the range
            // into prologue / body / epilogue sections
            auto currentDimension = schedule.CurrentDimension();

            // Find the active range for the current loop dimension and reduce our end amount if it exceeds the active range (boundary case)
            auto activeRangeIt = state.activeDimensionRanges.find(currentDimension);
            auto loopRange = schedule.LoopRange();
            int begin = loopRange.Begin();
            int end = loopRange.End();
            int increment = schedule.LoopIncrement();
            if (activeRangeIt != state.activeDimensionRanges.end())
            {
                auto activeRange = activeRangeIt->second;
                if (end > activeRange.End())
                {
                    end = activeRange.End();
                    loopRange = Range{ begin, end, increment };
                }
            }

            int nonBoundaryEnd = GetMainBodyLoopEnd(state, schedule, loopRange);

            // These mean "split current loop for this fragment type"
            auto currentLoopHasPrologue = schedule.CurrentLoopHasFragment(state.activeKernels, LoopFragmentType::prologue);
            auto currentLoopHasEpilogue = schedule.CurrentLoopHasFragment(state.activeKernels, LoopFragmentType::epilogue);

            // check if we need to emit an epilogue section to handle the end boundary for this loop
            auto currentLoopHasEndBoundary = schedule.CurrentIndexEndBoundarySize() != 0;

            auto futureLoopHasPrologue = schedule.FutureLoopHasFragmentForThisIndex(state.activeKernels, LoopFragmentType::prologue);
            auto futureLoopHasEpilogue = schedule.FutureLoopHasFragmentForThisIndex(state.activeKernels, LoopFragmentType::epilogue);

            LoopFragmentFlags bodyFlags = state.currentFragment;
            bodyFlags.SetFlag(LoopFragmentType::boundary, false);

            bool bodyInPrologue = !schedule.FragmentCanRunAlone(state.activeKernels, LoopFragmentType::prologue);
            bool bodyInEpilogue = !schedule.FragmentCanRunAlone(state.activeKernels, LoopFragmentType::epilogue);

            bool generatePrologueFragment = currentLoopHasPrologue || futureLoopHasPrologue;
            bool generateEpilogueFragment = currentLoopHasEpilogue || futureLoopHasEpilogue;

            std::vector<LoopRange> ranges;
            auto prologueBegin = begin;
            auto prologueEnd = begin + increment;

            if (generatePrologueFragment)
            {
                if (bodyInPrologue)
                {
                    begin += increment;
                }
                else
                {
                    bodyFlags.SetFlag(LoopFragmentType::prologue, false);
                }
            }

            // adjust loop boundary to unswitch last loop iteration if we have an epilogue kernel
            auto epilogueBegin = end - increment;
            auto epilogueEnd = end;
            if (generateEpilogueFragment)
            {
                if (bodyInEpilogue)
                {
                    if (currentLoopHasEndBoundary)
                    {
                        epilogueBegin = nonBoundaryEnd;
                    }
                    else
                    {
                        end -= increment;
                        nonBoundaryEnd -= increment;
                    }
                }
                else
                {
                    bodyFlags.SetFlag(LoopFragmentType::epilogue, false);
                }
            }

            // Add prologue section
            if (generatePrologueFragment)
            {
                LoopFragmentFlags flags = bodyInPrologue ? LoopFragmentType::prologue | LoopFragmentType::body : LoopFragmentType::prologue;
                ranges.push_back({ prologueBegin, prologueEnd, increment, flags, LoopFragmentType::prologue });
            }

            // Add main body section
            if (nonBoundaryEnd > begin)
            {
                ranges.push_back({ begin, nonBoundaryEnd, increment, bodyFlags, LoopFragmentType::body });
            }

            // Add boundary case (unless epilogue case already handles it)
            if (currentLoopHasEndBoundary && !(generateEpilogueFragment && bodyInEpilogue) && (end - nonBoundaryEnd > 0))
            {
                ranges.push_back({ nonBoundaryEnd, end, increment, bodyFlags | LoopFragmentType::boundary, LoopFragmentType::body });
            }

            // Add epilogue case
            if (generateEpilogueFragment)
            {
                LoopFragmentFlags flags = bodyInEpilogue ? LoopFragmentType::epilogue | LoopFragmentType::body : LoopFragmentType::epilogue;
                if (currentLoopHasEndBoundary)
                {
                    flags.SetFlag(LoopFragmentType::boundary, true);
                }
                ranges.push_back({ epilogueBegin, epilogueEnd, increment, flags, LoopFragmentType::epilogue });
            }

            for (auto r : ranges)
            {
                std::function<void(Scalar)> codegenFn = GetCodegenFnOld(r, state, schedule);
                const int startInt = r.start.Get<int>();
                const int stopInt = r.stop.Get<int>();
                const int stepInt = r.step.Get<int>();
                auto numIterations = CeilDiv(stopInt - startInt, stepInt);

                if (numIterations == 0)
                {
                    // throw?
                }
                else if (numIterations == 1)
                {
                    codegenFn(r.start);
                }
                else
                {
                    GenerateLoopRangeOld(r, state, schedule, codegenFn);
                }
            }
        }

        std::function<void(Scalar)> LoopNestVisitor::GetCodegenFnNew(const LoopRange& r, const RecursionStateNew& state, const LoopVisitSchedule& schedule) const
        {
            // define the function used to do the codegen, but don't run it yet
            return [this, &r, &state, &schedule](Scalar index) {
                auto loopIndex = schedule.CurrentLoopIndex();

                // TODO: deal with eventually not having an emit-time-constant range here
                const int startInt = r.start.Get<int>();
                const int stopInt = r.stop.Get<int>();
                const int stepInt = r.step.Get<int>();

                // Note: it's important that this code not be moved outside of the `codegenFn` lambda, otherwise Compute will incorrectly use old info for subsequent ranges
                auto newState = state;
                newState.loopIndices.insert_or_assign(loopIndex, LoopIndexSymbolTableEntry{ loopIndex, index, Range(startInt, stopInt, stepInt), LoopIndexState::inProgress });

                // define vars for all kernels
                std::vector<ScheduledKernel> kernels;
                for (const auto& g : state.kernelGroups)
                {
                    if (g.first)
                    {
                        kernels.insert(kernels.end(), g.second.kernels.begin(), g.second.kernels.end());
                    }
                }

                DefineComputedIndexVariables(newState.loopIndices, kernels, schedule);

                // invoke all kernels valid before inner loops
                for (auto& g : newState.kernelGroups)
                {
                    if (g.first)
                    {
                        auto invoked = InvokeKernelGroup(g.second, newState.loopIndices, schedule);
                        if (invoked)
                        {
#if 0
                            InvokeForContext<LLVMContext>([&](LLVMContext& context) {
                                auto& fn = context.GetFunctionEmitter();
                                fn.Print("Invoking kernel " + g.second.id + " at loop index " + loopIndex.GetName() + "\n");
                            });
#endif

                            g.first = false;
                        }
                    }
                }

                // TODO: need to know if we're going to invoke any kernels after the inner loops, and remove them from the valid kernel groups

                if (!schedule.IsInnermostLoop())
                {
                    GenerateLoopsNew(newState, schedule.Next());

                    // invoke all kernels valid after inner loops
                    for (auto& g : newState.kernelGroups)
                    {
                        if (g.first)
                        {
                            auto invoked = InvokeKernelGroup(g.second, newState.loopIndices, schedule);
                            if (invoked)
                            {
#if 0
                                InvokeForContext<LLVMContext>([&](LLVMContext& context) {
                                    auto& fn = context.GetFunctionEmitter();
                                    fn.Print("Invoking after-kernel " + g.second.id + " at loop index " + loopIndex.GetName() + "\n");
                                });
#endif
                                g.first = false;
                            }
                        }
                    }
                }

                // TODO: restore state of variables
                // debugging
                {
                    DefineComputedIndexVariables(newState.loopIndices, kernels, schedule);
                }
            };
        }

        Range LoopNestVisitor::GetLoopRange(const Index& loopIndex, const LoopIndexSymbolTable& activeRanges, const LoopVisitSchedule& schedule)
        {
            const auto& loopNest = schedule.GetLoopNest();
            const auto& domain = loopNest.GetDomain();
            auto loopRange = domain.GetIndexRange(loopIndex);
            int begin = loopRange.Begin();
            int end = loopRange.End();
            int rangeSize = end - begin;
            int increment = loopRange.Increment();

            auto fixBoundaryRange = [&](Index index) {
                // check activeRanges for parent
                auto outerIndex = domain.GetOuterSplitIndex(domain.GetParentIndex(index));
                if (domain.IsLoopIndex(outerIndex) && activeRanges.count(outerIndex) != 0)
                {
                    // check if it's a boundary --- if so, set size to its size
                    auto parentRange = activeRanges.at(outerIndex).loopRange;
                    if (parentRange.Size() < rangeSize)
                    {
                        rangeSize = parentRange.Size();
                        end = begin + rangeSize;
                        loopRange = { begin, end, increment };
                    }
                }
            };

            if (domain.IsInnerSplitIndex(loopIndex))
            {
                fixBoundaryRange(loopIndex);
            }
            else if (domain.HasParentIndex(loopIndex))
            {
                auto parentIndex = domain.GetParentIndex(loopIndex);
                if (domain.IsInnerSplitIndex(parentIndex))
                {
                    fixBoundaryRange(domain.GetParentIndex(loopIndex));
                }
            }

            return loopRange;
        }

        LoopNestVisitor::PartitionList LoopNestVisitor::GetPartitions(const Index& loopIndex, Range loopRange, const ActiveKernelGroupList& kernels, const LoopIndexSymbolTable& runtimeIndexVariables, const LoopVisitSchedule& schedule) const
        {
            int begin = loopRange.Begin();
            int end = loopRange.End();
            int rangeSize = end - begin;
            int increment = loopRange.Increment();

            // Find conditions involving this index and add any relevant partition split points
            std::set<int> splits;
            for (const auto& g : kernels)
            {
                if (g.first)
                {
                    for (const auto& k : g.second.kernels)
                    {
                        auto predicate = schedule.GetKernelPredicate(k).Simplify(runtimeIndexVariables, schedule);
                        AddSplits(loopIndex, loopRange, predicate, schedule, splits);
                    }
                }
            }

            // Add boundary split point, if necessary
            int extra = rangeSize % increment;
            if (extra != 0)
            {
                splits.insert(rangeSize - extra);
            }

            // get index range
            PartitionList result;
            for (auto partitionEnd : splits)
            {
                result.push_back({ loopIndex, { begin, partitionEnd, increment } });
                begin = partitionEnd;
            }
            result.push_back({ loopIndex, { begin, end, increment } });

            return result;
        }

        void LoopNestVisitor::AddSplits(const Index& loopIndex, Range loopRange, const KernelPredicate& predicate, const LoopVisitSchedule& schedule, std::set<int>& splits) const
        {
            const auto& loopNest = schedule.GetLoopNest();
            const auto& domain = loopNest.GetDomain();

            // visit predicate, adding testVal to splits
            auto addSplits = [&splits, &domain, &loopIndex, &loopRange](const auto& addSplits, const KernelPredicate& p) -> void {
                if (auto simplePredicate = p.As<FragmentTypePredicate>(); simplePredicate != nullptr)
                {
                    auto where = simplePredicate->GetCondition();
                    if (where != Fragment::all)
                    {
                        auto predIndex = simplePredicate->GetIndex();
                        if (predIndex == loopIndex || (domain.SameDimension(predIndex, loopIndex) && domain.DependsOn(predIndex, loopIndex)))
                        {
                            std::optional<int> splitVal;
                            switch (simplePredicate->GetCondition())
                            {
                            case Fragment::first:
                                splitVal = loopRange.Begin() + loopRange.Increment();
                                break;
                            case Fragment::last: {
                                // take into account last range being a boundary condition
                                auto extra = loopRange.End() % loopRange.Increment();
                                if (extra == 0)
                                {
                                    splitVal = loopRange.End() - loopRange.Increment();
                                }
                                else
                                {
                                    splitVal = loopRange.End() - extra;
                                }
                            }
                            break;
                            case Fragment::endBoundary:
                                // already set by automatic boundary-handling code
                                break;
                            default:
                                // nothing
                                break;
                            }

                            if (splitVal)
                            {
                                if (splitVal.value() > 0 && splitVal.value() < loopRange.End())
                                {
                                    splits.insert(splitVal.value());
                                }
                            }
                        }
                    }
                }
                else if (p.Is<IndexDefinedPredicate>())
                {
                    // nothing
                }
                else if (auto conjunction = p.As<KernelPredicateConjunction>(); conjunction != nullptr)
                {
                    for (const auto& t : conjunction->GetTerms())
                    {
                        addSplits(addSplits, *t);
                    }
                }
                else if (auto disjunction = p.As<KernelPredicateDisjunction>(); disjunction != nullptr)
                {
                    for (const auto& t : disjunction->GetTerms())
                    {
                        addSplits(addSplits, *t);
                    }
                }
            };

            addSplits(addSplits, predicate);
        }

        std::function<void(Scalar)> LoopNestVisitor::GetCodegenFnOld(const LoopRange& r, const RecursionState& state, const LoopVisitSchedule& schedule) const
        {
            // define the function used to do the codegen, but don't run it yet
            return [this, &r, &state, &schedule](Scalar index) {
                const LoopNest& loopNest = schedule.GetLoopNest();
                auto loopIndex = schedule.CurrentLoopIndex();

                auto dimensionIndex = schedule.CurrentDimension();

                LoopFragmentFlags flags = state.fragmentStates.count(dimensionIndex) == 0 ? LoopFragmentFlags::All() : state.fragmentStates.at(dimensionIndex);
                flags &= r.futureLoopFragmentFlags;
                if (r.futureLoopFragmentFlags.GetFlag(LoopFragmentType::boundary))
                {
                    flags.SetFlag(LoopFragmentType::boundary, true);
                }

                // Note: it's important that this code not be moved outside of the `codegenFn` lambda, otherwise Compute will incorrectly use old info for subsequent ranges
                auto newState = state;
                newState.currentFragment = flags;
                newState.fragmentStates[dimensionIndex] = flags;
                newState.loopIndices.insert_or_assign(loopIndex, LoopIndexSymbolTableEntry{ loopIndex, index, Range{ 0, r.stop.Get<int>() - r.start.Get<int>() }, LoopIndexState::inProgress });

                // set the active range for the current dimension based on the loop range given
                newState.activeDimensionRanges.insert_or_assign(dimensionIndex, Range{ 0, r.stop.Get<int>() - r.start.Get<int>() });

                // Should we use 'flags' or 'r.futureLoopFragmentFlags' in GetValidKernels call?
                auto prologueKernels = GetValidKernels(newState.activeKernels, newState.fragmentStates, r.futureLoopFragmentFlags, LoopFragmentType::prologue, schedule);
                auto bodyKernels = GetValidKernels(newState.activeKernels, newState.fragmentStates, r.futureLoopFragmentFlags, LoopFragmentType::body, schedule);
                auto epilogueKernels = GetValidKernels(newState.activeKernels, newState.fragmentStates, r.futureLoopFragmentFlags, { LoopFragmentType::epilogue }, schedule);

                // Concatenate kernel lists together
                std::vector<ScheduledKernel> thisLoopKernels;
                thisLoopKernels.insert(thisLoopKernels.begin(), prologueKernels.begin(), prologueKernels.end());
                thisLoopKernels.insert(thisLoopKernels.begin(), bodyKernels.begin(), bodyKernels.end());
                thisLoopKernels.insert(thisLoopKernels.begin(), epilogueKernels.begin(), epilogueKernels.end());

                DefineComputedIndexVariables(newState.loopIndices, thisLoopKernels, schedule);
                auto indexVariables = GetRuntimeIndexVariables(newState.loopIndices, loopNest);

                // erase all kernels in newState.activeKernels with the same ID as ones we're going to execute
                for (const auto& k : thisLoopKernels)
                {
                    auto id = k.kernel.GetId();
                    auto it = std::remove_if(newState.activeKernels.begin(), newState.activeKernels.end(), [&](auto el) {
                        return el.kernel.GetId() == id;
                    });

                    newState.activeKernels.erase(it, newState.activeKernels.end());
                }

                // Prologue
                for (auto k : prologueKernels)
                {
                    InvokeKernel(k.kernel, k.predicate, indexVariables, schedule);
                }

                // Body
                for (auto k : bodyKernels)
                {
                    InvokeKernel(k.kernel, k.predicate, indexVariables, schedule);
                }

                // Recursively generate the loops inside this one
                if (!newState.activeKernels.empty())
                {
                    GenerateLoopsOld(newState, schedule.Next());
                }

                // TODO: restore state of variables
                // debugging
                {
                    DefineComputedIndexVariables(newState.loopIndices, thisLoopKernels, schedule);
                    indexVariables = GetRuntimeIndexVariables(newState.loopIndices, loopNest);
                }

                for (auto k : epilogueKernels)
                {
                    InvokeKernel(k.kernel, k.predicate, indexVariables, schedule);
                }
            };
        }

        int LoopNestVisitor::GetMainBodyLoopEnd(const RecursionState& state, const LoopVisitSchedule& schedule, const Range& loopRange) const
        {
            if (!LoopInEndBoundaryFragment(state, schedule))
            {
                return schedule.NonBoundaryEnd();
            }

            auto rangeSize = loopRange.Size();
            auto increment = loopRange.Increment();
            int remainder = rangeSize % increment;
            int nonBoundarySize = rangeSize - remainder;
            return loopRange.Begin() + nonBoundarySize;
        }

        bool LoopNestVisitor::LoopInEndBoundaryFragment(const RecursionState& state, const LoopVisitSchedule& schedule) const
        {
            auto loopIndex = schedule.CurrentLoopIndex();
            auto dimensionIndex = schedule.GetDomain().GetBaseIndex(loopIndex);
            return ((state.fragmentStates.count(dimensionIndex) != 0) && state.fragmentStates.at(dimensionIndex).GetFlag(LoopFragmentType::boundary));
        }

        void LoopNestVisitor::DefineComputedIndexVariables(LoopIndexSymbolTable& indexVariables, const std::vector<ScheduledKernel>& activeKernels, const LoopVisitSchedule& schedule) const
        {
            const auto& loopNest = schedule.GetLoopNest();
            const auto& domain = schedule.GetDomain();
            int numDimensions = domain.NumDimensions();

            // define all computed index variables (that are used)
            std::set<Index> usedIndices;
            for (int d = 0; d < numDimensions; ++d)
            {
                auto computedIndices = domain.GetComputedIndicesForDimension(domain.GetBaseIndex(d));
                for (auto index : computedIndices)
                {
                    if (loopNest.IsUsed(index, activeKernels))
                    {
                        usedIndices.insert(index);
                    }
                }
            }

            for (const auto& index : usedIndices)
            {
                auto expr = loopNest.GetIndexExpression(index);
                auto indexValue = EmitIndexExpression(index, expr, indexVariables);
                indexVariables.insert_or_assign(index, LoopIndexSymbolTableEntry{ schedule.CurrentLoopIndex(), indexValue, Range{ 0, 0, 0 }, LoopIndexState::inProgress });
            }
        }

        bool LoopNestVisitor::IsPlacementValid(const ScheduledKernel& kernel, const LoopIndexSymbolTable& runtimeLoopIndices, const LoopVisitSchedule& schedule) const
        {
            const auto& domain = schedule.GetDomain();
            if (kernel.placement.IsEmpty() || IsBodyPlacementPredicate(kernel.placement))
            {
                // TODO: put this in a function that preprocesses the kernel predicates when adding the kernels to the schedule
                for (const auto& kernelIndex : kernel.kernel.GetIndices())
                {
                    for (const auto& loopIndex : domain.GetDependentLoopIndices(kernelIndex, true))
                    {
                        // if not defined(loopIndex) return false;
                        if (runtimeLoopIndices.count(loopIndex) == 0 || runtimeLoopIndices.at(loopIndex).state == LoopIndexState::done)
                        {
                            return false;
                        }
                    }
                }

                if (kernel.placement.IsEmpty())
                {
                    return true;
                }
            }

            auto evalPlacement = [&](const auto& evalPlacement, const KernelPredicate& p) -> bool {
                if (p.IsAlwaysTrue())
                {
                    return true;
                }
                else if (p.Is<FragmentTypePredicate>())
                {
                    throw utilities::InputException(utilities::InputExceptionErrors::invalidArgument, "Fragment predicates not valid for placement");
                }
                else if (auto placementPred = p.As<PlacementPredicate>(); placementPred != nullptr)
                {
                    if (schedule.IsInnermostLoop())
                    {
                        return !placementPred->HasIndex();
                    }

                    auto nextLoopIndex = schedule.Next().CurrentLoopIndex();
                    auto where = placementPred->GetPlacement();

                    std::vector<Index> dependentLoopIndices;
                    if (placementPred->HasIndex())
                    {
                        auto testIndex = placementPred->GetIndex();

                        // get list of dependent indices
                        dependentLoopIndices = domain.GetDependentLoopIndices(testIndex, true);

                        // First check that we're not already inside any dependent loops
                        for (const auto& i : dependentLoopIndices)
                        {
                            if (runtimeLoopIndices.count(i) != 0 && runtimeLoopIndices.at(i).state == LoopIndexState::inProgress)
                            {
                                return false;
                            }
                        }
                    }
                    else
                    {
                        dependentLoopIndices = { nextLoopIndex };
                    }

                    // Now check that the next loop at least partially defines the index in question
                    if (std::find(dependentLoopIndices.begin(), dependentLoopIndices.end(), nextLoopIndex) != dependentLoopIndices.end())
                    {
                        // Finally, check that we're in the correct position (before vs. after)
                        if (where == Placement::before)
                        {
                            return (runtimeLoopIndices.count(nextLoopIndex) == 0 || runtimeLoopIndices.at(nextLoopIndex).state == LoopIndexState::notVisited);
                        }
                        else // (where == Placement::after)
                        {
                            return (runtimeLoopIndices.count(nextLoopIndex) != 0 && runtimeLoopIndices.at(nextLoopIndex).state == LoopIndexState::done);
                        }
                    }
                    return false;
                }
                else if (auto definedPred = p.As<IndexDefinedPredicate>(); definedPred != nullptr)
                {
                    auto definedIndex = definedPred->GetIndex();
                    return (runtimeLoopIndices.count(definedIndex) > 0) && (runtimeLoopIndices.at(definedIndex).state != LoopIndexState::done);
                }
                else if (auto conjunction = p.As<KernelPredicateConjunction>(); conjunction != nullptr)
                {
                    bool result = true;
                    for (const auto& t : conjunction->GetTerms())
                    {
                        result &= evalPlacement(evalPlacement, *t);
                    }
                    return result;
                }
                else if (auto disjunction = p.As<KernelPredicateDisjunction>(); disjunction != nullptr)
                {
                    bool result = false;
                    for (const auto& t : disjunction->GetTerms())
                    {
                        result |= evalPlacement(evalPlacement, *t);
                    }
                    return result;
                }
                else
                {
                    return false;
                }
            };

            return evalPlacement(evalPlacement, kernel.placement);
        }

        std::vector<ScheduledKernel> LoopNestVisitor::GetValidKernels(const ScheduledKernelGroup& kernelGroup, const LoopIndexSymbolTable& runtimeIndexVariables, const LoopVisitSchedule& schedule) const
        {
            std::vector<ScheduledKernel> validKernels;
            std::copy_if(kernelGroup.kernels.begin(), kernelGroup.kernels.end(), std::back_inserter(validKernels), [&](const ScheduledKernel& k) {
                if (!IsPlacementValid(k, runtimeIndexVariables, schedule))
                {
                    return false;
                }
                auto predicate = schedule.GetKernelPredicate(k).Simplify(runtimeIndexVariables, schedule);
                if (predicate.IsAlwaysFalse())
                {
                    return false;
                }
                return true;
            });
            return validKernels;
        }

        LoopIndexSymbolTable LoopNestVisitor::GetRuntimeIndexVariables(const LoopIndexSymbolTable& runtimeLoopIndices, const LoopNest& loopNest) const
        {
            int numDimensions = loopNest.NumDimensions();

            // Start with the concrete loop indices
            LoopIndexSymbolTable indexVariables = runtimeLoopIndices;

            // ...and add the variables we need to compute (because they represent an index that has been split)
            for (int d = 0; d < numDimensions; ++d)
            {
                auto computedIndices = loopNest.GetDomain().GetComputedIndicesForDimension(loopNest.GetDomain().GetBaseIndex(d));
                for (auto index : computedIndices)
                {
                    auto runtimeVarIter = runtimeLoopIndices.find(index);
                    if (runtimeVarIter != runtimeLoopIndices.end())
                    {
                        indexVariables.insert_or_assign(index, runtimeVarIter->second);
                    }
                }
            }
            return indexVariables;
        }

        void LoopNestVisitor::DefinePostLoopIndex(const Index& loopIndex, LoopIndexSymbolTable& runtimeLoopIndices, const LoopVisitSchedule& schedule) const
        {
            auto loopRange = GetLoopRange(loopIndex, runtimeLoopIndices, schedule);
            auto lastVal = loopRange.End();
            runtimeLoopIndices.insert_or_assign(loopIndex, LoopIndexSymbolTableEntry{ loopIndex, lastVal, loopRange, LoopIndexState::done });
        }

        std::vector<ScheduledKernel> LoopNestVisitor::GetValidKernels(std::vector<ScheduledKernel> activeKernels, const std::unordered_map<Index, LoopFragmentFlags>& fragmentStates, LoopFragmentFlags currentLoopFlags, LoopFragmentFlags kernelFilter, const LoopVisitSchedule& schedule) const
        {
            std::vector<ScheduledKernel> result;
            for (auto fragmentType : { LoopFragmentType::prologue, LoopFragmentType::body, LoopFragmentType::epilogue })
            {
                for (const auto& kernel : activeKernels)
                {
                    if (kernelFilter.GetFlag(kernel.constraints.GetPlacement()))
                    {
                        // This should only run in a loop fragment of type 'fragmentType' and allowed by currentLoopFlags
                        if (ShouldRunKernel(kernel, fragmentType, fragmentStates, currentLoopFlags, schedule))
                        {
                            result.push_back(kernel);
                        }
                    }
                }
            }

            return result;
        }

        bool LoopNestVisitor::ShouldRunKernel(const ScheduledKernel& kernel, LoopFragmentType kernelPlacement, const std::unordered_map<Index, LoopFragmentFlags>& constraintIndices, LoopFragmentFlags currentLoopFlags, const LoopVisitSchedule& schedule) const
        {
            const auto& where = kernel.constraints;
            auto placement = where.GetPlacement();
            bool isBodyKernel = where.GetBoundaryIndices().size() == 0;
            if (isBodyKernel)
                placement = LoopFragmentType::body;

            // if (where.GetPlacement() != kernelPlacement)
            if (placement != kernelPlacement)
            {
                return false;
            }

            // bool enforceBoundary = where.GetBoundaryIndices().size() != 0 || (currentLoopFlags.GetFlag(LoopFragmentType::prologue) || currentLoopFlags.GetFlag(LoopFragmentType::epilogue));
            bool enforceBoundary = true;
            if (enforceBoundary && !currentLoopFlags.GetFlag(kernelPlacement))
            {
                return false;
            }

            // Are we at the correct loop level (are all the indices needed by the kernel defined)?
            // TODO: We want to only fire on a loop involving a leaf child of the index
            auto insideIndices = where.GetRequiredIndices();
            if (!insideIndices.empty())
            {
                if (currentLoopFlags.GetFlag(kernelPlacement) && where.GetBoundaryIndices().size() != 0)
                {
                    if (schedule.CurrentNestLevel() == 0)
                    {
                        return false;
                    }
                    if (!AreAllFullyDefined(insideIndices, schedule))
                    {
                        return false;
                    }
                }
                else
                {
                    if (!AreAllFullyDefined(insideIndices, schedule))
                    {
                        return false;
                    }

                    // We want to return true only when _this_ loop defines all the indices, so let's check that the parent
                    // loop wasn't also a valid candidate (but only perform this check if we're not on the first loop)
                    if (schedule.CurrentNestLevel() != 0)
                    {
                        if (AreAllFullyDefined(insideIndices, schedule.Prev()))
                        {
                            return false;
                        }
                    }
                }
            }

            // are we part of a prologue/epilogue for the indices we were constrained with?
            for (const auto& outsideIndex : where.GetBoundaryIndices())
            {
                auto it = constraintIndices.find(outsideIndex);
                if (it == constraintIndices.end() || !it->second.GetFlag(kernelPlacement))
                {
                    return false;
                }

                // is this the innermost loop level (or later) for the given constraint index?
                // (to check, just ensure there are no more loops after this one with the same dimension index)
                if (schedule.Next().WillVisitIndex(outsideIndex))
                {
                    return false;
                }
            }

            return true;
        }

        bool LoopNestVisitor::WillKernelRunInThisLoop(const ScheduledKernel& kernel, LoopFragmentFlags kernelFilter, const LoopVisitSchedule& schedule) const
        {
            // return true if:
            // 1) constraints position allowed by kernelFilter
            // 2) all required indices exist
            // 3) none of boundary indices exist (except perhaps for current loop?)
            const auto& where = kernel.constraints;
            if (!kernelFilter.GetFlag(where.GetPlacement()))
            {
                return false;
            }

            // are we at the correct loop level (are all the indices needed by the kernel defined)?
            // TODO: need to allow using non-"dimension" indices as well (for non-innermost kernels)
            auto insideIndices = where.GetRequiredIndices();
            if (!insideIndices.empty())
            {
                // If all the required indices aren't defined yet, fail
                if (!AreAllFullyDefined(insideIndices, schedule))
                {
                    return false;
                }

                // We want to return true only when _this_ loop defines all the indices, so let's check that the parent
                // loop wasn't also a valid candidate (but only perform this check if we're not on the first loop)
                if (schedule.CurrentNestLevel() != 0)
                {
                    if (AreAllFullyDefined(insideIndices, schedule.Prev()))
                    {
                        return false;
                    }
                }
            }

            // are we part of a prologue/epilogue for the indices we were constrained with?
            for (const auto& outsideIndex : where.GetBoundaryIndices())
            {
                if (schedule.IsDone())
                {
                    return false;
                }

                if (schedule.Next().WillVisitIndex(outsideIndex))
                {
                    return false;
                }
            }

            return true;
        }

        bool LoopNestVisitor::IsIdentity(const IndexExpression& expr, const Index& index) const
        {
            return (expr.indices.size() == 1 &&
                    expr.indices[0].index == index &&
                    expr.indices[0].scale == 1 &&
                    expr.begin == 0);
        }

        bool LoopNestVisitor::AreAllFullyDefined(const std::vector<Index>& indices, const LoopVisitSchedule& schedule) const
        {
            for (const auto& index : indices)
            {
                if (!schedule.IsFullyDefined(index))
                {
                    return false;
                }
            }
            return true;
        }

    } // namespace loopnests
} // namespace value
} // namespace ell
