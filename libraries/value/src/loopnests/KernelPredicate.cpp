////////////////////////////////////////////////////////////////////////////////////////////////////
//
//  Project:  Embedded Learning Library (ELL)
//  File:     KernelPredicate.cpp (value)
//  Authors:  Chuck Jacobs
//
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "loopnests/KernelPredicate.h"
#include "loopnests/LoopNest.h"
#include "loopnests/LoopNestVisitor.h"

#include <utilities/include/TypeTraits.h>

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

            bool Intersects(const Range& a, const Range& b)
            {
                int aIter = CeilDiv(a.End() - a.Begin(), a.Increment());
                int bIter = CeilDiv(b.End() - b.Begin(), b.Increment());

                if (aIter == 0 || bIter == 0)
                {
                    return false;
                }
                auto aLast = a.Begin() + (aIter - 1) * a.Increment();
                auto bLast = b.Begin() + (bIter - 1) * b.Increment();

                return aLast >= b.Begin() && a.Begin() <= bLast;
            }
        } // namespace

        //
        // ConstantPredicate
        //
        ConstantPredicate::ConstantPredicate(bool value) :
            _value(value)
        {
        }

        bool ConstantPredicate::GetValue() const
        {
            return _value;
        }

        //
        // FragmentTypePredicate
        //
        FragmentTypePredicate::FragmentTypePredicate(const Index& index, Fragment condition) :
            _index(index),
            _condition(condition)
        {
        }

        // bool FragmentTypePredicate::IsSatisfied(const std::vector<Index>& indices) const; // Perhaps pass in current loop state?
        const Index& FragmentTypePredicate::GetIndex() const
        {
            return _index;
        }

        Fragment FragmentTypePredicate::GetCondition() const
        {
            return _condition;
        }

        KernelPredicate FragmentTypePredicate::Simplify() const
        {
            if (_condition == Fragment::all)
            {
                return ConstantPredicate(true);
            }
            return *this;
        }

        KernelPredicate FragmentTypePredicate::Simplify(const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule) const
        {
            if (_condition == Fragment::all)
            {
                return ConstantPredicate(true);
            }

            const auto index = GetIndex();
            const auto condition = GetCondition();

            // Get all index variables dependent on the predicate index
            const auto& domain = schedule.GetLoopNest().GetDomain();
            auto loopIndices = domain.GetDependentLoopIndices(index, true);

            // Evaluate a little equality "sub-predicate" for each dependent variable. All of them must be true for the result to be true.
            for (auto loopIndex : loopIndices)
            {
                // TODO: move `GetLoopRage` somewhere else
                auto fullRange = LoopNestVisitor::GetLoopRange(loopIndex, indices, schedule);

                int testVal = 0;
                bool valid = true;
                switch (condition)
                {
                case Fragment::first:
                    testVal = fullRange.Begin();
                    break;
                case Fragment::last:
                    testVal = fullRange.End() - (fullRange.Size() % fullRange.Increment());
                    if (testVal == fullRange.End()) // not a boundary
                    {
                        testVal = fullRange.End() - fullRange.Increment();
                    }
                    break;
                case Fragment::endBoundary:
                    testVal = fullRange.End() - (fullRange.Size() % fullRange.Increment());
                    if (testVal == fullRange.End()) // not a boundary
                    {
                        valid = false;
                    }
                    break;
                default:
                    valid = false;
                    // throw?
                    break;
                }

                if (valid)
                {
                    // Loop up range of the active loop
                    auto activeRange = fullRange;
                    if (const auto it = indices.find(loopIndex); it != indices.end())
                    {
                        if (it->second.state == LoopIndexState::inProgress)
                        {
                            activeRange = it->second.loopRange;
                        }
                    }

                    // Now check if testVal intersects with the loop's range
                    if (activeRange.Increment() == 0) // bad range
                    {
                        return *this;
                    }
                    int numIterations = CeilDiv(activeRange.End() - activeRange.Begin(), activeRange.Increment());
                    if (numIterations == 0)
                    {
                        return *this;
                    }

                    if (Intersects(activeRange, { testVal, testVal + 1 }))
                    {
                        if (numIterations == 1)
                        {
                            // true -- don't add anything to AND list
                        }
                        else
                        {
                            return *this;
                            // TODO: add index, testVal to AND list, later return a conjunction of equality predicates
                        }
                    }
                    else
                    {
                        return ConstantPredicate(false);
                    }
                }
            }
            return ConstantPredicate(true);
        }

        //
        // PlacementPredicate
        //
        PlacementPredicate::PlacementPredicate(Placement placement) :
            _index(std::nullopt),
            _placement(placement)
        {
        }

        PlacementPredicate::PlacementPredicate(const Index& index, Placement placement) :
            _index(index),
            _placement(placement)
        {
        }

        // bool PlacementPredicate::IsSatisfied(const std::vector<Index>& indices) const; // Perhaps pass in current loop state?
        bool PlacementPredicate::HasIndex() const
        {
            return _index.has_value();
        }

        Index PlacementPredicate::GetIndex() const
        {
            return _index.value();
        }

        Placement PlacementPredicate::GetPlacement() const
        {
            return _placement;
        }

        const PlacementPredicate& PlacementPredicate::Simplify() const
        {
            return *this;
        }

        const PlacementPredicate& PlacementPredicate::Simplify(const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule) const
        {
            return *this;
        }

        //
        // IndexDefinedPredicate
        //
        IndexDefinedPredicate::IndexDefinedPredicate(const Index& index) :
            _index(index)
        {
        }

        const Index& IndexDefinedPredicate::GetIndex() const
        {
            return _index;
        }

        const IndexDefinedPredicate& IndexDefinedPredicate::Simplify() const
        {
            return *this;
        }

        const IndexDefinedPredicate& IndexDefinedPredicate::Simplify(const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule) const
        {
            return *this;
        }

        //
        // KernelPredicateConjunction
        //
        KernelPredicateConjunction::KernelPredicateConjunction(const KernelPredicate& lhs, const KernelPredicate& rhs)
        {
            _terms.push_back(std::make_unique<KernelPredicate>(lhs));
            _terms.push_back(std::make_unique<KernelPredicate>(rhs));
        }

        KernelPredicateConjunction::KernelPredicateConjunction(const KernelPredicateConjunction& other) :
            KernelPredicateConjunction(other._terms)
        {
        }

        KernelPredicateConjunction::KernelPredicateConjunction(const std::vector<std::unique_ptr<KernelPredicate>>& terms)
        {
            for (const auto& t : terms)
            {
                _terms.emplace_back(std::make_unique<KernelPredicate>(*t));
            }
        }

        KernelPredicateConjunction& KernelPredicateConjunction::operator=(const KernelPredicateConjunction& other)
        {
            for (const auto& t : other._terms)
            {
                _terms.emplace_back(std::make_unique<KernelPredicate>(*t));
            }
            return *this;
        }

        const std::vector<std::unique_ptr<KernelPredicate>>& KernelPredicateConjunction::GetTerms() const
        {
            return _terms;
        }

        KernelPredicate KernelPredicateConjunction::Simplify() const
        {
            std::vector<std::unique_ptr<KernelPredicate>> terms;
            for (const auto& t : GetTerms())
            {
                auto simplifiedTerm = t->Simplify();
                if (simplifiedTerm.IsAlwaysFalse())
                {
                    return ConstantPredicate(false);
                }
                if (!simplifiedTerm.IsAlwaysTrue())
                {
                    terms.emplace_back(std::make_unique<KernelPredicate>(simplifiedTerm));
                }
            }

            if (terms.empty())
            {
                return KernelPredicate{};
            }
            else if (terms.size() == 1)
            {
                return *terms[0];
            }
            else
            {
                return KernelPredicateConjunction(terms);
            }
        }

        KernelPredicate KernelPredicateConjunction::Simplify(const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule) const
        {
            if (GetTerms().empty())
            {
                return EmptyPredicate();
            }

            std::vector<std::unique_ptr<KernelPredicate>> terms;
            for (const auto& t : GetTerms())
            {
                auto simplifiedTerm = t->Simplify(indices, schedule);
                if (simplifiedTerm.IsAlwaysFalse())
                {
                    return ConstantPredicate(false);
                }
                else if (!simplifiedTerm.IsAlwaysTrue())
                {
                    terms.emplace_back(std::make_unique<KernelPredicate>(simplifiedTerm));
                }
                // If always true, do nothing
            }

            if (terms.empty())
            {
                return ConstantPredicate(true);
            }
            else if (terms.size() == 1)
            {
                return *terms[0];
            }
            else
            {
                return KernelPredicateConjunction(terms);
            }
        }

        //
        // KernelPredicateDisjunction
        //
        KernelPredicateDisjunction::KernelPredicateDisjunction(const KernelPredicate& lhs, const KernelPredicate& rhs)
        {
            _terms.push_back(std::make_unique<KernelPredicate>(lhs));
            _terms.push_back(std::make_unique<KernelPredicate>(rhs));
        }

        KernelPredicateDisjunction::KernelPredicateDisjunction(const KernelPredicateDisjunction& other) :
            KernelPredicateDisjunction(other._terms)
        {
        }

        KernelPredicateDisjunction::KernelPredicateDisjunction(const std::vector<std::unique_ptr<KernelPredicate>>& terms)
        {
            for (const auto& t : terms)
            {
                _terms.emplace_back(std::make_unique<KernelPredicate>(*t));
            }
        }

        KernelPredicateDisjunction& KernelPredicateDisjunction::operator=(const KernelPredicateDisjunction& other)
        {
            for (const auto& t : other._terms)
            {
                _terms.emplace_back(std::make_unique<KernelPredicate>(*t));
            }
            return *this;
        }

        const std::vector<std::unique_ptr<KernelPredicate>>& KernelPredicateDisjunction::GetTerms() const
        {
            return _terms;
        }

        KernelPredicate KernelPredicateDisjunction::Simplify() const
        {
            if (GetTerms().empty())
            {
                return EmptyPredicate();
            }

            std::vector<std::unique_ptr<KernelPredicate>> terms;
            for (const auto& t : GetTerms())
            {
                auto simplifiedTerm = t->Simplify();
                if (simplifiedTerm.IsAlwaysTrue())
                {
                    return { ConstantPredicate(true) };
                }
                else if (!simplifiedTerm.IsAlwaysFalse())
                {
                    terms.emplace_back(std::make_unique<KernelPredicate>(simplifiedTerm));
                }
                // If always false, do nothing
            }
            if (terms.empty())
            {
                return ConstantPredicate(false);
            }
            else if (terms.size() == 1)
            {
                return *terms[0];
            }
            else
            {
                return KernelPredicateDisjunction(terms);
            }
        }

        KernelPredicate KernelPredicateDisjunction::Simplify(const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule) const
        {
            std::vector<std::unique_ptr<KernelPredicate>> terms;
            for (const auto& t : GetTerms())
            {
                auto simplifiedTerm = t->Simplify(indices, schedule);
                if (simplifiedTerm.IsAlwaysTrue())
                {
                    return { ConstantPredicate(true) };
                }
                else if (!simplifiedTerm.IsAlwaysFalse())
                {
                    terms.emplace_back(std::make_unique<KernelPredicate>(simplifiedTerm));
                }
            }
            if (terms.empty())
            {
                return KernelPredicate{};
            }
            else if (terms.size() == 1)
            {
                return *terms[0];
            }
            else
            {
                return KernelPredicateDisjunction(terms);
            }
        }

        //
        // KernelPredicate
        //
        KernelPredicate::KernelPredicate(const EmptyPredicate& predicate) :
            _expr(predicate)
        {}

        KernelPredicate::KernelPredicate(const ConstantPredicate& predicate) :
            _expr(predicate)
        {}

        KernelPredicate::KernelPredicate(const FragmentTypePredicate& predicate) :
            _expr(predicate)
        {}

        KernelPredicate::KernelPredicate(const PlacementPredicate& predicate) :
            _expr(predicate)
        {}

        KernelPredicate::KernelPredicate(const IndexDefinedPredicate& predicate) :
            _expr(predicate)
        {}

        KernelPredicate::KernelPredicate(const KernelPredicateConjunction& predicate) :
            _expr(predicate)
        {}

        KernelPredicate::KernelPredicate(const KernelPredicateDisjunction& predicate) :
            _expr(predicate)
        {}

        KernelPredicate KernelPredicate::Simplify() const
        {
            return std::visit(
                [](auto&& pred) -> KernelPredicate {
                    return pred.Simplify();
                },
                _expr);
        }

        KernelPredicate KernelPredicate::Simplify(const LoopIndexSymbolTable& indices, const LoopVisitSchedule& schedule) const
        {
            return std::visit(
                [&indices, &schedule](auto&& pred) -> KernelPredicate {
                    return pred.Simplify(indices, schedule);
                },
                _expr);
        }

        bool KernelPredicate::IsAlwaysTrue() const
        {
            if (IsEmpty())
            {
                return true;
            }
            auto simplePredicate = Simplify();
            if (auto constPred = simplePredicate.As<ConstantPredicate>(); constPred != nullptr)
            {
                return constPred->GetValue() == true;
            }
            return false;
        }

        bool KernelPredicate::IsAlwaysFalse() const
        {
            if (IsEmpty())
            {
                return false;
            }
            auto simplePredicate = Simplify();
            if (auto constPred = simplePredicate.As<ConstantPredicate>(); constPred != nullptr)
            {
                return constPred->GetValue() == false;
            }
            return false;
        }

        bool KernelPredicate::IsEmpty() const
        {
            return std::holds_alternative<EmptyPredicate>(_expr);
        }

        //
        // free functions
        //
        KernelPredicate First(const Index& index)
        {
            return FragmentTypePredicate{ index, Fragment::first };
        }

        KernelPredicate Last(const Index& index)
        {
            return FragmentTypePredicate{ index, Fragment::last };
        }

        KernelPredicate EndBoundary(const Index& index)
        {
            return FragmentTypePredicate{ index, Fragment::endBoundary };
        }

        KernelPredicate All(const Index& index)
        {
            return FragmentTypePredicate{ index, Fragment::all };
        }

        KernelPredicate Before(const Index& index)
        {
            return PlacementPredicate{ index, Placement::before };
        }

        KernelPredicate After(const Index& index)
        {
            return PlacementPredicate{ index, Placement::after };
        }

        KernelPredicate IsDefined(const Index& index)
        {
            return IndexDefinedPredicate{ index };
        }

        KernelPredicate operator&&(const KernelPredicate& lhs, const KernelPredicate& rhs)
        {
            return KernelPredicateConjunction{ lhs, rhs };
        }

        KernelPredicate operator||(const KernelPredicate& lhs, const KernelPredicate& rhs)
        {
            return KernelPredicateDisjunction{ lhs, rhs };
        }

        std::string ToString(Fragment cond)
        {
            switch (cond)
            {
            case Fragment::all:
                return "all";
            case Fragment::first:
                return "first";
            case Fragment::last:
                return "last";
            case Fragment::endBoundary:
                return "endBoundary";
            default:
                return "<>";
            }
        }

        std::string ToString(Placement where)
        {
            switch (where)
            {
            case Placement::before:
                return "before";
            case Placement::after:
                return "after";
            default:
                return "<>";
            }
        }

        std::ostream& operator<<(std::ostream& os, const EmptyPredicate& predicate)
        {
            os << "{}";
            return os;
        }

        std::ostream& operator<<(std::ostream& os, const ConstantPredicate& predicate)
        {
            os << (predicate.GetValue() ? "true" : "false");
            return os;
        }

        std::ostream& operator<<(std::ostream& os, const FragmentTypePredicate& predicate)
        {
            os << ToString(predicate.GetCondition()) << "(" << predicate.GetIndex() << ")";
            return os;
        }

        std::ostream& operator<<(std::ostream& os, const PlacementPredicate& predicate)
        {
            if (predicate.HasIndex())
            {
                os << ToString(predicate.GetPlacement()) << "(" << predicate.GetIndex() << ")";
            }
            else
            {
                os << ToString(predicate.GetPlacement()) << "()";
            }
            return os;
        }

        std::ostream& operator<<(std::ostream& os, const IndexDefinedPredicate& predicate)
        {
            os << "IsDefined(" << predicate.GetIndex() << ")";
            return os;
        }

        std::ostream& operator<<(std::ostream& os, const KernelPredicateConjunction& predicate)
        {
            const auto& terms = predicate.GetTerms();
            if (terms.size() == 0)
            {
                os << "true";
            }
            else if (terms.size() == 1)
            {
                os << *terms[0];
            }
            else
            {
                os << "(";
                bool first = true;
                for (const auto& t : terms)
                {
                    os << *t;
                    if (!first)
                    {
                        os << " && ";
                    }
                    first = false;
                }
                os << ")";
            }
            return os;
        }

        std::ostream& operator<<(std::ostream& os, const KernelPredicateDisjunction& predicate)
        {
            const auto& terms = predicate.GetTerms();
            if (terms.size() == 0)
            {
                os << "true";
            }
            else if (terms.size() == 1)
            {
                os << *terms[0];
            }
            else
            {
                os << "(";
                bool first = true;
                for (const auto& t : terms)
                {
                    os << *t;
                    if (!first)
                    {
                        os << " || ";
                    }
                    first = false;
                }
                os << ")";
            }
            return os;
        }

        std::ostream& operator<<(std::ostream& os, const KernelPredicate& predicate)
        {
            std::visit([&os](auto&& pred) {
                os << pred;
            },
                       predicate._expr);
            return os;
        }

    } // namespace loopnests
} // namespace value
} // namespace ell
