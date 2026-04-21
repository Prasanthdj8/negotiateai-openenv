"""
graders.py — NegotiateAI: Adversarial Procurement Arena
Episode-level graders for all 3 tasks.
EasyGrader, MediumGrader, HardGrader — all scores clamped to (1e-4, 1-1e-4).
Used by GRPO as the final reward signal after each complete episode.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

from models import (
    Contract,
    ContractStatus,
    EpisodeResult,
    ItemCategory,
    ProcurementObservation,
    Requirement,
    StepReward,
)


# ─────────────────────────────────────────────────────────────
# SAFE SCORE HELPER
# ─────────────────────────────────────────────────────────────

def _safe(value: float, lo: float = 1e-4, hi: float = 1 - 1e-4) -> float:
    """Clamp score and protect against NaN / inf — same pattern as Round 1."""
    if math.isnan(value):
        return lo
    if math.isinf(value):
        return hi if value > 0 else lo
    return float(max(lo, min(hi, value)))


def _ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division — returns default if denominator is zero."""
    if denominator <= 0:
        return default
    return numerator / denominator


# ─────────────────────────────────────────────────────────────
# BASE GRADER
# ─────────────────────────────────────────────────────────────

class BaseGrader(ABC):
    """
    Abstract base grader.
    All graders receive the complete episode state and return
    a score in (1e-4, 1-1e-4) plus a detailed breakdown.
    """

    TASK_ID: str = ""

    @abstractmethod
    def grade(
        self,
        requirements:   list[Requirement],
        contracts:      list[Contract],
        budget_total:   float,
        budget_spent:   float,
        total_steps:    int,
        total_weeks:    int,
        deception_caught: int,
        deception_total:  int,
        rival_contracts_won: int,
        our_contracts_won:   int,
        disruption_count:    int,
        cancel_penalties:    float,
        stockout_penalties:  float,
        market_benchmark:    float,
    ) -> dict[str, Any]:
        ...

    # ── Shared component calculators ──────────────────────────

    def _cost_savings_ratio(
        self,
        contracts: list[Contract],
        market_benchmark: float,
    ) -> float:
        """
        How much did the agent save vs paying full market price?
        savings = (benchmark_spend - actual_spend) / benchmark_spend
        """
        actual_spend = sum(
            c.total_value for c in contracts
            if c.status in (ContractStatus.FULFILLED, ContractStatus.ACTIVE)
        )
        if market_benchmark <= 0:
            return 0.0
        savings = market_benchmark - actual_spend
        return _ratio(max(0.0, savings), market_benchmark)

    def _fulfillment_rate(self, requirements: list[Requirement]) -> float:
        """What fraction of requirements were fulfilled on time?"""
        if not requirements:
            return 0.0
        fulfilled = sum(1 for r in requirements if r.fulfilled)
        return _ratio(fulfilled, len(requirements))

    def _critical_fulfillment_rate(self, requirements: list[Requirement]) -> float:
        """Fulfillment rate for critical items only (weighted higher in hard grader)."""
        critical = [r for r in requirements if r.is_critical]
        if not critical:
            return 1.0
        fulfilled = sum(1 for r in critical if r.fulfilled)
        return _ratio(fulfilled, len(critical))

    def _deception_catch_rate(
        self, caught: int, total: int
    ) -> float:
        """Fraction of deceptive/distressed suppliers identified."""
        if total <= 0:
            return 1.0   # no bad suppliers = full score
        return _ratio(caught, total)

    def _rival_outperformance(
        self, our_wins: int, rival_wins: int
    ) -> float:
        """
        Score based on head-to-head contract wins vs rival.
        0.5 = tied, >0.5 = beating rival, <0.5 = losing.
        """
        total = our_wins + rival_wins
        if total == 0:
            return 0.5
        return _ratio(our_wins, total)

    def _budget_efficiency(
        self,
        budget_total: float,
        budget_spent: float,
        cancel_penalties: float,
        stockout_penalties: float,
    ) -> float:
        """
        Budget compliance score.
        Penalises cancellation fees and stockout costs.
        Rewards staying within budget.
        """
        if budget_total <= 0:
            return 0.0
        penalty_ratio = _ratio(
            cancel_penalties + stockout_penalties, budget_total
        )
        overspend_ratio = max(
            0.0, _ratio(budget_spent - budget_total, budget_total)
        )
        score = 1.0 - penalty_ratio - overspend_ratio
        return max(0.0, score)

    def _disruption_recovery(
        self,
        requirements: list[Requirement],
        disruption_count: int,
    ) -> float:
        """
        How well did the agent recover from disruptions?
        No disruptions → full score by default.
        With disruptions → score based on fulfillment despite chaos.
        """
        if disruption_count == 0:
            return 1.0
        fulfilled = sum(1 for r in requirements if r.fulfilled)
        base = _ratio(fulfilled, len(requirements)) if requirements else 0.0
        # Bonus for recovering with multiple disruptions
        recovery_bonus = min(0.2, disruption_count * 0.05)
        return min(1.0, base + recovery_bonus)

    def _workflow_compliance(
        self,
        contracts: list[Contract],
        total_steps: int,
    ) -> float:
        """
        Simple proxy: fulfilled contracts vs failed contracts.
        More failed = worse workflow.
        """
        fulfilled = sum(
            1 for c in contracts if c.status == ContractStatus.FULFILLED
        )
        failed = sum(
            1 for c in contracts if c.status == ContractStatus.FAILED
        )
        total = fulfilled + failed
        if total == 0:
            return 0.5
        return _ratio(fulfilled, total)


# ─────────────────────────────────────────────────────────────
# EASY GRADER
# ─────────────────────────────────────────────────────────────

class EasyGrader(BaseGrader):
    """
    Task: easy_negotiation
    Formula: score = cost_savings_ratio × fulfillment_rate
    Focus: learn basic negotiation — get below list price, meet deadlines.
    Baseline: ~0.15 (random agent)
    Target:   ~0.60 (trained agent after 100-200 GRPO steps)
    """

    TASK_ID = "easy_negotiation"

    def grade(self, **kwargs) -> dict[str, Any]:
        requirements   = kwargs["requirements"]
        contracts      = kwargs["contracts"]
        market_benchmark = kwargs["market_benchmark"]

        savings     = self._cost_savings_ratio(contracts, market_benchmark)
        fulfillment = self._fulfillment_rate(requirements)

        raw   = savings * fulfillment
        score = _safe(raw)

        return {
            "task_id":           self.TASK_ID,
            "score":             score,
            "baseline":          0.15,
            "target":            0.60,
            "components": {
                "cost_savings_ratio": round(savings,     4),
                "fulfillment_rate":   round(fulfillment, 4),
            },
            "formula":   "cost_savings_ratio × fulfillment_rate",
            "explanation": (
                f"Cost savings: {savings:.1%} vs market. "
                f"Fulfillment: {fulfillment:.1%} of requirements. "
                f"Score: {score:.4f}."
            ),
        }


# ─────────────────────────────────────────────────────────────
# MEDIUM GRADER
# ─────────────────────────────────────────────────────────────

class MediumGrader(BaseGrader):
    """
    Task: medium_adversarial
    Formula:
        score = cost_savings  × 0.35
              + fulfillment   × 0.30
              + deception_catch × 0.20
              + cycle_time    × 0.15
    Focus: detect deceptive suppliers, outmanoeuvre rule-based rival,
           recover from one supplier going dark.
    Baseline: ~0.10
    Target:   ~0.50
    """

    TASK_ID = "medium_adversarial"

    WEIGHTS = {
        "cost_savings":    0.35,
        "fulfillment":     0.30,
        "deception_catch": 0.20,
        "cycle_time":      0.15,
    }

    def grade(self, **kwargs) -> dict[str, Any]:
        requirements      = kwargs["requirements"]
        contracts         = kwargs["contracts"]
        market_benchmark  = kwargs["market_benchmark"]
        deception_caught  = kwargs["deception_caught"]
        deception_total   = kwargs["deception_total"]
        total_steps       = kwargs["total_steps"]
        total_weeks       = kwargs["total_weeks"]
        cancel_penalties  = kwargs["cancel_penalties"]
        stockout_penalties = kwargs["stockout_penalties"]

        savings     = self._cost_savings_ratio(contracts, market_benchmark)
        fulfillment = self._fulfillment_rate(requirements)
        deception   = self._deception_catch_rate(deception_caught, deception_total)

        # Cycle time: reward completing procurement faster
        max_steps   = total_weeks * 10          # generous upper bound
        cycle_time  = max(0.0, 1.0 - _ratio(total_steps, max_steps))

        w = self.WEIGHTS
        raw = (
            savings     * w["cost_savings"]    +
            fulfillment * w["fulfillment"]      +
            deception   * w["deception_catch"]  +
            cycle_time  * w["cycle_time"]
        )
        score = _safe(raw)

        return {
            "task_id":  self.TASK_ID,
            "score":    score,
            "baseline": 0.10,
            "target":   0.50,
            "components": {
                "cost_savings_ratio":  round(savings,     4),
                "fulfillment_rate":    round(fulfillment, 4),
                "deception_catch_rate": round(deception,  4),
                "cycle_time_score":    round(cycle_time,  4),
            },
            "weights":  w,
            "formula":  (
                "cost_savings×0.35 + fulfillment×0.30 "
                "+ deception_catch×0.20 + cycle_time×0.15"
            ),
            "explanation": (
                f"Cost savings: {savings:.1%}. "
                f"Fulfillment: {fulfillment:.1%}. "
                f"Deception catch: {deception:.1%} "
                f"({deception_caught}/{deception_total}). "
                f"Cycle time: {cycle_time:.1%}. "
                f"Score: {score:.4f}."
            ),
        }


# ─────────────────────────────────────────────────────────────
# HARD GRADER
# ─────────────────────────────────────────────────────────────

class HardGrader(BaseGrader):
    """
    Task: hard_full_arena
    Formula:
        score = cost_savings        × 0.25
              + fulfillment         × 0.25
              + rival_outperform    × 0.20
              + disruption_recovery × 0.15
              + budget_compliance   × 0.10
              + deception_catch     × 0.05
    Focus: survive all 3 crises simultaneously —
           supply chain disruption + budget cut + LLM rival buyer.
    Baseline: ~0.05
    Target:   ~0.40
    """

    TASK_ID = "hard_full_arena"

    WEIGHTS = {
        "cost_savings":        0.25,
        "fulfillment":         0.25,
        "rival_outperform":    0.20,
        "disruption_recovery": 0.15,
        "budget_compliance":   0.10,
        "deception_catch":     0.05,
    }

    def grade(self, **kwargs) -> dict[str, Any]:
        requirements         = kwargs["requirements"]
        contracts            = kwargs["contracts"]
        market_benchmark     = kwargs["market_benchmark"]
        deception_caught     = kwargs["deception_caught"]
        deception_total      = kwargs["deception_total"]
        rival_contracts_won  = kwargs["rival_contracts_won"]
        our_contracts_won    = kwargs["our_contracts_won"]
        disruption_count     = kwargs["disruption_count"]
        budget_total         = kwargs["budget_total"]
        budget_spent         = kwargs["budget_spent"]
        cancel_penalties     = kwargs["cancel_penalties"]
        stockout_penalties   = kwargs["stockout_penalties"]

        savings     = self._cost_savings_ratio(contracts, market_benchmark)
        fulfillment = self._fulfillment_rate(requirements)
        critical_f  = self._critical_fulfillment_rate(requirements)

        # Blend overall and critical fulfillment (critical items weighted 60%)
        fulfillment_blended = 0.4 * fulfillment + 0.6 * critical_f

        rival       = self._rival_outperformance(our_contracts_won, rival_contracts_won)
        disruption  = self._disruption_recovery(requirements, disruption_count)
        budget      = self._budget_efficiency(
            budget_total, budget_spent, cancel_penalties, stockout_penalties
        )
        deception   = self._deception_catch_rate(deception_caught, deception_total)

        w = self.WEIGHTS
        raw = (
            savings              * w["cost_savings"]        +
            fulfillment_blended  * w["fulfillment"]         +
            rival                * w["rival_outperform"]    +
            disruption           * w["disruption_recovery"] +
            budget               * w["budget_compliance"]   +
            deception            * w["deception_catch"]
        )
        score = _safe(raw)

        return {
            "task_id":  self.TASK_ID,
            "score":    score,
            "baseline": 0.05,
            "target":   0.40,
            "components": {
                "cost_savings_ratio":        round(savings,              4),
                "fulfillment_rate":          round(fulfillment,          4),
                "critical_fulfillment_rate": round(critical_f,           4),
                "fulfillment_blended":       round(fulfillment_blended,  4),
                "rival_outperformance":      round(rival,                4),
                "disruption_recovery":       round(disruption,           4),
                "budget_compliance":         round(budget,               4),
                "deception_catch_rate":      round(deception,            4),
            },
            "weights":  w,
            "formula": (
                "cost_savings×0.25 + fulfillment×0.25 "
                "+ rival_outperform×0.20 + disruption_recovery×0.15 "
                "+ budget_compliance×0.10 + deception_catch×0.05"
            ),
            "explanation": (
                f"Cost savings: {savings:.1%}. "
                f"Fulfillment: {fulfillment:.1%} "
                f"(critical: {critical_f:.1%}). "
                f"Rival: {'winning' if rival > 0.5 else 'losing'} "
                f"({our_contracts_won} vs {rival_contracts_won}). "
                f"Disruptions survived: {disruption_count}. "
                f"Budget ok: {budget:.1%}. "
                f"Score: {score:.4f}."
            ),
        }


# ─────────────────────────────────────────────────────────────
# GRADER REGISTRY
# ─────────────────────────────────────────────────────────────

GRADERS: dict[str, BaseGrader] = {
    "easy_negotiation":   EasyGrader(),
    "medium_adversarial": MediumGrader(),
    "hard_full_arena":    HardGrader(),
    # aliases
    "easy":   EasyGrader(),
    "medium": MediumGrader(),
    "hard":   HardGrader(),
}


def get_grader(task_id: str) -> BaseGrader:
    if task_id not in GRADERS:
        raise ValueError(
            f"Unknown task_id: {task_id!r}. "
            f"Valid: {list(GRADERS.keys())}"
        )
    return GRADERS[task_id]


def grade_episode(
    task_id:            str,
    requirements:       list[Requirement],
    contracts:          list[Contract],
    budget_total:       float,
    budget_spent:       float,
    total_steps:        int,
    total_weeks:        int,
    deception_caught:   int   = 0,
    deception_total:    int   = 0,
    rival_contracts_won: int  = 0,
    our_contracts_won:  int   = 0,
    disruption_count:   int   = 0,
    cancel_penalties:   float = 0.0,
    stockout_penalties: float = 0.0,
    market_benchmark:   float | None = None,
) -> dict[str, Any]:
    """
    Convenience function — grade a complete episode.
    Called by app.py at episode end and by GRPO reward function.
    """
    grader = get_grader(task_id)

    # Auto-compute market benchmark if not provided
    if market_benchmark is None:
        market_benchmark = sum(
            c.agreed_price * c.quantity * 1.15   # 15% above agreed = rough market
            for c in contracts
            if c.status in (ContractStatus.FULFILLED, ContractStatus.ACTIVE)
        ) or budget_total * 0.8

    return grader.grade(
        requirements=requirements,
        contracts=contracts,
        budget_total=budget_total,
        budget_spent=budget_spent,
        total_steps=total_steps,
        total_weeks=total_weeks,
        deception_caught=deception_caught,
        deception_total=deception_total,
        rival_contracts_won=rival_contracts_won,
        our_contracts_won=our_contracts_won,
        disruption_count=disruption_count,
        cancel_penalties=cancel_penalties,
        stockout_penalties=stockout_penalties,
        market_benchmark=market_benchmark,
    )
