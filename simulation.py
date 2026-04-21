"""
simulation.py — NegotiateAI: Adversarial Procurement Arena
Market dynamics engine:
- Price volatility and market benchmarks
- Demand pattern simulation (weekly, seasonal)
- Disruption event scheduling
- Rival buyer pressure propagation
- Market signal generation
"""

from __future__ import annotations

import math
import random
import uuid
from typing import Any

from models import (
    DisruptionEvent,
    DisruptionType,
    ItemCategory,
    MarketSignal,
    SupplierReputation,
    SupplierView,
)


# ─────────────────────────────────────────────────────────────
# MARKET PRICE DYNAMICS
# ─────────────────────────────────────────────────────────────

# Base price volatility per category (weekly σ as fraction of price)
PRICE_VOLATILITY: dict[ItemCategory, float] = {
    ItemCategory.SOFTWARE:  0.03,   # stable — subscription models
    ItemCategory.HARDWARE:  0.08,   # volatile — supply chain sensitive
    ItemCategory.SERVICES:  0.05,   # moderate — labour market
}

# Seasonal price multipliers by month (1=Jan … 12=Dec)
# Maps to fiscal weeks: week 1-4 = Q1, 5-8 = Q2, 9-12 = Q3
SEASONAL_MULTIPLIERS: dict[ItemCategory, dict[int, float]] = {
    ItemCategory.SOFTWARE: {
        1: 0.95,   # Q1: post-renewal discounts
        2: 0.95,
        3: 1.00,
        4: 1.02,   # Q2: mid-year renewals
        5: 1.02,
        6: 1.00,
        7: 0.98,   # Q3: summer slow
        8: 0.98,
        9: 1.05,   # Q4: year-end budget flush
        10: 1.08,
        11: 1.10,
        12: 1.05,
    },
    ItemCategory.HARDWARE: {
        1: 0.92,   # post-holiday stock clearance
        2: 0.90,
        3: 0.95,
        4: 0.98,
        5: 1.00,
        6: 1.02,
        7: 0.97,   # summer lull
        8: 0.98,
        9: 1.05,   # back-to-school / enterprise refresh
        10: 1.08,
        11: 1.12,  # pre-holiday demand spike
        12: 1.10,
    },
    ItemCategory.SERVICES: {
        1: 1.05,   # new year headcount push
        2: 1.02,
        3: 1.00,
        4: 1.00,
        5: 0.97,
        6: 0.95,   # summer holidays
        7: 0.93,
        8: 0.95,
        9: 1.02,
        10: 1.05,
        11: 1.08,  # year-end projects
        12: 1.10,
    },
}

# Day-of-week demand multipliers (week index 1-12 maps to simulated months)
DOW_MULTIPLIERS = [1.0, 1.05, 1.10, 1.08, 1.15, 1.20, 0.90]
# Mon  Tue   Wed   Thu   Fri   Sat   Sun
# (Sun = low demand, Fri-Sat = high demand in B2B simulated sense)


class MarketSimulator:
    """
    Simulates realistic tech procurement market dynamics.
    Provides price benchmarks, demand signals, and volatility.
    """

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self._price_history: dict[str, list[float]] = {}  # supplier_id → prices

    # ── Price benchmarks ──────────────────────────────────────

    def get_market_price(
        self,
        category: ItemCategory,
        base_price: float,
        week: int,
        noise: bool = True,
    ) -> float:
        """
        Compute current market price for a category.
        Applies seasonal multiplier + Gaussian noise.
        """
        # Map week (1-12) to month (1-12)
        month = min(12, max(1, week))
        seasonal = SEASONAL_MULTIPLIERS[category].get(month, 1.0)
        price = base_price * seasonal

        if noise:
            sigma = PRICE_VOLATILITY[category]
            noise_factor = self.rng.gauss(1.0, sigma)
            price *= max(0.7, min(1.3, noise_factor))   # cap at ±30%

        return round(price, 2)

    def compute_benchmark(
        self,
        suppliers: list[SupplierView],
        quantity: int,
        week: int,
    ) -> float:
        """
        Market benchmark = average market price across active suppliers × quantity.
        Used by graders to compute cost savings.
        """
        if not suppliers:
            return 0.0
        active = [s for s in suppliers if s.is_active]
        if not active:
            active = suppliers

        prices = [
            self.get_market_price(s.category, s.base_price, week, noise=False)
            for s in active
        ]
        avg_price = sum(prices) / len(prices)
        return round(avg_price * quantity, 2)

    def price_trend(
        self,
        supplier_id: str,
        category: ItemCategory,
        base_price: float,
        weeks: int = 4,
        seed: int | None = None,
    ) -> list[float]:
        """Generate a price trend for a supplier over N weeks."""
        rng = random.Random(seed)
        prices = []
        current = base_price
        for w in range(1, weeks + 1):
            sigma = PRICE_VOLATILITY[category]
            drift = rng.gauss(0, sigma * current)
            current = max(base_price * 0.7, current + drift)
            prices.append(round(current, 2))
        return prices


# ─────────────────────────────────────────────────────────────
# DEMAND ENGINE
# ─────────────────────────────────────────────────────────────

class DemandEngine:
    """
    Stochastic demand simulation for tech procurement items.
    Models urgency spikes, seasonal patterns, and category behaviour.
    """

    # Category-specific demand elasticity to discount
    DISCOUNT_ELASTICITY: dict[ItemCategory, float] = {
        ItemCategory.SOFTWARE:  0.10,   # +10% demand per 10% discount
        ItemCategory.HARDWARE:  0.15,   # +15%
        ItemCategory.SERVICES:  0.08,   # +8% (less elastic)
    }

    # Urgency decay: demand drops as deadline approaches without action
    URGENCY_DECAY_WEEKS = 2   # last N weeks before deadline

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def daily_demand(
        self,
        base_quantity: int,
        category: ItemCategory,
        week: int,
        discount_pct: float = 0.0,
        weeks_to_deadline: int = 99,
    ) -> int:
        """
        Compute stochastic demand for a given week.
        Applies: seasonal × DOW × discount elasticity × urgency × noise
        """
        # Seasonal factor
        month = min(12, max(1, week))
        seasonal = SEASONAL_MULTIPLIERS[category].get(month, 1.0)

        # Simulated day-of-week (week number maps to a day pattern)
        dow_idx = week % 7
        dow = DOW_MULTIPLIERS[dow_idx]

        # Discount elasticity
        elasticity = self.DISCOUNT_ELASTICITY[category]
        discount_boost = 1.0 + (discount_pct / 10.0) * elasticity

        # Urgency decay (demand drops last 2 days before deadline)
        urgency = 1.0
        if weeks_to_deadline <= self.URGENCY_DECAY_WEEKS:
            urgency = max(0.5, weeks_to_deadline / self.URGENCY_DECAY_WEEKS)

        # Gaussian noise (σ=15%)
        noise = self.rng.gauss(1.0, 0.15)
        noise = max(0.5, min(1.5, noise))

        demand = base_quantity * seasonal * dow * discount_boost * urgency * noise
        return max(1, int(round(demand)))

    def demand_spike(
        self,
        base_quantity: int,
        multiplier: float = 2.0,
        seed: int | None = None,
    ) -> int:
        """Simulate an unexpected demand spike (e.g. new hire batch)."""
        rng = random.Random(seed)
        spike = base_quantity * multiplier * rng.uniform(0.8, 1.2)
        return max(1, int(round(spike)))

    def expected_weekly_demand(
        self,
        base_quantity: int,
        category: ItemCategory,
        total_weeks: int,
    ) -> float:
        """Average expected demand per week over the episode."""
        total = sum(
            SEASONAL_MULTIPLIERS[category].get(w, 1.0)
            for w in range(1, total_weeks + 1)
        )
        return (base_quantity * total) / total_weeks


# ─────────────────────────────────────────────────────────────
# DISRUPTION SCHEDULER
# ─────────────────────────────────────────────────────────────

class DisruptionScheduler:
    """
    Schedules and tracks market disruption events.
    Supports both pre-configured (task-level) and dynamic disruptions.
    """

    def __init__(
        self,
        disruption_config: list[dict[str, Any]],
        total_weeks: int,
        seed: int | None = None,
    ):
        self.rng = random.Random(seed)
        self.total_weeks = total_weeks
        self._schedule: list[dict[str, Any]] = list(disruption_config)
        self._triggered: list[DisruptionEvent] = []

    def get_scheduled_weeks(self) -> list[int]:
        """Return all weeks that have disruptions scheduled."""
        return [cfg.get("week", 0) for cfg in self._schedule]

    def pop_events_for_week(self, week: int) -> list[dict[str, Any]]:
        """Return and remove all events scheduled for this week."""
        due = [cfg for cfg in self._schedule if cfg.get("week") == week]
        self._schedule = [cfg for cfg in self._schedule if cfg.get("week") != week]
        return due

    def inject_dynamic_disruption(
        self,
        dtype: DisruptionType,
        week: int,
        **kwargs: Any,
    ) -> None:
        """Add a runtime disruption (e.g. triggered by rival activity)."""
        self._schedule.append({"type": dtype, "week": week, **kwargs})

    def record_triggered(self, event: DisruptionEvent) -> None:
        self._triggered.append(event)

    def triggered_count(self) -> int:
        return len(self._triggered)

    def all_triggered(self) -> list[DisruptionEvent]:
        return list(self._triggered)

    def remaining_count(self) -> int:
        return len(self._schedule)


# ─────────────────────────────────────────────────────────────
# RIVAL PRESSURE PROPAGATOR
# ─────────────────────────────────────────────────────────────

class RivalPressurePropagator:
    """
    Models how rival buyer activity spreads across the supplier network.
    When rival targets a supplier, pressure bleeds to related suppliers.
    Creates realistic market tension that the buyer agent must navigate.
    """

    # How much pressure bleeds to same-category suppliers
    CATEGORY_BLEED_FACTOR = 0.3

    # Pressure decay per week (rival pressure fades if not reinforced)
    DECAY_RATE = 0.15

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def propagate(
        self,
        suppliers: dict[str, SupplierView],
        rival_target_id: str,
        pressure_added: float = 0.3,
    ) -> dict[str, float]:
        """
        When rival targets a supplier, compute pressure updates
        for all suppliers in the same category.
        Returns dict of supplier_id → new pressure.
        """
        updates: dict[str, float] = {}

        if rival_target_id not in suppliers:
            return updates

        target = suppliers[rival_target_id]

        # Direct pressure on targeted supplier
        current = target.rival_pressure
        updates[rival_target_id] = min(1.0, current + pressure_added)

        # Bleed to same-category suppliers
        same_cat = [
            (sid, sv) for sid, sv in suppliers.items()
            if sv.category == target.category and sid != rival_target_id
        ]
        for sid, sv in same_cat:
            bleed = pressure_added * self.CATEGORY_BLEED_FACTOR
            bleed *= self.rng.uniform(0.5, 1.0)   # randomise bleed
            updates[sid] = min(1.0, sv.rival_pressure + bleed)

        return updates

    def decay_all(
        self,
        suppliers: dict[str, SupplierView],
    ) -> dict[str, float]:
        """
        Apply weekly pressure decay to all suppliers.
        Returns dict of supplier_id → decayed pressure.
        """
        updates: dict[str, float] = {}
        for sid, sv in suppliers.items():
            decayed = sv.rival_pressure * (1.0 - self.DECAY_RATE)
            updates[sid] = max(0.0, round(decayed, 3))
        return updates


# ─────────────────────────────────────────────────────────────
# MARKET SIGNAL GENERATOR
# ─────────────────────────────────────────────────────────────

class MarketSignalGenerator:
    """
    Generates realistic market intelligence signals.
    Some signals are genuine, some are noise — agent must learn to filter.
    """

    # Probability that a signal about a deceptive supplier is a true warning
    TRUE_SIGNAL_RATE: dict[str, float] = {
        "cooperative":  0.3,    # rarely generate warnings
        "aggressive":   0.5,    # sometimes leak pressure signals
        "deceptive":    0.7,    # often generate quality/capacity alerts
        "distressed":   0.8,    # frequently leak financial stress signals
    }

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def generate_weekly_signals(
        self,
        suppliers: list[SupplierView],
        reputations: list[SupplierReputation],
        week: int,
        rival_active: bool = False,
    ) -> list[MarketSignal]:
        """
        Generate market intelligence signals for this week.
        Mix of genuine warnings and noise.
        """
        signals: list[MarketSignal] = []
        rep_map = {r.supplier_id: r for r in reputations}

        for sv in suppliers:
            if not sv.is_active:
                continue

            rep = rep_map.get(sv.supplier_id)
            if not rep:
                continue

            # Community reputation signal
            if rep.community_rating < 0.65:
                signals.append(MarketSignal(
                    signal_id=str(uuid.uuid4()),
                    week=week,
                    signal_type="quality_alert",
                    supplier_id=sv.supplier_id,
                    description=(
                        f"{sv.name} community rating dropped to "
                        f"{rep.community_rating:.2f}. "
                        f"Known issues: {', '.join(rep.known_issues) or 'none reported'}."
                    ),
                    severity=round(1.0 - rep.community_rating, 2),
                ))

            # Capacity warning — high rival pressure
            if sv.rival_pressure > 0.6:
                signals.append(MarketSignal(
                    signal_id=str(uuid.uuid4()),
                    week=week,
                    signal_type="capacity_warning",
                    supplier_id=sv.supplier_id,
                    description=(
                        f"High buyer interest at {sv.name}. "
                        f"Available capacity may be limited this week."
                    ),
                    severity=round(sv.rival_pressure, 2),
                ))

            # Price shift signal (random, category-level)
            if self.rng.random() < PRICE_VOLATILITY[sv.category] * 2:
                direction = self.rng.choice(["rising", "falling"])
                pct = self.rng.uniform(3, 12)
                signals.append(MarketSignal(
                    signal_id=str(uuid.uuid4()),
                    week=week,
                    signal_type="price_shift",
                    supplier_id=sv.supplier_id,
                    category=sv.category,
                    description=(
                        f"{sv.category.value.title()} market prices {direction} "
                        f"~{pct:.0f}% this week. "
                        f"Check {sv.name} quotes."
                    ),
                    severity=round(pct / 100, 2),
                ))

        # Rival activity signal
        if rival_active and self.rng.random() < 0.4:
            high_pressure = [s for s in suppliers if s.rival_pressure > 0.5]
            if high_pressure:
                target = self.rng.choice(high_pressure)
                signals.append(MarketSignal(
                    signal_id=str(uuid.uuid4()),
                    week=week,
                    signal_type="rival_activity",
                    supplier_id=target.supplier_id,
                    description=(
                        f"Competitor activity detected at {target.name}. "
                        f"They may be negotiating for similar capacity."
                    ),
                    severity=round(target.rival_pressure, 2),
                ))

        return signals

    def generate_disruption_warning(
        self,
        week: int,
        disruption_week: int,
        dtype: DisruptionType,
        severity: float = 0.7,
    ) -> MarketSignal | None:
        """
        Generate an early warning 1-2 weeks before a disruption.
        Gives the agent a chance to hedge proactively.
        """
        weeks_ahead = disruption_week - week
        if weeks_ahead not in (1, 2):
            return None

        # Less certain warning 2 weeks out
        if weeks_ahead == 2 and self.rng.random() > 0.5:
            return None

        descriptions = {
            DisruptionType.SUPPLIER_DARK: (
                "Industry reports of supply chain stress. "
                "Some suppliers may face operational issues soon."
            ),
            DisruptionType.BUDGET_CUT: (
                "Finance team reviewing Q3 procurement spend. "
                "Budget adjustments possible next week."
            ),
            DisruptionType.RIVAL_LOCKOUT: (
                "Competitor procurement activity increasing. "
                "Key suppliers may have reduced availability soon."
            ),
            DisruptionType.QUALITY_SCANDAL: (
                "Industry watchdog investigating supplier quality claims. "
                "Announcements expected soon."
            ),
            DisruptionType.DEMAND_SPIKE: (
                "Headcount projections revised upward. "
                "Additional procurement requirements likely."
            ),
        }

        return MarketSignal(
            signal_id=str(uuid.uuid4()),
            week=week,
            signal_type="disruption_warning",
            description=descriptions.get(dtype, "Market disruption signal detected."),
            severity=severity * (0.5 if weeks_ahead == 2 else 1.0),
        )


# ─────────────────────────────────────────────────────────────
# MARKET BENCHMARK CALCULATOR
# ─────────────────────────────────────────────────────────────

def compute_episode_benchmark(
    suppliers: list[SupplierView],
    requirements_config: list[dict[str, Any]],
    week: int = 1,
    simulator: MarketSimulator | None = None,
) -> float:
    """
    Compute the total market benchmark for an episode.
    This is the 'fair market price' a naive buyer would pay —
    used by graders to measure cost savings.
    """
    if simulator is None:
        simulator = MarketSimulator()

    total = 0.0
    for req in requirements_config:
        category = req["category"]
        quantity = req["quantity"]

        # Find matching suppliers
        matching = [s for s in suppliers if s.category == category and s.is_active]
        if not matching:
            # Fallback: use any supplier
            matching = [s for s in suppliers if s.is_active] or suppliers

        if not matching:
            continue

        # Benchmark = average market price across matching suppliers
        avg_base = sum(s.base_price for s in matching) / len(matching)
        market_price = simulator.get_market_price(category, avg_base, week, noise=False)
        total += market_price * quantity

    return round(total, 2)


# ─────────────────────────────────────────────────────────────
# STRESS TEST UTILITY
# ─────────────────────────────────────────────────────────────

def stress_test_simulation(
    n_episodes: int = 100,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Run N random simulations and check for crashes / degenerate rewards.
    Used to validate env before onsite training.
    """
    import sys, unittest.mock as mock
    # Proper mock: return valid string content from LLM
    mock_client = mock.MagicMock()
    mock_choice = mock.MagicMock()
    mock_choice.message.content = "Our price is $1000/unit with 7-day delivery. Ready to proceed."
    mock_client.return_value.chat.completions.create.return_value.choices = [mock_choice]
    sys.modules['openai'] = mock_client

    from env import NegotiateAIEnv
    from models import ActionType, ProcurementAction

    results = {
        "episodes": n_episodes,
        "crashes":  0,
        "nan_rewards": 0,
        "zero_rewards": 0,
        "scores": [],
        "errors": [],
    }

    rng = random.Random(seed)

    for i in range(n_episodes):
        try:
            task = rng.choice(["easy_negotiation", "medium_adversarial"])
            env = NegotiateAIEnv()
            obs = env.reset(task_id=task, seed=i)

            for _ in range(20):   # max 20 steps per episode
                if env.done:
                    break

                # Random valid action
                if not obs.suppliers:
                    break
                sup  = rng.choice([s for s in obs.suppliers if s.is_active] or obs.suppliers)
                req  = rng.choice(obs.requirements)
                atype = rng.choice([
                    ActionType.NEGOTIATE,
                    ActionType.RAISE_PR,
                    ActionType.AWARD_CONTRACT,
                    ActionType.DEFER,
                ])

                action = ProcurementAction(
                    action_type=atype,
                    supplier_id=sup.supplier_id,
                    item_id=req.item_id,
                    message="Random test action",
                    proposed_price=rng.uniform(500, 1500),
                    proposed_quantity=rng.randint(1, 50),
                )

                obs, reward, done, info = env.step(action)

                import math as _math
                if _math.isnan(reward.total) or _math.isinf(reward.total):
                    results["nan_rewards"] += 1
                if reward.total <= 0:
                    results["zero_rewards"] += 1

            result = env.get_episode_result()
            results["scores"].append(result.total_score)

        except Exception as e:
            results["crashes"] += 1
            results["errors"].append(f"Episode {i}: {str(e)[:80]}")

    if results["scores"]:
        results["mean_score"]  = round(sum(results["scores"]) / len(results["scores"]), 4)
        results["min_score"]   = round(min(results["scores"]), 4)
        results["max_score"]   = round(max(results["scores"]), 4)

    return results


if __name__ == "__main__":
    print("Running simulation stress test (100 episodes)...")
    result = stress_test_simulation(n_episodes=100, seed=42)
    print(f"Episodes:     {result['episodes']}")
    print(f"Crashes:      {result['crashes']}")
    print(f"NaN rewards:  {result['nan_rewards']}")
    print(f"Mean score:   {result.get('mean_score', 'N/A')}")
    print(f"Score range:  {result.get('min_score', 'N/A')} – {result.get('max_score', 'N/A')}")
    if result["errors"]:
        print(f"Errors:       {result['errors'][:3]}")
    else:
        print("✅ No crashes or errors detected")
