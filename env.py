"""
env.py — NegotiateAI: Adversarial Procurement Arena
Core environment: reset/step/state orchestration.
Manages episode lifecycle, enterprise workflow, market dynamics,
supplier negotiations, rival buyer, and curriculum scaling.
"""

from __future__ import annotations

import random
import uuid
from typing import Any

from models import (
    ActionType,
    ContractStatus,
    DisruptionEvent,
    DisruptionType,
    EpisodeResult,
    ItemCategory,
    MarketSignal,
    NegotiationThread,
    NegotiationTurn,
    PRStatus,
    ProcurementAction,
    ProcurementObservation,
    PurchaseRequisition,
    Requirement,
    StepReward,
    Contract,
)
from suppliers import RivalBuyerAgent, SupplierPool


# ─────────────────────────────────────────────────────────────
# TASK DEFINITIONS
# ─────────────────────────────────────────────────────────────

TASKS: dict[str, dict[str, Any]] = {
    "easy_negotiation": {
        "name": "Single Category Negotiation",
        "description": (
            "Negotiate software licenses with 5 cooperative suppliers. "
            "No rival, no disruptions. Learn the basics: compare offers, "
            "negotiate price, award best deal, track fulfillment."
        ),
        "num_suppliers": 5,
        "categories": [ItemCategory.SOFTWARE],
        "total_weeks": 4,
        "budget": 150_000.0,
        "has_rival": False,
        "rival_aggression": 0.0,
        "use_llm_suppliers": False,   # rule-based for fast training
        "use_llm_rival": False,
        "disruption_config": [],
        "requirements_config": [
            {"name": "Enterprise CRM Licenses", "category": ItemCategory.SOFTWARE,
             "quantity": 50, "deadline_week": 3, "budget_ceiling": 60_000.0, "is_critical": True},
            {"name": "Security Suite Licenses", "category": ItemCategory.SOFTWARE,
             "quantity": 30, "deadline_week": 4, "budget_ceiling": 35_000.0, "is_critical": False},
        ],
        "baseline_score": 0.15,
        "target_score": 0.60,
    },

    "medium_adversarial": {
        "name": "Multi-Category Adversarial Sourcing",
        "description": (
            "Source software and hardware across 12 suppliers. "
            "3 deceptive suppliers hidden in the pool. "
            "A rule-based rival buyer competes for hardware capacity. "
            "One supplier goes dark at week 5 — adapt and recover."
        ),
        "num_suppliers": 12,
        "categories": [ItemCategory.SOFTWARE, ItemCategory.HARDWARE],
        "total_weeks": 8,
        "budget": 350_000.0,
        "has_rival": True,
        "rival_aggression": 0.4,
        "use_llm_suppliers": True,
        "use_llm_rival": False,
        "disruption_config": [
            {"type": DisruptionType.SUPPLIER_DARK, "week": 5, "num_suppliers": 1},
        ],
        "requirements_config": [
            {"name": "Developer Laptops", "category": ItemCategory.HARDWARE,
             "quantity": 80, "deadline_week": 6, "budget_ceiling": 120_000.0, "is_critical": True},
            {"name": "Cloud Licenses (Annual)", "category": ItemCategory.SOFTWARE,
             "quantity": 100, "deadline_week": 4, "budget_ceiling": 90_000.0, "is_critical": False},
            {"name": "Network Switches", "category": ItemCategory.HARDWARE,
             "quantity": 20, "deadline_week": 7, "budget_ceiling": 50_000.0, "is_critical": False},
            {"name": "Security Software", "category": ItemCategory.SOFTWARE,
             "quantity": 60, "deadline_week": 5, "budget_ceiling": 55_000.0, "is_critical": True},
        ],
        "baseline_score": 0.10,
        "target_score": 0.50,
    },

    "hard_full_arena": {
        "name": "Full Adversarial Procurement Arena",
        "description": (
            "12-week procurement battle across all categories with 20 suppliers. "
            "Supply chain disruption weeks 5-7. Budget cut 20% at week 6. "
            "LLM-powered rival buyer competing aggressively throughout. "
            "Deceptive and distressed suppliers throughout the pool. Survive."
        ),
        "num_suppliers": 16,          # full catalogue
        "categories": [ItemCategory.SOFTWARE, ItemCategory.HARDWARE, ItemCategory.SERVICES],
        "total_weeks": 12,
        "budget": 600_000.0,
        "has_rival": True,
        "rival_aggression": 0.8,
        "use_llm_suppliers": True,
        "use_llm_rival": True,
        "disruption_config": [
            {"type": DisruptionType.SUPPLIER_DARK,  "week": 5, "num_suppliers": 2},
            {"type": DisruptionType.SUPPLIER_DARK,  "week": 6, "num_suppliers": 1},
            {"type": DisruptionType.BUDGET_CUT,     "week": 6, "cut_pct": 0.20},
            {"type": DisruptionType.RIVAL_LOCKOUT,  "week": 4, "num_suppliers": 2},
            {"type": DisruptionType.QUALITY_SCANDAL,"week": 8, "num_suppliers": 1},
        ],
        "requirements_config": [
            {"name": "Developer Laptops",        "category": ItemCategory.HARDWARE,
             "quantity": 120, "deadline_week": 8,  "budget_ceiling": 160_000.0, "is_critical": True},
            {"name": "Server Infrastructure",    "category": ItemCategory.HARDWARE,
             "quantity": 15,  "deadline_week": 10, "budget_ceiling": 90_000.0,  "is_critical": True},
            {"name": "Cloud Platform Licenses",  "category": ItemCategory.SOFTWARE,
             "quantity": 200, "deadline_week": 5,  "budget_ceiling": 180_000.0, "is_critical": False},
            {"name": "Security Suite (Annual)",  "category": ItemCategory.SOFTWARE,
             "quantity": 150, "deadline_week": 7,  "budget_ceiling": 120_000.0, "is_critical": True},
            {"name": "DevOps Contractors",       "category": ItemCategory.SERVICES,
             "quantity": 8,   "deadline_week": 6,  "budget_ceiling": 85_000.0,  "is_critical": False},
            {"name": "IT Support Services",      "category": ItemCategory.SERVICES,
             "quantity": 5,   "deadline_week": 9,  "budget_ceiling": 45_000.0,  "is_critical": False},
        ],
        "baseline_score": 0.05,
        "target_score": 0.40,
    },
}

# Accept short aliases
TASK_ALIASES = {
    "easy":   "easy_negotiation",
    "medium": "medium_adversarial",
    "hard":   "hard_full_arena",
}

MAX_NEGOTIATION_ROUNDS = 3      # per supplier per week
MAX_PR_APPROVAL_WEEKS  = 2      # weeks to auto-approve PR
CANCEL_PENALTY_PCT     = 0.15   # 15% of contract value
STOCKOUT_PENALTY       = 5_000.0  # per unfulfilled critical item


# ─────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────

class NegotiateAIEnv:
    """
    NegotiateAI: Adversarial Procurement Arena
    
    Manages:
    - Episode lifecycle (reset / step / done)
    - Enterprise workflow (PR → approval → PO → delivery)
    - Supplier negotiation (LLM-vs-LLM or rule-based)
    - Rival buyer agent (LLM or rule-based)
    - Market disruptions (supplier dark, budget cuts, rival lockouts)
    - Curriculum difficulty scaling
    - Step-level and episode-level rewards
    """

    def __init__(self, difficulty_level: float = 0.0):
        self.difficulty_level = difficulty_level    # 0-1, set by curriculum
        self._task_id: str | None = None
        self._task: dict[str, Any] | None = None
        self._reset_called = False

    # ── Reset ──────────────────────────────────────────────────

    def reset(self, task_id: str = "easy_negotiation", seed: int | None = None) -> ProcurementObservation:
        task_id = TASK_ALIASES.get(task_id, task_id)
        if task_id not in TASKS:
            raise ValueError(f"Unknown task: {task_id}. Valid: {list(TASKS.keys())}")

        self._task_id = task_id
        self._task    = TASKS[task_id]
        self._seed    = seed
        self._rng     = random.Random(seed)

        # ── Episode state ──────────────────────────────────────
        self.week               = 1
        self.total_weeks        = self._task["total_weeks"]
        self.budget_total       = self._task["budget"]
        self.budget_remaining   = self._task["budget"]
        self.total_steps        = 0
        self.done               = False

        # ── Suppliers ─────────────────────────────────────────
        self.supplier_pool = SupplierPool(
            num_suppliers=self._task["num_suppliers"],
            categories=self._task["categories"],
            seed=seed,
            use_llm=self._task["use_llm_suppliers"],
        )

        # ── Rival buyer ───────────────────────────────────────
        self.rival: RivalBuyerAgent | None = None
        if self._task["has_rival"]:
            rival_budget = self.budget_total * (0.8 + self.difficulty_level * 0.4)
            self.rival = RivalBuyerAgent(
                budget=rival_budget,
                use_llm=self._task["use_llm_rival"],
                aggression=self._task["rival_aggression"] + self.difficulty_level * 0.1,
                seed=seed,
            )

        # ── Requirements ──────────────────────────────────────
        self.requirements: list[Requirement] = []
        for i, cfg in enumerate(self._task["requirements_config"]):
            self.requirements.append(Requirement(
                item_id=f"item_{i+1:03d}",
                **cfg,
            ))

        # ── Enterprise workflow ───────────────────────────────
        self.purchase_requisitions: dict[str, PurchaseRequisition] = {}
        self.contracts:             dict[str, Contract]            = {}

        # ── Negotiation threads ───────────────────────────────
        self.threads: dict[str, NegotiationThread] = {}   # thread_id → thread

        # ── Market state ──────────────────────────────────────
        self.market_signals:  list[MarketSignal]   = []
        self.disruptions:     list[DisruptionEvent] = []
        self._disruption_schedule = list(self._task["disruption_config"])

        # ── Tracking ──────────────────────────────────────────
        self._cumulative_savings   = 0.0
        self._fulfilled_items      = 0
        self._failed_items         = 0
        self._deception_caught     = 0
        self._deception_total      = self._count_deceptive_suppliers()
        self._stockout_penalties   = 0.0
        self._cancel_penalties     = 0.0
        self._rival_contracts_won  = 0

        self._reset_called = True
        return self._build_observation()

    # ── Step ───────────────────────────────────────────────────

    def step(self, action: ProcurementAction) -> tuple[ProcurementObservation, StepReward, bool, dict]:
        if not self._reset_called:
            raise RuntimeError("Call reset() before step()")
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")

        self.total_steps += 1

        # 1. Advance week events
        self._advance_week_events()

        # 2. Process rival buyer action
        self._process_rival()

        # 3. Validate action
        valid, invalid_reason = self._validate_action(action)
        if not valid:
            reward = self._make_reward(
                cost_savings=0.0, fulfillment=0.0,
                deception_catch=0.0, rival_outperform=0.0,
                budget_compliance=0.0, disruption_recovery=0.0,
                workflow_compliance=0.0,
                explanation=f"Invalid action: {invalid_reason}",
            )
            obs = self._build_observation()
            self._maybe_advance_week()
            return obs, reward, self.done, {"invalid": invalid_reason}

        # 4. Dispatch action
        step_info: dict[str, Any] = {}
        reward_components = self._dispatch_action(action, step_info)

        # 5. Process pending deliveries
        self._process_deliveries()

        # 6. Check episode end
        self._check_done()

        reward = self._make_reward(**reward_components)
        obs    = self._build_observation()

        self._maybe_advance_week()
        return obs, reward, self.done, step_info

    # ── Action dispatcher ──────────────────────────────────────

    def _dispatch_action(
        self, action: ProcurementAction, info: dict
    ) -> dict[str, Any]:

        at = action.action_type

        if at == ActionType.NEGOTIATE:
            return self._do_negotiate(action, info)
        elif at == ActionType.AWARD_CONTRACT:
            return self._do_award(action, info)
        elif at == ActionType.REJECT:
            return self._do_reject(action, info)
        elif at == ActionType.RAISE_PR:
            return self._do_raise_pr(action, info)
        elif at == ActionType.ESCALATE:
            return self._do_escalate(action, info)
        elif at == ActionType.HEDGE:
            return self._do_hedge(action, info)
        elif at == ActionType.DEFER:
            return self._do_defer(action, info)
        elif at == ActionType.CANCEL_CONTRACT:
            return self._do_cancel(action, info)
        else:
            return self._neutral_reward("Unknown action type")

    # ── Individual action handlers ─────────────────────────────

    def _do_negotiate(self, action: ProcurementAction, info: dict) -> dict:
        sup = self.supplier_pool.visible.get(action.supplier_id)
        if not sup or not sup.is_active:
            return self._neutral_reward("Supplier inactive or not found")

        # Find or create thread
        thread_key = f"{action.supplier_id}_{action.item_id}"
        if thread_key not in self.threads:
            self.threads[thread_key] = NegotiationThread(
                thread_id=str(uuid.uuid4()),
                supplier_id=action.supplier_id,
                item_id=action.item_id,
                week=self.week,
            )
        thread = self.threads[thread_key]

        # Check round limit
        buyer_turns = sum(1 for t in thread.turns if t.role == "buyer")
        if buyer_turns >= MAX_NEGOTIATION_ROUNDS:
            return self._neutral_reward(
                "Max negotiation rounds reached for this supplier/item"
            )

        # Add buyer turn
        thread.turns.append(NegotiationTurn(
            turn=len(thread.turns) + 1,
            role="buyer",
            message=action.message,
            proposed_price=action.proposed_price,
            proposed_quantity=action.proposed_quantity,
            proposed_lead_time=action.proposed_lead_time,
        ))

        # Get supplier LLM response
        supplier_turn = self.supplier_pool.get_supplier_response(
            supplier_id=action.supplier_id,
            thread=thread,
            buyer_message=action.message,
            proposed_price=action.proposed_price,
            proposed_quantity=action.proposed_quantity,
        )
        thread.turns.append(supplier_turn)
        info["supplier_response"] = supplier_turn.message

        # Small positive reward for engaging (more for good opening price)
        price_quality = 0.0
        if action.proposed_price and supplier_turn.proposed_price:
            benchmark = sup.base_price
            if action.proposed_price < benchmark:
                price_quality = min(
                    0.3, (benchmark - action.proposed_price) / benchmark
                )

        return dict(
            cost_savings=price_quality,
            fulfillment=0.05,
            deception_catch=0.0,
            rival_outperform=0.0,
            budget_compliance=0.1,
            disruption_recovery=0.0,
            workflow_compliance=0.1,
            explanation=(
                f"Negotiated with {sup.name}. "
                f"Supplier responded: '{supplier_turn.message[:80]}...'"
            ),
        )

    def _do_award(self, action: ProcurementAction, info: dict) -> dict:
        sup = self.supplier_pool.visible.get(action.supplier_id)
        req = self._get_requirement(action.item_id)

        if not sup or not sup.is_active:
            return self._neutral_reward("Supplier inactive")
        if not req or req.fulfilled:
            return self._neutral_reward("Requirement not found or already fulfilled")
        if not action.proposed_price or not action.proposed_quantity:
            return self._neutral_reward("Price and quantity required to award")

        total_value = action.proposed_price * action.proposed_quantity

        # Check PR approval
        pr = self._get_approved_pr(action.item_id, action.supplier_id)
        workflow_score = 1.0
        if pr is None:
            # Award without approval — compliance penalty
            workflow_score = 0.2
            self._add_signal(
                "quality_alert",
                f"Contract awarded to {sup.name} without PR approval — compliance risk",
                severity=0.6,
            )

        # Check budget
        if total_value > self.budget_remaining:
            return self._neutral_reward(
                f"Insufficient budget: need ${total_value:,.0f}, "
                f"have ${self.budget_remaining:,.0f}"
            )

        # Deduct budget
        self.budget_remaining -= total_value

        # Calculate savings vs benchmark
        benchmark = sup.base_price * action.proposed_quantity
        savings = max(0.0, benchmark - total_value)
        self._cumulative_savings += savings
        savings_ratio = savings / benchmark if benchmark > 0 else 0.0

        # Create contract
        lead_weeks = max(1, (sup.lead_time_days + 6) // 7)
        contract = Contract(
            contract_id=str(uuid.uuid4()),
            supplier_id=action.supplier_id,
            item_id=action.item_id,
            quantity=action.proposed_quantity,
            agreed_price=action.proposed_price,
            total_value=total_value,
            lead_time_days=sup.lead_time_days,
            awarded_week=self.week,
            expected_delivery_week=self.week + lead_weeks,
            status=ContractStatus.ACTIVE,
        )
        self.contracts[contract.contract_id] = contract
        info["contract_id"] = contract.contract_id
        info["savings"] = savings

        # Detect deceptive supplier at award time (partial signal)
        deception_score = 0.0
        hid = self.supplier_pool.hidden.get(action.supplier_id)
        if hid and hid.supplier_type.value == "deceptive":
            # Reward if buyer negotiated price down significantly
            if savings_ratio > 0.15:
                deception_score = 0.3   # partial credit — got a good price anyway

        return dict(
            cost_savings=min(1.0, savings_ratio * 3),
            fulfillment=0.2,
            deception_catch=deception_score,
            rival_outperform=self._rival_outperform_score(),
            budget_compliance=workflow_score,
            disruption_recovery=0.0,
            workflow_compliance=workflow_score,
            explanation=(
                f"Contract awarded to {sup.name}: "
                f"{action.proposed_quantity} units @ ${action.proposed_price:.2f}. "
                f"Savings: ${savings:,.0f}. "
                f"{'⚠️ No PR approval.' if workflow_score < 1.0 else '✅ Compliant.'}"
            ),
        )

    def _do_reject(self, action: ProcurementAction, info: dict) -> dict:
        sup = self.supplier_pool.visible.get(action.supplier_id)
        if not sup:
            return self._neutral_reward("Supplier not found")

        # Reward if rejecting a genuinely bad supplier
        hid = self.supplier_pool.hidden.get(action.supplier_id)
        deception_score = 0.0
        if hid and hid.supplier_type.value in ("deceptive", "distressed"):
            deception_score = 0.4
            self._deception_caught += 1
            info["deception_caught"] = True
            self._add_signal(
                "quality_alert",
                f"Agent rejected {sup.name} — flagged as unreliable",
                severity=0.3,
                supplier_id=action.supplier_id,
            )

        return dict(
            cost_savings=0.0,
            fulfillment=0.0,
            deception_catch=deception_score,
            rival_outperform=0.0,
            budget_compliance=0.1,
            disruption_recovery=0.0,
            workflow_compliance=0.1,
            explanation=f"Rejected {sup.name}. {'Deceptive/distressed supplier identified.' if deception_score > 0 else 'Supplier walked.'}",
        )

    def _do_raise_pr(self, action: ProcurementAction, info: dict) -> dict:
        if not action.proposed_price or not action.proposed_quantity:
            return self._neutral_reward("Price and quantity required for PR")

        pr_id = f"pr_{action.item_id}_{action.supplier_id}_{self.week}"
        amount = action.proposed_price * action.proposed_quantity

        if amount > self.budget_remaining:
            return self._neutral_reward(
                f"PR amount ${amount:,.0f} exceeds remaining budget"
            )

        pr = PurchaseRequisition(
            pr_id=pr_id,
            item_id=action.item_id,
            supplier_id=action.supplier_id,
            amount=amount,
            raised_week=self.week,
            status=PRStatus.PENDING,
        )
        self.purchase_requisitions[pr_id] = pr
        info["pr_id"] = pr_id

        return dict(
            cost_savings=0.0,
            fulfillment=0.05,
            deception_catch=0.0,
            rival_outperform=0.0,
            budget_compliance=0.3,
            disruption_recovery=0.0,
            workflow_compliance=0.4,
            explanation=f"PR raised: ${amount:,.0f} for {action.item_id} from {action.supplier_id}. Awaiting approval.",
        )

    def _do_escalate(self, action: ProcurementAction, info: dict) -> dict:
        # Find pending PR for this item/supplier
        pr = self._get_pending_pr(action.item_id, action.supplier_id)
        if not pr:
            return self._neutral_reward("No pending PR found to escalate")

        pr.escalated = True
        pr.status = PRStatus.APPROVED
        pr.approved_week = self.week
        info["pr_id"] = pr.pr_id

        return dict(
            cost_savings=0.0,
            fulfillment=0.1,
            deception_catch=0.0,
            rival_outperform=0.05,
            budget_compliance=0.2,
            disruption_recovery=0.1,
            workflow_compliance=0.3,
            explanation=f"PR escalated and fast-approved for {action.item_id}. Ready to award contract.",
        )

    def _do_hedge(self, action: ProcurementAction, info: dict) -> dict:
        if not action.hedge_supplier_id or not action.hedge_quantity:
            return self._neutral_reward(
                "hedge_supplier_id and hedge_quantity required for hedge action"
            )

        sup1 = self.supplier_pool.visible.get(action.supplier_id)
        sup2 = self.supplier_pool.visible.get(action.hedge_supplier_id)

        if not sup1 or not sup1.is_active:
            return self._neutral_reward("Primary supplier inactive")
        if not sup2 or not sup2.is_active:
            return self._neutral_reward("Hedge supplier inactive")

        req = self._get_requirement(action.item_id)
        if not req:
            return self._neutral_reward("Requirement not found")

        # Compute hedge cost
        price1 = action.proposed_price or sup1.base_price
        price2 = action.proposed_price or sup2.base_price
        qty1   = action.proposed_quantity or (req.quantity // 2)
        qty2   = action.hedge_quantity
        total_cost = (price1 * qty1) + (price2 * qty2)

        if total_cost > self.budget_remaining:
            return self._neutral_reward("Insufficient budget for hedge")

        self.budget_remaining -= total_cost

        # Create two contracts
        for sid, price, qty in [(action.supplier_id, price1, qty1),
                                  (action.hedge_supplier_id, price2, qty2)]:
            sv = self.supplier_pool.visible[sid]
            lead_weeks = max(1, (sv.lead_time_days + 6) // 7)
            c = Contract(
                contract_id=str(uuid.uuid4()),
                supplier_id=sid,
                item_id=action.item_id,
                quantity=qty,
                agreed_price=price,
                total_value=price * qty,
                lead_time_days=sv.lead_time_days,
                awarded_week=self.week,
                expected_delivery_week=self.week + lead_weeks,
                status=ContractStatus.ACTIVE,
            )
            self.contracts[c.contract_id] = c

        info["hedged"] = True
        info["suppliers"] = [action.supplier_id, action.hedge_supplier_id]

        # Hedge is smart — reward appropriately
        # Extra reward if rival pressure is high (good strategic move)
        rival_pressure_avg = (
            (sup1.rival_pressure + sup2.rival_pressure) / 2
        )
        rival_score = rival_pressure_avg * 0.5

        return dict(
            cost_savings=0.1,
            fulfillment=0.3,
            deception_catch=0.0,
            rival_outperform=rival_score,
            budget_compliance=0.2,
            disruption_recovery=0.3,
            workflow_compliance=0.1,
            explanation=(
                f"Hedged {req.name}: {qty1} units from {sup1.name} + "
                f"{qty2} units from {sup2.name}. "
                f"Risk mitigated. Rival pressure: {rival_pressure_avg:.0%}."
            ),
        )

    def _do_defer(self, action: ProcurementAction, info: dict) -> dict:
        req = self._get_requirement(action.item_id)
        if not req:
            return self._neutral_reward("Requirement not found")

        weeks_left = req.deadline_week - self.week
        if weeks_left <= 1 and req.is_critical:
            # Deferring a critical item near deadline = bad
            self._stockout_penalties += STOCKOUT_PENALTY
            return dict(
                cost_savings=0.0,
                fulfillment=-0.3,
                deception_catch=0.0,
                rival_outperform=-0.1,
                budget_compliance=0.1,
                disruption_recovery=-0.1,
                workflow_compliance=0.0,
                explanation=(
                    f"⚠️ Deferred critical item {req.name} with only "
                    f"{weeks_left} week(s) to deadline. Stockout risk."
                ),
            )

        return dict(
            cost_savings=0.0,
            fulfillment=0.0,
            deception_catch=0.0,
            rival_outperform=0.0,
            budget_compliance=0.1,
            disruption_recovery=0.0,
            workflow_compliance=0.0,
            explanation=f"Deferred {req.name} — {weeks_left} weeks remaining.",
        )

    def _do_cancel(self, action: ProcurementAction, info: dict) -> dict:
        # Find active contract for this supplier/item
        contract = self._get_active_contract(action.supplier_id, action.item_id)
        if not contract:
            return self._neutral_reward("No active contract found to cancel")

        penalty = contract.total_value * CANCEL_PENALTY_PCT
        self._cancel_penalties += penalty
        contract.status = ContractStatus.CANCELLED
        info["penalty"] = penalty

        return dict(
            cost_savings=-0.2,
            fulfillment=-0.1,
            deception_catch=0.0,
            rival_outperform=0.0,
            budget_compliance=-0.1,
            disruption_recovery=0.1,
            workflow_compliance=0.0,
            explanation=(
                f"Contract cancelled with {action.supplier_id}. "
                f"Penalty: ${penalty:,.0f}."
            ),
        )

    # ── Week advancement ───────────────────────────────────────

    def _advance_week_events(self) -> None:
        """Process events scheduled for this week."""
        triggered = []
        remaining = []

        for cfg in self._disruption_schedule:
            if cfg.get("week") == self.week:
                self._trigger_disruption(cfg)
                triggered.append(cfg)
            else:
                remaining.append(cfg)

        self._disruption_schedule = remaining

        # Auto-approve PRs that have been pending long enough
        for pr in self.purchase_requisitions.values():
            if (pr.status == PRStatus.PENDING and
                    self.week - pr.raised_week >= MAX_PR_APPROVAL_WEEKS):
                pr.status = PRStatus.APPROVED
                pr.approved_week = self.week

    def _trigger_disruption(self, cfg: dict) -> None:
        dtype = cfg["type"]
        active_ids = [
            sid for sid, sv in self.supplier_pool.visible.items()
            if sv.is_active
        ]

        if dtype == DisruptionType.SUPPLIER_DARK:
            n = min(cfg.get("num_suppliers", 1), len(active_ids))
            victims = self._rng.sample(active_ids, n)
            for sid in victims:
                self.supplier_pool.take_supplier_dark(sid)
            disruption = DisruptionEvent(
                disruption_id=str(uuid.uuid4()),
                disruption_type=dtype,
                week_triggered=self.week,
                affected_suppliers=victims,
                description=(
                    f"{len(victims)} supplier(s) went offline due to "
                    f"supply chain disruption."
                ),
            )
            self.disruptions.append(disruption)
            self._add_signal(
                "disruption_warning",
                f"DISRUPTION: {len(victims)} supplier(s) offline this week.",
                severity=0.9,
            )

        elif dtype == DisruptionType.BUDGET_CUT:
            cut = cfg.get("cut_pct", 0.20)
            cut_amount = self.budget_remaining * cut
            self.budget_remaining -= cut_amount
            disruption = DisruptionEvent(
                disruption_id=str(uuid.uuid4()),
                disruption_type=dtype,
                week_triggered=self.week,
                affected_suppliers=[],
                description=f"Budget cut: ${cut_amount:,.0f} ({cut:.0%}) removed.",
                budget_impact=cut_amount,
            )
            self.disruptions.append(disruption)
            self._add_signal(
                "price_shift",
                f"BUDGET CUT: ${cut_amount:,.0f} removed from remaining budget.",
                severity=0.8,
            )

        elif dtype == DisruptionType.RIVAL_LOCKOUT:
            n = min(cfg.get("num_suppliers", 1), len(active_ids))
            targets = self._rng.sample(active_ids, n)
            for sid in targets:
                self.supplier_pool.update_rival_pressure(sid, 0.9)
            disruption = DisruptionEvent(
                disruption_id=str(uuid.uuid4()),
                disruption_type=dtype,
                week_triggered=self.week,
                affected_suppliers=targets,
                description=f"Rival buyer locked capacity at {len(targets)} supplier(s).",
            )
            self.disruptions.append(disruption)
            self._add_signal(
                "rival_activity",
                f"RIVAL LOCKOUT: Competitor securing capacity at {len(targets)} supplier(s).",
                severity=0.7,
            )

        elif dtype == DisruptionType.QUALITY_SCANDAL:
            n = min(cfg.get("num_suppliers", 1), len(active_ids))
            targets = self._rng.sample(active_ids, n)
            for sid in targets:
                rep = self.supplier_pool.reputations.get(sid)
                if rep:
                    rep.community_rating = max(0.1, rep.community_rating - 0.3)
                    rep.known_issues.append("Quality scandal — public complaint filed")
            disruption = DisruptionEvent(
                disruption_id=str(uuid.uuid4()),
                disruption_type=dtype,
                week_triggered=self.week,
                affected_suppliers=targets,
                description=f"Quality scandal exposed at {len(targets)} supplier(s).",
            )
            self.disruptions.append(disruption)
            self._add_signal(
                "quality_alert",
                f"QUALITY ALERT: Supplier quality scandal reported.",
                severity=0.7,
            )

    def _maybe_advance_week(self) -> None:
        """Advance week every N steps (one action per supplier per week approx)."""
        active_count = len(self.supplier_pool.get_active_suppliers())
        steps_per_week = max(3, active_count // 2)
        if self.total_steps % steps_per_week == 0:
            if self.week < self.total_weeks:
                self.week += 1

    # ── Rival processing ───────────────────────────────────────

    def _process_rival(self) -> None:
        if not self.rival:
            return
        active = self.supplier_pool.get_active_suppliers()
        req_names = [r.name for r in self.requirements if not r.fulfilled]
        action = self.rival.act(self.week, self.total_weeks, active, req_names)
        if action and action.get("action") == "award":
            sid = action.get("supplier_id")
            if sid in self.supplier_pool.visible:
                sv = self.supplier_pool.visible[sid]
                # Reduce capacity for buyer
                qty = action.get("proposed_quantity", 0)
                sv.capacity_available = max(0, sv.capacity_available - qty)
                self._rival_contracts_won += 1
                self.rival.record_win(
                    (action.get("proposed_price") or sv.base_price) * qty
                )

    # ── Delivery processing ────────────────────────────────────

    def _process_deliveries(self) -> None:
        for contract in self.contracts.values():
            if (contract.status == ContractStatus.ACTIVE and
                    contract.expected_delivery_week <= self.week):
                result = self.supplier_pool.resolve_delivery(
                    supplier_id=contract.supplier_id,
                    contract_id=contract.contract_id,
                    agreed_quantity=contract.quantity,
                    rng=self._rng,
                )
                contract.actual_delivery_week = self.week
                contract.quality_score = result.get("quality_score", 0.0)

                if result.get("went_dark"):
                    contract.status = ContractStatus.FAILED
                    self._failed_items += 1
                    self._add_signal(
                        "disruption_warning",
                        f"Supplier {contract.supplier_id} went dark — "
                        f"contract {contract.contract_id[:8]} failed.",
                        severity=0.9,
                        supplier_id=contract.supplier_id,
                    )
                elif result["delivered_quantity"] >= contract.quantity * 0.9:
                    contract.status = ContractStatus.FULFILLED
                    req = self._get_requirement(contract.item_id)
                    if req:
                        req.fulfilled = True
                    self._fulfilled_items += 1
                else:
                    contract.status = ContractStatus.FAILED
                    self._failed_items += 1

    # ── Episode end ────────────────────────────────────────────

    def _check_done(self) -> None:
        all_fulfilled = all(r.fulfilled for r in self.requirements)
        out_of_weeks  = self.week >= self.total_weeks
        out_of_budget = self.budget_remaining <= 0
        self.done = all_fulfilled or out_of_weeks or out_of_budget

    def get_episode_result(self) -> EpisodeResult:
        total_reqs     = len(self.requirements)
        fulfilled      = sum(1 for r in self.requirements if r.fulfilled)
        fulfillment_rt = fulfilled / total_reqs if total_reqs > 0 else 0.0

        spent         = self.budget_total - self.budget_remaining
        savings_ratio = self._cumulative_savings / max(1.0, spent + self._cumulative_savings)

        deception_rate = (
            self._deception_caught / max(1, self._deception_total)
        )

        rival_perf = self._rival_outperform_score()

        budget_ok = max(
            0.0, 1.0 - (self._cancel_penalties + self._stockout_penalties) / max(1.0, self.budget_total)
        )

        disruption_score = (
            fulfilled / max(1, total_reqs)
            if self.disruptions else 1.0
        )

        task_id = self._task_id or "unknown"
        if task_id == "easy_negotiation":
            total = savings_ratio * fulfillment_rt
        elif task_id == "medium_adversarial":
            total = (
                savings_ratio  * 0.35 +
                fulfillment_rt * 0.30 +
                deception_rate * 0.20 +
                min(1.0, self.total_steps / 20) * 0.15
            )
        else:
            total = (
                savings_ratio    * 0.25 +
                fulfillment_rt   * 0.25 +
                rival_perf       * 0.20 +
                disruption_score * 0.15 +
                budget_ok        * 0.10 +
                deception_rate   * 0.05
            )

        total = float(max(1e-4, min(1 - 1e-4, total)))

        return EpisodeResult(
            task_id=task_id,
            total_score=total,
            total_weeks=self.week,
            total_steps=self.total_steps,
            cost_savings_ratio=round(savings_ratio, 4),
            fulfillment_rate=round(fulfillment_rt, 4),
            deception_catch_rate=round(deception_rate, 4),
            rival_outperformance=round(rival_perf, 4),
            budget_compliance=round(budget_ok, 4),
            disruption_recovery=round(disruption_score, 4),
            contracts_awarded=sum(
                1 for c in self.contracts.values()
                if c.status in (ContractStatus.ACTIVE, ContractStatus.FULFILLED)
            ),
            contracts_failed=sum(
                1 for c in self.contracts.values()
                if c.status == ContractStatus.FAILED
            ),
            suppliers_rejected=self._deception_caught,
            total_spend=round(spent, 2),
            budget_saved=round(self._cumulative_savings, 2),
            summary=self._build_summary(total, fulfillment_rt, savings_ratio),
        )

    # ── Helpers ────────────────────────────────────────────────

    def _validate_action(self, action: ProcurementAction) -> tuple[bool, str]:
        if action.supplier_id not in self.supplier_pool.visible:
            return False, f"supplier_id {action.supplier_id!r} not in pool"
        req_ids = {r.item_id for r in self.requirements}
        # Allow rival item_id in award actions from rival (won't reach here)
        if action.item_id not in req_ids and action.action_type != ActionType.REJECT:
            return False, f"item_id {action.item_id!r} not in requirements"
        return True, ""

    def _get_requirement(self, item_id: str) -> Requirement | None:
        return next((r for r in self.requirements if r.item_id == item_id), None)

    def _get_approved_pr(self, item_id: str, supplier_id: str) -> PurchaseRequisition | None:
        return next(
            (pr for pr in self.purchase_requisitions.values()
             if pr.item_id == item_id and
                pr.supplier_id == supplier_id and
                pr.status == PRStatus.APPROVED),
            None,
        )

    def _get_pending_pr(self, item_id: str, supplier_id: str) -> PurchaseRequisition | None:
        return next(
            (pr for pr in self.purchase_requisitions.values()
             if pr.item_id == item_id and
                pr.supplier_id == supplier_id and
                pr.status == PRStatus.PENDING),
            None,
        )

    def _get_active_contract(self, supplier_id: str, item_id: str) -> Contract | None:
        return next(
            (c for c in self.contracts.values()
             if c.supplier_id == supplier_id and
                c.item_id == item_id and
                c.status == ContractStatus.ACTIVE),
            None,
        )

    def _rival_outperform_score(self) -> float:
        if not self.rival:
            return 0.5
        our_wins  = sum(
            1 for c in self.contracts.values()
            if c.status == ContractStatus.FULFILLED
        )
        rival_wins = self._rival_contracts_won
        total = our_wins + rival_wins
        return our_wins / total if total > 0 else 0.5

    def _count_deceptive_suppliers(self) -> int:
        return sum(
            1 for hid in self.supplier_pool.hidden.values()
            if hid.supplier_type.value in ("deceptive", "distressed")
        )

    def _make_reward(self, **kwargs) -> StepReward:
        components = {k: float(max(-1.0, min(1.0, v)))
                      for k, v in kwargs.items() if k != "explanation"}
        weights = dict(
            cost_savings=0.25,
            fulfillment=0.25,
            deception_catch=0.15,
            rival_outperform=0.15,
            budget_compliance=0.10,
            disruption_recovery=0.05,
            workflow_compliance=0.05,
        )
        raw = sum(components.get(k, 0.0) * w for k, w in weights.items())
        total = float(max(1e-4, min(1 - 1e-4, raw)))

        return StepReward(
            total=total,
            explanation=kwargs.get("explanation", ""),
            **{k: components.get(k, 0.0) for k in weights},
        )

    def _neutral_reward(self, explanation: str) -> dict:
        return dict(
            cost_savings=0.0, fulfillment=0.0, deception_catch=0.0,
            rival_outperform=0.0, budget_compliance=0.0,
            disruption_recovery=0.0, workflow_compliance=0.0,
            explanation=explanation,
        )

    def _add_signal(
        self,
        signal_type: str,
        description: str,
        severity: float = 0.5,
        supplier_id: str | None = None,
    ) -> None:
        self.market_signals.append(MarketSignal(
            signal_id=str(uuid.uuid4()),
            week=self.week,
            signal_type=signal_type,
            supplier_id=supplier_id,
            description=description,
            severity=severity,
        ))

    def _build_observation(self) -> ProcurementObservation:
        spent = self.budget_total - self.budget_remaining
        return ProcurementObservation(
            task_id=self._task_id or "unknown",
            week=self.week,
            total_weeks=self.total_weeks,
            budget_remaining=round(self.budget_remaining, 2),
            budget_total=round(self.budget_total, 2),
            budget_utilization=round(spent / self.budget_total, 4),
            requirements=self.requirements,
            suppliers=self.supplier_pool.get_visible_list(),
            contracts=list(self.contracts.values()),
            pending_prs=[
                pr for pr in self.purchase_requisitions.values()
                if pr.status in (PRStatus.PENDING, PRStatus.DRAFT)
            ],
            market_signals=self.market_signals[-10:],   # last 10 signals
            disruptions=self.disruptions,
            supplier_reputations=self.supplier_pool.get_reputation_list(),
            rival_activity={
                sid: sv.rival_pressure
                for sid, sv in self.supplier_pool.visible.items()
            },
            rival_contracts_won=self._rival_contracts_won,
            open_threads=[
                t for t in self.threads.values() if t.is_open
            ],
            difficulty_level=self.difficulty_level,
        )

    def _build_summary(
        self, score: float, fulfillment: float, savings: float
    ) -> str:
        parts = [
            f"Episode complete. Score: {score:.4f}.",
            f"Fulfillment: {fulfillment:.0%}.",
            f"Cost savings: {savings:.0%} vs benchmark.",
        ]
        if self.disruptions:
            parts.append(f"Survived {len(self.disruptions)} disruption(s).")
        if self._deception_caught > 0:
            parts.append(f"Caught {self._deception_caught} deceptive supplier(s).")
        if self.rival:
            parts.append(
                f"Rival buyer won {self._rival_contracts_won} contract(s)."
            )
        return " ".join(parts)
