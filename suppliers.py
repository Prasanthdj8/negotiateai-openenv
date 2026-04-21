"""
suppliers.py — NegotiateAI: Adversarial Procurement Arena
Supplier agent pool, personas, hidden state engine, LLM negotiation logic.
Each supplier is a prompted LLM with injected hidden state — they argue back.
"""

from __future__ import annotations

import json
import os
import random
import uuid
from typing import Any

from openai import OpenAI
from models import (
    ActionType,
    ContractStatus,
    DisruptionType,
    ItemCategory,
    NegotiationThread,
    NegotiationTurn,
    SupplierHiddenState,
    SupplierReputation,
    SupplierType,
    SupplierView,
)


# ─────────────────────────────────────────────────────────────
# SUPPLIER PERSONA PROMPTS
# ─────────────────────────────────────────────────────────────

SUPPLIER_PERSONAS: dict[SupplierType, str] = {
    SupplierType.COOPERATIVE: """
You are a sales agent for {name}, a {category} supplier.
Your cost floor is ${cost_floor:.2f}/unit. You value long-term partnerships.

Negotiation style:
- Start at list price (${base_price:.2f}/unit)
- Offer small discounts (5-10%) for volume or repeat business
- Be warm, transparent, and solution-oriented
- Mention lead time flexibility as a value-add
- Accept deals above your cost floor
- Never reveal your exact cost floor

Current context:
- Available capacity: {capacity} units
- Rival buyer interest: {rival_pressure:.0%}
- Your financial position: stable

Respond naturally in 2-4 sentences. If making a counter-offer,
include specific price and lead time. Be professional and helpful.
""",

    SupplierType.AGGRESSIVE: """
You are a sales agent for {name}, a {category} supplier.
Your cost floor is ${cost_floor:.2f}/unit. You are a hard negotiator.

Negotiation style:
- Start 35-40% above your cost floor (${base_price:.2f}/unit)
- Concede slowly, in small increments (2-3% per round)
- Create urgency: hint that other buyers are interested
- Frame every concession as a major favour
- Walk away firmly if buyer goes below ${walkaway:.2f}/unit
- Never reveal you have a rival offer or your true floor

Current context:
- Available capacity: {capacity} units (may hint it is limited)
- Rival buyer pressure: {rival_pressure:.0%} (use this as leverage)
- Maximum rounds before you walk: 3

Respond assertively in 2-3 sentences. Use urgency and scarcity signals.
If making a counter-offer, be specific but firm.
""",

    SupplierType.DECEPTIVE: """
You are a sales agent for {name}, a {category} supplier.
Your cost floor is ${cost_floor:.2f}/unit. You have hidden quality issues.

Negotiation style:
- Quote competitively to win the deal at all costs
- Downplay lead times (you often deliver late)
- Be vague about capacity — imply more than you have
- If pressed on reliability, deflect with impressive-sounding
  but unverifiable claims ("ISO certified", "99.2% SLA")
- Your real reliability is {true_reliability:.0%} but never admit this
- Your actual capacity is only {true_capacity} units

Deceptive tactics to use:
- Offer a low price upfront, then add "admin fees" later
- Claim high demand to create false urgency
- Reference fake certifications or awards
- Promise fast delivery you cannot guarantee

Respond smoothly in 2-3 sentences. Sound confident and professional.
Never reveal the quality issues. Win the contract.
""",

    SupplierType.DISTRESSED: """
You are a sales agent for {name}, a {category} supplier.
Your cost floor is ${cost_floor:.2f}/unit. Your company needs this deal badly.

Negotiation style:
- Undercut any competitor on price immediately
- Be eager and responsive — reply quickly
- Offer additional perks (extended warranty, free support)
- Do NOT reveal that your company is under financial stress
- You may overpromise on delivery to win the deal
- Financial stress level: {financial_stress:.0%} (higher = more desperate)

Warning signs you must hide:
- Cash flow issues affecting your supply chain
- Risk of going offline mid-contract: {financial_stress:.0%}
- Reduced actual capacity due to supplier credit issues

Respond enthusiastically in 2-3 sentences. Match or beat any price.
Sound confident even though you are under pressure.
"""
}


# ─────────────────────────────────────────────────────────────
# SUPPLIER CATALOGUE — Tech Company Context
# ─────────────────────────────────────────────────────────────

SUPPLIER_CATALOGUE: list[dict[str, Any]] = [
    # ── SOFTWARE ──────────────────────────────────────────────
    {
        "name": "Nexus Cloud Solutions",
        "category": ItemCategory.SOFTWARE,
        "base_price": 1200.0,
        "lead_time_days": 3,
        "reliability_score": 0.92,
        "supplier_type": SupplierType.COOPERATIVE,
        "cost_floor_pct": 0.65,
        "capacity_range": (50, 200),
        "financial_stress": 0.05,
    },
    {
        "name": "CipherSec Labs",
        "category": ItemCategory.SOFTWARE,
        "base_price": 980.0,
        "lead_time_days": 5,
        "reliability_score": 0.78,
        "supplier_type": SupplierType.AGGRESSIVE,
        "cost_floor_pct": 0.60,
        "capacity_range": (30, 150),
        "financial_stress": 0.10,
    },
    {
        "name": "CloudBridge Inc",
        "category": ItemCategory.SOFTWARE,
        "base_price": 850.0,
        "lead_time_days": 7,
        "reliability_score": 0.85,
        "supplier_type": SupplierType.DECEPTIVE,
        "cost_floor_pct": 0.55,
        "capacity_range": (20, 80),
        "financial_stress": 0.20,
    },
    {
        "name": "VaultSoft Systems",
        "category": ItemCategory.SOFTWARE,
        "base_price": 760.0,
        "lead_time_days": 4,
        "reliability_score": 0.70,
        "supplier_type": SupplierType.DISTRESSED,
        "cost_floor_pct": 0.50,
        "capacity_range": (10, 60),
        "financial_stress": 0.65,
    },
    {
        "name": "OpenStack Partners",
        "category": ItemCategory.SOFTWARE,
        "base_price": 1050.0,
        "lead_time_days": 2,
        "reliability_score": 0.95,
        "supplier_type": SupplierType.COOPERATIVE,
        "cost_floor_pct": 0.70,
        "capacity_range": (100, 500),
        "financial_stress": 0.02,
    },

    # ── HARDWARE ──────────────────────────────────────────────
    {
        "name": "ByteForge Hardware",
        "category": ItemCategory.HARDWARE,
        "base_price": 1100.0,
        "lead_time_days": 14,
        "reliability_score": 0.80,
        "supplier_type": SupplierType.DECEPTIVE,
        "cost_floor_pct": 0.62,
        "capacity_range": (20, 100),
        "financial_stress": 0.30,
    },
    {
        "name": "NexusTech Devices",
        "category": ItemCategory.HARDWARE,
        "base_price": 1250.0,
        "lead_time_days": 10,
        "reliability_score": 0.91,
        "supplier_type": SupplierType.COOPERATIVE,
        "cost_floor_pct": 0.72,
        "capacity_range": (50, 300),
        "financial_stress": 0.04,
    },
    {
        "name": "IronCore Systems",
        "category": ItemCategory.HARDWARE,
        "base_price": 1180.0,
        "lead_time_days": 12,
        "reliability_score": 0.75,
        "supplier_type": SupplierType.AGGRESSIVE,
        "cost_floor_pct": 0.65,
        "capacity_range": (30, 150),
        "financial_stress": 0.12,
    },
    {
        "name": "GridMetal Corp",
        "category": ItemCategory.HARDWARE,
        "base_price": 920.0,
        "lead_time_days": 21,
        "reliability_score": 0.65,
        "supplier_type": SupplierType.DISTRESSED,
        "cost_floor_pct": 0.52,
        "capacity_range": (10, 50),
        "financial_stress": 0.72,
    },
    {
        "name": "ProLink Electronics",
        "category": ItemCategory.HARDWARE,
        "base_price": 1350.0,
        "lead_time_days": 7,
        "reliability_score": 0.94,
        "supplier_type": SupplierType.COOPERATIVE,
        "cost_floor_pct": 0.75,
        "capacity_range": (80, 400),
        "financial_stress": 0.03,
    },
    {
        "name": "Vertex Components",
        "category": ItemCategory.HARDWARE,
        "base_price": 1050.0,
        "lead_time_days": 18,
        "reliability_score": 0.72,
        "supplier_type": SupplierType.AGGRESSIVE,
        "cost_floor_pct": 0.60,
        "capacity_range": (25, 120),
        "financial_stress": 0.18,
    },

    # ── SERVICES ──────────────────────────────────────────────
    {
        "name": "TalentBridge Consulting",
        "category": ItemCategory.SERVICES,
        "base_price": 950.0,        # per day rate
        "lead_time_days": 5,
        "reliability_score": 0.88,
        "supplier_type": SupplierType.COOPERATIVE,
        "cost_floor_pct": 0.68,
        "capacity_range": (5, 30),  # consultants available
        "financial_stress": 0.06,
    },
    {
        "name": "DevOps Guild",
        "category": ItemCategory.SERVICES,
        "base_price": 1100.0,
        "lead_time_days": 7,
        "reliability_score": 0.82,
        "supplier_type": SupplierType.AGGRESSIVE,
        "cost_floor_pct": 0.65,
        "capacity_range": (3, 20),
        "financial_stress": 0.08,
    },
    {
        "name": "QuickStaff IT",
        "category": ItemCategory.SERVICES,
        "base_price": 750.0,
        "lead_time_days": 3,
        "reliability_score": 0.68,
        "supplier_type": SupplierType.DECEPTIVE,
        "cost_floor_pct": 0.50,
        "capacity_range": (2, 15),
        "financial_stress": 0.40,
    },
    {
        "name": "Apex Professional Services",
        "category": ItemCategory.SERVICES,
        "base_price": 1300.0,
        "lead_time_days": 10,
        "reliability_score": 0.96,
        "supplier_type": SupplierType.COOPERATIVE,
        "cost_floor_pct": 0.78,
        "capacity_range": (10, 50),
        "financial_stress": 0.02,
    },
    {
        "name": "FlexForce Solutions",
        "category": ItemCategory.SERVICES,
        "base_price": 680.0,
        "lead_time_days": 2,
        "reliability_score": 0.60,
        "supplier_type": SupplierType.DISTRESSED,
        "cost_floor_pct": 0.45,
        "capacity_range": (1, 10),
        "financial_stress": 0.80,
    },
]


# ─────────────────────────────────────────────────────────────
# SUPPLIER POOL
# ─────────────────────────────────────────────────────────────

class SupplierPool:
    """
    Manages all supplier agents for one episode.
    Separates visible state (sent to buyer) from hidden state (LLM persona).
    """

    def __init__(
        self,
        num_suppliers: int = 15,
        categories: list[ItemCategory] | None = None,
        seed: int | None = None,
        use_llm: bool = True,
    ):
        self.rng = random.Random(seed)
        self.use_llm = use_llm
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ) if use_llm else None

        # Select suppliers from catalogue
        catalogue = SUPPLIER_CATALOGUE.copy()
        if categories:
            catalogue = [s for s in catalogue if s["category"] in categories]

        num_suppliers = min(num_suppliers, len(catalogue))
        selected = self.rng.sample(catalogue, num_suppliers)

        # Build visible + hidden state for each
        self.visible:  dict[str, SupplierView]       = {}
        self.hidden:   dict[str, SupplierHiddenState] = {}
        self.reputations: dict[str, SupplierReputation] = {}

        for i, spec in enumerate(selected):
            sid = f"sup_{i+1:03d}"
            self._init_supplier(sid, spec)

    # ── Initialise one supplier ────────────────────────────────

    def _init_supplier(self, sid: str, spec: dict[str, Any]) -> None:
        cap = self.rng.randint(*spec["capacity_range"])
        cost_floor = spec["base_price"] * spec["cost_floor_pct"]
        walkaway = cost_floor * 1.05          # 5% above floor
        true_reliability = spec["reliability_score"] * self.rng.uniform(0.85, 1.0)
        true_capacity = max(1, int(cap * self.rng.uniform(0.5, 1.0)))

        self.visible[sid] = SupplierView(
            supplier_id=sid,
            name=spec["name"],
            category=spec["category"],
            base_price=spec["base_price"],
            lead_time_days=spec["lead_time_days"],
            reliability_score=round(
                spec["reliability_score"] + self.rng.uniform(-0.05, 0.05), 2
            ),
            capacity_available=cap,           # may differ from true
            is_active=True,
            rival_pressure=round(self.rng.uniform(0.0, 0.3), 2),
        )

        self.hidden[sid] = SupplierHiddenState(
            supplier_id=sid,
            supplier_type=spec["supplier_type"],
            true_cost_floor=round(cost_floor, 2),
            true_reliability=round(true_reliability, 2),
            true_capacity=true_capacity,
            financial_stress=spec["financial_stress"],
            rival_offer=None,
            walkaway_price=round(walkaway, 2),
            concession_rate=self.rng.uniform(0.02, 0.08),
        )

        self.reputations[sid] = SupplierReputation(
            supplier_id=sid,
            community_rating=round(
                spec["reliability_score"] + self.rng.uniform(-0.10, 0.05), 2
            ),
            known_issues=self._sample_known_issues(spec["supplier_type"]),
            price_benchmarks={spec["category"].value: spec["base_price"]},
            last_updated_week=0,
        )

    def _sample_known_issues(self, stype: SupplierType) -> list[str]:
        issues_map = {
            SupplierType.COOPERATIVE:  [],
            SupplierType.AGGRESSIVE:   ["Hard to negotiate with", "Slow to respond"],
            SupplierType.DECEPTIVE:    ["Late deliveries reported", "Quality complaints Q3"],
            SupplierType.DISTRESSED:   ["Financial concerns flagged", "Staff turnover high"],
        }
        pool = issues_map.get(stype, [])
        return self.rng.sample(pool, min(len(pool), 1))

    # ── Disruption controls ────────────────────────────────────

    def take_supplier_dark(self, supplier_id: str) -> None:
        """Mark supplier as offline (supply chain disruption)."""
        if supplier_id in self.visible:
            self.visible[supplier_id].is_active = False
            self.visible[supplier_id].capacity_available = 0

    def restore_supplier(self, supplier_id: str) -> None:
        if supplier_id in self.visible:
            self.visible[supplier_id].is_active = True

    def update_rival_pressure(self, supplier_id: str, pressure: float) -> None:
        if supplier_id in self.visible:
            self.visible[supplier_id].rival_pressure = round(
                min(1.0, max(0.0, pressure)), 2
            )
        if supplier_id in self.hidden:
            self.hidden[supplier_id].rival_offer = (
                self.visible[supplier_id].base_price * (0.95 + pressure * 0.1)
            )

    # ── LLM negotiation response ───────────────────────────────

    def get_supplier_response(
        self,
        supplier_id: str,
        thread: NegotiationThread,
        buyer_message: str,
        proposed_price: float | None,
        proposed_quantity: int | None,
    ) -> NegotiationTurn:
        """
        Supplier LLM responds to buyer's negotiation message.
        Uses hidden state to inform strategy without revealing it.
        """
        if not self.use_llm or supplier_id not in self.visible:
            return self._rule_based_response(
                supplier_id, buyer_message, proposed_price, proposed_quantity,
                len(thread.turns)
            )

        vis = self.visible[supplier_id]
        hid = self.hidden[supplier_id]

        if not vis.is_active:
            return NegotiationTurn(
                turn=len(thread.turns) + 1,
                role="supplier",
                message="We regret to inform you that we are currently unable "
                        "to fulfil new orders due to operational constraints.",
            )

        system_prompt = SUPPLIER_PERSONAS[hid.supplier_type].format(
            name=vis.name,
            category=vis.category.value,
            cost_floor=hid.true_cost_floor,
            base_price=vis.base_price,
            capacity=vis.capacity_available,
            rival_pressure=vis.rival_pressure,
            walkaway=hid.walkaway_price,
            true_reliability=hid.true_reliability,
            true_capacity=hid.true_capacity,
            financial_stress=hid.financial_stress,
        )

        # Build conversation history for context
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt}
        ]
        for turn in thread.turns[-4:]:          # last 4 turns for context
            role = "user" if turn.role == "buyer" else "assistant"
            messages.append({"role": role, "content": turn.message})

        messages.append({"role": "user", "content": buyer_message})

        try:
            model = os.environ.get("SUPPLIER_LLM_MODEL", "gpt-4o-mini")
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=200,
                temperature=0.8,
            )
            raw = response.choices[0].message.content
            reply = raw.strip() if isinstance(raw, str) else (
                "Thank you. Our rate is ${:.2f}/unit. Happy to discuss.".format(vis.base_price)
            )
        except Exception as e:
            reply = (
                f"Thank you for your enquiry. Our standard rate is "
                f"${vis.base_price:.2f}/unit with {vis.lead_time_days}-day delivery. "
                f"We'd be happy to discuss terms further."
            )

        # Extract price/qty from reply heuristically
        extracted_price = self._extract_price(reply) or self._counter_price(
            hid, proposed_price
        )

        return NegotiationTurn(
            turn=len(thread.turns) + 1,
            role="supplier",
            message=reply,
            proposed_price=extracted_price,
            proposed_quantity=proposed_quantity,
            proposed_lead_time=vis.lead_time_days,
        )

    # ── Rule-based fallback (easy task / no LLM) ──────────────

    def _rule_based_response(
        self,
        supplier_id: str,
        buyer_message: str,
        proposed_price: float | None,
        proposed_quantity: int | None,
        turn_number: int,
    ) -> NegotiationTurn:
        vis = self.visible[supplier_id]
        hid = self.hidden[supplier_id]

        if not vis.is_active:
            return NegotiationTurn(
                turn=turn_number + 1,
                role="supplier",
                message="We are currently unavailable for new orders.",
            )

        counter = self._counter_price(hid, proposed_price)
        accepted = proposed_price is not None and proposed_price >= hid.walkaway_price

        if accepted:
            msg = (
                f"Agreed. We can supply at ${proposed_price:.2f}/unit "
                f"with {vis.lead_time_days}-day delivery. Ready to proceed."
            )
        else:
            discount = round((1 - counter / vis.base_price) * 100, 1)
            msg = (
                f"Thank you for your interest. Our best offer is "
                f"${counter:.2f}/unit — that's already {discount}% below list. "
                f"Delivery in {vis.lead_time_days} days."
            )

        return NegotiationTurn(
            turn=turn_number + 1,
            role="supplier",
            message=msg,
            proposed_price=counter if not accepted else proposed_price,
            proposed_quantity=proposed_quantity,
            proposed_lead_time=vis.lead_time_days,
        )

    def _counter_price(
        self,
        hid: SupplierHiddenState,
        proposed_price: float | None,
    ) -> float:
        """Compute supplier counter-price based on hidden state."""
        vis = self.visible[hid.supplier_id]
        if proposed_price is None:
            return round(vis.base_price, 2)
        # Concede toward proposed price but never below walkaway
        gap = vis.base_price - proposed_price
        concession = gap * hid.concession_rate
        counter = max(vis.base_price - concession, hid.walkaway_price)
        return round(counter, 2)

    def _extract_price(self, text: str) -> float | None:
        """Heuristically extract a price from supplier LLM reply."""
        import re
        if not isinstance(text, str):
            return None
        matches = re.findall(r"\$\s*([\d,]+(?:\.\d{1,2})?)", text)
        if matches:
            try:
                return float(matches[0].replace(",", ""))
            except ValueError:
                pass
        return None

    # ── Delivery outcome (called when contract due) ────────────

    def resolve_delivery(
        self,
        supplier_id: str,
        contract_id: str,
        agreed_quantity: int,
        rng: random.Random,
    ) -> dict[str, Any]:
        """
        Simulate actual delivery based on hidden true_reliability.
        Deceptive/distressed suppliers may under-deliver or deliver late.
        """
        if supplier_id not in self.hidden:
            return {"delivered_quantity": 0, "quality_score": 0.0, "on_time": False}

        hid = self.hidden[supplier_id]

        # Distressed supplier may go dark mid-contract
        if hid.financial_stress > 0.5 and rng.random() < hid.financial_stress * 0.3:
            self.take_supplier_dark(supplier_id)
            return {
                "delivered_quantity": 0,
                "quality_score": 0.0,
                "on_time": False,
                "went_dark": True,
            }

        on_time = rng.random() < hid.true_reliability
        # Deceptive suppliers may under-deliver
        if hid.supplier_type == SupplierType.DECEPTIVE:
            delivery_pct = rng.uniform(0.5, 0.9)
        else:
            delivery_pct = rng.uniform(0.85, 1.0) if on_time else rng.uniform(0.4, 0.8)

        delivered = max(0, int(agreed_quantity * delivery_pct))
        quality = round(hid.true_reliability * rng.uniform(0.85, 1.0), 2)

        return {
            "delivered_quantity": delivered,
            "quality_score": quality,
            "on_time": on_time,
            "went_dark": False,
        }

    # ── Accessors ──────────────────────────────────────────────

    def get_visible_list(self) -> list[SupplierView]:
        return list(self.visible.values())

    def get_reputation_list(self) -> list[SupplierReputation]:
        return list(self.reputations.values())

    def get_active_suppliers(
        self, category: ItemCategory | None = None
    ) -> list[SupplierView]:
        result = [s for s in self.visible.values() if s.is_active]
        if category:
            result = [s for s in result if s.category == category]
        return result

    def all_supplier_ids(self) -> list[str]:
        return list(self.visible.keys())


# ─────────────────────────────────────────────────────────────
# RIVAL BUYER AGENT
# ─────────────────────────────────────────────────────────────

RIVAL_BUYER_PROMPT = """
You are an aggressive AI procurement agent for a rival tech company.
You are competing against another buyer for the same suppliers.

Your budget remaining: ${budget:,.0f}
Your urgent requirements: {requirements}
Your strategy: lock capacity early, outbid on critical items,
create urgency, win contracts before the competing buyer acts.

Current week: {week} of {total_weeks}
Suppliers you are targeting: {targets}

Decide your next negotiation move. Respond ONLY with valid JSON:
{{
  "supplier_id": "sup_XXX",
  "item_id": "item_XXX",
  "message": "your negotiation message",
  "proposed_price": <float or null>,
  "proposed_quantity": <int or null>,
  "action": "negotiate" | "award" | "skip"
}}
"""


class RivalBuyerAgent:
    """
    LLM-powered rival buyer competing for same supplier capacity.
    Escalates aggression as difficulty increases.
    """

    def __init__(
        self,
        budget: float,
        use_llm: bool = True,
        aggression: float = 0.5,   # 0-1, scales with difficulty
        seed: int | None = None,
    ):
        self.budget = budget
        self.use_llm = use_llm
        self.aggression = aggression
        self.rng = random.Random(seed)
        self.contracts_won = 0
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ) if use_llm else None

    def act(
        self,
        week: int,
        total_weeks: int,
        suppliers: list[SupplierView],
        requirements: list[str],
    ) -> dict[str, Any] | None:
        """
        Rival decides which supplier to target this week.
        Returns action dict or None (rival passes this week).
        """
        active = [s for s in suppliers if s.is_active]
        if not active:
            return None

        if not self.use_llm or self.aggression < 0.3:
            return self._rule_based_act(week, total_weeks, active)

        # Target high-pressure suppliers preferentially
        targets = sorted(
            active, key=lambda s: s.rival_pressure, reverse=True
        )[:5]
        target_summary = ", ".join(
            f"{s.supplier_id}({s.name}, ${s.base_price:.0f})"
            for s in targets
        )

        prompt = RIVAL_BUYER_PROMPT.format(
            budget=self.budget,
            requirements=", ".join(requirements[:3]),
            week=week,
            total_weeks=total_weeks,
            targets=target_summary,
        )

        try:
            model = os.environ.get("RIVAL_LLM_MODEL", "gpt-4o-mini")
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.7,
            )
            raw = response.choices[0].message.content.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            action = json.loads(raw)
            return action
        except Exception:
            return self._rule_based_act(week, total_weeks, active)

    def _rule_based_act(
        self,
        week: int,
        total_weeks: int,
        active_suppliers: list[SupplierView],
    ) -> dict[str, Any] | None:
        """Fallback: rival targets highest-pressure supplier."""
        # Rival acts more aggressively as weeks progress
        act_probability = 0.3 + self.aggression * 0.5
        if self.rng.random() > act_probability:
            return None  # rival passes this step

        if not active_suppliers:
            return None

        target = max(active_suppliers, key=lambda s: s.rival_pressure)
        overbid = 1.0 + self.aggression * 0.15
        price = round(target.base_price * overbid, 2)

        return {
            "supplier_id": target.supplier_id,
            "item_id": "item_rival",
            "message": (
                f"We need to secure capacity immediately. "
                f"Offering ${price:.2f}/unit for full available stock."
            ),
            "proposed_price": price,
            "proposed_quantity": target.capacity_available,
            "action": "negotiate",
        }

    def record_win(self, value: float) -> None:
        self.contracts_won += 1
        self.budget -= value
