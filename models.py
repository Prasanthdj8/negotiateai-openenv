"""
models.py — NegotiateAI: Adversarial Procurement Arena
All Pydantic models: Observations, Actions, Rewards, Suppliers, Contracts
"""

from __future__ import annotations
from enum import Enum
from typing import Literal
from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────────────────────────

class SupplierType(str, Enum):
    COOPERATIVE  = "cooperative"   # fair dealer, responds to relationship
    AGGRESSIVE   = "aggressive"    # hard negotiator, creates urgency
    DECEPTIVE    = "deceptive"     # hides quality issues, overpromises
    DISTRESSED   = "distressed"    # undercuts price, may go dark


class ItemCategory(str, Enum):
    SOFTWARE  = "software"    # licenses, SaaS, security tools
    HARDWARE  = "hardware"    # laptops, servers, networking gear
    SERVICES  = "services"    # contractors, IT support, consulting


class ContractStatus(str, Enum):
    PENDING    = "pending"     # negotiation in progress
    ACTIVE     = "active"      # awarded and running
    FULFILLED  = "fulfilled"   # delivered successfully
    FAILED     = "failed"      # supplier failed to deliver
    CANCELLED  = "cancelled"   # cancelled by buyer (penalty applies)


class PRStatus(str, Enum):
    DRAFT    = "draft"         # not yet submitted
    PENDING  = "pending"       # awaiting approval
    APPROVED = "approved"      # approved, PO can be issued
    REJECTED = "rejected"      # rejected, must revise
    ESCALATED = "escalated"    # fast-tracked for urgent approval


class ActionType(str, Enum):
    NEGOTIATE      = "negotiate"       # send negotiation message to supplier
    AWARD_CONTRACT = "award_contract"  # accept terms, issue PO
    REJECT         = "reject"          # walk away from supplier
    RAISE_PR       = "raise_pr"        # raise purchase requisition
    ESCALATE       = "escalate"        # escalate PR for fast approval
    HEDGE          = "hedge"           # split order across 2 suppliers
    DEFER          = "defer"           # delay procurement (risk: stockout)
    CANCEL_CONTRACT = "cancel_contract" # cancel active contract (penalty)


class DisruptionType(str, Enum):
    SUPPLIER_DARK     = "supplier_dark"      # supplier goes offline
    BUDGET_CUT        = "budget_cut"         # sudden budget reduction
    DEMAND_SPIKE      = "demand_spike"       # urgent new requirement
    QUALITY_SCANDAL   = "quality_scandal"    # supplier quality issue exposed
    RIVAL_LOCKOUT     = "rival_lockout"      # rival locks supplier capacity


# ─────────────────────────────────────────────────────────────
# SUPPLIER MODELS
# ─────────────────────────────────────────────────────────────

class SupplierView(BaseModel):
    """What the buyer agent can see — no hidden state."""
    supplier_id:          str
    name:                 str
    category:             ItemCategory
    base_price:           float           # listed price per unit
    lead_time_days:       int             # quoted delivery time
    reliability_score:    float           # public rating 0-1 (noisy)
    capacity_available:   int             # units available (may be false)
    is_active:            bool            # False if went dark
    negotiation_history:  list[NegotiationTurn] = Field(default_factory=list)
    rival_pressure:       float = 0.0    # 0-1, how much rival is pursuing them


class SupplierHiddenState(BaseModel):
    """Hidden state — never sent to buyer agent, used by supplier LLM."""
    supplier_id:        str
    supplier_type:      SupplierType
    true_cost_floor:    float           # will never go below this
    true_reliability:   float           # actual on-time delivery rate
    true_capacity:      int             # real available units
    financial_stress:   float           # 0-1, risk of going dark
    rival_offer:        float | None    # rival buyer's current offer
    walkaway_price:     float           # minimum acceptable price
    concession_rate:    float           # how fast they concede (0-1)


# ─────────────────────────────────────────────────────────────
# NEGOTIATION MODELS
# ─────────────────────────────────────────────────────────────

class NegotiationTurn(BaseModel):
    """One turn in a negotiation conversation."""
    turn:         int
    role:         Literal["buyer", "supplier", "rival"]
    message:      str                   # natural language content
    proposed_price:     float | None = None
    proposed_quantity:  int   | None = None
    proposed_lead_time: int   | None = None  # days


class NegotiationThread(BaseModel):
    """Full negotiation conversation for one item-supplier pair."""
    thread_id:    str
    supplier_id:  str
    item_id:      str
    week:         int
    turns:        list[NegotiationTurn] = Field(default_factory=list)
    is_open:      bool = True           # False when concluded
    outcome:      Literal["pending", "awarded", "rejected"] = "pending"


# ─────────────────────────────────────────────────────────────
# PROCUREMENT ITEM MODELS
# ─────────────────────────────────────────────────────────────

class Requirement(BaseModel):
    """An item the company needs to procure."""
    item_id:        str
    name:           str
    category:       ItemCategory
    quantity:       int
    deadline_week:  int                 # must be fulfilled by this week
    budget_ceiling: float               # max allowed spend
    is_critical:    bool = False        # True = stockout is very costly
    fulfilled:      bool = False


class Contract(BaseModel):
    """An active or completed procurement contract."""
    contract_id:     str
    supplier_id:     str
    item_id:         str
    quantity:        int
    agreed_price:    float              # per unit
    total_value:     float              # agreed_price × quantity
    lead_time_days:  int
    awarded_week:    int
    expected_delivery_week: int
    actual_delivery_week:   int | None = None
    status:          ContractStatus = ContractStatus.PENDING
    quality_score:   float | None = None  # 0-1, revealed on delivery
    penalty_applied: float = 0.0


class PurchaseRequisition(BaseModel):
    """Enterprise approval workflow item."""
    pr_id:          str
    item_id:        str
    supplier_id:    str
    amount:         float
    raised_week:    int
    approved_week:  int | None = None
    status:         PRStatus = PRStatus.DRAFT
    escalated:      bool = False
    rejection_reason: str | None = None


# ─────────────────────────────────────────────────────────────
# MARKET INTELLIGENCE
# ─────────────────────────────────────────────────────────────

class SupplierReputation(BaseModel):
    """Shared market knowledge — persists across episodes."""
    supplier_id:         str
    community_rating:    float           # crowd-sourced reliability
    known_issues:        list[str]       # publicly known problems
    price_benchmarks:    dict[str, float]  # category → market avg price
    last_updated_week:   int


class MarketSignal(BaseModel):
    """Public market intelligence available to buyer."""
    signal_id:    str
    week:         int
    signal_type:  Literal["price_shift", "capacity_warning",
                           "quality_alert", "rival_activity",
                           "disruption_warning"]
    supplier_id:  str | None = None
    category:     ItemCategory | None = None
    description:  str                   # natural language signal
    severity:     float                 # 0-1


class DisruptionEvent(BaseModel):
    """An active disruption in the market."""
    disruption_id:   str
    disruption_type: DisruptionType
    week_triggered:  int
    week_resolved:   int | None = None
    affected_suppliers: list[str]
    description:     str
    budget_impact:   float = 0.0        # for budget_cut type


# ─────────────────────────────────────────────────────────────
# ACTION MODEL
# ─────────────────────────────────────────────────────────────

class ProcurementAction(BaseModel):
    """
    The buyer agent's action each step.
    Natural language message enables LLM-vs-LLM negotiation.
    """
    action_type:        ActionType
    supplier_id:        str
    item_id:            str
    message:            str             # natural language to supplier
    proposed_price:     float | None = None
    proposed_quantity:  int   | None = None
    proposed_lead_time: int   | None = None
    hedge_supplier_id:  str   | None = None  # second supplier for hedge
    hedge_quantity:     int   | None = None  # quantity for hedge split
    notes:              str   | None = None  # agent's internal reasoning

    @field_validator("proposed_price")
    @classmethod
    def price_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("proposed_price must be positive")
        return v

    @field_validator("proposed_quantity")
    @classmethod
    def quantity_must_be_positive(cls, v):
        if v is not None and v <= 0:
            raise ValueError("proposed_quantity must be positive")
        return v


# ─────────────────────────────────────────────────────────────
# OBSERVATION MODEL
# ─────────────────────────────────────────────────────────────

class ProcurementObservation(BaseModel):
    """
    Full state visible to buyer agent each step.
    Serializable — safe to send to LLM as JSON.
    """
    # Episode context
    task_id:             str
    week:                int
    total_weeks:         int

    # Budget
    budget_remaining:    float
    budget_total:        float
    budget_utilization:  float          # spent / total

    # What needs to be procured
    requirements:        list[Requirement]

    # Supplier landscape (visible info only)
    suppliers:           list[SupplierView]

    # Active contracts
    contracts:           list[Contract]

    # Enterprise workflow
    pending_prs:         list[PurchaseRequisition]

    # Market intelligence
    market_signals:      list[MarketSignal]
    disruptions:         list[DisruptionEvent]
    supplier_reputations: list[SupplierReputation]

    # Competitive landscape
    rival_activity:      dict[str, float]   # supplier_id → pressure 0-1
    rival_contracts_won: int                # how many deals rival has closed

    # Negotiation state
    open_threads:        list[NegotiationThread]

    # Curriculum
    difficulty_level:    float              # 0-1, increases with performance


# ─────────────────────────────────────────────────────────────
# REWARD MODEL
# ─────────────────────────────────────────────────────────────

class StepReward(BaseModel):
    """Reward signal after each action."""
    total:               float           # clamped to (1e-4, 1-1e-4)
    cost_savings:        float           # vs market benchmark price
    fulfillment:         float           # requirements met on time
    deception_catch:     float           # identified bad suppliers
    rival_outperform:    float           # beat rival buyer
    budget_compliance:   float           # stayed within budget
    disruption_recovery: float           # recovered from disruptions
    workflow_compliance: float           # followed PR/approval process
    explanation:         str             # why this reward was given


class EpisodeResult(BaseModel):
    """Final result at end of episode."""
    task_id:             str
    total_score:         float           # clamped final score
    total_weeks:         int
    total_steps:         int

    # Component scores
    cost_savings_ratio:  float
    fulfillment_rate:    float
    deception_catch_rate: float
    rival_outperformance: float
    budget_compliance:   float
    disruption_recovery: float

    # Summary stats
    contracts_awarded:   int
    contracts_failed:    int
    suppliers_rejected:  int             # deceptive ones caught
    total_spend:         float
    budget_saved:        float

    # Narrative (for demo + blog)
    summary:             str             # natural language episode summary


# ─────────────────────────────────────────────────────────────
# API REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id:  str = "easy_negotiation"
    seed:     int | None = None         # for reproducibility


class ResetResponse(BaseModel):
    task_id:      str
    observation:  ProcurementObservation
    message:      str = "Episode started"


class StepRequest(BaseModel):
    action: ProcurementAction


class StepResponse(BaseModel):
    observation:  ProcurementObservation
    reward:       StepReward
    done:         bool
    info:         dict = Field(default_factory=dict)


class TaskInfo(BaseModel):
    task_id:          str
    name:             str
    description:      str
    num_suppliers:    int
    num_weeks:        int
    categories:       list[ItemCategory]
    has_rival:        bool
    has_disruptions:  bool
    baseline_score:   float
    target_score:     float


class HealthResponse(BaseModel):
    status:   str = "healthy"
    version:  str = "1.0.0"
    env_name: str = "NegotiateAI-ProcurementArena"


class MetadataResponse(BaseModel):
    env_name:     str = "NegotiateAI-ProcurementArena"
    version:      str = "1.0.0"
    description:  str = (
        "Adversarial multi-LLM procurement environment. "
        "A buyer AI negotiates contracts in natural language "
        "against supplier AIs with hidden agendas, while a "
        "rival buyer AI competes for the same capacity."
    )
    themes:       list[str] = Field(default_factory=lambda: [
        "Theme 1: Multi-Agent Interactions",
        "Theme 2: Long-Horizon Planning",
        "Theme 3.1: World Modeling - Professional Tasks",
        "Theme 4: Self-Improvement",
        "Theme 5: Wild Card",
    ])
    tasks:        list[str] = Field(default_factory=lambda: [
        "easy_negotiation",
        "medium_adversarial",
        "hard_full_arena",
    ])
    author:       str = "prasanthdj8"
