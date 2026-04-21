"""
inference.py — NegotiateAI: Adversarial Procurement Arena
Baseline LLM inference script for OpenEnv evaluation.
Uses [START]/[STEP]/[END] log format required by OpenEnv.
Reads OPENAI_API_KEY and OPENAI_BASE_URL from environment variables.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any

import requests
from openai import OpenAI

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

ENV_URL    = os.environ.get("ENV_URL", "http://localhost:7860")
MODEL      = os.environ.get("INFERENCE_MODEL", "gpt-4o-mini")
MAX_STEPS  = int(os.environ.get("MAX_STEPS", "50"))
TASK_ID    = os.environ.get("TASK_ID", "easy_negotiation")
SEED       = int(os.environ.get("SEED", "42"))
VERBOSE    = os.environ.get("VERBOSE", "1") == "1"

logging.basicConfig(
    level=logging.INFO if VERBOSE else logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("inference")

# ─────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are an expert AI procurement agent for a tech company.
Your goal: source software, hardware, and services at the best cost,
within budget, before deadlines — while outperforming a rival buyer
and detecting deceptive suppliers.

STRATEGY GUIDE:
1. Negotiate price DOWN from list price — always open below list
2. Raise a PR before awarding contracts (compliance)
3. Watch rival_pressure signals — hedge if > 0.6
4. Reject suppliers with low community_rating or known issues
5. Escalate PRs when deadline is approaching (≤ 2 weeks left)
6. Hedge critical items across 2 suppliers when rival pressure is high
7. Read supplier messages carefully — deceptive suppliers deflect on reliability

Always respond with ONLY valid JSON matching this schema:
{
  "action_type": "negotiate" | "award_contract" | "reject" | "raise_pr" | "escalate" | "hedge" | "defer" | "cancel_contract",
  "supplier_id": "<supplier_id from observation>",
  "item_id": "<item_id from requirements>",
  "message": "<natural language negotiation message>",
  "proposed_price": <float or null>,
  "proposed_quantity": <int or null>,
  "proposed_lead_time": <int or null>,
  "hedge_supplier_id": "<second supplier_id for hedge, or null>",
  "hedge_quantity": <int for hedge split, or null>,
  "notes": "<your internal reasoning>"
}

No preamble. No markdown. Pure JSON only.
""".strip()


def build_user_prompt(obs: dict[str, Any]) -> str:
    """Convert observation dict to a concise LLM prompt."""

    # Format suppliers
    supplier_lines = []
    for s in obs.get("suppliers", []):
        status = "✅ ACTIVE" if s["is_active"] else "❌ OFFLINE"
        pressure = s.get("rival_pressure", 0)
        pressure_flag = " ⚠️ HIGH RIVAL PRESSURE" if pressure > 0.6 else ""
        supplier_lines.append(
            f"  {s['supplier_id']} | {s['name']} | {s['category']} | "
            f"${s['base_price']:.0f}/unit | lead={s['lead_time_days']}d | "
            f"reliability={s['reliability_score']:.2f} | "
            f"capacity={s['capacity_available']} | {status}{pressure_flag}"
        )

    # Format requirements
    req_lines = []
    for r in obs.get("requirements", []):
        status = "✅ DONE" if r["fulfilled"] else "⏳ PENDING"
        critical = " [CRITICAL]" if r["is_critical"] else ""
        weeks_left = r["deadline_week"] - obs.get("week", 1)
        req_lines.append(
            f"  {r['item_id']} | {r['name']} | qty={r['quantity']} | "
            f"deadline=week{r['deadline_week']} ({weeks_left}w left) | "
            f"budget=${r['budget_ceiling']:,.0f} | {status}{critical}"
        )

    # Format contracts
    contract_lines = []
    for c in obs.get("contracts", [])[-5:]:   # last 5
        contract_lines.append(
            f"  {c['contract_id'][:8]}... | {c['supplier_id']} | "
            f"${c['total_value']:,.0f} | status={c['status']} | "
            f"delivery=week{c['expected_delivery_week']}"
        )

    # Format recent signals
    signal_lines = []
    for s in obs.get("market_signals", [])[-4:]:
        signal_lines.append(f"  [{s['signal_type']}] sev={s['severity']:.1f}: {s['description'][:80]}")

    # Format disruptions
    disruption_lines = [
        f"  ⚡ {d['description']}" for d in obs.get("disruptions", [])
    ]

    # Format rival activity
    rival_high = [
        f"{sid}(pressure={p:.2f})"
        for sid, p in obs.get("rival_activity", {}).items()
        if p > 0.4
    ]

    prompt = f"""
── WEEK {obs.get('week', '?')} / {obs.get('total_weeks', '?')} ──────────────────────────────────────────
BUDGET: ${obs.get('budget_remaining', 0):,.0f} remaining of ${obs.get('budget_total', 0):,.0f}
DIFFICULTY: {obs.get('difficulty_level', 0):.2f}

REQUIREMENTS:
{chr(10).join(req_lines) if req_lines else '  None'}

SUPPLIERS:
{chr(10).join(supplier_lines) if supplier_lines else '  None'}

ACTIVE CONTRACTS:
{chr(10).join(contract_lines) if contract_lines else '  None'}

MARKET SIGNALS:
{chr(10).join(signal_lines) if signal_lines else '  None'}

DISRUPTIONS:
{chr(10).join(disruption_lines) if disruption_lines else '  None'}

RIVAL ACTIVITY (high pressure):
{', '.join(rival_high) if rival_high else '  Low'}

RIVAL CONTRACTS WON: {obs.get('rival_contracts_won', 0)}

What is your next procurement action? Respond with JSON only.
""".strip()

    return prompt


# ─────────────────────────────────────────────────────────────
# ENV CLIENT
# ─────────────────────────────────────────────────────────────

def env_reset(task_id: str, seed: int | None = None) -> dict[str, Any]:
    resp = requests.post(
        f"{ENV_URL}/reset",
        json={"task_id": task_id, "seed": seed},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_step(action: dict[str, Any]) -> dict[str, Any]:
    resp = requests.post(
        f"{ENV_URL}/step",
        json={"action": action},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def env_health() -> bool:
    try:
        resp = requests.get(f"{ENV_URL}/health", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# LLM AGENT
# ─────────────────────────────────────────────────────────────

def get_llm_action(
    client: OpenAI,
    obs: dict[str, Any],
    step_num: int,
    conversation_history: list[dict[str, str]],
) -> dict[str, Any] | None:
    """
    Call LLM with current observation, get procurement action.
    Maintains conversation history for multi-turn reasoning.
    """
    user_prompt = build_user_prompt(obs)

    # Keep last 6 turns to stay within context
    recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *recent_history,
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.3,    # low temp for consistent decisions
        )
        raw = response.choices[0].message.content.strip()

        # Add to history
        conversation_history.append({"role": "user",      "content": user_prompt})
        conversation_history.append({"role": "assistant", "content": raw})

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        action = json.loads(raw)
        return action

    except json.JSONDecodeError as e:
        log.warning(f"Step {step_num}: JSON parse error — {e}. Raw: {raw[:100]}")
        return None
    except Exception as e:
        log.warning(f"Step {step_num}: LLM error — {e}")
        return None


def fallback_action(obs: dict[str, Any]) -> dict[str, Any]:
    """
    Rule-based fallback when LLM fails to produce valid JSON.
    Always returns a valid action.
    """
    suppliers = [s for s in obs.get("suppliers", []) if s["is_active"]]
    requirements = [r for r in obs.get("requirements", []) if not r["fulfilled"]]

    if not suppliers or not requirements:
        # Defer if nothing actionable
        all_sups = obs.get("suppliers", [])
        all_reqs = obs.get("requirements", [])
        sup_id  = all_sups[0]["supplier_id"] if all_sups else "sup_001"
        item_id = all_reqs[0]["item_id"]     if all_reqs else "item_001"
        return {
            "action_type":  "defer",
            "supplier_id":  sup_id,
            "item_id":      item_id,
            "message":      "Deferring — waiting for better conditions",
            "notes":        "Fallback: no active suppliers or requirements",
        }

    # Pick first unfulfilled requirement + lowest price supplier
    req = requirements[0]
    sup = min(suppliers, key=lambda s: s["base_price"])
    discount_price = round(sup["base_price"] * 0.92, 2)

    return {
        "action_type":       "negotiate",
        "supplier_id":       sup["supplier_id"],
        "item_id":           req["item_id"],
        "message":           (
            f"We need {req['quantity']} {req['name']} units. "
            f"Our budget ceiling is ${req['budget_ceiling']:,.0f}. "
            f"Can you offer ${discount_price:.2f}/unit?"
        ),
        "proposed_price":    discount_price,
        "proposed_quantity": req["quantity"],
        "notes":             "Fallback: rule-based negotiation opener",
    }


# ─────────────────────────────────────────────────────────────
# MAIN EPISODE RUNNER
# ─────────────────────────────────────────────────────────────

def run_episode(
    client: OpenAI,
    task_id: str = TASK_ID,
    seed: int = SEED,
    max_steps: int = MAX_STEPS,
) -> dict[str, Any]:
    """
    Run one complete episode and return the result.
    Logs in OpenEnv [START]/[STEP]/[END] format.
    """

    # ── Reset ─────────────────────────────────────────────────
    log.info(f"[START] task={task_id} seed={seed}")
    try:
        reset_data = env_reset(task_id, seed)
    except Exception as e:
        log.error(f"Reset failed: {e}")
        return {"success": False, "error": str(e)}

    obs               = reset_data["observation"]
    conversation_history: list[dict[str, str]] = []
    step_rewards: list[float] = []
    total_score = 0.0

    print(f"\n{'='*60}")
    print(f"NegotiateAI — {task_id}")
    print(f"Week 1/{obs['total_weeks']} | "
          f"Budget: ${obs['budget_remaining']:,.0f} | "
          f"Suppliers: {len(obs['suppliers'])} | "
          f"Requirements: {len(obs['requirements'])}")
    print(f"{'='*60}")

    # ── Episode loop ──────────────────────────────────────────
    for step_num in range(1, max_steps + 1):
        if obs.get("done") or all(r["fulfilled"] for r in obs.get("requirements", [])):
            break

        # Get LLM action
        action = get_llm_action(client, obs, step_num, conversation_history)

        if action is None:
            log.warning(f"Step {step_num}: Using fallback action")
            action = fallback_action(obs)

        # Step environment
        try:
            step_data = env_step(action)
        except Exception as e:
            log.error(f"Step {step_num}: env_step failed — {e}")
            break

        reward  = step_data["reward"]
        done    = step_data["done"]
        obs     = step_data["observation"]
        info    = step_data.get("info", {})

        step_rewards.append(reward["total"])

        # Log in OpenEnv [STEP] format
        log.info(
            f"[STEP] step={step_num} "
            f"action={action['action_type']} "
            f"reward={reward['total']:.4f} "
            f"week={obs.get('week','?')}/{obs.get('total_weeks','?')} "
            f"done={done}"
        )

        if VERBOSE:
            print(
                f"Step {step_num:3d} | "
                f"Week {obs.get('week','?'):2}/{obs.get('total_weeks','?')} | "
                f"Action: {action['action_type']:20s} | "
                f"Reward: {reward['total']:.4f} | "
                f"{reward.get('explanation','')[:50]}"
            )

        if done:
            # Extract final score from episode end info
            grade = info.get("grade", {})
            total_score = grade.get("score", reward["total"])

            curriculum = info.get("curriculum", {})
            tier = curriculum.get("tier", "unknown")

            break

    # ── Final score ───────────────────────────────────────────
    if not step_rewards:
        total_score = 0.0
    elif total_score == 0.0:
        total_score = sum(step_rewards) / len(step_rewards)

    avg_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0.0

    # Log in OpenEnv [END] format
    log.info(
        f"[END] task={task_id} "
        f"success=True "
        f"steps={step_num} "
        f"score={total_score:.4f} "
        f"rewards={total_score:.4f}"
    )

    print(f"\n{'='*60}")
    print(f"Episode complete")
    print(f"Steps: {step_num} | Score: {total_score:.4f} | Avg reward: {avg_reward:.4f}")
    print(f"{'='*60}\n")

    return {
        "success":      True,
        "task_id":      task_id,
        "steps":        step_num,
        "score":        total_score,
        "avg_reward":   avg_reward,
        "step_rewards": step_rewards,
    }


# ─────────────────────────────────────────────────────────────
# MULTI-TASK EVALUATION
# ─────────────────────────────────────────────────────────────

def run_evaluation(
    client: OpenAI,
    tasks: list[str] | None = None,
    episodes_per_task: int = 1,
    seed: int = SEED,
) -> dict[str, Any]:
    """
    Run full evaluation across all tasks.
    Reports per-task scores and overall performance.
    """
    if tasks is None:
        tasks = ["easy_negotiation", "medium_adversarial", "hard_full_arena"]

    results: dict[str, list[float]] = {t: [] for t in tasks}

    for task_id in tasks:
        print(f"\n{'─'*60}")
        print(f"Evaluating: {task_id}")
        print(f"{'─'*60}")

        for ep in range(episodes_per_task):
            result = run_episode(
                client,
                task_id=task_id,
                seed=seed + ep,
                max_steps=MAX_STEPS,
            )
            if result.get("success"):
                results[task_id].append(result["score"])
            time.sleep(0.5)  # small delay between episodes

    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    overall_scores = []
    for task_id, scores in results.items():
        if scores:
            avg = sum(scores) / len(scores)
            overall_scores.append(avg)
            print(f"  {task_id:30s}: {avg:.4f} (n={len(scores)})")
        else:
            print(f"  {task_id:30s}: N/A (no successful episodes)")
    if overall_scores:
        print(f"  {'Overall':30s}: {sum(overall_scores)/len(overall_scores):.4f}")
    print(f"{'='*60}\n")

    return {"results": results, "summary": {
        t: sum(s)/len(s) if s else 0.0 for t, s in results.items()
    }}


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("NegotiateAI — Inference Script")
    print(f"ENV_URL:   {ENV_URL}")
    print(f"MODEL:     {MODEL}")
    print(f"TASK_ID:   {TASK_ID}")
    print(f"MAX_STEPS: {MAX_STEPS}")
    print(f"SEED:      {SEED}")

    # Health check
    print("\nChecking environment health...")
    if not env_health():
        print(f"❌ Environment not reachable at {ENV_URL}")
        print("   Start with: uvicorn main:app --host 0.0.0.0 --port 7860")
        sys.exit(1)
    print("✅ Environment healthy")

    # Init LLM client
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

    # Run
    mode = os.environ.get("EVAL_MODE", "single")

    if mode == "eval":
        # Full evaluation across all tasks
        run_evaluation(client, episodes_per_task=1)
    else:
        # Single episode
        result = run_episode(client, task_id=TASK_ID, seed=SEED)
        sys.exit(0 if result.get("success") else 1)
