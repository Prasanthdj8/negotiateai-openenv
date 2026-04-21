"""
app.py — NegotiateAI: Adversarial Procurement Arena
FastAPI server — all HTTP endpoints, WebSocket, MCP JSON-RPC.
OpenEnv-compliant: /reset /step /state /tasks /health /metadata /schema /mcp /ws
"""

from __future__ import annotations

import json
import logging
import os
import traceback
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from env import NegotiateAIEnv, TASKS, TASK_ALIASES
from graders import grade_episode
from models import (
    ActionType,
    EpisodeResult,
    HealthResponse,
    ItemCategory,
    MetadataResponse,
    ProcurementAction,
    ResetRequest,
    ResetResponse,
    StepRequest,
    StepResponse,
    TaskInfo,
)
from simulation import MarketSimulator, compute_episode_benchmark
from curriculum import CurriculumRegistry, shape_reward_for_grpo, simulate_training_curve

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("negotiateai")


# ─────────────────────────────────────────────────────────────
# GLOBAL STATE
# ─────────────────────────────────────────────────────────────

_envs: dict[str, NegotiateAIEnv] = {}
_curriculum = CurriculumRegistry()
_market_sim = MarketSimulator(seed=42)


def _get_env(session_id: str = "default") -> NegotiateAIEnv:
    if session_id not in _envs:
        _envs[session_id] = NegotiateAIEnv(
            difficulty_level=_curriculum.difficulty_level(
                _envs.get(session_id + "_task", "easy_negotiation")
            )
        )
    return _envs[session_id]


def _new_env(task_id: str, session_id: str = "default") -> NegotiateAIEnv:
    difficulty = _curriculum.difficulty_level(task_id)
    _envs[session_id] = NegotiateAIEnv(difficulty_level=difficulty)
    _envs[session_id + "_task"] = task_id
    return _envs[session_id]


# ─────────────────────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("NegotiateAI environment starting up...")
    log.info(f"Tasks available: {list(TASKS.keys())}")
    log.info(f"OpenAI base URL: {os.environ.get('OPENAI_BASE_URL', 'default')}")
    yield
    log.info("NegotiateAI shutting down.")


app = FastAPI(
    title="NegotiateAI — Adversarial Procurement Arena",
    description=(
        "The world's first multi-LLM adversarial procurement environment. "
        "A buyer AI negotiates contracts in natural language against supplier AIs "
        "with hidden agendas, while a rival buyer AI competes for the same capacity."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def _handle_episode_end(env: NegotiateAIEnv) -> dict[str, Any]:
    """Grade episode, update curriculum, return structured result."""
    task_id = env._task_id or "easy_negotiation"

    benchmark = compute_episode_benchmark(
        env.supplier_pool.get_visible_list(),
        env._task["requirements_config"],
        week=env.week,
        simulator=_market_sim,
    )

    grade_result = grade_episode(
        task_id=task_id,
        requirements=env.requirements,
        contracts=list(env.contracts.values()),
        budget_total=env.budget_total,
        budget_spent=env.budget_total - env.budget_remaining,
        total_steps=env.total_steps,
        total_weeks=env.week,
        deception_caught=env._deception_caught,
        deception_total=env._deception_total,
        rival_contracts_won=env._rival_contracts_won,
        our_contracts_won=sum(
            1 for c in env.contracts.values()
            if c.status.value == "fulfilled"
        ),
        disruption_count=len(env.disruptions),
        cancel_penalties=env._cancel_penalties,
        stockout_penalties=env._stockout_penalties,
        market_benchmark=benchmark,
    )

    score = grade_result["score"]

    curriculum_update = _curriculum.record(
        task_id=task_id,
        score=score,
        cost_savings=grade_result["components"].get("cost_savings_ratio", 0.0),
        fulfillment=grade_result["components"].get("fulfillment_rate", 0.0),
    )

    episode_result = env.get_episode_result()

    log.info(
        f"[END] task={task_id} success=True "
        f"steps={env.total_steps} "
        f"score={score:.4f} "
        f"rewards={score:.4f}"
    )

    return {
        "grade":      grade_result,
        "curriculum": curriculum_update,
        "result":     episode_result.model_dump(),
    }


# ─────────────────────────────────────────────────────────────
# CORE OPENENV ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.post("/reset")
async def reset(request: ResetRequest) -> ResetResponse:
    """Start a new episode. Required by OpenEnv."""
    task_id = TASK_ALIASES.get(request.task_id, request.task_id)
    if task_id not in TASKS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: {request.task_id!r}. "
                   f"Valid: {list(TASKS.keys()) + list(TASK_ALIASES.keys())}",
        )

    env = _new_env(task_id)
    obs = env.reset(task_id=task_id, seed=request.seed)

    log.info(f"[START] task={task_id} seed={request.seed} "
             f"suppliers={len(obs.suppliers)} budget=${obs.budget_total:,.0f}")

    return ResetResponse(
        task_id=task_id,
        observation=obs,
        message=f"Episode started: {TASKS[task_id]['name']}",
    )


@app.post("/step")
async def step(request: StepRequest) -> StepResponse:
    """Take one action. Required by OpenEnv."""
    env = _get_env()

    if not env._reset_called:
        raise HTTPException(status_code=400, detail="Call /reset before /step")
    if env.done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call /reset to start a new episode.",
        )

    try:
        obs, reward, done, info = env.step(request.action)
    except Exception as e:
        log.error(f"Step error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")

    log.info(
        f"[STEP] task={env._task_id} week={env.week}/{env.total_weeks} "
        f"action={request.action.action_type.value} "
        f"reward={reward.total:.4f} done={done}"
    )

    episode_end_info: dict[str, Any] = {}
    if done:
        episode_end_info = _handle_episode_end(env)
        info.update(episode_end_info)

    return StepResponse(
        observation=obs,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state")
async def state() -> dict[str, Any]:
    """Current environment state."""
    env = _get_env()
    if not env._reset_called:
        return {"status": "not_started", "message": "Call /reset to start an episode"}

    obs = env._build_observation()
    return {
        "observation":      obs.model_dump(),
        "done":             env.done,
        "week":             env.week,
        "total_weeks":      env.total_weeks,
        "budget_remaining": env.budget_remaining,
        "total_steps":      env.total_steps,
    }


@app.get("/tasks")
async def tasks() -> dict[str, Any]:
    """List all available tasks."""
    task_list = []
    for task_id, cfg in TASKS.items():
        task_list.append(TaskInfo(
            task_id=task_id,
            name=cfg["name"],
            description=cfg["description"],
            num_suppliers=cfg["num_suppliers"],
            num_weeks=cfg["total_weeks"],
            categories=cfg["categories"],
            has_rival=cfg["has_rival"],
            has_disruptions=len(cfg["disruption_config"]) > 0,
            baseline_score=cfg.get("baseline_score", 0.10),
            target_score=cfg.get("target_score", 0.50),
        ).model_dump())

    return {
        "tasks":   task_list,
        "aliases": TASK_ALIASES,
        "count":   len(task_list),
    }


@app.get("/health")
async def health() -> HealthResponse:
    """Health check."""
    return HealthResponse()


@app.get("/metadata")
async def metadata() -> MetadataResponse:
    """Environment metadata."""
    return MetadataResponse()


@app.get("/schema")
async def schema() -> dict[str, Any]:
    """Action and observation schemas."""
    import models as _models
    return {
        "action":          ProcurementAction.model_json_schema(),
        "observation":     _models.ProcurementObservation.model_json_schema(),
        "reward":          _models.StepReward.model_json_schema(),
        "action_types":    [a.value for a in ActionType],
        "item_categories": [c.value for c in ItemCategory],
    }


# ─────────────────────────────────────────────────────────────
# CURRICULUM ENDPOINTS
# ─────────────────────────────────────────────────────────────

@app.get("/curriculum")
async def get_curriculum() -> dict[str, Any]:
    """Curriculum state across all tasks."""
    return _curriculum.all_summaries()


@app.get("/curriculum/{task_id}")
async def get_task_curriculum(task_id: str) -> dict[str, Any]:
    """Curriculum state for a specific task."""
    task_id = TASK_ALIASES.get(task_id, task_id)
    return _curriculum.get(task_id).summary()


@app.get("/curriculum/{task_id}/curve")
async def get_reward_curve(task_id: str) -> dict[str, Any]:
    """
    Reward curve for a task.
    Returns simulated curve if fewer than 5 real episodes recorded.
    """
    task_id = TASK_ALIASES.get(task_id, task_id)
    engine  = _curriculum.get(task_id)
    curve   = engine.reward_curve()

    if len(curve) < 5:
        curve = simulate_training_curve(task_id=task_id, n_episodes=150, seed=42)

    return {
        "task_id":        task_id,
        "curve":          curve,
        "total_episodes": engine.total_episodes,
        "current_tier":   engine.current_tier,
        "rolling_avg":    engine.rolling_average(),
        "is_simulated":   engine.total_episodes < 5,
    }


# ─────────────────────────────────────────────────────────────
# DASHBOARD ENDPOINT
# ─────────────────────────────────────────────────────────────

@app.get("/dashboard")
async def dashboard() -> dict[str, Any]:
    """Full dashboard state for the HF Space UI."""
    env = _get_env()

    if not env._reset_called:
        return {
            "status":  "idle",
            "message": "No active episode. POST /reset to start.",
            "tasks":   list(TASKS.keys()),
        }

    obs = env._build_observation()

    return {
        "status":           "active",
        "task_id":          obs.task_id,
        "week":             obs.week,
        "total_weeks":      obs.total_weeks,
        "budget_remaining": obs.budget_remaining,
        "budget_total":     obs.budget_total,
        "budget_pct":       round(obs.budget_utilization * 100, 1),
        "total_steps":      env.total_steps,
        "done":             env.done,
        "suppliers": [
            {
                "id":             sv.supplier_id,
                "name":           sv.name,
                "category":       sv.category.value,
                "base_price":     sv.base_price,
                "is_active":      sv.is_active,
                "reliability":    sv.reliability_score,
                "rival_pressure": sv.rival_pressure,
                "capacity":       sv.capacity_available,
            }
            for sv in obs.suppliers
        ],
        "contracts": [
            {
                "id":            c.contract_id[:8],
                "supplier":      c.supplier_id,
                "item":          c.item_id,
                "value":         c.total_value,
                "status":        c.status.value,
                "delivery_week": c.expected_delivery_week,
            }
            for c in obs.contracts
        ],
        "requirements": [
            {
                "id":        r.item_id,
                "name":      r.name,
                "category":  r.category.value,
                "quantity":  r.quantity,
                "deadline":  r.deadline_week,
                "fulfilled": r.fulfilled,
                "critical":  r.is_critical,
            }
            for r in obs.requirements
        ],
        "disruptions":         [d.description for d in obs.disruptions],
        "recent_signals": [
            {
                "type": s.signal_type,
                "desc": s.description,
                "sev":  s.severity,
            }
            for s in obs.market_signals[-5:]
        ],
        "rival_contracts_won": obs.rival_contracts_won,
        "difficulty":          obs.difficulty_level,
        "open_threads":        len(obs.open_threads),
    }


@app.get("/negotiation/{supplier_id}")
async def get_negotiation_thread(supplier_id: str) -> dict[str, Any]:
    """Get negotiation threads for a supplier."""
    env = _get_env()
    if not env._reset_called:
        raise HTTPException(status_code=400, detail="No active episode")

    threads = [
        t.model_dump()
        for t in env.threads.values()
        if t.supplier_id == supplier_id
    ]
    return {
        "supplier_id": supplier_id,
        "threads":     threads,
        "count":       len(threads),
    }


# ─────────────────────────────────────────────────────────────
# MCP JSON-RPC
# ─────────────────────────────────────────────────────────────

@app.post("/mcp")
async def mcp_handler(request: dict[str, Any]) -> dict[str, Any]:
    """MCP JSON-RPC endpoint."""
    method  = request.get("method", "")
    params  = request.get("params", {})
    req_id  = request.get("id", str(uuid.uuid4()))

    try:
        if method == "reset":
            result = await reset(ResetRequest(
                task_id=params.get("task_id", "easy_negotiation"),
                seed=params.get("seed"),
            ))
            return {"jsonrpc": "2.0", "id": req_id, "result": result.model_dump()}

        elif method == "step":
            result = await step(StepRequest(
                action=ProcurementAction(**params.get("action", {}))
            ))
            return {"jsonrpc": "2.0", "id": req_id, "result": result.model_dump()}

        elif method == "state":
            return {"jsonrpc": "2.0", "id": req_id, "result": await state()}

        elif method == "tasks":
            return {"jsonrpc": "2.0", "id": req_id, "result": await tasks()}

        elif method == "health":
            result = await health()
            return {"jsonrpc": "2.0", "id": req_id, "result": result.model_dump()}

        elif method == "metadata":
            result = await metadata()
            return {"jsonrpc": "2.0", "id": req_id, "result": result.model_dump()}

        elif method == "curriculum":
            return {"jsonrpc": "2.0", "id": req_id, "result": await get_curriculum()}

        else:
            return {
                "jsonrpc": "2.0", "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

    except ValidationError as e:
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32602, "message": "Invalid params", "data": str(e)},
        }
    except Exception as e:
        log.error(f"MCP error: {traceback.format_exc()}")
        return {
            "jsonrpc": "2.0", "id": req_id,
            "error": {"code": -32603, "message": str(e)},
        }


# ─────────────────────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint — used by OpenEnv client."""
    await websocket.accept()
    session_id = str(uuid.uuid4())
    log.info(f"WebSocket connected: session={session_id}")

    try:
        while True:
            raw = await websocket.receive_text()

            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                continue

            method = message.get("method", "")
            params = message.get("params", {})
            response: dict[str, Any] = {}

            try:
                if method == "reset":
                    task_id = TASK_ALIASES.get(
                        params.get("task_id", "easy_negotiation"),
                        params.get("task_id", "easy_negotiation"),
                    )
                    env = _new_env(task_id, session_id)
                    obs = env.reset(task_id=task_id, seed=params.get("seed"))
                    log.info(f"[START] ws={session_id} task={task_id}")
                    response = {
                        "type":        "reset",
                        "task_id":     task_id,
                        "observation": obs.model_dump(),
                        "message":     f"Episode started: {TASKS[task_id]['name']}",
                    }

                elif method == "step":
                    env = _envs.get(session_id) or _get_env()
                    if not env._reset_called:
                        response = {"error": "Call reset first"}
                    elif env.done:
                        response = {"error": "Episode done. Call reset."}
                    else:
                        action = ProcurementAction(**params.get("action", {}))
                        obs, reward, done, info = env.step(action)
                        log.info(
                            f"[STEP] ws={session_id} "
                            f"action={action.action_type.value} "
                            f"reward={reward.total:.4f} done={done}"
                        )
                        episode_end: dict[str, Any] = {}
                        if done:
                            episode_end = _handle_episode_end(env)
                        response = {
                            "type":        "step",
                            "observation": obs.model_dump(),
                            "reward":      reward.model_dump(),
                            "done":        done,
                            "info":        {**info, **episode_end},
                        }

                elif method == "state":
                    env = _envs.get(session_id) or _get_env()
                    if env._reset_called:
                        obs = env._build_observation()
                        response = {
                            "type":        "state",
                            "observation": obs.model_dump(),
                            "done":        env.done,
                        }
                    else:
                        response = {"type": "state", "status": "not_started"}

                elif method == "ping":
                    response = {"type": "pong", "session_id": session_id}

                else:
                    response = {"error": f"Unknown method: {method}"}

            except ValidationError as e:
                response = {"error": "Validation error", "detail": str(e)}
            except Exception as e:
                log.error(f"WS error: {traceback.format_exc()}")
                response = {"error": str(e)}

            await websocket.send_text(json.dumps(response, default=str))

    except WebSocketDisconnect:
        log.info(f"WebSocket disconnected: session={session_id}")
        _envs.pop(session_id, None)
        _envs.pop(session_id + "_task", None)


# ─────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────

@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "type": "validation_error"},
    )


@app.exception_handler(Exception)
async def general_error_handler(request, exc):
    log.error(f"Unhandled error: {traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc), "type": "server_error"},
    )


# ─────────────────────────────────────────────────────────────
# ROOT
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def root() -> dict[str, Any]:
    return {
        "name":    "NegotiateAI — Adversarial Procurement Arena",
        "version": "1.0.0",
        "endpoints": {
            "POST /reset":              "Start episode",
            "POST /step":               "Take action",
            "GET  /state":              "Current state",
            "GET  /tasks":              "Task list",
            "GET  /health":             "Health check",
            "GET  /metadata":           "Environment metadata",
            "GET  /schema":             "Action/observation schemas",
            "POST /mcp":                "MCP JSON-RPC",
            "WS   /ws":                 "WebSocket (OpenEnv client)",
            "GET  /dashboard":          "Full dashboard state (UI)",
            "GET  /curriculum":         "Curriculum state",
            "GET  /curriculum/{task}/curve": "Reward curve",
        },
        "themes": [
            "Theme 1: Multi-Agent (supplier LLMs + rival LLM)",
            "Theme 2: Long-Horizon Planning (12-week cycle)",
            "Theme 3.1: World Modeling (enterprise PR workflow)",
            "Theme 4: Self-Improvement (curriculum engine)",
            "Theme 5: Wild Card",
        ],
        "author": "prasanthdj8",
    }
