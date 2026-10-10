"""Online ReCo-WM with PNN evidence-based Context selection."""

from __future__ import annotations
import argparse
import hashlib
import datetime as dt
import json
import os
from pathlib import Path
import re
import sys
import traceback
from types import SimpleNamespace
from typing import Any

PROJECT = Path(__file__).resolve().parent


class Config(SimpleNamespace):

    def get(self, key, default=None):
        return getattr(self, key, default)


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env", required=True, choices=("Walker2d", "Hopper", "DMC", "HC")
    )
    parser.add_argument("--seed", type=int, default=2001)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "runs")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--steps", type=int, default=320000)
    parser.add_argument("--pnn-device", choices=("cpu", "cuda:0"), default="cuda:0")
    return parser


def _resolve(args):
    if not 1 <= args.steps <= 320000:
        raise ValueError("--steps must be between 1 and 320000")
    run_id = (
        args.run_id
        or f"{args.env}_seed{args.seed}_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    if not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id):
        raise ValueError("run-id must contain 1–96 filename-safe characters")
    payload = json.loads((PROJECT / "configs" / f"{args.env}.json").read_text())
    protocol = payload["environment"]
    cfg = Config(**payload["agent"])
    cfg.seed = args.seed
    cfg.steps = args.steps
    cfg.work_dir = args.output_dir.resolve() / run_id
    cfg.exp_name = run_id
    cfg.data_dir = None
    cfg.m3wm_s0_pnn_device = args.pnn_device
    return (cfg, protocol, run_id)


def _write_run_manifest(cfg, protocol, *, run_id, initial_state, environment_observed):
    path = Path(cfg.work_dir) / "manifest.json"
    payload = {
        "method": "ReCo-WM",
        "run_id": run_id,
        "agent": vars(cfg),
        "environment": protocol,
        "environment_observed": environment_observed,
        "initial_state": initial_state,
        "context_selection": "PNN likelihood competition and future-data confirmation",
        "fresh_initialization": True,
    }
    path.write_text(json.dumps(payload, default=str, indent=2) + "\n")
    return path


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish a small JSON control file without exposing partial writes."""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_run_status(
    output: Path,
    *,
    status: str,
    last_step: int,
    expected_steps: int,
    completed: bool,
    error: str | None,
    stop_reason: str | None,
    run_id: str | None = None,
    replay_failure: dict[str, Any] | None = None,
) -> Path:
    """Persist monitor-friendly lifecycle state while retaining run identity."""
    path = output / "run_status.json"
    previous: dict[str, Any] = {}
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous = loaded
        except (OSError, UnicodeError, json.JSONDecodeError):
            previous = {}
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    payload = {
        "schema": "m3wm_online_four_task_run_status",
        "version": 1,
        "run_id": (
            str(run_id) if run_id is not None else str(previous.get("run_id", ""))
        ),
        "status": str(status),
        "last_step": int(last_step),
        "expected_steps": int(expected_steps),
        "completed": bool(completed),
        "error": None if error is None else str(error),
        "stop_reason": None if stop_reason is None else str(stop_reason),
        "replay_failure": (
            replay_failure
            if replay_failure is not None
            else previous.get("replay_failure")
        ),
        "pid": int(os.getpid()),
        "created_utc": str(previous.get("created_utc", now)),
        "updated_utc": now,
    }
    _atomic_write_json(path, payload)
    return path


def _build_env(cfg, protocol: dict[str, Any]):
    from envs.wrappers.tensor import TensorWrapper

    if str(protocol["env_id"]) == "DMCWalkerRun-v0":
        from envs.dmc_walker_run_bridge import DMCWalkerRunBridge

        raw = DMCWalkerRunBridge(
            seed=int(cfg.seed),
            sequence=tuple(protocol["sequence"]),
            change_freq=int(protocol["change_freq"]),
            phase_lengths=tuple(protocol["phase_lengths"]),
            target_velocities=dict(protocol["target_velocities"]),
            malfunction_mask=tuple(protocol["jm_mask"]),
            wind_force=tuple(protocol["wind_force"]),
            diagnostic_revisit=bool(cfg.get("m3wm_passive_revisit_diagnostic", False)),
        )
    else:
        from envs.locomotion_v2_bridge import LocomotionV2Bridge

        raw = LocomotionV2Bridge(
            env_id=str(protocol["env_id"]),
            seed=int(cfg.seed),
            sequence=tuple(protocol["sequence"]),
            change_freq=int(protocol["change_freq"]),
            phase_lengths=tuple(protocol["phase_lengths"]),
            target_velocities=dict(protocol["target_velocities"]),
            malfunction_mask=tuple(protocol["jm_mask"]),
            wind_force=tuple(protocol["wind_force"]),
            worker_python=cfg.get("four_task_v2_worker_python", None),
        )
    env = TensorWrapper(raw)
    observed = {
        "env_id": str(raw.env_id),
        "observation_dim": int(raw.observation_space.shape[0]),
        "action_dim": int(raw.action_space.shape[0]),
        "sequence": tuple(raw.sequence),
        "change_freq": int(raw._change_freq),
        "phase_lengths": tuple((int(value) for value in raw._phase_lengths)),
        "jm_mask": tuple((float(value) for value in raw._malfunction_mask)),
        "wind_force": tuple((float(value) for value in raw._wind)),
        "target_velocities": dict(raw._target_velocities),
    }
    expected = {
        "env_id": str(protocol["env_id"]),
        "observation_dim": int(protocol["expected_observation_dim"]),
        "action_dim": int(protocol["expected_action_dim"]),
        "sequence": tuple(protocol["sequence"]),
        "change_freq": int(protocol["change_freq"]),
        "phase_lengths": tuple((int(value) for value in protocol["phase_lengths"])),
        "jm_mask": tuple((float(value) for value in protocol["jm_mask"])),
        "wind_force": tuple((float(value) for value in protocol["wind_force"])),
        "target_velocities": dict(protocol["target_velocities"]),
    }
    if observed != expected:
        raw.close()
        raise RuntimeError(
            f"constructed environment differs from the frozen protocol: observed={observed!r}, expected={expected!r}"
        )
    env._m3wm_online_protocol_observed = observed
    env.action_space.seed(int(cfg.seed))
    cfg.obs_shape = {"state": env.observation_space.shape}
    cfg.action_dim = int(env.action_space.shape[0])
    cfg.episode_length = int(env.max_episode_steps)
    cfg.seed_steps = int(cfg.seed_steps)
    return env


def _module_state_sha256(module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _initial_state(agent, buffer, *, identity_shadow=None) -> dict[str, Any]:
    metrics = agent.context_metrics() if hasattr(agent, "context_metrics") else {}
    scalar = lambda value, default=0: int(
        float(value.detach().cpu().reshape(-1)[0])
        if hasattr(value, "detach")
        else value if value is not None else default
    )
    state = {
        "random_initialization": True,
        "checkpoint_loaded": False,
        "replay_episodes": int(buffer.num_eps),
        "replay_total_transitions": int(buffer.total_transitions),
        "replay_recent_refresh_count": int(buffer.recent_refresh_count),
        "model_train_updates": int(getattr(agent, "_m3wm_train_updates", -1)),
        "optimizer_state_entries": int(len(agent.optim.state)),
        "actor_optimizer_state_entries": int(len(agent.pi_optim.state)),
        "context_memory_size": scalar(metrics.get("context_memory_size", 0)),
        "active_context_id": scalar(metrics.get("active_context_id", -1), -1),
        "context_creation_count": scalar(metrics.get("context_creation_count", 0)),
        "context_recall_count": scalar(metrics.get("context_recall_count", 0)),
        "context_anchor_ready": scalar(metrics.get("context_anchor_ready", 0)),
        "pending_change_active": scalar(metrics.get("pending_change_active", 0)),
        "control_model_state_sha256": _module_state_sha256(agent.model),
        "proposal_sidecar": (
            None
            if identity_shadow is None
            else {
                "full_tdmpc2": bool(identity_shadow.full_tdmpc2_enabled),
                "theta_learning_enabled": bool(identity_shadow.theta_learning_enabled),
                "model_state_sha256": (
                    None
                    if identity_shadow.model is None
                    else _module_state_sha256(identity_shadow.model)
                ),
                "optimizer_state_entries": (
                    0
                    if identity_shadow.optimizer is None
                    else int(len(identity_shadow.optimizer.state))
                ),
                "replay_total_transitions": (
                    0
                    if identity_shadow.side_buffer is None
                    else int(identity_shadow.side_buffer.total_transitions)
                ),
            }
        ),
    }
    expected = {
        "replay_episodes": 0,
        "replay_total_transitions": 0,
        "replay_recent_refresh_count": 0,
        "model_train_updates": 0,
        "optimizer_state_entries": 0,
        "actor_optimizer_state_entries": 0,
        "context_memory_size": 0,
        "active_context_id": -1,
        "context_creation_count": 0,
        "context_recall_count": 0,
        "context_anchor_ready": 0,
        "pending_change_active": 0,
    }
    mismatch = {
        key: {"expected": value, "observed": state.get(key)}
        for (key, value) in expected.items()
        if state.get(key) != value
    }
    if mismatch:
        raise RuntimeError(f"non-fresh initial algorithm state: {mismatch}")
    return state


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg, protocol, run_id = _resolve(args)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "method": "ReCo-WM",
                    "environment": protocol,
                    "steps": cfg.steps,
                    "seed": cfg.seed,
                    "work_dir": str(cfg.work_dir),
                    "context_selection": "pnn_evidence",
                },
                indent=2,
            )
        )
        return 0
    os.environ.setdefault("MUJOCO_GL", "egl")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("online TD-MPC2 training requires CUDA")
    from common.buffer import Buffer
    from common.logger import Logger
    from common.seed import set_seed
    from common.m3wm_control_identity_shadow import PlainControlIdentityShadow
    from common.m3wm_context_agent_bank import ContextAgentBank
    from common.m3wm_context_agent_bundle import ContextAgentBundle
    from common.m3wm_context_agent_factory import TDMPC2ContextBundleFactory
    from common.m3wm_s0_context_router import S0ContextRouter
    from common.m3wm_s0_context_selector import S0PNNEvidenceContextSelector
    from common.m3wm_s0_pnn import PersistentPNNEnsemble
    from common.m3wm_s0_trial_controller import TrialController
    from tdmpc2 import TDMPC2
    from trainer.online_trainer import OnlineTrainer

    set_seed(int(cfg.seed))
    output = Path(cfg.work_dir)
    output.mkdir(parents=True, exist_ok=False)
    _write_run_status(
        output,
        status="initializing",
        last_step=0,
        expected_steps=int(cfg.steps),
        completed=False,
        error=None,
        stop_reason=None,
        run_id=run_id,
    )
    env = None
    trainer = None
    try:
        try:
            env = _build_env(cfg, protocol)
            agent = TDMPC2(cfg)
            identity_shadow = None
            with torch.random.fork_rng(
                devices=(
                    [torch.cuda.current_device()] if torch.cuda.is_available() else []
                ),
                enabled=True,
            ):
                sidecar_model_seed = int(cfg.seed)
                torch.manual_seed(sidecar_model_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(sidecar_model_seed)
                identity_shadow = PlainControlIdentityShadow(
                    cfg,
                    device=getattr(agent, "device", "cuda:0"),
                    seed=int(cfg.seed) + 7919,
                    work_dir=output,
                )
            buffer = Buffer(cfg)
            context_bank = None
            context_selector = None
            context_router = None
            trial_controller = None
            initial_bundle = ContextAgentBundle(
                context_id=0,
                control_agent=agent,
                control_replay=buffer,
                theta_agent=identity_shadow,
                theta_replay=None,
                predictor_state=None,
                metadata={
                    "creation": "random_initialization",
                    "new_optimizer_policy": "initial",
                    "replay_policy": "independent",
                },
                active=True,
            )
            context_bank = ContextAgentBank(
                initial_bundle=initial_bundle,
                bundle_factory=TDMPC2ContextBundleFactory(
                    cfg, work_dir=output / "context_agents"
                ),
                storage_dir=output / "context_agents" / "snapshots",
                release_inactive=True,
            )
            selector_name = str(cfg.get("m3wm_context_selector", ""))
            if selector_name == "pnn_evidence":
                state_dim = int(env.observation_space.shape[0])
                action_dim = int(env.action_space.shape[0])

                def predictor_factory(context_id):
                    return PersistentPNNEnsemble(
                        state_dim,
                        action_dim,
                        members=5,
                        hidden=200,
                        layers=4,
                        seed=int(cfg.seed) + int(context_id) * 1000003,
                        device=str(cfg.get("m3wm_s0_pnn_device", "cpu")),
                    )

                context_selector = S0PNNEvidenceContextSelector(
                    initial_predictor=predictor_factory(0),
                    predictor_factory=predictor_factory,
                    threshold=float(
                        cfg.get("m3wm_pnn_evidence_cusum_threshold", 300.0)
                    ),
                    initial_fit_updates=200,
                    maintenance_every=250,
                    maintenance_updates=20,
                    batch_size=256,
                    min_ready_rows=512,
                    max_contexts=8,
                    max_disagreement=0.5,
                    detection_recent_capacity=1024,
                    detection_reservoir_capacity=4096,
                    detection_seed=int(cfg.seed) + 90017,
                    trial_requests_enabled=True,
                    trial_probe_fraction=0.5,
                    settle_after_new_steps=int(
                        cfg.get("m3wm_s0_settle_after_new_steps", 1024)
                    ),
                    settle_after_return_steps=int(
                        cfg.get("m3wm_s0_settle_after_return_steps", 256)
                    ),
                    new_min_maintenance_rounds=int(
                        cfg.get("m3wm_s0_new_min_maintenance_rounds", 2)
                    ),
                    return_confirmation_enabled=True,
                    return_confirm_block_size=int(
                        cfg.get("m3wm_s0_return_confirm_block_size", 32)
                    ),
                    return_confirm_blocks=int(
                        cfg.get("m3wm_s0_return_confirm_blocks", 2)
                    ),
                    failed_candidate_hold_steps=int(
                        cfg.get("m3wm_s0_failed_candidate_hold_steps", 1024)
                    ),
                    post_new_failed_candidate_hold_steps=int(
                        cfg.get("m3wm_s0_post_new_failed_candidate_hold_steps", 2048)
                    ),
                    post_new_return_confirm_blocks=int(
                        cfg.get("m3wm_s0_post_new_return_confirm_blocks", 3)
                    ),
                    new_confirmation_enabled=True,
                    new_confirm_block_size=int(
                        cfg.get("m3wm_s0_new_confirm_block_size", 32)
                    ),
                    new_confirm_blocks=int(cfg.get("m3wm_s0_new_confirm_blocks", 2)),
                    residual_confirmation_enabled=bool(
                        cfg.get("m3wm_s0_residual_confirmation_enabled", False)
                    ),
                    residual_block_size=int(cfg.get("m3wm_s0_residual_block_size", 32)),
                    residual_reference_min_blocks=int(
                        cfg.get("m3wm_s0_residual_reference_min_blocks", 128)
                    ),
                    residual_reference_max_blocks=int(
                        cfg.get("m3wm_s0_residual_reference_max_blocks", 256)
                    ),
                    residual_tail_rank_alpha=float(
                        cfg.get("m3wm_s0_residual_tail_rank_alpha", 0.01)
                    ),
                )
                trial_controller = TrialController(
                    enabled=True,
                    check_interval=32,
                    max_trial_steps=128,
                    cooldown_steps=1024,
                    total_ratio=0.02,
                    absolute_limit=4096,
                )
                context_router = S0ContextRouter(
                    selector=context_selector,
                    bank=context_bank,
                    trial_controller=trial_controller,
                    observer_only=False,
                )
            else:
                raise RuntimeError("Unknown Context selector")
            initial_state = _initial_state(
                agent, buffer, identity_shadow=identity_shadow
            )
            manifest = _write_run_manifest(
                cfg,
                protocol,
                run_id=run_id,
                initial_state=initial_state,
                environment_observed=env._m3wm_online_protocol_observed,
            )
            print(f"[FreshOnlineManifest] {manifest}")
            trainer = OnlineTrainer(
                cfg=cfg,
                env=env,
                agent=agent,
                buffer=buffer,
                logger=Logger(cfg),
                identity_shadow=identity_shadow,
                context_bank=context_bank,
                context_selector=context_selector,
                context_router=context_router,
                trial_controller=trial_controller,
            )
            if context_router is not None:
                context_router.before_switch = trainer._prepare_s0_context_switch
            _write_run_status(
                output,
                status="running",
                last_step=int(trainer._step),
                expected_steps=int(cfg.steps),
                completed=False,
                error=None,
                stop_reason=None,
            )
            trainer.train()
        finally:
            if env is not None:
                env.close()
                env = None
    except KeyboardInterrupt:
        _write_run_status(
            output,
            status="stopped",
            last_step=int(0 if trainer is None else trainer._step),
            expected_steps=int(cfg.steps),
            completed=False,
            error=None,
            stop_reason="keyboard_interrupt",
        )
        raise
    except Exception as exc:
        traceback_text = traceback.format_exc()
        traceback_path = output / "failure_traceback.log"
        try:
            traceback_path.write_text(traceback_text, encoding="utf-8")
        except OSError as traceback_error:
            print(
                f"[FailureTracebackWriteError] {traceback_error}",
                file=sys.stderr,
                flush=True,
            )
        print(traceback_text, file=sys.stderr, end="", flush=True)
        replay_failure = None
        if trainer is not None:
            snapshot = getattr(trainer.buffer, "replay_failure_snapshot", None)
            if isinstance(snapshot, dict):
                replay_failure = snapshot
        _write_run_status(
            output,
            status="failed",
            last_step=int(0 if trainer is None else trainer._step),
            expected_steps=int(cfg.steps),
            completed=False,
            error=f"{type(exc).__name__}: {exc}",
            stop_reason="exception",
            replay_failure=replay_failure,
        )
        raise
    last_step = int(trainer._step)
    completed = last_step == int(cfg.steps)
    _write_run_status(
        output,
        status="completed" if completed else "stopped",
        last_step=last_step,
        expected_steps=int(cfg.steps),
        completed=completed,
        error=None,
        stop_reason="step_budget_reached" if completed else "early_return",
    )
    if not completed:
        raise RuntimeError(
            f"trainer returned before step budget: {last_step}/{int(cfg.steps)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
