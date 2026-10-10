from __future__ import annotations
import json
import os
import subprocess
from pathlib import Path
import gymnasium as gym
import numpy as np
from common.m3wm_episode_control_contract import control_contract

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WORKER_PYTHON = (
    PROJECT_DIR / "runtime" / "halfcheetah_v2_worker_venv" / "bin" / "python"
)
DEFAULT_MJPRO_PATH = PROJECT_DIR / "runtime" / "mjpro150"
WORKER = PROJECT_DIR / "external_baselines" / "locomotion_v2_bridge_worker.py"
SUPPORTED_ENVIRONMENTS = {
    "HalfCheetah-v2": {
        "robot": "halfcheetah",
        "task": "mujoco-halfcheetah",
        "action_dim": 6,
    },
    "Walker2d-v2": {"robot": "walker2d", "task": "mujoco-walker2d", "action_dim": 6},
    "Hopper-v2": {"robot": "hopper", "task": "mujoco-hopper", "action_dim": 3},
    "Ant-v2": {"robot": "ant", "task": "mujoco-ant", "action_dim": 8},
}


class LocomotionV2Bridge(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        env_id: str,
        seed: int,
        sequence: tuple[str, ...],
        change_freq: int,
        phase_lengths: tuple[int, ...] | None = None,
        target_velocities: dict,
        malfunction_mask: tuple[float, ...],
        wind_force: tuple[float, ...],
        worker_python: str | None = None,
    ) -> None:
        super().__init__()
        if env_id not in SUPPORTED_ENVIRONMENTS:
            raise ValueError(f"unsupported locomotion environment: {env_id}")
        self.env_id = str(env_id)
        python = Path(
            worker_python
            or os.environ.get("M3WM_LOCOMOTION_V2_WORKER_PYTHON", "")
            or os.environ.get("M3WM_HCV2_WORKER_PYTHON", "")
            or DEFAULT_WORKER_PYTHON
        )
        if not python.is_file():
            raise FileNotFoundError(f"locomotion-v2 worker Python: {python}")
        if not WORKER.is_file():
            raise FileNotFoundError(WORKER)
        child_env = os.environ.copy()
        child_env.pop("PYTHONPATH", None)
        mjpro_path = Path(
            os.environ.get("M3WM_MJPRO_PATH", "") or DEFAULT_MJPRO_PATH
        ).resolve()
        if not (mjpro_path / "bin" / "libmujoco150.so").is_file():
            raise FileNotFoundError(f"MuJoCo 1.50 runtime: {mjpro_path}")
        child_env["MUJOCO_PY_MJPRO_PATH"] = str(mjpro_path)
        old_ld = child_env.get("LD_LIBRARY_PATH", "")
        child_env["LD_LIBRARY_PATH"] = ":".join(
            [str(mjpro_path / "bin")] + ([old_ld] if old_ld else [])
        )
        self._process = subprocess.Popen(
            [str(python), "-u", str(WORKER)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            env=child_env,
        )
        self.sequence = tuple(sequence)
        self._change_freq = int(change_freq)
        self._phase_lengths = tuple(
            (
                int(value)
                for value in phase_lengths or (self._change_freq,) * len(self.sequence)
            )
        )
        if not self._phase_lengths or any(
            (value <= 0 for value in self._phase_lengths)
        ):
            raise ValueError("phase_lengths must contain positive values")
        self._target_velocities = dict(target_velocities)
        self._default_target_vel = 1.5
        self._malfunction_mask = np.asarray(malfunction_mask, dtype=np.float32)
        self._wind = np.asarray(wind_force, dtype=np.float32)
        self._pending_global_step: int | None = 0
        self._global_step = 0
        spec = self._request(
            {
                "cmd": "init",
                "env_id": self.env_id,
                "seed": int(seed),
                "sequence": list(self.sequence),
                "change_freq": self._change_freq,
                "phase_lengths": list(self._phase_lengths),
                "target_velocities": self._target_velocities,
                "default_target_velocity": self._default_target_vel,
                "malfunction_mask": self._malfunction_mask.tolist(),
                "wind_force": self._wind.tolist(),
            }
        )
        if str(spec.get("env_id")) != self.env_id:
            self.close()
            raise RuntimeError(
                f"worker environment mismatch: {spec.get('env_id')} != {self.env_id}"
            )
        self.observation_space = gym.spaces.Box(
            low=np.asarray(spec["observation_low"], dtype=np.float32),
            high=np.asarray(spec["observation_high"], dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.asarray(spec["action_low"], dtype=np.float32),
            high=np.asarray(spec["action_high"], dtype=np.float32),
            dtype=np.float32,
        )
        expected_dim = int(SUPPORTED_ENVIRONMENTS[self.env_id]["action_dim"])
        if self.action_space.shape != (expected_dim,):
            self.close()
            raise RuntimeError(
                f"{self.env_id} action dimension drifted: {self.action_space.shape}"
            )
        self._max_episode_steps = int(spec["max_episode_steps"])
        self._current_task = self.sequence[0]

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @property
    def current_task(self) -> str:
        return self._current_task

    def _request(self, payload: dict) -> dict:
        if self._process.poll() is not None:
            raise RuntimeError(
                f"{self.env_id} worker exited: {self._process.returncode}"
            )
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(json.dumps(payload) + "\n")
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f"{self.env_id} worker closed its output")
        result = json.loads(line)
        if result.get("bridge_error"):
            raise RuntimeError(
                f"{self.env_id} worker failure: {result['bridge_error']}\n{result.get('traceback', '')}"
            )
        return result

    def set_global_step(self, step: int) -> None:
        self._pending_global_step = int(step)
        self._global_step = int(step)
        index = self._phase_index(self._global_step)
        self._current_task = self.sequence[index]

    def _phase_index(self, step: int) -> int:
        remaining = max(0, int(step))
        for index, length in enumerate(self._phase_lengths):
            if remaining < length:
                return index % len(self.sequence)
            remaining -= length
        return (len(self._phase_lengths) - 1) % len(self.sequence)

    def reset(self, **kwargs):
        result = self._request({"cmd": "reset"})
        self._current_task = str(result["task"])
        return np.asarray(result["obs"], dtype=np.float32)

    def step(self, action):
        result = self._request(
            {
                "cmd": "step",
                "global_step": self._pending_global_step,
                "action": np.asarray(action, dtype=np.float32).tolist(),
            }
        )
        self._pending_global_step = None
        self._global_step += 1
        info = dict(result["info"])
        contract = control_contract(bool(result["done"]), info)
        info.update(contract.as_info())
        info["applied_action"] = np.asarray(result["applied_action"], dtype=np.float32)
        self._current_task = str(info["task"])
        return (
            np.asarray(result["obs"], dtype=np.float32),
            float(result["reward"]),
            bool(result["done"]),
            info,
        )

    def close(self):
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                self._request({"cmd": "close"})
            except (BrokenPipeError, RuntimeError):
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
        self._process = None


HalfCheetahV2Bridge = LocomotionV2Bridge
