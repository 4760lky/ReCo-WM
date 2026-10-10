from __future__ import annotations
import sys
import json
import os
from pathlib import Path
import subprocess
import gymnasium as gym
import numpy as np
from common.m3wm_episode_control_contract import control_contract

PROJECT = Path(__file__).resolve().parents[1]
SERVER = PROJECT / "external_baselines" / "dmc_walker_run_server.py"
DMC_PYTHON = Path(os.environ.get("M3WM_DMC_PYTHON") or sys.executable)
EXPECTED_SEQUENCE = ("normal", "joint-malfunction", "wind", "velocity")
EXPECTED_TARGETS = {
    "normal": 8.0,
    "joint-malfunction": 8.0,
    "wind": 8.0,
    "velocity": 32.0 / 3.0,
}
EXPECTED_MASK = (-1.0, -1.0, 1.0, 1.0, 1.0, 1.0)
EXPECTED_WIND = (-4.0, 0.0, 0.0, 0.0, 0.0, 0.0)


class DMCWalkerRunBridge(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        seed: int,
        sequence: tuple[str, ...],
        change_freq: int,
        phase_lengths: tuple[int, ...] | None = None,
        target_velocities: dict[str, float],
        malfunction_mask: tuple[float, ...],
        wind_force: tuple[float, ...],
        diagnostic_revisit: bool = False,
    ) -> None:
        super().__init__()
        self._diagnostic_revisit = bool(diagnostic_revisit)
        if tuple(sequence) != EXPECTED_SEQUENCE:
            raise ValueError(f"DMC sequence drift: {sequence}")
        if int(change_freq) != 40000:
            raise ValueError(f"DMC change frequency drift: {change_freq}")
        if dict(target_velocities) != EXPECTED_TARGETS:
            raise ValueError(f"DMC target drift: {target_velocities}")
        if tuple((float(x) for x in malfunction_mask)) != EXPECTED_MASK:
            raise ValueError(f"DMC JM mask drift: {malfunction_mask}")
        if tuple((float(x) for x in wind_force)) != EXPECTED_WIND:
            raise ValueError(f"DMC wind drift: {wind_force}")
        if not DMC_PYTHON.is_file() or not SERVER.is_file():
            raise FileNotFoundError("frozen DMC runtime or server is missing")
        child_env = dict(os.environ)
        child_env.pop("PYTHONPATH", None)
        child_env["MUJOCO_GL"] = child_env.get("MUJOCO_GL", "egl")
        self._process = subprocess.Popen(
            [str(DMC_PYTHON), str(SERVER), "--seed", str(int(seed))],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=child_env,
        )
        spec = self._request("spec")
        self.env_id = "DMCWalkerRun-v0"
        self.sequence = tuple(sequence)
        self._change_freq = int(change_freq)
        self._phase_lengths = tuple(
            (int(value) for value in phase_lengths or (self._change_freq,) * 8)
        )
        self._target_velocities = dict(target_velocities)
        self._malfunction_mask = np.asarray(malfunction_mask, dtype=np.float32)
        self._wind = np.asarray(wind_force, dtype=np.float32)
        self._global_step = 0
        self._max_episode_steps = int(spec["max_episode_steps"])
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=tuple(spec["observation_shape"]),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=np.asarray(spec["action_min"], dtype=np.float32),
            high=np.asarray(spec["action_max"], dtype=np.float32),
            dtype=np.float32,
        )
        if self.observation_space.shape != (24,) or self.action_space.shape != (6,):
            self.close()
            raise RuntimeError("DMC walker-run observation/action shape drift")

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @property
    def current_task(self) -> str:
        index = min(self._global_step // self._change_freq, 7)
        return (EXPECTED_SEQUENCE + EXPECTED_SEQUENCE)[index]

    def _request(self, command: str, **payload):
        if self._process.poll() is not None:
            error = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(f"DMC server exited: {error.strip()}")
        message = {"command": command, **payload}
        assert self._process.stdin is not None
        assert self._process.stdout is not None
        self._process.stdin.write(json.dumps(message, separators=(",", ":")) + "\n")
        self._process.stdin.flush()
        raw = self._process.stdout.readline()
        if not raw:
            error = self._process.stderr.read() if self._process.stderr else ""
            raise RuntimeError(f"DMC server produced no reply: {error.strip()}")
        response = json.loads(raw)
        if not response.get("ok"):
            raise RuntimeError(
                f"DMC {response.get('error_type')}: {response.get('error')}"
            )
        return response["result"]

    def set_global_step(self, step: int) -> None:
        server_step = int(step)
        if self._diagnostic_revisit:
            if not 0 <= server_step < 18000:
                raise ValueError("DMC diagnostic is restricted to 18K counted steps")
            stage = server_step // 6000
            server_step = (40000 if stage == 1 else 0) + server_step % 6000
        result = self._request("set_global_step", value=server_step)
        self._global_step = int(result["global_step"])

    def reset(self, **kwargs):
        result = self._request("reset")
        return np.asarray(result["observation"], dtype=np.float32)

    def step(self, action):
        requested = np.asarray(action, dtype=np.float32).reshape(
            self.action_space.shape
        )
        phase = self.current_task
        bounded = np.clip(requested, self.action_space.low, self.action_space.high)
        applied = (
            self._malfunction_mask * bounded
            if phase == "joint-malfunction"
            else bounded
        )
        result = self._request("step", action=requested.tolist())
        observation = np.asarray(result["observation"], dtype=np.float32)
        reward = float(result["reward"])
        terminated = bool(result["terminated"])
        truncated = bool(result["truncated"])
        done = terminated or truncated
        info = dict(result["info"])
        self._global_step = int(info["global_step"])
        info["applied_action"] = applied.astype(np.float32)
        info.update(control_contract(done, info).as_info())
        return (observation, reward, done, info)

    def close(self) -> None:
        process = getattr(self, "_process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                self._request("close")
            except (BrokenPipeError, RuntimeError):
                pass
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                stream.close()
        self._process = None
