import argparse
import json
import sys
import numpy as np
from dm_control import suite
from dm_control.utils import rewards

TOTAL_STEPS = 320000
CHANGE_FREQ = 40000
SEQUENCE = (
    "normal",
    "joint-malfunction",
    "wind",
    "velocity",
    "normal",
    "joint-malfunction",
    "wind",
    "velocity",
)
NORMAL_TARGET_SPEED = 8.0
VELOCITY_TARGET_SPEED = 32.0 / 3.0
JM_MASK = np.asarray((-1.0, -1.0, 1.0, 1.0, 1.0, 1.0), dtype=np.float64)
TORSO_WIND = np.asarray((-4.0, 0.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float64)


def _flatten(observation):
    return np.concatenate(
        [
            np.asarray(value, dtype=np.float32).reshape(-1)
            for value in observation.values()
        ],
        axis=0,
    )


class WalkerRunProtocol(object):

    def __init__(self, seed):
        self.seed = int(seed)
        self.env = suite.load(
            "walker", "run", task_kwargs={"random": self.seed}, visualize_reward=False
        )
        self.global_step = 0
        self.episode_step = 0
        self.episode_id = -1
        self._torso_id = int(self.env.physics.model.name2id("torso", "body"))
        spec = self.env.action_spec()
        self.action_min = np.asarray(spec.minimum, dtype=np.float64)
        self.action_max = np.asarray(spec.maximum, dtype=np.float64)

    @property
    def phase(self):
        index = min(self.global_step // CHANGE_FREQ, len(SEQUENCE) - 1)
        return SEQUENCE[index]

    def spec(self):
        obs_spec = self.env.observation_spec()
        return {
            "domain": "walker",
            "task": "run",
            "observation_keys": list(obs_spec.keys()),
            "observation_shape": [
                sum((int(np.prod(v.shape or (1,))) for v in obs_spec.values()))
            ],
            "action_shape": list(self.env.action_spec().shape),
            "action_min": self.action_min.tolist(),
            "action_max": self.action_max.tolist(),
            "control_timestep": float(self.env.control_timestep()),
            "time_limit_seconds": 25.0,
            "max_episode_steps": 1000,
            "seed": self.seed,
        }

    def reset(self):
        self.env.physics.data.xfrc_applied[:, :] = 0.0
        timestep = self.env.reset()
        self.episode_step = 0
        self.episode_id += 1
        return {
            "observation": _flatten(timestep.observation).tolist(),
            "info": {
                "episode_id": self.episode_id,
                "episode_step": self.episode_step,
                "global_step": self.global_step,
            },
        }

    def _benchmark_reward(self, target_speed):
        physics = self.env.physics
        standing = rewards.tolerance(
            physics.torso_height(), bounds=(1.2, float("inf")), margin=0.6
        )
        upright = (1.0 + physics.torso_upright()) / 2.0
        stand_reward = (3.0 * standing + upright) / 4.0
        move_reward = rewards.tolerance(
            physics.horizontal_velocity(),
            bounds=(target_speed, float("inf")),
            margin=target_speed / 2.0,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        return float(stand_reward * (5.0 * move_reward + 1.0) / 6.0)

    def step(self, action):
        phase = self.phase
        requested = np.asarray(action, dtype=np.float64)
        if requested.shape != self.action_min.shape:
            raise ValueError(
                "expected action shape {}, got {}".format(
                    self.action_min.shape, requested.shape
                )
            )
        if not np.all(np.isfinite(requested)):
            raise ValueError("action contains non-finite values")
        bounded = np.clip(requested, self.action_min, self.action_max)
        applied = JM_MASK * bounded if phase == "joint-malfunction" else bounded
        self.env.physics.data.xfrc_applied[:, :] = 0.0
        if phase == "wind":
            self.env.physics.data.xfrc_applied[self._torso_id, :] = TORSO_WIND
        timestep = self.env.step(applied)
        target = VELOCITY_TARGET_SPEED if phase == "velocity" else NORMAL_TARGET_SPEED
        reward = self._benchmark_reward(target)
        native_reward = float(timestep.reward)
        forward_velocity = float(self.env.physics.horizontal_velocity())
        self.global_step += 1
        self.episode_step += 1
        truncated = bool(timestep.last())
        info = {
            "task": phase,
            "actual_task": phase,
            "global_step": self.global_step,
            "episode_id": self.episode_id,
            "episode_step": self.episode_step,
            "native_reward": native_reward,
            "forward_velocity": forward_velocity,
            "target_velocity": float(target),
            "velocity_error": abs(forward_velocity - target),
            "requested_action_l1": float(np.abs(requested).sum()),
            "applied_action_l1": float(np.abs(applied).sum()),
            "applied_action_0": float(applied[0]),
            "applied_action_1": float(applied[1]),
            "terminated": False,
            "truncated": truncated,
            "discount": float(timestep.discount),
            "numerical_recovery": False,
            "numerical_recovery_count": 0,
            "numerical_error": None,
        }
        return {
            "observation": _flatten(timestep.observation).tolist(),
            "reward": reward,
            "terminated": False,
            "truncated": truncated,
            "info": info,
        }


def _reply(value):
    sys.stdout.write(
        json.dumps({"ok": True, "result": value}, separators=(",", ":")) + "\n"
    )
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    protocol = WalkerRunProtocol(args.seed)
    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            command = request["command"]
            if command == "spec":
                result = protocol.spec()
            elif command == "reset":
                result = protocol.reset()
            elif command == "step":
                result = protocol.step(request["action"])
            elif command == "set_global_step":
                value = int(request["value"])
                if value < 0 or value > TOTAL_STEPS:
                    raise ValueError("global step out of range")
                protocol.global_step = value
                result = {"global_step": value}
            elif command == "close":
                _reply({"closed": True})
                return 0
            else:
                raise ValueError("unknown command {!r}".format(command))
            _reply(result)
        except BaseException as exc:
            sys.stdout.write(
                json.dumps(
                    {"ok": False, "error_type": type(exc).__name__, "error": str(exc)},
                    separators=(",", ":"),
                )
                + "\n"
            )
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
