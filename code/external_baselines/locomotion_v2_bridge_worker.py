from __future__ import print_function
import json
import sys
import traceback
import gym
import numpy as np

SUPPORTED_ENV_IDS = ("HalfCheetah-v2", "Walker2d-v2", "Hopper-v2", "Ant-v2")
CONTROL_COST_WEIGHTS = {
    "HalfCheetah-v2": 0.1,
    "Walker2d-v2": 0.001,
    "Hopper-v2": 0.001,
    "Ant-v2": 0.5,
}


def _reply(payload):
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _read():
    line = sys.stdin.readline()
    if not line:
        raise EOFError
    return json.loads(line)


def _json_info(info):
    result = {}
    for key, value in dict(info).items():
        if isinstance(value, (str, bool, int, float)):
            result[key] = value
        elif np.isscalar(value):
            result[key] = value.item()
    return result


def _serve():
    init = _read()
    if init.get("cmd") != "init":
        raise RuntimeError("first bridge command must be init")
    env_id = str(init["env_id"])
    if env_id not in SUPPORTED_ENV_IDS:
        raise ValueError("unsupported locomotion environment: %s" % env_id)
    sequence = tuple(init["sequence"])
    change_freq = int(init["change_freq"])
    phase_lengths = tuple(
        (
            int(value)
            for value in init.get("phase_lengths", (change_freq,) * len(sequence))
        )
    )
    if not phase_lengths or any((value <= 0 for value in phase_lengths)):
        raise ValueError("phase_lengths must contain positive values")
    target_velocities = dict(init["target_velocities"])
    default_target = float(init["default_target_velocity"])
    malfunction_mask = np.asarray(init["malfunction_mask"], dtype=np.float32)
    wind = np.asarray(init["wind_force"], dtype=np.float64)
    if wind.shape != (6,):
        raise ValueError("MuJoCo external force must have shape (6,)")
    no_wind = np.zeros(6, dtype=np.float64)
    severity = {"normal": 0.0, "joint-malfunction": 0.5, "wind": 1.0, "velocity": 0.25}
    env = gym.make(env_id)
    seed = int(init["seed"])
    env.seed(seed)
    env.action_space.seed(seed)
    action_shape = tuple(env.action_space.shape)
    if action_shape != tuple(malfunction_mask.shape):
        raise ValueError(
            "%s action shape %r does not match malfunction mask %r"
            % (env_id, action_shape, malfunction_mask.shape)
        )
    global_step = 0
    _reply(
        {
            "env_id": env_id,
            "observation_low": np.asarray(env.observation_space.low).tolist(),
            "observation_high": np.asarray(env.observation_space.high).tolist(),
            "action_low": np.asarray(env.action_space.low).tolist(),
            "action_high": np.asarray(env.action_space.high).tolist(),
            "max_episode_steps": int(getattr(env, "_max_episode_steps", 1000)),
        }
    )

    def task_spec(step):
        remaining = max(0, int(step))
        phase_index = len(phase_lengths) - 1
        for index, length in enumerate(phase_lengths):
            if remaining < length:
                phase_index = index
                break
            remaining -= length
        task = sequence[phase_index % len(sequence)]
        target = target_velocities.get(task, default_target)
        if isinstance(target, (list, tuple)):
            occurrence = sum((1 for value in sequence[:phase_index] if value == task))
            target = target[min(occurrence, len(target) - 1)]
        return (task, float(target))

    try:
        while True:
            command = _read()
            kind = command.get("cmd")
            if kind == "reset":
                obs = env.reset()
                task, target = task_spec(global_step)
                _reply(
                    {
                        "obs": np.asarray(obs, dtype=np.float32).tolist(),
                        "task": task,
                        "target_velocity": target,
                    }
                )
            elif kind == "step":
                if command.get("global_step") is not None:
                    global_step = int(command["global_step"])
                task, target = task_spec(global_step)
                action = np.asarray(command["action"], dtype=np.float32)
                if tuple(action.shape) != action_shape:
                    raise ValueError(
                        "unexpected %s action shape %r" % (env_id, action.shape)
                    )
                applied_action = (
                    malfunction_mask * action if task == "joint-malfunction" else action
                )
                force = wind if task == "wind" else no_wind
                for body_id in env.unwrapped.sim.model._body_name2id.values():
                    env.unwrapped.sim.data.xfrc_applied[body_id, :] = force
                pos_before = float(env.unwrapped.sim.data.qpos[0])
                obs, native_reward, done, info = env.step(applied_action)
                pos_after = float(env.unwrapped.sim.data.qpos[0])
                forward_velocity = (pos_after - pos_before) / float(env.unwrapped.dt)
                velocity_error = abs(forward_velocity - target)
                reward = float(native_reward) - forward_velocity - velocity_error
                if not np.isfinite(reward):
                    raise FloatingPointError("non-finite target-tracking reward")
                result_info = _json_info(info)
                terminated = bool(
                    done and (not result_info.get("TimeLimit.truncated", False))
                )
                truncated = bool(done and result_info.get("TimeLimit.truncated", False))
                control_cost = CONTROL_COST_WEIGHTS[env_id] * float(
                    np.square(applied_action).sum()
                )
                result_info.update(
                    {
                        "env_id": env_id,
                        "task": task,
                        "actual_task": task,
                        "forward_velocity": forward_velocity,
                        "target_velocity": target,
                        "velocity_error": velocity_error,
                        "control_cost": control_cost,
                        "tracking_reward": -velocity_error,
                        "native_reward": float(native_reward),
                        "nonstationary_severity": severity.get(task, 0.0),
                        "terminated": terminated,
                        "truncated": truncated,
                        "bootstrap_allowed": not terminated,
                        "terminal_observation_is_post_step": True,
                        "success": 0.0,
                    }
                )
                global_step += 1
                _reply(
                    {
                        "obs": np.asarray(obs, dtype=np.float32).tolist(),
                        "reward": reward,
                        "done": bool(done),
                        "info": result_info,
                        "applied_action": applied_action.tolist(),
                    }
                )
            elif kind == "close":
                _reply({"closed": True})
                break
            else:
                raise ValueError("unknown bridge command: %r" % (kind,))
    except EOFError:
        pass
    finally:
        env.close()


def main():
    try:
        _serve()
    except Exception as exc:
        try:
            _reply(
                {
                    "bridge_error": "%s: %s" % (type(exc).__name__, exc),
                    "traceback": traceback.format_exc(),
                }
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
