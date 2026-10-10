from __future__ import annotations
import copy
import math
from typing import Any, Iterable, Mapping
import torch


class PersistentPNNEnsemble:
    schema = "m3wm-s0-persistent-pnn-v1"

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        *,
        members: int = 5,
        hidden: int = 200,
        layers: int = 4,
        learning_rate: float = 0.001,
        seed: int = 0,
        device: str | torch.device = "cpu",
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.members = int(members)
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("S0 PNN device must be CPU or CUDA")
        if self.device.type == "cuda" and (not torch.cuda.is_available()):
            raise RuntimeError("CUDA S0 PNN requested but CUDA is unavailable")
        self.floor = 1e-06
        self.models = [
            self._build(index).to(self.device) for index in range(self.members)
        ]
        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            for model in self.models
        ]
        self.generators = [
            torch.Generator(device="cpu").manual_seed(self.seed + index * 104729 + 17)
            for index in range(self.members)
        ]
        self.x_mean: torch.Tensor | None = None
        self.x_std: torch.Tensor | None = None
        self.y_mean: torch.Tensor | None = None
        self.y_std: torch.Tensor | None = None
        self.fitted = False
        self.total_updates = 0

    @property
    def input_dim(self) -> int:
        return self.state_dim + self.action_dim

    @property
    def output_dim(self) -> int:
        return self.state_dim + 1

    def _build(self, member: int) -> torch.nn.Sequential:
        with torch.random.fork_rng(devices=[], enabled=True):
            init_generator = torch.Generator(device="cpu").manual_seed(
                self.seed + int(member) * 104729
            )
            torch.random.set_rng_state(init_generator.get_state())
            modules: list[torch.nn.Module] = [
                torch.nn.Linear(self.input_dim, self.hidden),
                torch.nn.SiLU(),
            ]
            for _ in range(max(0, self.layers - 1)):
                modules.extend(
                    (torch.nn.Linear(self.hidden, self.hidden), torch.nn.SiLU())
                )
            modules.append(torch.nn.Linear(self.hidden, 2 * self.output_dim))
            return torch.nn.Sequential(*modules).double()

    def _tensors(self, rows: Iterable[Mapping[str, Any]]):
        rows = list(rows)
        if not rows:
            raise ValueError("PNN training requires transition rows")
        obs = torch.stack(
            [
                torch.as_tensor(row["obs"], dtype=torch.float64, device="cpu").flatten()
                for row in rows
            ]
        )
        action = torch.stack(
            [
                torch.as_tensor(
                    row["action"], dtype=torch.float64, device="cpu"
                ).flatten()
                for row in rows
            ]
        )
        next_obs = torch.stack(
            [
                torch.as_tensor(
                    row["next_obs"], dtype=torch.float64, device="cpu"
                ).flatten()
                for row in rows
            ]
        )
        reward = torch.tensor(
            [float(row["reward"]) for row in rows], dtype=torch.float64, device="cpu"
        ).reshape(-1, 1)
        x = torch.cat((obs, action), dim=-1)
        y = torch.cat((next_obs - obs, reward), dim=-1)
        return (x, y)

    def _ensure_normalization(self, x: torch.Tensor, y: torch.Tensor) -> None:
        if self.x_mean is not None:
            return
        self.x_mean = x.mean(dim=0)
        self.x_std = x.std(dim=0, unbiased=False).clamp_min(0.0001)
        self.y_mean = y.mean(dim=0)
        self.y_std = y.std(dim=0, unbiased=False).clamp_min(0.0001)

    def _fit(self, rows, *, updates: int, batch_size: int) -> None:
        view_sampler = getattr(rows, "training_rows", None)
        readonly_view_sampler = getattr(rows, "_borrow_training_rows", None)
        archive_getter = getattr(rows, "all_unique_rows", None)
        dynamic_view = callable(view_sampler) and callable(archive_getter)
        archive_rows = archive_getter() if dynamic_view else list(rows)
        x, y = self._tensors(archive_rows)
        self._ensure_normalization(x, y)
        batch = min(int(batch_size), len(x))
        if batch <= 0 or int(updates) <= 0:
            raise ValueError("PNN fit budgets must be positive")
        for member, (model, optimizer, generator) in enumerate(
            zip(self.models, self.optimizers, self.generators)
        ):
            model.train()
            for _ in range(int(updates)):
                if dynamic_view:
                    sampler = (
                        readonly_view_sampler
                        if callable(readonly_view_sampler)
                        else view_sampler
                    )
                    batch_rows = sampler(max_rows=batch, generator=generator)
                    batch_x, batch_y = self._tensors(batch_rows)
                    batch_x = (batch_x - self.x_mean) / self.x_std
                    batch_y = (batch_y - self.y_mean) / self.y_std
                else:
                    x_norm = (x - self.x_mean) / self.x_std
                    y_norm = (y - self.y_mean) / self.y_std
                    if batch == len(x):
                        indices = torch.randperm(len(x), generator=generator)
                    else:
                        indices = torch.randint(
                            0, len(x), (batch,), generator=generator
                        )
                    batch_x = x_norm[indices]
                    batch_y = y_norm[indices]
                batch_x_device = batch_x.to(self.device)
                batch_y_device = batch_y.to(self.device)
                output = model(batch_x_device)
                mean, logvar = output.chunk(2, dim=-1)
                logvar = logvar.clamp(-8.0, 6.0)
                variance = torch.exp(logvar).clamp_min(self.floor)
                residual = batch_y_device - mean
                state_nll = (
                    residual[:, :-1].square() / variance[:, :-1] + logvar[:, :-1]
                ).mean()
                reward_nll = (
                    residual[:, -1:].square() / variance[:, -1:] + logvar[:, -1:]
                ).mean()
                loss = 0.5 * (state_nll + reward_nll)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
            model.eval()
        self.total_updates += int(updates) * self.members
        self.fitted = True

    def fit_initial(self, rows, *, updates: int = 200, batch_size: int = 256) -> None:
        if self.fitted:
            raise RuntimeError("initial PNN fit may run only once")
        self._fit(rows, updates=updates, batch_size=batch_size)

    def fit_incremental(
        self, rows, *, updates: int = 20, batch_size: int = 256
    ) -> None:
        if not self.fitted:
            raise RuntimeError("incremental PNN fit requires initial fit")
        self._fit(rows, updates=updates, batch_size=batch_size)

    @torch.no_grad()
    def predict_physical(self, obs, action):
        if not self.fitted or any(
            (
                value is None
                for value in (self.x_mean, self.x_std, self.y_mean, self.y_std)
            )
        ):
            raise RuntimeError("S0 PNN is not ready")
        obs = torch.as_tensor(obs, dtype=torch.float64, device="cpu").flatten()
        action = torch.as_tensor(action, dtype=torch.float64, device="cpu").flatten()
        x = torch.cat((obs, action)).reshape(1, -1)
        x = (x - self.x_mean) / self.x_std
        means, variances = ([], [])
        x_device = x.to(self.device)
        for model in self.models:
            mean, logvar = model(x_device).to("cpu").chunk(2, dim=-1)
            mean = mean * self.y_std + self.y_mean
            variance = (
                torch.exp(logvar.clamp(-8.0, 6.0)) * self.y_std.square()
            ).clamp_min(self.floor)
            mean = mean.clone()
            mean[:, : self.state_dim] += obs.reshape(1, -1)
            means.append(mean)
            variances.append(variance)
        return (torch.stack(means), torch.stack(variances))

    @staticmethod
    def moment_matched_moments(means, variances):
        means = torch.as_tensor(means, dtype=torch.float64)
        variances = torch.as_tensor(variances, dtype=torch.float64)
        mean = means.mean(dim=0)
        variance = (
            (means.square() + variances).mean(dim=0) - mean.square() + 1e-06
        ).clamp_min(1e-08)
        if not torch.isfinite(mean).all() or not torch.isfinite(variance).all():
            raise FloatingPointError("non-finite S0 PNN moment-matched moments")
        return (mean, variance)

    @staticmethod
    def moment_matched_log_prob(target, means, variances) -> float:
        target = torch.as_tensor(target, dtype=torch.float64).reshape(1, -1)
        mean, variance = PersistentPNNEnsemble.moment_matched_moments(means, variances)
        k = int(target.shape[-1])
        value = -0.5 * (
            k * math.log(2.0 * math.pi)
            + torch.log(variance).sum(dim=-1)
            + ((target - mean).square() / variance).sum(dim=-1)
        )
        result = float(value.reshape(-1)[0].item())
        if not math.isfinite(result):
            raise FloatingPointError("non-finite S0 PNN log probability")
        return result

    def log_prob_from_prediction(
        self, row: Mapping[str, Any], means, variances
    ) -> float:
        target = torch.cat(
            (
                torch.as_tensor(row["next_obs"], dtype=torch.float64).flatten(),
                torch.tensor([float(row["reward"])], dtype=torch.float64),
            )
        )
        return self.moment_matched_log_prob(target, means, variances)

    @staticmethod
    def disagreement_from_prediction(means, variances=None) -> float:
        del variances
        means = torch.as_tensor(means, dtype=torch.float64)
        value = torch.linalg.vector_norm(means.std(dim=0, unbiased=False), dim=-1)
        return float(value.reshape(-1)[0].item())

    @staticmethod
    def new_alternative_log_prob_from_prediction(
        means, variances, *, num_stds: float = 2.0
    ) -> float:
        _, variance = PersistentPNNEnsemble.moment_matched_moments(means, variances)
        variance = variance.reshape(-1)
        k = int(variance.numel())
        return float(
            -0.5
            * (
                k * math.log(2.0 * math.pi)
                + torch.log(variance).sum().item()
                + k * float(num_stds) ** 2
            )
        )

    @staticmethod
    def grouped_stats_from_prediction(row, means, variances) -> dict[str, float]:
        target = torch.cat(
            (
                torch.as_tensor(row["next_obs"], dtype=torch.float64).flatten(),
                torch.tensor([float(row["reward"])], dtype=torch.float64),
            )
        ).reshape(1, -1)
        mean, variance = PersistentPNNEnsemble.moment_matched_moments(means, variances)
        q = ((target - mean).square() / variance).reshape(-1)
        terms = math.log(2.0 * math.pi) + torch.log(variance).reshape(-1) + q
        return {
            "logp_joint": float((-0.5 * terms.sum()).item()),
            "logp_dynamics": float((-0.5 * terms[:-1].sum()).item()),
            "logp_reward": float((-0.5 * terms[-1]).item()),
            "q_dynamics_mean": float(q[:-1].mean().item()),
            "q_reward": float(q[-1].item()),
            "reward_residual": float(
                (target.reshape(-1)[-1] - mean.reshape(-1)[-1]).item()
            ),
            "reward_mean": float(mean.reshape(-1)[-1].item()),
            "reward_variance": float(variance.reshape(-1)[-1].item()),
            "dynamics_variance_mean": float(variance.reshape(-1)[:-1].mean().item()),
            "reward_total_variance": float(variance.reshape(-1)[-1].item()),
        }

    def log_prob(self, row: Mapping[str, Any]) -> float:
        means, variances = self.predict_physical(row["obs"], row["action"])
        return self.log_prob_from_prediction(row, means, variances)

    def disagreement(self, row: Mapping[str, Any]) -> float:
        means, variances = self.predict_physical(row["obs"], row["action"])
        return self.disagreement_from_prediction(means, variances)

    def new_alternative_log_prob(
        self, row: Mapping[str, Any], *, num_stds: float = 2.0
    ) -> float:
        means, variances = self.predict_physical(row["obs"], row["action"])
        return self.new_alternative_log_prob_from_prediction(
            means, variances, num_stds=num_stds
        )

    def state_dict(self) -> dict[str, Any]:

        def cpu_clone(value):
            if torch.is_tensor(value):
                return value.detach().to("cpu").clone()
            if isinstance(value, dict):
                return {key: cpu_clone(item) for (key, item) in value.items()}
            if isinstance(value, list):
                return [cpu_clone(item) for item in value]
            if isinstance(value, tuple):
                return tuple((cpu_clone(item) for item in value))
            return copy.deepcopy(value)

        return {
            "schema": self.schema,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "members": self.members,
                "hidden": self.hidden,
                "layers": self.layers,
                "learning_rate": self.learning_rate,
                "seed": self.seed,
            },
            "models": [cpu_clone(model.state_dict()) for model in self.models],
            "optimizers": [
                cpu_clone(optimizer.state_dict()) for optimizer in self.optimizers
            ],
            "generators": [
                generator.get_state().clone() for generator in self.generators
            ],
            "normalization": {
                "x_mean": None if self.x_mean is None else self.x_mean.clone(),
                "x_std": None if self.x_std is None else self.x_std.clone(),
                "y_mean": None if self.y_mean is None else self.y_mean.clone(),
                "y_std": None if self.y_std is None else self.y_std.clone(),
            },
            "fitted": self.fitted,
            "total_updates": self.total_updates,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != self.schema:
            raise ValueError("invalid S0 PNN state schema")
        config = state["config"]
        expected = {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "members": self.members,
            "hidden": self.hidden,
            "layers": self.layers,
            "learning_rate": self.learning_rate,
            "seed": self.seed,
        }
        if dict(config) != expected:
            raise ValueError("S0 PNN configuration mismatch")
        for model, payload in zip(self.models, state["models"]):
            model.load_state_dict(payload)
        for optimizer, payload in zip(self.optimizers, state["optimizers"]):
            optimizer.load_state_dict(payload)
            for optimizer_state in optimizer.state.values():
                for name, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[name] = value.to(self.device)
        for generator, payload in zip(self.generators, state["generators"]):
            generator.set_state(payload.clone())
        for name, value in state["normalization"].items():
            setattr(
                self, name, None if value is None else value.detach().to("cpu").clone()
            )
        self.fitted = bool(state["fitted"])
        self.total_updates = int(state["total_updates"])
