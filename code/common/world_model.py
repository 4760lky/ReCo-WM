from copy import deepcopy
import torch
import torch.nn as nn
from common import layers, math, init
from common.m3wm_modules import (
    ContextRouter,
    ContextConditionedModularResidualMLP,
    ContextVectorMLP,
    HardSwitchMLPBank,
    ModularResidualMLP,
)
from tensordict import TensorDict
from tensordict.nn import TensorDictParams


class WorldModel(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = layers.enc(cfg)
        model_input_dim = cfg.latent_dim + cfg.action_dim + cfg.task_dim
        reward_output_dim = max(cfg.num_bins, 1)
        self.enable_m3wm = bool(cfg.get("enable_m3wm", False))
        self.m3wm_composition_mode = str(
            cfg.get("m3wm_composition_mode", "modular")
        ).lower()
        valid_composition_modes = {
            "modular",
            "hybrid_modular",
            "context_vector",
            "hard_switching",
        }
        if self.m3wm_composition_mode not in valid_composition_modes:
            raise ValueError(
                f"m3wm_composition_mode must be one of {sorted(valid_composition_modes)}, received {self.m3wm_composition_mode!r}"
            )
        self.m3wm_num_primitives = int(cfg.get("m3wm_num_primitives", 4))
        self.m3wm_separate_composition_heads = False
        self.m3wm_num_composition_branches = (
            3 if self.m3wm_separate_composition_heads else 1
        )
        self.m3wm_theta_dim = (
            self.m3wm_num_primitives * self.m3wm_num_composition_branches
        )
        self.m3wm_primitive_hidden_dim = int(
            cfg.get("m3wm_primitive_hidden_dim", max(cfg.mlp_dim // 4, 64))
        )
        self.m3wm_residual_scale = float(cfg.get("m3wm_residual_scale", 0.1))
        raw_reward_residual_scale = cfg.get("m3wm_reward_residual_scale", None)
        self.m3wm_reward_residual_scale = (
            self.m3wm_residual_scale
            if raw_reward_residual_scale is None
            else float(raw_reward_residual_scale)
        )
        self.m3wm_modular_value = (
            bool(cfg.get("m3wm_modular_value", False))
            and self.enable_m3wm
            and (self.m3wm_composition_mode in {"modular", "hybrid_modular"})
        )
        if self.m3wm_separate_composition_heads and (
            self.m3wm_composition_mode not in {"modular", "hybrid_modular"}
            or not self.m3wm_modular_value
        ):
            raise ValueError(
                "Separate dyn/rew/val compositions require modular dynamics, reward, and value heads."
            )
        self.m3wm_value_uses_theta = self.m3wm_modular_value or (
            self.enable_m3wm
            and self.m3wm_composition_mode in {"context_vector", "hard_switching"}
        )
        self.m3wm_theta_temperature_start = float(
            cfg.get(
                "m3wm_theta_temperature_start", cfg.get("m3wm_theta_temperature", 1.0)
            )
        )
        self.m3wm_theta_temperature_end = float(
            cfg.get("m3wm_theta_temperature_end", 0.5)
        )
        self.m3wm_primitive_init_std = float(cfg.get("m3wm_primitive_init_std", 0.001))
        self.m3wm_context_window = int(cfg.get("m3wm_context_window", 8))
        self.m3wm_context_projection_dim = int(
            cfg.get("m3wm_context_projection_dim", 128)
        )
        self.m3wm_context_hidden_dim = int(cfg.get("m3wm_context_hidden_dim", 128))
        self.m3wm_context_embedding_dim = int(cfg.get("m3wm_context_embedding_dim", 16))
        self.m3wm_context_embedding_scale = float(
            cfg.get("m3wm_context_embedding_scale", 2.0)
        )
        if not 1 <= self.m3wm_context_embedding_dim <= self.m3wm_context_hidden_dim:
            raise ValueError(
                "m3wm_context_embedding_dim must be in [1, m3wm_context_hidden_dim]"
            )
        if self.m3wm_context_embedding_scale <= 0:
            raise ValueError("m3wm_context_embedding_scale must be > 0")
        self.m3wm_context_feature_mode = str(
            cfg.get("m3wm_context_feature_mode", "anchor_residual")
        ).lower()
        valid_context_modes = {
            "legacy",
            "residual",
            "anchor_residual",
            "composed_residual",
        }
        if self.m3wm_context_feature_mode not in valid_context_modes:
            raise ValueError(
                f"m3wm_context_feature_mode must be one of {sorted(valid_context_modes)}, received {self.m3wm_context_feature_mode!r}"
            )
        self.m3wm_context_short_window = int(cfg.get("m3wm_context_short_window", 8))
        self.m3wm_context_mid_window = int(cfg.get("m3wm_context_mid_window", 16))
        self.m3wm_context_long_window = int(
            cfg.get("m3wm_context_long_window", self.m3wm_context_window)
        )
        self.m3wm_multiscale_context = True
        self.m3wm_explicit_three_scale_context = True
        self.m3wm_context_stream_mode = str(
            cfg.get("m3wm_context_stream_mode", "shared")
        ).lower()
        valid_context_stream_modes = {"shared", "architecture_partitioned"}
        if self.m3wm_context_stream_mode not in valid_context_stream_modes:
            raise ValueError(
                f"m3wm_context_stream_mode must be one of {sorted(valid_context_stream_modes)}, received {self.m3wm_context_stream_mode!r}"
            )
        self.m3wm_use_reward_residual = True
        self.m3wm_use_action_gain_feature = False
        self.m3wm_context_residual_transform = str(
            cfg.get("m3wm_context_residual_transform", "abs_action_norm")
        ).lower()
        valid_residual_transforms = {
            "signed_action",
            "abs_action_norm",
            "stats_group_rms",
            "architecture",
        }
        if self.m3wm_context_residual_transform not in valid_residual_transforms:
            raise ValueError(
                f"m3wm_context_residual_transform must be one of {sorted(valid_residual_transforms)}, received {self.m3wm_context_residual_transform!r}"
            )
        self.m3wm_context_group_count = int(cfg.get("m3wm_context_group_count", 16))
        if self.m3wm_context_group_count < 1:
            raise ValueError("m3wm_context_group_count must be >= 1")
        if self.m3wm_context_group_count > int(cfg.latent_dim):
            raise ValueError("m3wm_context_group_count must not exceed latent_dim")
        self.m3wm_use_context_anchor = False
        if self.enable_m3wm:
            if self.m3wm_context_window < 1:
                raise ValueError("m3wm_context_window must be >= 1")
            if self.m3wm_context_short_window < 1:
                raise ValueError("m3wm_context_short_window must be >= 1")
            if self.m3wm_context_short_window > self.m3wm_context_long_window:
                raise ValueError("context short window must not exceed long window")
            if self.m3wm_explicit_three_scale_context and (
                not self.m3wm_context_short_window
                <= self.m3wm_context_mid_window
                <= self.m3wm_context_long_window
            ):
                raise ValueError("Require context short <= mid <= long window")
            if self.m3wm_context_long_window > self.m3wm_context_window:
                raise ValueError(
                    "m3wm_context_window must cover the long context window"
                )
            if self.m3wm_context_stream_mode == "architecture_partitioned":
                if self.m3wm_context_residual_transform != "architecture":
                    raise ValueError(
                        "architecture_partitioned context requires m3wm_context_residual_transform=architecture"
                    )
                if not (
                    self.m3wm_multiscale_context
                    and self.m3wm_explicit_three_scale_context
                ):
                    raise ValueError(
                        "architecture_partitioned context requires both m3wm_multiscale_context and m3wm_explicit_three_scale_context"
                    )
            if self.m3wm_composition_mode == "modular":
                self._dynamics = ModularResidualMLP(
                    input_dim=model_input_dim,
                    output_dim=cfg.latent_dim,
                    base_hidden_dim=cfg.mlp_dim,
                    primitive_hidden_dim=self.m3wm_primitive_hidden_dim,
                    num_primitives=self.m3wm_num_primitives,
                    residual_scale=self.m3wm_residual_scale,
                    base_output_activation=layers.SimNorm(cfg),
                    output_simplex_group_size=cfg.simnorm_dim,
                )
                self._reward = ModularResidualMLP(
                    input_dim=model_input_dim,
                    output_dim=reward_output_dim,
                    base_hidden_dim=cfg.mlp_dim,
                    primitive_hidden_dim=self.m3wm_primitive_hidden_dim,
                    num_primitives=self.m3wm_num_primitives,
                    residual_scale=self.m3wm_reward_residual_scale,
                )
            elif self.m3wm_composition_mode == "hybrid_modular":
                self._dynamics = ContextConditionedModularResidualMLP(
                    input_dim=model_input_dim,
                    output_dim=cfg.latent_dim,
                    base_hidden_dim=cfg.mlp_dim,
                    primitive_hidden_dim=self.m3wm_primitive_hidden_dim,
                    num_primitives=self.m3wm_num_primitives,
                    residual_scale=self.m3wm_residual_scale,
                    base_output_activation=layers.SimNorm(cfg),
                    output_simplex_group_size=cfg.simnorm_dim,
                )
                self._reward = ContextConditionedModularResidualMLP(
                    input_dim=model_input_dim,
                    output_dim=reward_output_dim,
                    base_hidden_dim=cfg.mlp_dim,
                    primitive_hidden_dim=self.m3wm_primitive_hidden_dim,
                    num_primitives=self.m3wm_num_primitives,
                    residual_scale=self.m3wm_reward_residual_scale,
                )
            elif self.m3wm_composition_mode == "context_vector":
                self._dynamics = ContextVectorMLP(
                    input_dim=model_input_dim,
                    context_dim=self.m3wm_num_primitives,
                    output_dim=cfg.latent_dim,
                    hidden_dim=cfg.mlp_dim,
                    output_activation=layers.SimNorm(cfg),
                )
                self._reward = ContextVectorMLP(
                    input_dim=model_input_dim,
                    context_dim=self.m3wm_num_primitives,
                    output_dim=reward_output_dim,
                    hidden_dim=cfg.mlp_dim,
                )
            else:
                self._dynamics = HardSwitchMLPBank(
                    input_dim=model_input_dim,
                    output_dim=cfg.latent_dim,
                    hidden_dim=cfg.mlp_dim,
                    num_experts=self.m3wm_num_primitives,
                    output_activation=layers.SimNorm(cfg),
                )
                self._reward = HardSwitchMLPBank(
                    input_dim=model_input_dim,
                    output_dim=reward_output_dim,
                    hidden_dim=cfg.mlp_dim,
                    num_experts=self.m3wm_num_primitives,
                )
            if self.m3wm_context_feature_mode == "legacy":
                context_feature_dim = 2 * cfg.latent_dim + cfg.action_dim + 1
            elif self.m3wm_context_residual_transform == "signed_action":
                context_feature_dim = (
                    cfg.action_dim
                    + cfg.latent_dim
                    + 1
                    + int(self.m3wm_use_reward_residual)
                )
            elif self.m3wm_context_residual_transform == "abs_action_norm":
                context_feature_dim = (
                    cfg.latent_dim + 2 + int(self.m3wm_use_reward_residual)
                )
            elif self.m3wm_context_residual_transform == "architecture":
                context_feature_dim = (
                    2 * cfg.latent_dim
                    + cfg.action_dim
                    + 2
                    + int(self.m3wm_use_reward_residual)
                )
            else:
                context_feature_dim = (
                    8
                    + self.m3wm_context_group_count
                    + 1
                    + 3 * int(self.m3wm_use_action_gain_feature)
                    + int(self.m3wm_use_reward_residual)
                )
            self.m3wm_context_feature_dim = int(context_feature_dim)
            self._m3wm_context = ContextRouter(
                input_dim=context_feature_dim,
                projection_dim=self.m3wm_context_projection_dim,
                hidden_dim=self.m3wm_context_hidden_dim,
                num_primitives=self.m3wm_num_primitives,
                temperature=self.m3wm_theta_temperature_start,
                num_branches=self.m3wm_num_composition_branches,
                short_window=self.m3wm_context_short_window,
                mid_window=self.m3wm_context_mid_window,
                long_window=self.m3wm_context_long_window,
                use_multiscale=self.m3wm_multiscale_context,
                explicit_three_scale=self.m3wm_explicit_three_scale_context,
                stream_mode=self.m3wm_context_stream_mode,
                architecture_latent_dim=int(cfg.latent_dim),
                architecture_action_dim=int(cfg.action_dim),
                architecture_include_reward_residual=self.m3wm_use_reward_residual,
            )
            self.register_buffer(
                "_m3wm_uniform_theta",
                torch.full(
                    (
                        (self.m3wm_num_composition_branches, self.m3wm_num_primitives)
                        if self.m3wm_separate_composition_heads
                        else (self.m3wm_num_primitives,)
                    ),
                    1.0 / self.m3wm_num_primitives,
                ),
            )
        else:
            self._dynamics = layers.mlp(
                model_input_dim,
                2 * [cfg.mlp_dim],
                cfg.latent_dim,
                act=layers.SimNorm(cfg),
            )
            self._reward = layers.mlp(
                model_input_dim, 2 * [cfg.mlp_dim], reward_output_dim
            )
        self._termination = (
            layers.mlp(cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 1)
            if cfg.episodic
            else None
        )
        self._pi = layers.mlp(
            cfg.latent_dim + cfg.task_dim, 2 * [cfg.mlp_dim], 2 * cfg.action_dim
        )
        if self.m3wm_modular_value:
            q_modules = []
            for _ in range(cfg.num_q):
                q_head_cls = (
                    ContextConditionedModularResidualMLP
                    if self.m3wm_composition_mode == "hybrid_modular"
                    else ModularResidualMLP
                )
                q_module = q_head_cls(
                    input_dim=model_input_dim,
                    output_dim=reward_output_dim,
                    base_hidden_dim=cfg.mlp_dim,
                    primitive_hidden_dim=self.m3wm_primitive_hidden_dim,
                    num_primitives=self.m3wm_num_primitives,
                    residual_scale=float(
                        cfg.get("m3wm_value_residual_scale", self.m3wm_residual_scale)
                    ),
                    base_dropout=cfg.dropout,
                ).apply(init.weight_init)
                init.zero_([q_module.base_output_layer.weight])
                q_module.initialize_residual_outputs(self.m3wm_primitive_init_std)
                q_modules.append(q_module)
        elif self.enable_m3wm and self.m3wm_composition_mode == "context_vector":
            q_modules = [
                ContextVectorMLP(
                    input_dim=model_input_dim,
                    context_dim=self.m3wm_num_primitives,
                    output_dim=reward_output_dim,
                    hidden_dim=cfg.mlp_dim,
                    dropout=cfg.dropout,
                ).apply(init.weight_init)
                for _ in range(cfg.num_q)
            ]
            for q_module in q_modules:
                init.zero_([q_module.output_layer.weight])
        elif self.enable_m3wm and self.m3wm_composition_mode == "hard_switching":
            q_modules = [
                HardSwitchMLPBank(
                    input_dim=model_input_dim,
                    output_dim=reward_output_dim,
                    hidden_dim=cfg.mlp_dim,
                    num_experts=self.m3wm_num_primitives,
                    dropout=cfg.dropout,
                ).apply(init.weight_init)
                for _ in range(cfg.num_q)
            ]
            for q_module in q_modules:
                init.zero_([layer.weight for layer in q_module.output_layers])
        else:
            q_modules = [
                layers.mlp(
                    model_input_dim,
                    2 * [cfg.mlp_dim],
                    reward_output_dim,
                    dropout=cfg.dropout,
                ).apply(init.weight_init)
                for _ in range(cfg.num_q)
            ]
        self._Qs = layers.Ensemble(q_modules)
        self.apply(init.weight_init)
        if self.enable_m3wm and self.m3wm_composition_mode in {
            "modular",
            "hybrid_modular",
        }:
            self._dynamics.initialize_residual_outputs(self.m3wm_primitive_init_std)
            self._reward.initialize_residual_outputs(self.m3wm_primitive_init_std)
            reward_output_weight = self._reward.base_output_layer.weight
        elif self.enable_m3wm and self.m3wm_composition_mode == "context_vector":
            reward_output_weight = self._reward.output_layer.weight
        elif self.enable_m3wm:
            init.zero_([layer.weight for layer in self._reward.output_layers])
            reward_output_weight = None
        else:
            reward_output_weight = self._reward[-1].weight
        if reward_output_weight is not None:
            init.zero_([reward_output_weight])
        if not self.m3wm_value_uses_theta:
            init.zero_([self._Qs.params["2", "weight"]])
        self._m3wm_anchor_encoder = None
        self._m3wm_anchor_dynamics_base = None
        self._m3wm_anchor_reward_base = None
        self._m3wm_anchor_task_emb = None
        if self.enable_m3wm and self.m3wm_use_context_anchor:
            self._m3wm_anchor_encoder = deepcopy(self._encoder)
            self._m3wm_anchor_dynamics_base = deepcopy(
                self._dynamics.base
                if self.m3wm_composition_mode in {"modular", "hybrid_modular"}
                else self._dynamics
            )
            if self.m3wm_use_reward_residual:
                self._m3wm_anchor_reward_base = deepcopy(
                    self._reward.base
                    if self.m3wm_composition_mode in {"modular", "hybrid_modular"}
                    else self._reward
                )
            self.register_buffer(
                "_m3wm_context_anchor_ready", torch.tensor(False, dtype=torch.bool)
            )
            self.register_buffer(
                "_m3wm_context_anchor_version", torch.tensor(0, dtype=torch.long)
            )
            self._freeze_context_anchor_modules()
        self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
        self.register_buffer(
            "log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min
        )
        self._m3wm_runtime_theta = None
        self.init()

    def init(self):
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(
            self._Qs.params.data.clone(), no_convert=True
        )
        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)
        delattr(self._detach_Qs, "params")
        self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        delattr(self._target_Qs, "params")
        self._target_Qs.__dict__["params"] = self._target_Qs_params

    def _context_anchor_modules(self):
        return [
            module
            for module in (
                self._m3wm_anchor_encoder,
                self._m3wm_anchor_dynamics_base,
                self._m3wm_anchor_reward_base,
                self._m3wm_anchor_task_emb,
            )
            if module is not None
        ]

    def _freeze_context_anchor_modules(self):
        for module in self._context_anchor_modules():
            module.requires_grad_(False)
            module.eval()

    @property
    def context_anchor_ready(self):
        if not self.m3wm_use_context_anchor:
            return False
        return bool(self._m3wm_context_anchor_ready.item())

    @torch.no_grad()
    def refresh_context_anchor(self):
        if not self.m3wm_use_context_anchor:
            return False
        self._m3wm_anchor_encoder.load_state_dict(self._encoder.state_dict())
        dynamics_source = (
            self._dynamics.base
            if self.m3wm_composition_mode in {"modular", "hybrid_modular"}
            else self._dynamics
        )
        self._m3wm_anchor_dynamics_base.load_state_dict(dynamics_source.state_dict())
        if self._m3wm_anchor_reward_base is not None:
            reward_source = (
                self._reward.base
                if self.m3wm_composition_mode in {"modular", "hybrid_modular"}
                else self._reward
            )
            self._m3wm_anchor_reward_base.load_state_dict(reward_source.state_dict())
        if self._m3wm_anchor_task_emb is not None:
            self._m3wm_anchor_task_emb.load_state_dict(self._task_emb.state_dict())
        self._m3wm_context_anchor_ready.fill_(True)
        self._m3wm_context_anchor_version.add_(1)
        self._freeze_context_anchor_modules()
        return True

    def __repr__(self):
        representation = "TD-MPC2 World Model\n"
        modules = [
            "Encoder",
            "Dynamics",
            "Reward",
            "Termination",
            "Policy prior",
            "Q-functions",
        ]
        for i, module in enumerate(
            [
                self._encoder,
                self._dynamics,
                self._reward,
                self._termination,
                self._pi,
                self._Qs,
            ]
        ):
            if module == self._termination and (not self.cfg.episodic):
                continue
            representation += f"{modules[i]}: {module}\n"
        if self.enable_m3wm:
            architecture_label = (
                "ReCo-WM figure-exact"
                if self.m3wm_composition_mode == "modular"
                and self.m3wm_separate_composition_heads
                and (self.m3wm_context_feature_mode == "composed_residual")
                and (self.m3wm_context_residual_transform == "architecture")
                and self.m3wm_explicit_three_scale_context
                and (self.m3wm_context_stream_mode == "architecture_partitioned")
                else "ReCo-WM"
            )
            representation += f"{architecture_label}: enabled (K={self.m3wm_num_primitives}, composition={self.m3wm_composition_mode}, separate_heads={self.m3wm_separate_composition_heads}, context_mode={self.m3wm_context_feature_mode}, residual_transform={self.m3wm_context_residual_transform}, context_groups={self.m3wm_context_group_count}, anchor={self.m3wm_use_context_anchor}, reward_residual={self.m3wm_use_reward_residual}, context_window={self.m3wm_context_window}, short/mid/long={self.m3wm_context_short_window}/{self.m3wm_context_mid_window}/{self.m3wm_context_long_window}, multiscale={self.m3wm_multiscale_context}, context_streams={self.m3wm_context_stream_mode}, stream_dims={self._m3wm_context.stream_dims}, context_hidden={self.m3wm_context_hidden_dim}, modular_value={self.m3wm_modular_value}, residual_scale={self.m3wm_residual_scale}, primitive_init_std={self.m3wm_primitive_init_std})\n"
            representation += f"Context router: {self._m3wm_context}\n"
        representation += f"Learnable parameters: {self.total_params:,}"
        return representation

    @property
    def total_params(self):
        return sum(
            (
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )
        )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode=True):
        super().train(mode)
        self._target_Qs.train(False)
        self._freeze_context_anchor_modules()
        return self

    def soft_update_target_Q(self):
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.cfg.tau)

    @staticmethod
    def _task_emb_with_module(x, task, embedding_module):
        if isinstance(task, int):
            task = torch.tensor([task], device=x.device)
        embedding = embedding_module(task.long())
        if x.ndim == 3:
            embedding = embedding.unsqueeze(0).repeat(x.shape[0], 1, 1)
        elif embedding.shape[0] == 1:
            embedding = embedding.repeat(x.shape[0], 1)
        return torch.cat([x, embedding], dim=-1)

    def task_emb(self, x, task):
        return self._task_emb_with_module(x, task, self._task_emb)

    def anchor_task_emb(self, x, task):
        if self._m3wm_anchor_task_emb is None:
            return self.task_emb(x, task)
        return self._task_emb_with_module(x, task, self._m3wm_anchor_task_emb)

    def uniform_theta(self, reference):
        if not self.enable_m3wm:
            raise RuntimeError("uniform_theta() called while ReCo-WM is disabled.")
        theta = self._m3wm_uniform_theta.to(
            device=reference.device, dtype=reference.dtype
        )
        if self.m3wm_separate_composition_heads:
            return theta.expand(
                *reference.shape[:-1],
                self.m3wm_num_composition_branches,
                self.m3wm_num_primitives,
            )
        return theta.expand(*reference.shape[:-1], self.m3wm_num_primitives)

    def set_runtime_theta(self, theta):
        if not self.enable_m3wm:
            return
        if theta is None:
            self._m3wm_runtime_theta = None
            return
        valid_shape = theta.shape[-1] == self.m3wm_num_primitives and (
            not self.m3wm_separate_composition_heads
            or (
                theta.ndim >= 2
                and theta.shape[-2] == self.m3wm_num_composition_branches
            )
        )
        if not valid_shape:
            raise ValueError("Invalid runtime theta shape.")
        self._m3wm_runtime_theta = theta.detach()

    def _resolve_theta(self, reference, theta=None, branch="dynamics"):
        if theta is None:
            theta = self._m3wm_runtime_theta
        if theta is None:
            theta = self.uniform_theta(reference)
        if self.m3wm_separate_composition_heads:
            branch_index = {"dynamics": 0, "reward": 1, "value": 2}[branch]
            if theta.shape[-2:] != (
                self.m3wm_num_composition_branches,
                self.m3wm_num_primitives,
            ):
                raise ValueError(
                    f"Separate composition theta must end in [3,K], received {tuple(theta.shape)}"
                )
            theta = theta[..., branch_index, :]
        theta = theta.to(device=reference.device, dtype=reference.dtype)
        while theta.ndim < reference.ndim:
            theta = theta.unsqueeze(0)
        target_shape = (*reference.shape[:-1], self.m3wm_num_primitives)
        return theta.expand(target_shape)

    def _base_model_input(self, z, action, task):
        model_z = z
        return torch.cat([model_z, action.to(dtype=model_z.dtype)], dim=-1)

    def base_next(self, z, action, task=None):
        model_input = self._base_model_input(z, action, task)
        if self.enable_m3wm:
            if self.m3wm_composition_mode == "modular":
                return self._dynamics.base(model_input)
            theta = self.uniform_theta(model_input)
            if self.m3wm_composition_mode == "hybrid_modular":
                return self._dynamics.base(model_input, theta)
            return self._dynamics(model_input, theta)
        return self._dynamics(model_input)

    def base_reward(self, z, action, task=None):
        model_input = self._base_model_input(z, action, task)
        if self.enable_m3wm:
            if self.m3wm_composition_mode == "modular":
                logits = self._reward.base(model_input)
            else:
                theta = self.uniform_theta(model_input)
                if self.m3wm_composition_mode == "hybrid_modular":
                    logits = self._reward.base(model_input, theta)
                else:
                    logits = self._reward(model_input, theta)
        else:
            logits = self._reward(model_input)
        return math.two_hot_inv(logits, self.cfg)

    def anchor_encode(self, obs, task=None):
        if not self.m3wm_use_context_anchor or not self.context_anchor_ready:
            return self.encode(obs, task)
        if self.cfg.obs == "rgb" and obs.ndim == 5:
            return torch.stack(
                [
                    self._m3wm_anchor_encoder[self.cfg.obs](observation)
                    for observation in obs
                ]
            )
        return self._m3wm_anchor_encoder[self.cfg.obs](obs)

    def _anchor_base_model_input(self, z, action, task):
        model_z = z
        return torch.cat([model_z, action.to(dtype=model_z.dtype)], dim=-1)

    def anchor_base_next(self, z, action, task=None):
        if not self.m3wm_use_context_anchor or not self.context_anchor_ready:
            return self.base_next(z, action, task)
        model_input = self._anchor_base_model_input(z, action, task)
        if self.m3wm_composition_mode == "modular":
            return self._m3wm_anchor_dynamics_base(model_input)
        return self._m3wm_anchor_dynamics_base(
            model_input, self.uniform_theta(model_input)
        )

    def anchor_base_reward(self, z, action, task=None):
        if (
            not self.m3wm_use_context_anchor
            or not self.context_anchor_ready
            or self._m3wm_anchor_reward_base is None
        ):
            return self.base_reward(z, action, task)
        model_input = self._anchor_base_model_input(z, action, task)
        if self.m3wm_composition_mode == "modular":
            logits = self._m3wm_anchor_reward_base(model_input)
        else:
            logits = self._m3wm_anchor_reward_base(
                model_input, self.uniform_theta(model_input)
            )
        return math.two_hot_inv(logits, self.cfg)

    def build_context_features(
        self, z, action, reward, next_z, task=None, use_anchor_base=False, theta=None
    ):
        if reward.ndim == z.ndim - 1:
            reward = reward.unsqueeze(-1)
        action = action.to(dtype=z.dtype)
        reward = reward.to(dtype=z.dtype)
        if self.m3wm_context_feature_mode == "legacy":
            delta_z = next_z - z
            return torch.cat([z, action, reward, delta_z], dim=-1)
        if self.m3wm_context_feature_mode == "composed_residual":
            predicted_next_z = self.next(z, action, task, theta=theta)
        else:
            predicted_next_z = (
                self.anchor_base_next(z, action, task)
                if use_anchor_base
                else self.base_next(z, action, task)
            )
        latent_residual = next_z - predicted_next_z
        residual_rms = (
            latent_residual.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
        )
        if self.m3wm_context_residual_transform == "architecture":
            features = [latent_residual, residual_rms, z, action, reward]
        elif self.m3wm_context_residual_transform == "abs_action_norm":
            action_norm = (
                action.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
            )
            features = [latent_residual.abs(), residual_rms, action_norm]
        elif self.m3wm_context_residual_transform == "stats_group_rms":
            abs_residual = latent_residual.abs()
            mean_abs = abs_residual.mean(dim=-1, keepdim=True)
            std_abs = abs_residual.std(dim=-1, keepdim=True, unbiased=False)
            max_abs = abs_residual.amax(dim=-1, keepdim=True)
            quantiles = [
                torch.quantile(abs_residual, q, dim=-1, keepdim=True)
                for q in (0.5, 0.75, 0.9, 0.95)
            ]
            residual_groups = torch.tensor_split(
                latent_residual, self.m3wm_context_group_count, dim=-1
            )
            group_rms = torch.cat(
                [
                    group.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
                    for group in residual_groups
                ],
                dim=-1,
            )
            action_norm = (
                action.pow(2).sum(dim=-1, keepdim=True).clamp_min(1e-12).sqrt()
            )
            features = [
                mean_abs,
                std_abs,
                residual_rms,
                max_abs,
                *quantiles,
                group_rms,
                action_norm,
            ]
            if self.m3wm_use_action_gain_feature:
                zero_action = torch.zeros_like(action)
                if self.m3wm_context_feature_mode == "composed_residual":
                    zero_next_z = self.next(z, zero_action, task, theta=theta)
                else:
                    zero_next_z = (
                        self.anchor_base_next(z, zero_action, task)
                        if use_anchor_base
                        else self.base_next(z, zero_action, task)
                    )
                predicted_action_effect = predicted_next_z - zero_next_z
                observed_action_effect = next_z - zero_next_z
                effect_energy = (
                    predicted_action_effect.square()
                    .sum(dim=-1, keepdim=True)
                    .clamp_min(1e-06)
                )
                action_gain = (observed_action_effect * predicted_action_effect).sum(
                    dim=-1, keepdim=True
                ) / effect_energy
                action_gain = action_gain.clamp(-1.0, 2.0)
                action_fit_residual = (
                    observed_action_effect - action_gain * predicted_action_effect
                )
                action_fit_rms = (
                    action_fit_residual.square()
                    .mean(dim=-1, keepdim=True)
                    .clamp_min(1e-12)
                    .sqrt()
                )
                features.extend([action_gain, action_fit_rms, effect_energy])
        else:
            features = [action, latent_residual, residual_rms]
        if self.m3wm_use_reward_residual:
            if self.m3wm_context_feature_mode == "composed_residual":
                predicted_reward = math.two_hot_inv(
                    self.reward(z, action, task, theta=theta), self.cfg
                )
            else:
                predicted_reward = (
                    self.anchor_base_reward(z, action, task)
                    if use_anchor_base
                    else self.base_reward(z, action, task)
                )
            if predicted_reward.ndim == reward.ndim - 1:
                predicted_reward = predicted_reward.unsqueeze(-1)
            reward_residual = reward - predicted_reward.to(dtype=reward.dtype)
            features.append(reward_residual)
        return torch.cat(features, dim=-1)

    def build_context_features_from_observations(
        self, obs, action, reward, next_obs, task=None, theta=None
    ):
        use_anchor = (
            self.m3wm_context_feature_mode == "anchor_residual"
            and self.m3wm_use_context_anchor
            and self.context_anchor_ready
        )
        if use_anchor:
            z = self.anchor_encode(obs, task)
            next_z = self.anchor_encode(next_obs, task)
        else:
            z = self.encode(obs, task)
            next_z = self.encode(next_obs, task)
        return self.build_context_features(
            z,
            action,
            reward,
            next_z,
            task=task,
            use_anchor_base=use_anchor,
            theta=theta,
        )

    def context_feature_metric_indices(self):
        if self.m3wm_context_feature_mode == "legacy":
            return {}
        if self.m3wm_context_residual_transform == "architecture":
            residual_rms = int(self.cfg.latent_dim)
            action_norm = None
            reward_residual = (
                2 * int(self.cfg.latent_dim) + int(self.cfg.action_dim) + 2
            )
            extra = {}
        elif self.m3wm_context_residual_transform == "abs_action_norm":
            residual_rms = int(self.cfg.latent_dim)
            action_norm = residual_rms + 1
            reward_residual = action_norm + 1
            extra = {}
        elif self.m3wm_context_residual_transform == "stats_group_rms":
            residual_rms = 2
            action_norm = 8 + self.m3wm_context_group_count
            action_gain = action_norm + 1 if self.m3wm_use_action_gain_feature else None
            action_gain_fit = (
                action_norm + 2 if self.m3wm_use_action_gain_feature else None
            )
            action_effect_energy = (
                action_norm + 3 if self.m3wm_use_action_gain_feature else None
            )
            reward_residual = (
                action_norm + 1 + 3 * int(self.m3wm_use_action_gain_feature)
            )
            extra = {
                "residual_mean_abs": 0,
                "residual_std_abs": 1,
                "residual_max_abs": 3,
                "residual_q90": 6,
                "action_gain": action_gain,
                "action_gain_fit": action_gain_fit,
                "action_effect_energy": action_effect_energy,
            }
        else:
            residual_rms = int(self.cfg.action_dim) + int(self.cfg.latent_dim)
            action_norm = None
            reward_residual = residual_rms + 1
            extra = {}
        return {
            "residual_rms": residual_rms,
            "action_norm": action_norm,
            "reward_residual": (
                reward_residual if self.m3wm_use_reward_residual else None
            ),
            **extra,
        }

    def context_theta_from_features(self, features, valid_mask=None):
        if not self.enable_m3wm:
            raise RuntimeError("context_theta_from_features() called while disabled.")
        theta, _ = self._m3wm_context(features, valid_mask=valid_mask)
        return theta

    def context_embedding_from_features(self, features, valid_mask=None):
        if not self.enable_m3wm:
            raise RuntimeError(
                "context_embedding_from_features() called while disabled."
            )
        embedding, _ = self._m3wm_context.context_embedding(
            features, self.m3wm_context_embedding_dim, valid_mask=valid_mask
        )
        return embedding * self.m3wm_context_embedding_scale

    def set_context_temperature(self, temperature):
        if self.enable_m3wm:
            self._m3wm_context.set_temperature(float(temperature))

    @torch.no_grad()
    def synchronize_hard_switch_experts(self):
        if not self.enable_m3wm or self.m3wm_composition_mode != "hard_switching":
            return
        self._dynamics.synchronize_from_first()
        self._reward.synchronize_from_first()
        q_parameters = dict(self._Qs.params.named_parameters())
        for name, target in q_parameters.items():
            marker = "experts.0."
            if "experts.0." not in name:
                continue
            for expert_index in range(1, self.m3wm_num_primitives):
                target_name = name.replace("experts.0.", f"experts.{expert_index}.", 1)
                q_parameters[target_name].copy_(target)
        self._target_Qs_params.data.copy_(self._Qs.params.data)

    def set_modular_residual_scale(self, scale, value_scale=None, reward_scale=None):
        if self.m3wm_composition_mode not in {"modular", "hybrid_modular"}:
            return
        value_scale = scale if value_scale is None else value_scale
        reward_scale = scale if reward_scale is None else reward_scale
        self._dynamics.set_residual_scale(float(scale))
        self._reward.set_residual_scale(float(reward_scale))
        if self.m3wm_modular_value:
            for q_ensemble_name in ("_Qs", "_detach_Qs", "_target_Qs"):
                q_ensemble = getattr(self, q_ensemble_name, None)
                q_module = getattr(q_ensemble, "module", None)
                setter = getattr(q_module, "set_residual_scale", None)
                if not callable(setter):
                    raise RuntimeError(
                        f"{q_ensemble_name} does not expose a modular set_residual_scale method"
                    )
                setter(float(value_scale))

    def set_modular_branch_residual_scales(
        self, dynamics_scale, reward_scale, value_scale
    ):
        if self.m3wm_composition_mode not in {"modular", "hybrid_modular"}:
            return
        scales = {
            "dynamics": float(dynamics_scale),
            "reward": float(reward_scale),
            "value": float(value_scale),
        }
        if any((scale < 0 for scale in scales.values())):
            raise ValueError("modular branch residual scales must be non-negative")
        self._dynamics.set_residual_scale(scales["dynamics"])
        self._reward.set_residual_scale(scales["reward"])
        if self.m3wm_modular_value:
            for q_ensemble_name in ("_Qs", "_detach_Qs", "_target_Qs"):
                q_ensemble = getattr(self, q_ensemble_name, None)
                q_module = getattr(q_ensemble, "module", None)
                setter = getattr(q_module, "set_residual_scale", None)
                if not callable(setter):
                    raise RuntimeError(
                        f"{q_ensemble_name} does not expose a modular set_residual_scale method"
                    )
                setter(scales["value"])

    def primitive_diversity_loss(self, z, action, task, valid_mask=None):
        if not self.enable_m3wm or self.m3wm_composition_mode not in {
            "modular",
            "hybrid_modular",
        }:
            return z.new_zeros(())
        if valid_mask is not None:
            mask = valid_mask.to(device=z.device, dtype=torch.bool)
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            if mask.ndim != 2 or tuple(mask.shape) != tuple(z.shape[:2]):
                raise ValueError(
                    "primitive diversity valid_mask must match [time,batch]"
                )
            if not bool(mask.any()):
                return z.new_zeros(())
            z = z[mask]
            action = action[mask]
        model_input = torch.cat([z, action], dim=-1)
        dynamics_loss = self._dynamics.diversity_loss(model_input)
        reward_loss = self._reward.diversity_loss(model_input)
        losses = [dynamics_loss, reward_loss]
        if self.m3wm_modular_value:
            value_losses = self._Qs.call_method("diversity_loss", model_input)
            losses.append(value_losses.mean())
        return torch.stack(losses).mean()

    def context_regularization(self, theta, valid_mask=None):
        preserved_dims = 2 if self.m3wm_separate_composition_heads else 1
        mask = None
        if valid_mask is not None:
            mask = valid_mask.to(device=theta.device, dtype=torch.bool)
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            if mask.ndim != 2 or tuple(mask.shape[:2]) != tuple(theta.shape[:2]):
                raise ValueError(
                    "context regularization valid_mask must match theta [time,batch]"
                )
            if bool(mask.all()):
                mask = None
        reduce_dims = tuple(range(theta.ndim - preserved_dims))
        if mask is None:
            mean_theta = theta.mean(dim=reduce_dims) if reduce_dims else theta
        else:
            weights = mask.to(dtype=theta.dtype)
            while weights.ndim < theta.ndim:
                weights = weights.unsqueeze(-1)
            denom = weights.sum(dim=reduce_dims).clamp_min(1.0)
            mean_theta = (theta * weights).sum(dim=reduce_dims) / denom
        uniform = torch.full_like(mean_theta, 1.0 / self.m3wm_num_primitives)
        balance_loss = (
            (mean_theta * (mean_theta.clamp_min(1e-08).log() - uniform.log()))
            .sum(dim=-1)
            .mean()
        )
        if theta.shape[0] > 1:
            smooth = (theta[1:] - theta[:-1]).pow(2)
            if mask is None:
                smooth_loss = smooth.mean()
            else:
                smooth_mask = (mask[1:] & mask[:-1]).to(dtype=theta.dtype)
                while smooth_mask.ndim < smooth.ndim:
                    smooth_mask = smooth_mask.unsqueeze(-1)
                smooth_loss = (
                    smooth * smooth_mask
                ).sum() / smooth_mask.sum().clamp_min(1.0)
        else:
            smooth_loss = theta.new_zeros(())
        entropy_values = -(theta * theta.clamp_min(1e-08).log()).sum(dim=-1)
        if mask is None:
            entropy = entropy_values.mean()
        else:
            entropy_weights = mask.to(dtype=theta.dtype)
            while entropy_weights.ndim < entropy_values.ndim:
                entropy_weights = entropy_weights.unsqueeze(-1)
            entropy = (
                entropy_values * entropy_weights
            ).sum() / entropy_weights.sum().clamp_min(1.0)
        return (balance_loss, smooth_loss, entropy)

    def encode(self, obs, task):
        if self.cfg.obs == "rgb" and obs.ndim == 5:
            return torch.stack(
                [self._encoder[self.cfg.obs](observation) for observation in obs]
            )
        return self._encoder[self.cfg.obs](obs)

    def next(self, z, a, task, theta=None):
        model_input = torch.cat([z, a], dim=-1)
        if self.enable_m3wm:
            theta = self._resolve_theta(model_input, theta, branch="dynamics")
            return self._dynamics(model_input, theta)
        return self._dynamics(model_input)

    def reward(self, z, a, task, theta=None):
        model_input = torch.cat([z, a], dim=-1)
        if self.enable_m3wm:
            theta = self._resolve_theta(model_input, theta, branch="reward")
            return self._reward(model_input, theta)
        return self._reward(model_input)

    def termination(self, z, task, unnormalized=False):
        assert task is None
        if unnormalized:
            return self._termination(z)
        return torch.sigmoid(self._termination(z))

    def pi(self, z, task):
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        epsilon = torch.randn_like(mean)
        log_prob = math.gaussian_logprob(epsilon, log_std)
        size = epsilon.shape[-1]
        scaled_log_prob = log_prob * size
        action = mean + epsilon * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)
        entropy_scale = scaled_log_prob / (log_prob + 1e-08)
        info = TensorDict(
            {
                "mean": mean,
                "log_std": log_std,
                "action_prob": 1.0,
                "entropy": -log_prob,
                "scaled_entropy": -log_prob * entropy_scale,
            }
        )
        return (action, info)

    def Q(self, z, a, task, return_type="min", target=False, detach=False, theta=None):
        assert return_type in {"min", "avg", "all"}
        model_input = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        if self.m3wm_value_uses_theta:
            theta = self._resolve_theta(model_input, theta, branch="value")
            output = qnet(model_input, theta)
        else:
            output = qnet(model_input)
        if return_type == "all":
            return output
        q_indices = torch.randperm(self.cfg.num_q, device=output.device)[:2]
        q_values = math.two_hot_inv(output[q_indices], self.cfg)
        if return_type == "min":
            return q_values.min(0).values
        return q_values.sum(0) / 2
