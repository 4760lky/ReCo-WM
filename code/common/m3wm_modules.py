from __future__ import annotations
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from common import layers


def _last_linear(module: nn.Module) -> nn.Linear:
    for child in reversed(list(module.modules())):
        if isinstance(child, nn.Linear):
            return child
    raise RuntimeError("No nn.Linear layer was found in the supplied module.")


class ResidualPrimitiveBank(nn.Module):

    def __init__(
        self, input_dim: int, output_dim: int, num_primitives: int, hidden_dim: int
    ) -> None:
        super().__init__()
        if num_primitives < 1:
            raise ValueError("num_primitives must be >= 1")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")
        self.num_primitives = int(num_primitives)
        self.primitives = nn.ModuleList(
            [
                layers.mlp(input_dim, [hidden_dim, hidden_dim], output_dim)
                for _ in range(self.num_primitives)
            ]
        )
        self._inference_param_cache = None
        self._inference_param_cache_key = None
        self._inference_fast_path_supported = all(
            (
                len(tuple(primitive)) == 3
                and isinstance(primitive[0], layers.NormedLinear)
                and isinstance(primitive[1], layers.NormedLinear)
                and isinstance(primitive[2], nn.Linear)
                and (primitive[0].dropout is None)
                and (primitive[1].dropout is None)
                for primitive in self.primitives
            )
        )

    def initialize_output_layers(self, std: float) -> None:
        if std < 0:
            raise ValueError("primitive output init std must be non-negative")
        for primitive in self.primitives:
            output_layer = _last_linear(primitive)
            if std == 0:
                nn.init.zeros_(output_layer.weight)
            else:
                nn.init.normal_(output_layer.weight, mean=0.0, std=std)
            if output_layer.bias is not None:
                nn.init.zeros_(output_layer.bias)

    def _packed_inference_parameters(self, x: torch.Tensor):
        if not self._inference_fast_path_supported:
            return None
        try:
            first = self.primitives[0][0]
            device = x.device
            dtype = x.dtype
            key = (
                str(device),
                dtype,
                tuple(
                    (
                        int(parameter._version)
                        for primitive in self.primitives
                        for layer in primitive
                        for parameter in (
                            layer.weight,
                            layer.bias,
                            *(
                                (layer.ln.weight, layer.ln.bias)
                                if isinstance(layer, layers.NormedLinear)
                                else ()
                            ),
                        )
                    )
                ),
            )
            cached = self._inference_param_cache
            if key == self._inference_param_cache_key and cached is not None:
                return cached
            l0 = [primitive[0] for primitive in self.primitives]
            l1 = [primitive[1] for primitive in self.primitives]
            l2 = [primitive[2] for primitive in self.primitives]
            packed = (
                torch.stack(
                    [layer.weight.to(device=device, dtype=dtype) for layer in l0]
                ),
                torch.stack(
                    [layer.bias.to(device=device, dtype=dtype) for layer in l0]
                ),
                torch.stack(
                    [layer.ln.weight.to(device=device, dtype=dtype) for layer in l0]
                ),
                torch.stack(
                    [layer.ln.bias.to(device=device, dtype=dtype) for layer in l0]
                ),
                torch.stack(
                    [layer.weight.to(device=device, dtype=dtype) for layer in l1]
                ),
                torch.stack(
                    [layer.bias.to(device=device, dtype=dtype) for layer in l1]
                ),
                torch.stack(
                    [layer.ln.weight.to(device=device, dtype=dtype) for layer in l1]
                ),
                torch.stack(
                    [layer.ln.bias.to(device=device, dtype=dtype) for layer in l1]
                ),
                torch.stack(
                    [layer.weight.to(device=device, dtype=dtype) for layer in l2]
                ),
                torch.stack(
                    [layer.bias.to(device=device, dtype=dtype) for layer in l2]
                ),
                float(first.ln.eps),
            )
            self._inference_param_cache_key = key
            self._inference_param_cache = packed
            return packed
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError):
            self._inference_param_cache_key = None
            self._inference_param_cache = None
            return None

    @staticmethod
    def _packed_layer_norm(x, weight, bias, eps):
        mean = x.mean(dim=-1, keepdim=True)
        centered = x - mean
        variance = centered.square().mean(dim=-1, keepdim=True)
        return centered * torch.rsqrt(variance + eps) * weight + bias

    def _inference_primitive_outputs(self, x: torch.Tensor):
        packed = self._packed_inference_parameters(x)
        if packed is None:
            return None
        w0, b0, n0w, n0b, w1, b1, n1w, n1b, w2, b2, eps = packed
        hidden0 = torch.einsum("...i,khi->...kh", x, w0) + b0
        hidden0 = F.mish(self._packed_layer_norm(hidden0, n0w, n0b, eps))
        hidden1 = torch.einsum("...ki,koi->...ko", hidden0, w1) + b1
        hidden1 = F.mish(self._packed_layer_norm(hidden1, n1w, n1b, eps))
        return torch.einsum("...ki,koi->...ko", hidden1, w2) + b2

    def primitive_outputs(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            packed_output = self._inference_primitive_outputs(x)
            if packed_output is not None:
                return packed_output
        return torch.stack([primitive(x) for primitive in self.primitives], dim=-2)

    def diversity_loss(self, x: torch.Tensor) -> torch.Tensor:
        if self.num_primitives < 2:
            return x.new_zeros(())
        outputs = self.primitive_outputs(x)
        flattened = outputs.reshape(-1, self.num_primitives, outputs.shape[-1])
        normalized = F.normalize(flattened, p=2, dim=-1, eps=1e-06)
        similarities = torch.matmul(normalized, normalized.transpose(-1, -2))
        off_diagonal = ~torch.eye(
            self.num_primitives, dtype=torch.bool, device=similarities.device
        )
        return similarities[:, off_diagonal].pow(2).mean()

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if theta.shape[-1] != self.num_primitives:
            raise ValueError(
                f"Expected theta last dimension {self.num_primitives}, but received {theta.shape[-1]}."
            )
        primitive_outputs = self.primitive_outputs(x)
        return (
            primitive_outputs * theta.to(dtype=primitive_outputs.dtype).unsqueeze(-1)
        ).sum(dim=-2)


class ModularResidualMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        base_hidden_dim: int,
        primitive_hidden_dim: int,
        num_primitives: int,
        residual_scale: float,
        base_output_activation: Optional[nn.Module] = None,
        base_dropout: float = 0.0,
        output_simplex_group_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        if output_simplex_group_size is not None and output_simplex_group_size < 2:
            raise ValueError("output_simplex_group_size must be >= 2")
        if base_output_activation is None:
            self.base = layers.mlp(
                input_dim, 2 * [base_hidden_dim], output_dim, dropout=base_dropout
            )
        else:
            self.base = layers.mlp(
                input_dim,
                2 * [base_hidden_dim],
                output_dim,
                act=base_output_activation,
                dropout=base_dropout,
            )
        self.residual_bank = ResidualPrimitiveBank(
            input_dim=input_dim,
            output_dim=output_dim,
            num_primitives=num_primitives,
            hidden_dim=primitive_hidden_dim,
        )
        self.residual_scale = float(residual_scale)
        self.output_simplex_group_size = output_simplex_group_size
        self._inference_base_fast_path_supported = (
            len(tuple(self.base)) == 3
            and isinstance(self.base[0], layers.NormedLinear)
            and isinstance(self.base[1], layers.NormedLinear)
            and isinstance(self.base[2], (nn.Linear, layers.NormedLinear))
            and (self.base[0].dropout is None)
            and (self.base[1].dropout is None)
            and (
                not isinstance(self.base[2], layers.NormedLinear)
                or self.base[2].dropout is None
            )
        )

    @property
    def base_output_layer(self) -> nn.Linear:
        return _last_linear(self.base)

    def set_residual_scale(self, scale: float) -> None:
        scale = float(scale)
        if scale < 0:
            raise ValueError("residual scale must be non-negative")
        self.residual_scale = scale

    def initialize_residual_outputs(self, std: float) -> None:
        self.residual_bank.initialize_output_layers(std)

    def diversity_loss(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_bank.diversity_loss(x)

    @staticmethod
    def _inference_normed_layer(x: torch.Tensor, layer: layers.NormedLinear):
        value = F.linear(x, layer.weight, layer.bias)
        if layer.dropout is not None:
            value = F.dropout(value, p=layer.dropout.p, training=False)
        value = F.layer_norm(
            value, (layer.out_features,), layer.ln.weight, layer.ln.bias, layer.ln.eps
        )
        return layer.act(value)

    def _inference_base(self, x: torch.Tensor):
        if not self._inference_base_fast_path_supported:
            return None
        try:
            first, second, final = (self.base[0], self.base[1], self.base[2])
            value = self._inference_normed_layer(x, first)
            value = self._inference_normed_layer(value, second)
            if isinstance(final, layers.NormedLinear):
                value = self._inference_normed_layer(value, final)
            else:
                value = F.linear(value, final.weight, final.bias)
            return value
        except (AttributeError, RuntimeError, TypeError, ValueError):
            return None

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            base_output = self._inference_base(x)
            if base_output is None:
                base_output = self.base(x)
        else:
            base_output = self.base(x)
        residual = self.residual_bank(x, theta)
        output = base_output + self.residual_scale * residual
        if self.output_simplex_group_size is not None:
            if output.shape[-1] % self.output_simplex_group_size != 0:
                raise ValueError(
                    "Output dimension must be divisible by simplex group size"
                )
            grouped = output.reshape(
                *output.shape[:-1], -1, self.output_simplex_group_size
            )
            grouped = grouped.clamp_min(1e-06)
            grouped = grouped / grouped.sum(dim=-1, keepdim=True).clamp_min(1e-08)
            output = grouped.reshape_as(output)
        return output


class ContextVectorMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        output_dim: int,
        hidden_dim: int,
        output_activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if context_dim < 1:
            raise ValueError("context_dim must be >= 1")
        self.context_dim = int(context_dim)
        self.net = layers.mlp(
            input_dim + self.context_dim,
            2 * [hidden_dim],
            output_dim,
            act=output_activation,
            dropout=dropout,
        )

    @property
    def output_layer(self) -> nn.Linear:
        return _last_linear(self.net)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if theta.shape[-1] != self.context_dim:
            raise ValueError(
                f"Expected theta last dimension {self.context_dim}, but received {theta.shape[-1]}."
            )
        theta = theta.to(device=x.device, dtype=x.dtype)
        return self.net(torch.cat([x, theta], dim=-1))


class ContextConditionedModularResidualMLP(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        base_hidden_dim: int,
        primitive_hidden_dim: int,
        num_primitives: int,
        residual_scale: float,
        base_output_activation: Optional[nn.Module] = None,
        base_dropout: float = 0.0,
        output_simplex_group_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        if residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        if output_simplex_group_size is not None and output_simplex_group_size < 2:
            raise ValueError("output_simplex_group_size must be >= 2")
        self.base = ContextVectorMLP(
            input_dim=input_dim,
            context_dim=num_primitives,
            output_dim=output_dim,
            hidden_dim=base_hidden_dim,
            output_activation=base_output_activation,
            dropout=base_dropout,
        )
        self.residual_bank = ResidualPrimitiveBank(
            input_dim=input_dim,
            output_dim=output_dim,
            num_primitives=num_primitives,
            hidden_dim=primitive_hidden_dim,
        )
        self.residual_scale = float(residual_scale)
        self.output_simplex_group_size = output_simplex_group_size

    @property
    def base_output_layer(self) -> nn.Linear:
        return self.base.output_layer

    def set_residual_scale(self, scale: float) -> None:
        scale = float(scale)
        if scale < 0:
            raise ValueError("residual scale must be non-negative")
        self.residual_scale = scale

    def initialize_residual_outputs(self, std: float) -> None:
        self.residual_bank.initialize_output_layers(std)

    def diversity_loss(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual_bank.diversity_loss(x)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        base_output = self.base(x, theta)
        if self.residual_scale == 0.0:
            return base_output
        residual = self.residual_bank(x, theta)
        output = base_output + self.residual_scale * residual
        if self.output_simplex_group_size is not None:
            if output.shape[-1] % self.output_simplex_group_size != 0:
                raise ValueError(
                    "Output dimension must be divisible by simplex group size"
                )
            grouped = output.reshape(
                *output.shape[:-1], -1, self.output_simplex_group_size
            )
            grouped = grouped.clamp_min(1e-06)
            grouped = grouped / grouped.sum(dim=-1, keepdim=True).clamp_min(1e-08)
            output = grouped.reshape_as(output)
        return output


class HardSwitchMLPBank(nn.Module):

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_experts: int,
        output_activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_experts < 1:
            raise ValueError("num_experts must be >= 1")
        self.num_experts = int(num_experts)
        self.experts = nn.ModuleList(
            [
                layers.mlp(
                    input_dim,
                    2 * [hidden_dim],
                    output_dim,
                    act=output_activation,
                    dropout=dropout,
                )
                for _ in range(self.num_experts)
            ]
        )

    @property
    def output_layers(self) -> Tuple[nn.Linear, ...]:
        return tuple((_last_linear(expert) for expert in self.experts))

    @torch.no_grad()
    def synchronize_from_first(self) -> None:
        source = self.experts[0].state_dict()
        for expert in self.experts[1:]:
            expert.load_state_dict(source)

    def forward(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        if theta.shape[-1] != self.num_experts:
            raise ValueError(
                f"Expected theta last dimension {self.num_experts}, but received {theta.shape[-1]}."
            )
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        theta = theta.to(device=x.device, dtype=x.dtype)
        hard = F.one_hot(theta.argmax(dim=-1), num_classes=self.num_experts).to(
            dtype=x.dtype
        )
        use_straight_through = self.training or theta.requires_grad
        weights = hard + theta - theta.detach() if use_straight_through else hard
        return (expert_outputs * weights.unsqueeze(-1)).sum(dim=-2)


class ContextRouter(nn.Module):

    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        hidden_dim: int,
        num_primitives: int,
        temperature: float,
        num_branches: int = 1,
        short_window: int = 8,
        mid_window: int = 16,
        long_window: int = 32,
        use_multiscale: bool = True,
        explicit_three_scale: bool = False,
        stream_mode: str = "shared",
        architecture_latent_dim: Optional[int] = None,
        architecture_action_dim: Optional[int] = None,
        architecture_include_reward_residual: bool = False,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")
        if projection_dim < 1 or hidden_dim < 1:
            raise ValueError("context dimensions must be >= 1")
        if num_primitives < 1:
            raise ValueError("num_primitives must be >= 1")
        if num_branches < 1:
            raise ValueError("num_branches must be >= 1")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if short_window < 1 or mid_window < 1 or long_window < 1:
            raise ValueError("context windows must be >= 1")
        if short_window > long_window:
            raise ValueError("short_window must be <= long_window")
        if explicit_three_scale and (not short_window <= mid_window <= long_window):
            raise ValueError("Require short_window <= mid_window <= long_window")
        stream_mode = str(stream_mode).lower()
        valid_stream_modes = {"shared", "architecture_partitioned"}
        if stream_mode not in valid_stream_modes:
            raise ValueError(
                f"stream_mode must be one of {sorted(valid_stream_modes)}, received {stream_mode!r}"
            )
        if stream_mode == "architecture_partitioned":
            if not use_multiscale or not explicit_three_scale:
                raise ValueError(
                    "architecture_partitioned requires multiscale and explicit three-scale context"
                )
            if architecture_latent_dim is None or architecture_latent_dim < 1:
                raise ValueError(
                    "architecture_latent_dim must be >= 1 for architecture_partitioned"
                )
            if architecture_action_dim is None or architecture_action_dim < 1:
                raise ValueError(
                    "architecture_action_dim must be >= 1 for architecture_partitioned"
                )
            expected_input_dim = (
                2 * int(architecture_latent_dim)
                + int(architecture_action_dim)
                + 2
                + int(bool(architecture_include_reward_residual))
            )
            if input_dim != expected_input_dim:
                raise ValueError(
                    f"architecture_partitioned feature layout mismatch: expected {expected_input_dim}, received {input_dim}"
                )
        self.input_dim = int(input_dim)
        self.num_primitives = int(num_primitives)
        self.num_branches = int(num_branches)
        self.temperature = float(temperature)
        self.short_window = int(short_window)
        self.mid_window = int(mid_window)
        self.long_window = int(long_window)
        self.use_multiscale = bool(use_multiscale)
        self.explicit_three_scale = bool(explicit_three_scale)
        self.stream_mode = stream_mode
        self.architecture_latent_dim = (
            None if architecture_latent_dim is None else int(architecture_latent_dim)
        )
        self.architecture_action_dim = (
            None if architecture_action_dim is None else int(architecture_action_dim)
        )
        self.architecture_include_reward_residual = bool(
            architecture_include_reward_residual
        )
        if self.stream_mode == "architecture_partitioned":
            short_dim = 1 + int(self.architecture_include_reward_residual)
            mid_dim = int(self.architecture_latent_dim)
            long_dim = (
                int(self.architecture_latent_dim)
                + int(self.architecture_action_dim)
                + 1
            )
            router_input_dim = short_dim + mid_dim + long_dim
        else:
            short_dim = self.input_dim
            mid_dim = self.input_dim
            long_dim = self.input_dim
            router_input_dim = (
                3 * self.input_dim if self.use_multiscale else self.input_dim
            )
        self.stream_dims = {
            "short": int(short_dim),
            "mid": int(mid_dim),
            "long": int(long_dim),
        }
        self.stream_windows = {
            "short": self.short_window,
            "mid": self.mid_window,
            "long": self.long_window,
        }
        self.input_norm = nn.LayerNorm(router_input_dim)
        self.project = nn.Sequential(
            nn.Linear(router_input_dim, projection_dim),
            nn.Mish(),
            nn.LayerNorm(projection_dim),
        )
        self.gru = nn.GRU(
            input_size=projection_dim, hidden_size=hidden_dim, num_layers=1
        )
        self.head = nn.Linear(hidden_dim, self.num_branches * self.num_primitives)

    def set_temperature(self, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = float(temperature)

    @staticmethod
    def _causal_moving_average(
        features: torch.Tensor, window: int, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if features.shape[0] == 0:
            return features
        if valid_mask is None:
            mask = None
        else:
            mask = valid_mask.to(device=features.device, dtype=torch.bool)
            if mask.ndim == features.ndim and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            if features.ndim == 2 and mask.ndim == 1:
                mask = mask.unsqueeze(1)
            if mask.ndim != 2 or tuple(mask.shape[:2]) != tuple(features.shape[:2]):
                raise ValueError(
                    f"valid_mask must have shape [time,batch] for context routing; received {tuple(mask.shape)} for features {tuple(features.shape)}"
                )
            if bool(mask.all()):
                mask = None
        if mask is not None:
            weights = mask.to(dtype=features.dtype).unsqueeze(-1)
            safe_features = torch.where(
                mask.unsqueeze(-1), features, torch.zeros_like(features)
            )
            cumulative = (safe_features * weights).cumsum(dim=0)
            prefix = torch.cat([torch.zeros_like(cumulative[:1]), cumulative], dim=0)
            end = torch.arange(1, features.shape[0] + 1, device=features.device)
            start = (end - int(window)).clamp_min(0)
            totals = prefix[end] - prefix[start]
            count_prefix = torch.cat([torch.zeros_like(weights[:1]), weights], dim=0)
            counts = count_prefix[end] - count_prefix[start]
            return totals / counts.clamp_min(1.0)
        cumulative = features.cumsum(dim=0)
        prefix = torch.cat([torch.zeros_like(features[:1]), cumulative], dim=0)
        end = torch.arange(1, features.shape[0] + 1, device=features.device)
        start = (end - int(window)).clamp_min(0)
        totals = prefix[end] - prefix[start]
        counts = (end - start).to(features.dtype)
        counts = counts.view(features.shape[0], *[1] * (features.ndim - 1))
        return totals / counts.clamp_min(1)

    def _multiscale_features(
        self, features: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if not self.use_multiscale:
            return features
        if self.stream_mode == "architecture_partitioned":
            latent_dim = int(self.architecture_latent_dim)
            action_dim = int(self.architecture_action_dim)
            residual_end = latent_dim
            residual_rms_end = residual_end + 1
            latent_end = residual_rms_end + latent_dim
            action_end = latent_end + action_dim
            reward_end = action_end + 1
            dynamics_residual = features[..., :residual_end]
            residual_rms = features[..., residual_end:residual_rms_end]
            latent_action_reward = features[..., residual_rms_end:reward_end]
            short_error = residual_rms
            if self.architecture_include_reward_residual:
                reward_residual = features[..., reward_end : reward_end + 1]
                short_error = torch.cat([short_error, reward_residual.abs()], dim=-1)
            short_summary = self._causal_moving_average(
                short_error, self.short_window, valid_mask
            )
            mid_summary = self._causal_moving_average(
                dynamics_residual, self.mid_window, valid_mask
            )
            long_summary = self._causal_moving_average(
                latent_action_reward, self.long_window, valid_mask
            )
            return torch.cat([short_summary, mid_summary, long_summary], dim=-1)
        short_mean = self._causal_moving_average(
            features, self.short_window, valid_mask
        )
        mid_mean = self._causal_moving_average(features, self.mid_window, valid_mask)
        long_mean = self._causal_moving_average(features, self.long_window, valid_mask)
        if self.explicit_three_scale:
            return torch.cat([short_mean, mid_mean, long_mean], dim=-1)
        return torch.cat([features, short_mean, long_mean], dim=-1)

    def _encode(
        self,
        features: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        squeeze_batch = False
        if features.ndim == 2:
            features = features.unsqueeze(1)
            squeeze_batch = True
        if features.ndim != 3:
            raise ValueError(
                f"ContextRouter expects [T,D] or [T,B,D], received shape {tuple(features.shape)}."
            )
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected context feature dim {self.input_dim}, received {features.shape[-1]}."
            )
        mask = None
        if valid_mask is not None:
            mask = valid_mask.to(device=features.device, dtype=torch.bool)
            if mask.ndim == 1:
                mask = mask.unsqueeze(1)
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.squeeze(-1)
            if mask.ndim != 2 or tuple(mask.shape) != tuple(features.shape[:2]):
                raise ValueError(
                    f"ContextRouter valid_mask must have shape [time,batch], received {tuple(mask.shape)} for features {tuple(features.shape)}"
                )
            router_features = self._multiscale_features(features, mask)
        else:
            router_features = self._multiscale_features(features)
        projected = self.project(self.input_norm(router_features))
        if mask is None or bool(mask.all()):
            outputs, hidden = self.gru(projected, hidden)
        else:
            batch = int(projected.shape[1])
            if hidden is None:
                hidden = projected.new_zeros(1, batch, self.gru.hidden_size)
            outputs = []
            for time_index in range(int(projected.shape[0])):
                candidate, candidate_hidden = self.gru(
                    projected[time_index : time_index + 1], hidden
                )
                step_mask = mask[time_index].view(1, batch, 1)
                hidden = torch.where(step_mask, candidate_hidden, hidden)
                outputs.append(hidden)
            outputs = torch.cat(outputs, dim=0)
        return (outputs, hidden, squeeze_batch)

    def context_embedding(
        self,
        features: torch.Tensor,
        embedding_dim: int,
        hidden: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if embedding_dim < 1 or embedding_dim > self.gru.hidden_size:
            raise ValueError(
                f"embedding_dim must be in [1, hidden_dim], received {embedding_dim}"
            )
        outputs, hidden, squeeze_batch = self._encode(
            features, hidden, valid_mask=valid_mask
        )
        embedding = F.normalize(outputs[..., :embedding_dim], p=2, dim=-1, eps=1e-08)
        if squeeze_batch:
            embedding = embedding.squeeze(1)
        return (embedding, hidden)

    def forward(
        self,
        features: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs, hidden, squeeze_batch = self._encode(
            features, hidden, valid_mask=valid_mask
        )
        logits = self.head(outputs).reshape(
            *outputs.shape[:-1], self.num_branches, self.num_primitives
        ) / max(self.temperature, 1e-06)
        theta = torch.softmax(logits, dim=-1)
        if self.num_branches == 1:
            theta = theta.squeeze(-2)
        if squeeze_batch:
            theta = theta.squeeze(1)
        return (theta, hidden)
