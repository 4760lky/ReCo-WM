from bisect import insort
from collections import Counter, deque
from copy import deepcopy
import torch
from tensordict.tensordict import TensorDict
from torchrl.data.replay_buffers import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import SliceSampler
from common.m3wm_replay_transactions import StagedReplayRow


class ReplayStoragePoisoned(RuntimeError):
    pass


class _TrajectoryRun:
    __slots__ = ("trajectory_id", "first_index", "last_index", "length")

    def __init__(self, trajectory_id, physical_index):
        self.trajectory_id = int(trajectory_id)
        self.first_index = int(physical_index)
        self.last_index = int(physical_index)
        self.length = 1


class _IncrementalTrajectoryIndex:

    def __init__(self, capacity, *, name):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError(f"{name} trajectory index capacity must be positive.")
        self.capacity = capacity
        self.name = str(name)
        self._runs = deque()
        self._rows = deque()
        self._row_count = 0
        self._run_length_total = 0
        self._last_index = None
        self._version = 0
        self._cached_boundaries = {}

    @property
    def row_count(self):
        return int(self._row_count)

    @property
    def run_count(self):
        return len(self._runs)

    @property
    def minimum_run_length(self):
        return min((int(run.length) for run in self._runs), default=0)

    def clear(self):
        self._runs.clear()
        self._rows.clear()
        self._row_count = 0
        self._run_length_total = 0
        self._last_index = None
        self._version += 1
        self._cached_boundaries.clear()

    def state_dict(self):
        return {
            "capacity": int(self.capacity),
            "name": str(self.name),
            "rows": [
                (int(run.trajectory_id), int(index)) for (run, index) in self._rows
            ],
            "version": int(self._version),
        }

    def load_state_dict(self, state):
        if int(state.get("capacity", -1)) != int(self.capacity) or str(
            state.get("name", "")
        ) != str(self.name):
            raise ValueError(f"{self.name} trajectory-index schema mismatch.")
        self.clear()
        for trajectory_id, physical_index in state.get("rows", ()):
            self.append(int(trajectory_id), int(physical_index))

    def append(self, trajectory_id, physical_index):
        trajectory_id = int(trajectory_id)
        physical_index = int(physical_index)
        if not 0 <= physical_index < self.capacity:
            raise RuntimeError(
                f"{self.name} replay writer returned physical index {physical_index}, outside capacity {self.capacity}."
            )
        if self._last_index is not None:
            expected = (int(self._last_index) + 1) % self.capacity
            if physical_index != expected:
                raise RuntimeError(
                    f"{self.name} replay trajectory index lost writer order: expected position {expected}, got {physical_index}."
                )
        if self._runs and self._runs[-1].trajectory_id == trajectory_id:
            run = self._runs[-1]
            expected = (int(run.last_index) + 1) % self.capacity
            if physical_index != expected:
                raise RuntimeError(
                    f"{self.name} replay trajectory run is not physically contiguous: expected {expected}, got {physical_index}."
                )
            run.last_index = physical_index
            run.length += 1
        else:
            run = _TrajectoryRun(trajectory_id, physical_index)
            self._runs.append(run)
        self._rows.append((run, physical_index))
        self._row_count += 1
        self._run_length_total += 1
        if self._row_count > self.capacity:
            old_run, old_index = self._rows.popleft()
            if not self._runs or self._runs[0] is not old_run:
                raise RuntimeError(
                    f"{self.name} replay trajectory eviction order diverged."
                )
            if int(old_run.first_index) != int(old_index):
                raise RuntimeError(
                    f"{self.name} replay trajectory first position diverged: expected {old_run.first_index}, got {old_index}."
                )
            old_run.length -= 1
            self._run_length_total -= 1
            if old_run.length < 0:
                raise RuntimeError(
                    f"{self.name} replay trajectory length became negative."
                )
            if old_run.length == 0:
                self._runs.popleft()
            else:
                old_run.first_index = (int(old_index) + 1) % self.capacity
            self._row_count = self.capacity
        self._last_index = physical_index
        self._version += 1
        self._cached_boundaries.clear()

    def boundaries(self, storage):
        try:
            storage_length = int(storage.shape[0])
        except Exception as error:
            raise RuntimeError(
                f"Could not read {self.name} replay storage length."
            ) from error
        if storage_length != self._row_count:
            raise RuntimeError(
                f"{self.name} replay trajectory index/storage length mismatch: index={self._row_count}, storage={storage_length}."
            )
        if not self._runs:
            raise RuntimeError(
                f"{self.name} replay trajectory index is empty while storage is non-empty."
            )
        if self._run_length_total != self._row_count:
            raise RuntimeError(
                f"{self.name} replay trajectory run lengths do not conserve rows."
            )
        try:
            device = torch.device(storage.device)
        except Exception:
            device = torch.device("cpu")
        cache_key = str(device)
        cached = self._cached_boundaries.get(cache_key)
        if cached is not None and cached[0] == self._version:
            return cached[1]
        logical_length = int(self._row_count)
        stop_positions = sorted((int(run.last_index) for run in self._runs))
        if any(
            (position < 0 or position >= logical_length for position in stop_positions)
        ):
            raise RuntimeError(
                f"{self.name} replay trajectory stop lies outside logical storage."
            )
        if len(set(stop_positions)) != len(stop_positions):
            raise RuntimeError(f"{self.name} replay trajectory stops are not unique.")
        if bool(getattr(storage, "_is_full", False)):
            cursor = getattr(storage, "_last_cursor", None)
            if torch.is_tensor(cursor):
                if cursor.numel() != 1:
                    raise RuntimeError(
                        f"{self.name} replay storage has a non-scalar last cursor."
                    )
                cursor = int(cursor.reshape(-1)[0].item())
            elif cursor is None:
                raise RuntimeError(
                    f"{self.name} replay storage is full without a last cursor."
                )
            else:
                cursor = int(cursor)
            if not 0 <= cursor < logical_length:
                raise RuntimeError(
                    f"{self.name} replay last cursor {cursor} is outside logical storage length {logical_length}."
                )
            if cursor not in stop_positions:
                insort(stop_positions, cursor)
        starts_positions = [(stop_positions[-1] + 1) % logical_length]
        starts_positions.extend(
            (
                (stop_positions[index - 1] + 1) % logical_length
                for index in range(1, len(stop_positions))
            )
        )
        length_positions = [
            (
                stop - start + 1
                if stop - start + 1 > 0
                else stop - start + 1 + logical_length
            )
            for (start, stop) in zip(starts_positions, stop_positions)
        ]
        starts = torch.tensor(
            [[int(start)] for start in starts_positions],
            dtype=torch.long,
            device=device,
        )
        stops = torch.tensor(
            [[int(stop)] for stop in stop_positions], dtype=torch.long, device=device
        )
        lengths = torch.tensor(length_positions, dtype=torch.long, device=device)
        values = (starts, stops, lengths)
        self._cached_boundaries[cache_key] = (self._version, values)
        return values


class _IncrementalSliceSampler(SliceSampler):

    def __init__(
        self,
        trajectory_index,
        *,
        adaptive_short_episodes=False,
        short_episode_sampling_mode="uniform_episode",
        context_len=0,
        **kwargs,
    ):
        self._trajectory_index = trajectory_index
        self._context_len = max(0, int(context_len))
        self._adaptive_short_episodes = bool(adaptive_short_episodes)
        mode = str(short_episode_sampling_mode).lower()
        if mode not in {"uniform_episode", "transition_weighted"}:
            raise ValueError(
                f"unknown short-episode sampling mode: {short_episode_sampling_mode!r}"
            )
        self._short_episode_sampling_mode = mode
        super().__init__(**kwargs)

    def sample(self, storage, batch_size):
        if not self._adaptive_short_episodes:
            return super().sample(storage, batch_size)
        short_runs = self._trajectory_index.minimum_run_length < int(self.slice_len)
        if not short_runs:
            return super().sample(storage, batch_size)
        return self._sample_transition_anchored(storage, batch_size)

    def _sample_transition_anchored(self, storage, batch_size):
        if self.truncated_key is not None:
            raise RuntimeError(
                "transition-anchored online sampler requires truncated_key=None."
            )
        seq_length, num_slices = self._adjusted_batch_size(batch_size)
        if int(seq_length) != int(self.slice_len):
            raise RuntimeError(
                f"online transition-anchored sampler received an unexpected sequence length: {seq_length} != {self.slice_len}."
            )
        start_idx, _stop_idx, lengths = self._get_stop_and_length(storage)
        lengths = lengths.to(dtype=torch.long)
        eligible = (lengths >= 2).nonzero(as_tuple=False).reshape(-1)
        if not int(eligible.numel()):
            raise RuntimeError(
                "online replay has no trajectory with a real transition target."
            )
        if self._short_episode_sampling_mode == "transition_weighted":
            weights = (lengths[eligible] - 1).to(dtype=torch.float32)
            choice = torch.multinomial(
                weights, num_samples=num_slices, replacement=True, generator=self._rng
            )
        else:
            choice = torch.randint(
                int(eligible.numel()),
                (num_slices,),
                device=lengths.device,
                generator=self._rng,
            )
        traj_idx = eligible[choice]
        selected_lengths = lengths[traj_idx]
        selected_starts = start_idx[traj_idx, 0]
        short = selected_lengths < int(seq_length)
        random_unit = torch.rand(num_slices, device=lengths.device, generator=self._rng)
        max_start = (selected_lengths - int(seq_length) + 1).clamp_min(1)
        long_starts = torch.floor(random_unit * max_start).to(dtype=torch.long)
        anchor = torch.floor(random_unit * (selected_lengths - 1).clamp_min(1)).to(
            dtype=torch.long
        )
        short_window_start = anchor - int(self._context_len)
        window_start = torch.where(short, short_window_start, long_starts)
        time = torch.arange(int(seq_length), device=lengths.device, dtype=torch.long)
        offsets = window_start.unsqueeze(1) + time.unsqueeze(0)
        real_mask = (offsets >= 0) & (offsets < selected_lengths.unsqueeze(1))
        safe_offsets = offsets.clamp_min(0)
        safe_offsets = torch.minimum(safe_offsets, (selected_lengths - 1).unsqueeze(1))
        storage_length = int(self._trajectory_index.row_count)
        if storage_length <= 0:
            raise RuntimeError("online replay sampler has no physical storage rows.")
        physical = (selected_starts.unsqueeze(1) + safe_offsets) % storage_length
        if (
            int(physical.min().item()) < 0
            or int(physical.max().item()) >= storage_length
        ):
            raise RuntimeError(
                f"online replay sampler produced an out-of-range physical index: min={int(physical.min().item())}, max={int(physical.max().item())}, length={storage_length}."
            )
        index = physical.reshape(-1, 1).to(dtype=torch.long)
        info = {("collector", "mask"): real_mask.reshape(-1)}
        return (index, info)

    def _get_stop_and_length(self, storage, fallback=True):
        return self._trajectory_index.boundaries(storage)

    def _empty(self):
        self._trajectory_index.clear()


class Buffer:

    def __init__(self, cfg):
        self.cfg = cfg
        self._device = torch.device("cuda:0")
        self._capacity = min(cfg.buffer_size, cfg.steps)
        self._context_len = (
            int(cfg.get("m3wm_context_window", 8))
            if cfg.get("enable_m3wm", False)
            else 0
        )
        self._sequence_length = self._context_len + cfg.horizon + 1
        self._batch_size = cfg.batch_size * self._sequence_length
        self._num_eps = 0
        self._buffer = None
        self._recent_buffer = None
        self._active_episode_id = None
        self._active_episode_length = 0
        self._max_episode_length = 0
        self._total_transitions = 0
        self._history_episode_rows = deque()
        self._history_episode_counts = Counter()
        self._recent_episode_rows = deque()
        self._recent_episode_counts = Counter()
        self._recent_refresh_count = 0
        if not isinstance(True, bool):
            raise ValueError(
                "online_replay_short_episode_padding_enabled must be boolean."
            )
        self._short_episode_padding_enabled = True
        short_sampling_mode = str(
            cfg.get("online_replay_short_episode_sampling_mode", "uniform_episode")
        ).lower()
        if short_sampling_mode not in {"uniform_episode", "transition_weighted"}:
            raise ValueError(
                "online_replay_short_episode_sampling_mode must be 'uniform_episode' or 'transition_weighted'."
            )
        self._short_episode_sampling_mode = short_sampling_mode
        if not isinstance(False, bool):
            raise ValueError(
                "online_replay_transactional_context_enabled must be boolean."
            )
        self._transactional_replay_enabled = False
        if not isinstance(False, bool):
            raise ValueError(
                "online_replay_cross_episode_transaction_enabled must be boolean."
            )
        self._cross_episode_transaction_enabled = False
        self._replay_transactions = None
        self._pending_replay_seed = None
        self._pending_replay_seeds = []
        self._pending_replay_payloads = []
        self._latest_observation_row = None
        self._replay_trajectory_ids = {}
        self._next_replay_trajectory_id = 0
        self._replay_flush_state = "idle"
        self._active_replay_flush = None
        self._replay_poison = None
        self._last_replay_receipt = None
        self._replay_resolution_count = 0
        self._replay_flushed_transition_count = 0
        self._last_replay_context_switch_receipt = None
        self._replay_context_switch_count = 0
        self._online_replay_mixture_enabled = True
        False
        self._recent_fraction = float(cfg.get("online_replay_recent_fraction", 0.5))
        if not 0.0 <= self._recent_fraction <= 1.0:
            raise ValueError(
                f"online_replay_recent_fraction must lie in [0, 1], got {self._recent_fraction}."
            )
        self._recent_slices = int(round(int(cfg.batch_size) * self._recent_fraction))
        self._recent_slices = min(int(cfg.batch_size), max(0, self._recent_slices))
        self._history_slices = int(cfg.batch_size) - self._recent_slices
        default_recent_window = min(10000, self._capacity)
        self._recent_window = int(
            cfg.get("online_replay_recent_window", default_recent_window)
        )
        self._recent_capacity = int(
            cfg.get("online_replay_recent_capacity", self._recent_window)
        )
        self._recent_effective_capacity = min(
            self._capacity, self._recent_window, self._recent_capacity
        )
        self._history_trajectory_index = _IncrementalTrajectoryIndex(
            self._capacity, name="history"
        )
        self._recent_trajectory_index = _IncrementalTrajectoryIndex(
            max(1, self._recent_effective_capacity), name="recent"
        )
        if self._recent_window <= 0 or self._recent_capacity <= 0:
            raise ValueError("online recent replay window/capacity must be positive.")
        False
        online_slice_len = self._sequence_length
        self._history_sampler = self._make_sampler(
            max(1, self._history_slices),
            cache_values=False,
            slice_len=online_slice_len,
            trajectory_index=self._history_trajectory_index,
            traj_key="episode",
            adaptive_short_episodes=True,
            short_episode_sampling_mode=self._short_episode_sampling_mode,
            context_len=self._context_len,
        )
        self._recent_sampler = self._make_sampler(
            max(1, self._recent_slices),
            cache_values=False,
            slice_len=online_slice_len,
            trajectory_index=self._recent_trajectory_index,
            traj_key="episode",
            adaptive_short_episodes=True,
            short_episode_sampling_mode=self._short_episode_sampling_mode,
            context_len=self._context_len,
        )
        self._sampler = self._history_sampler
        self._last_sample_composition = {
            "history_slices": int(cfg.batch_size),
            "recent_slices": 0,
        }
        self._last_sample_valid_fraction = 1.0
        self._last_sample_unique_transition_ids = ()
        self._last_sample_real_loss_rows = 0

    @staticmethod
    def _make_sampler(
        num_slices,
        *,
        cache_values,
        slice_len=None,
        traj_key="episode",
        trajectory_index=None,
        adaptive_short_episodes=False,
        short_episode_sampling_mode="uniform_episode",
        context_len=0,
    ):
        sampler_cls = (
            _IncrementalSliceSampler if trajectory_index is not None else SliceSampler
        )
        kwargs = dict(
            num_slices=int(num_slices) if slice_len is None else None,
            slice_len=int(slice_len) if slice_len is not None else None,
            end_key=None,
            traj_key=str(traj_key),
            truncated_key=None,
            strict_length=True,
            cache_values=bool(cache_values),
        )
        if trajectory_index is not None:
            kwargs["trajectory_index"] = trajectory_index
            kwargs["adaptive_short_episodes"] = bool(adaptive_short_episodes)
            kwargs["short_episode_sampling_mode"] = str(short_episode_sampling_mode)
            kwargs["context_len"] = int(context_len)
        return sampler_cls(**kwargs)

    @property
    def capacity(self):
        return self._capacity

    @property
    def num_eps(self):
        return self._num_eps

    @property
    def streaming_enabled(self):
        return True

    @property
    def transactional_replay_enabled(self):
        return False

    @property
    def replay_transaction_pending(self):
        core = self._replay_transactions
        return bool(core is not None and core.has_pending_transaction)

    @property
    def replay_pending_proposal_token(self):
        core = self._replay_transactions
        if core is None:
            return None
        snapshot = core.pending_snapshot
        return None if snapshot is None else snapshot.proposal_token

    @property
    def replay_staged_transitions(self):
        core = self._replay_transactions
        return 0 if core is None else int(core.staged_transition_count)

    @property
    def replay_integrity_ok(self):
        return self._replay_poison is None

    @property
    def replay_flush_state(self):
        return str(self._replay_flush_state)

    @property
    def replay_failure_snapshot(self):
        return deepcopy(self._replay_poison)

    @property
    def replay_resolution_count(self):
        return int(self._replay_resolution_count)

    @property
    def replay_flushed_transition_count(self):
        return int(self._replay_flushed_transition_count)

    @property
    def last_replay_receipt(self):
        return self._last_replay_receipt

    @property
    def replay_active_context_id(self):
        core = self._replay_transactions
        return None if core is None else int(core.active_context_id)

    @property
    def replay_context_switch_count(self):
        return int(self._replay_context_switch_count)

    @property
    def last_replay_context_switch_receipt(self):
        return self._last_replay_context_switch_receipt

    @property
    def replay_transaction_metrics(self):
        receipt = self._last_replay_receipt
        last_receipt = (
            None
            if receipt is None
            else {
                "proposal_token": repr(receipt.proposal_token),
                "outcome": str(receipt.outcome),
                "source_context_id": int(receipt.source_context_id),
                "target_context_id": int(receipt.target_context_id),
                "segment_id": int(receipt.segment_id),
                "flushed_count": int(receipt.flushed_count),
            }
        )
        return {
            "integrity_ok": bool(self.replay_integrity_ok),
            "flush_state": self.replay_flush_state,
            "pending_transaction": bool(self.replay_transaction_pending),
            "pending_staged_transition_count": self.replay_staged_transitions,
            "resolution_count": self.replay_resolution_count,
            "flushed_transition_count": self.replay_flushed_transition_count,
            "last_receipt": last_receipt,
            "active_context_id": self.replay_active_context_id,
            "context_switch_count": self.replay_context_switch_count,
            "last_context_switch": (
                None
                if self._last_replay_context_switch_receipt is None
                else {
                    "source_context_id": int(
                        self._last_replay_context_switch_receipt.source_context_id
                    ),
                    "target_context_id": int(
                        self._last_replay_context_switch_receipt.target_context_id
                    ),
                    "segment_id": int(
                        self._last_replay_context_switch_receipt.segment_id
                    ),
                }
            ),
            "failure": self.replay_failure_snapshot,
        }

    def _assert_replay_healthy(self, operation):
        if self._replay_poison is None:
            return
        failure = self._replay_poison
        raise ReplayStoragePoisoned(
            f"Replay storage is permanently poisoned; create a fresh Buffer instead of attempting {operation!r}. phase={failure['failure_phase']!r}, proposal_token={failure['proposal_token']}."
        )

    def _poison_replay(self, error, *, phase, context=None):
        if self._replay_poison is None:
            ledger = dict(self._active_replay_flush or {})
            if context:
                ledger.update(context)
            self._replay_poison = {
                "proposal_token": repr(ledger.get("proposal_token")),
                "source_context_id": ledger.get("source_context_id"),
                "target_context_id": ledger.get("target_context_id"),
                "outcome": ledger.get("outcome"),
                "segment_id": ledger.get("segment_id"),
                "expected_staged_transitions": int(
                    ledger.get("expected_staged_transitions", 0)
                ),
                "expected_tensor_rows": int(ledger.get("expected_tensor_rows", 0)),
                "completed_tensor_rows": int(ledger.get("completed_tensor_rows", 0)),
                "failure_phase": str(phase),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
            }
        self._replay_flush_state = "poisoned"
        failure = self._replay_poison
        return ReplayStoragePoisoned(
            f"Replay storage became permanently poisoned after an ownership mutation: phase={failure['failure_phase']!r}, proposal_token={failure['proposal_token']}, completed_tensor_rows={failure['completed_tensor_rows']}/{failure['expected_tensor_rows']}. No retry or continued sampling is safe."
        )

    @property
    def ready(self):
        self._assert_replay_healthy("ready check")
        if self._buffer is None:
            return False
        if not self.streaming_enabled:
            return self._max_episode_length >= self._sequence_length
        return bool(
            any((int(length) >= 2 for length in self._history_episode_counts.values()))
        )

    @property
    def _recent_ready(self):
        if self._recent_slices == 0:
            return True
        return bool(
            any((int(length) >= 2 for length in self._recent_episode_counts.values()))
        )

    @property
    def total_transitions(self):
        return self._total_transitions

    @property
    def last_sample_valid_fraction(self):
        return float(self._last_sample_valid_fraction)

    @property
    def last_sample_unique_transition_ids(self):
        return tuple(self._last_sample_unique_transition_ids)

    @property
    def last_sample_real_loss_rows(self):
        return int(self._last_sample_real_loss_rows)

    @property
    def recent_effective_capacity(self):
        return self._recent_effective_capacity

    @property
    def recent_refresh_count(self):
        return self._recent_refresh_count

    def refresh_recent_segment(self):
        self._assert_replay_healthy("recent-segment refresh")
        if not self.streaming_enabled:
            raise RuntimeError(
                "refresh_recent_segment requires online replay mixture to be enabled."
            )
        if self._recent_buffer is not None:
            self._recent_buffer.empty()
        self._recent_episode_rows.clear()
        self._recent_episode_counts.clear()
        self._recent_trajectory_index.clear()
        self._recent_refresh_count += 1
        return self._recent_refresh_count

    def _reserve_buffer(self, storage, *, sampler, num_slices):
        return ReplayBuffer(
            storage=storage,
            sampler=sampler,
            pin_memory=False,
            prefetch=0,
            batch_size=(
                None
                if self.streaming_enabled
                else int(num_slices) * self._sequence_length
            ),
        )

    def _init(self, tds):
        print(f"Buffer capacity: {self._capacity:,}")
        mem_free, _ = torch.cuda.mem_get_info()
        bytes_per_step = sum(
            [
                (
                    v.numel() * v.element_size()
                    if not isinstance(v, TensorDict)
                    else sum([x.numel() * x.element_size() for x in v.values()])
                )
                for v in tds.values()
            ]
        ) / len(tds)
        total_bytes = bytes_per_step * self._capacity
        print(f"Storage required: {total_bytes / 1000000000.0:.2f} GB")
        storage_device = "cuda:0" if 2.5 * total_bytes < mem_free else "cpu"
        print(f"Using {storage_device.upper()} memory for storage.")
        self._storage_device = torch.device(storage_device)
        return self._reserve_buffer(
            LazyTensorStorage(self._capacity, device=self._storage_device),
            sampler=self._history_sampler,
            num_slices=max(1, self._history_slices),
        )

    def _init_recent(self):
        return self._reserve_buffer(
            LazyTensorStorage(
                self._recent_effective_capacity, device=self._storage_device
            ),
            sampler=self._recent_sampler,
            num_slices=max(1, self._recent_slices),
        )

    @staticmethod
    def _writer_cursor(replay):
        try:
            return int(replay._writer._cursor)
        except Exception as error:
            raise RuntimeError(
                "Could not read the online replay writer cursor."
            ) from error

    @staticmethod
    def _assert_writer_advanced(replay, before, capacity, *, name):
        expected = (int(before) + 1) % int(capacity)
        after = Buffer._writer_cursor(replay)
        if after != expected:
            raise RuntimeError(
                f"{name} replay writer cursor advanced unexpectedly: before={before}, expected_after={expected}, actual_after={after}."
            )

    def _extend_online_row(self, td):
        if self._buffer is None:
            self._buffer = self._init(td)
            if self._recent_slices > 0:
                self._recent_buffer = self._init_recent()
        history_position = self._writer_cursor(self._buffer)
        recent_position = (
            self._writer_cursor(self._recent_buffer)
            if self._recent_buffer is not None
            else None
        )
        self._buffer.extend(td)
        self._assert_writer_advanced(
            self._buffer, history_position, self._capacity, name="history"
        )
        if self._recent_buffer is not None:
            self._recent_buffer.extend(td)
            self._assert_writer_advanced(
                self._recent_buffer,
                recent_position,
                self._recent_effective_capacity,
                name="recent",
            )
        episode_id = int(td["episode"].reshape(-1)[0].item())
        self._history_trajectory_index.append(episode_id, history_position)
        self._track_episode_row(
            episode_id,
            queue=self._history_episode_rows,
            counts=self._history_episode_counts,
            capacity=self._capacity,
        )
        if self._recent_buffer is not None:
            self._recent_trajectory_index.append(episode_id, recent_position)
            self._track_episode_row(
                episode_id,
                queue=self._recent_episode_rows,
                counts=self._recent_episode_counts,
                capacity=self._recent_effective_capacity,
            )

    @staticmethod
    def _track_episode_row(episode_id, *, queue, counts, capacity):
        queue.append(int(episode_id))
        counts[int(episode_id)] += 1
        if len(queue) > int(capacity):
            removed = queue.popleft()
            counts[removed] -= 1
            if counts[removed] <= 0:
                del counts[removed]

    def begin_episode(self, td):
        self._assert_replay_healthy("episode begin")
        if not self.streaming_enabled:
            raise RuntimeError(
                "begin_episode is available only when online replay mixture is enabled."
            )
        if self._active_episode_id is not None:
            raise RuntimeError("Cannot begin an episode before ending the active one.")
        if len(td) != 1:
            raise ValueError("begin_episode expects exactly one initial replay row.")
        episode_id = int(self._num_eps)
        td["episode"] = torch.full_like(td["reward"], episode_id, dtype=torch.int64)
        pending_transaction = bool(self.replay_transaction_pending)
        if pending_transaction:
            raise RuntimeError(
                "Cannot begin an episode while a replay transaction is unresolved."
            )
        begin_context = {
            "proposal_token": None,
            "source_context_id": None,
            "target_context_id": None,
            "outcome": "episode_seed",
            "segment_id": None,
            "expected_staged_transitions": 0,
            "expected_tensor_rows": 1,
            "completed_tensor_rows": 0,
        }
        try:
            latest = td.clone()
        except Exception as error:
            if pending_transaction:
                raise self._poison_replay(
                    error,
                    phase="begin_pending_episode_seed_prepare",
                    context=begin_context,
                ) from error
            raise
        try:
            if pending_transaction:
                seed = td.clone()
                if any(
                    (
                        self._strict_single_replay_integer(value, "episode")
                        == episode_id
                        for value in self._pending_replay_seeds
                    )
                ):
                    raise RuntimeError(
                        "Pending replay already owns this physical episode seed."
                    )
                self._pending_replay_seeds.append(seed)
                if self._pending_replay_seed is None:
                    self._pending_replay_seed = seed
                if self._active_replay_flush is not None:
                    self._active_replay_flush["expected_tensor_rows"] = len(
                        self._pending_replay_seeds
                    ) + int(self._replay_transactions.staged_transition_count)
            else:
                self._extend_online_row(td)
                begin_context["completed_tensor_rows"] = 1
        except Exception as error:
            raise self._poison_replay(
                error,
                phase=(
                    "begin_pending_episode_seed"
                    if pending_transaction
                    else "begin_episode_storage"
                ),
                context=begin_context,
            ) from error
        self._active_episode_id = episode_id
        self._active_episode_length = int(len(td))
        self._latest_observation_row = latest
        self._max_episode_length = max(
            self._max_episode_length, self._active_episode_length
        )
        return self._active_episode_id

    def append_transition(self, td, *, proposal_token=None):
        self._assert_replay_healthy("transition append")
        if not self.streaming_enabled:
            raise RuntimeError(
                "append_transition requires online replay mixture to be enabled."
            )
        if self._active_episode_id is None:
            raise RuntimeError("No active episode; call begin_episode first.")
        if len(td) != 1:
            raise ValueError("append_transition expects exactly one causal replay row.")
        td["episode"] = torch.full_like(
            td["reward"], self._active_episode_id, dtype=torch.int64
        )
        if proposal_token is not None:
            raise RuntimeError("proposal_token requires transactional Context replay.")
        try:
            self._extend_online_row(td)
        except Exception as error:
            raise self._poison_replay(
                error,
                phase="append_online_row",
                context={
                    "proposal_token": None,
                    "episode_id": int(self._active_episode_id),
                    "completed_tensor_rows": 0,
                    "expected_tensor_rows": 1,
                },
            ) from error
        self._active_episode_length += 1
        self._total_transitions += 1
        self._max_episode_length = max(
            self._max_episode_length, self._active_episode_length
        )
        return self._total_transitions

    def _replay_trajectory_id(self, segment_id, trajectory_id):
        key = (int(segment_id), int(trajectory_id))
        value = self._replay_trajectory_ids.get(key)
        if value is None:
            value = self._next_replay_trajectory_id
            self._next_replay_trajectory_id += 1
            self._replay_trajectory_ids[key] = value
        return int(value)

    def _install_replay_envelope(
        self, td, *, context_id, segment_id, trajectory_id, sequence_id
    ):
        replay_trajectory = self._replay_trajectory_id(segment_id, trajectory_id)
        for key, value in (
            ("replay_context", int(context_id)),
            ("replay_segment", int(segment_id)),
            ("replay_sequence", int(sequence_id)),
            ("replay_trajectory", replay_trajectory),
        ):
            td[key] = torch.full_like(td["reward"], value, dtype=torch.int64)

    def _install_replay_envelope_from_row(self, td, row):
        self._install_replay_envelope(
            td,
            context_id=row.context_id,
            segment_id=row.segment_id,
            trajectory_id=row.trajectory_id,
            sequence_id=row.sequence_id,
        )

    @staticmethod
    def _strict_single_replay_integer(td, key):
        if not isinstance(td, TensorDict) or tuple(td.batch_size) != (1,):
            raise RuntimeError(
                "Pending replay rows must be one-row TensorDict instances."
            )
        value = td.get(key, None)
        if (
            not torch.is_tensor(value)
            or value.numel() != 1
            or value.dtype
            not in {torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}
        ):
            raise RuntimeError(f"Pending replay row has no scalar integral {key!r}.")
        integer = int(value.reshape(-1)[0].item())
        if integer < 0:
            raise RuntimeError(f"Pending replay row has a negative {key!r}.")
        return integer

    @staticmethod
    def _replay_tensor_schema(td):
        if not isinstance(td, TensorDict) or tuple(td.batch_size) != (1,):
            raise RuntimeError(
                "Pending replay rows must be one-row TensorDict instances."
            )
        if not torch.is_tensor(td.get("reward", None)):
            raise RuntimeError("Pending replay row is missing tensor reward.")
        schema = {}
        for key in td.keys(include_nested=True, leaves_only=True):
            value = td.get(key)
            if not torch.is_tensor(value):
                raise RuntimeError(f"Pending replay leaf {key!r} is not a tensor.")
            if value.ndim < 1 or int(value.shape[0]) != 1:
                raise RuntimeError(
                    f"Pending replay leaf {key!r} has no one-row batch axis."
                )
            schema[key] = (tuple(value.shape), value.dtype, value.device.type)
        return schema

    def _preflight_pending_resolution(
        self, proposal_token, *, context_committed, target_context_id
    ):
        core = self._replay_transactions
        if core is None:
            raise RuntimeError("Transactional replay has no ownership core.")
        snapshot = core.pending_snapshot
        if snapshot is None:
            raise RuntimeError("No replay ownership transaction is pending.")
        if proposal_token != snapshot.proposal_token:
            raise RuntimeError("Replay proposal token does not match pending owner.")
        if not isinstance(context_committed, bool):
            raise RuntimeError("context_committed must be boolean.")
        if context_committed:
            if (
                isinstance(target_context_id, bool)
                or not isinstance(target_context_id, int)
                or target_context_id < 0
            ):
                raise RuntimeError(
                    "Confirmed replay requires a non-negative integer target Context."
                )
            target = int(target_context_id)
        else:
            if target_context_id is not None:
                raise RuntimeError(
                    "Dismissed replay must return to its captured source Context."
                )
            target = int(snapshot.source_context_id)
        pending_seeds = list(self._pending_replay_seeds)
        if not pending_seeds and self._pending_replay_seed is not None:
            pending_seeds = [self._pending_replay_seed]
        if not pending_seeds:
            raise RuntimeError("Pending replay transaction has no causal seed row.")
        if len(self._pending_replay_payloads) != snapshot.staged_count:
            raise RuntimeError("Replay transaction payload/stage length mismatch.")
        if not core.transition_conservation_holds:
            raise RuntimeError("Replay transition conservation failed before flush.")
        if core.observed_transition_count != self._total_transitions:
            raise RuntimeError("Buffer/core transition counts diverged before flush.")
        seed_episodes = [
            self._strict_single_replay_integer(seed, "episode")
            for seed in pending_seeds
        ]
        if len(set(seed_episodes)) != len(seed_episodes):
            raise RuntimeError("Pending replay contains duplicate trajectory seeds.")
        seed_episode_set = set(seed_episodes)
        staged_rows = tuple(snapshot.staged_rows)
        previous_sequence_id = None
        for payload, row in zip(self._pending_replay_payloads, staged_rows):
            if not isinstance(row, StagedReplayRow):
                raise RuntimeError("Replay core returned an invalid staged row.")
            if int(row.segment_id) != int(snapshot.segment_id):
                raise RuntimeError("Pending replay row escaped its segment owner.")
            if int(row.trajectory_id) not in seed_episode_set:
                raise RuntimeError(
                    "Pending replay row has no causal physical-trajectory seed."
                )
            if (
                previous_sequence_id is not None
                and int(row.sequence_id) != previous_sequence_id + 1
            ):
                raise RuntimeError("Pending replay sequence ids are not contiguous.")
            previous_sequence_id = int(row.sequence_id)
            if self._strict_single_replay_integer(payload, "episode") != int(
                row.trajectory_id
            ):
                raise RuntimeError(
                    "Pending replay payload/row physical trajectories diverged."
                )
        prepared_seeds = []
        for seed, seed_episode in zip(pending_seeds, seed_episodes):
            prepared_seed = seed.clone()
            self._install_replay_envelope(
                prepared_seed,
                context_id=target,
                segment_id=snapshot.segment_id,
                trajectory_id=seed_episode,
                sequence_id=-1,
            )
            prepared_seeds.append(prepared_seed)
        prepared_payloads = []
        for payload, row in zip(self._pending_replay_payloads, staged_rows):
            prepared = payload.clone()
            self._install_replay_envelope(
                prepared,
                context_id=target,
                segment_id=row.segment_id,
                trajectory_id=row.trajectory_id,
                sequence_id=row.sequence_id,
            )
            prepared_payloads.append(prepared)
        reference_schema = self._replay_tensor_schema(prepared_seeds[0])
        for prepared in (*prepared_seeds[1:], *prepared_payloads):
            if self._replay_tensor_schema(prepared) != reference_schema:
                raise RuntimeError(
                    "Pending replay TensorDict schema differs from its causal seed."
                )
        return {
            "snapshot": snapshot,
            "target_context_id": target,
            "prepared_seed": prepared_seeds[0],
            "prepared_seeds": tuple(prepared_seeds),
            "prepared_payloads": tuple(prepared_payloads),
        }

    def commit_context_switch_at_boundary(
        self, *, source_context_id, target_context_id
    ):
        self._assert_replay_healthy("physical-boundary Context switch")
        raise RuntimeError("transactional Context replay is disabled.")

    def commit_context_switch_after_transition(
        self, td, *, source_context_id, target_context_id
    ):
        self._assert_replay_healthy("row-bearing Context switch")
        raise RuntimeError("transactional Context replay is disabled.")

    def begin_pending_transaction(self, proposal_token, *, source_context_id):
        self._assert_replay_healthy("transaction begin")
        raise RuntimeError("transactional Context replay is disabled.")

    def resolve_pending_transaction(
        self, proposal_token, *, context_committed, target_context_id=None
    ):
        self._assert_replay_healthy("transaction resolution")
        raise RuntimeError("transactional Context replay is disabled.")

    def end_episode(self):
        self._assert_replay_healthy("episode end")
        if not self.streaming_enabled:
            raise RuntimeError(
                "end_episode requires online replay mixture to be enabled."
            )
        if self._active_episode_id is None:
            raise RuntimeError("No active episode to end.")
        self._active_episode_id = None
        self._active_episode_length = 0
        self._num_eps += 1
        return self._num_eps

    @staticmethod
    def _context_state_cpu_clone(value):
        if isinstance(value, TensorDict):
            return value.detach().clone().cpu()
        if torch.is_tensor(value):
            return value.detach().clone().cpu()
        if isinstance(value, dict):
            return {
                key: Buffer._context_state_cpu_clone(item)
                for (key, item) in value.items()
            }
        if isinstance(value, list):
            return [Buffer._context_state_cpu_clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple((Buffer._context_state_cpu_clone(item) for item in value))
        return deepcopy(value)

    def state_dict(self):
        self._assert_replay_healthy("Context replay snapshot")
        if (
            self.replay_transaction_pending
            or self._pending_replay_seed is not None
            or self._pending_replay_seeds
            or self._pending_replay_payloads
        ):
            raise RuntimeError(
                "Cannot snapshot replay with a pending ownership transaction."
            )
        return {
            "schema": "m3wm-context-buffer-v1",
            "capacity": int(self._capacity),
            "recent_capacity": int(self._recent_effective_capacity),
            "sequence_length": int(self._sequence_length),
            "transactional": False,
            "history": (
                None
                if self._buffer is None
                else self._context_state_cpu_clone(self._buffer.state_dict())
            ),
            "recent": (
                None
                if self._recent_buffer is None
                else self._context_state_cpu_clone(self._recent_buffer.state_dict())
            ),
            "history_index": self._history_trajectory_index.state_dict(),
            "recent_index": self._recent_trajectory_index.state_dict(),
            "num_eps": int(self._num_eps),
            "active_episode_id": self._active_episode_id,
            "active_episode_length": int(self._active_episode_length),
            "max_episode_length": int(self._max_episode_length),
            "total_transitions": int(self._total_transitions),
            "history_episode_rows": tuple(self._history_episode_rows),
            "history_episode_counts": dict(self._history_episode_counts),
            "recent_episode_rows": tuple(self._recent_episode_rows),
            "recent_episode_counts": dict(self._recent_episode_counts),
            "recent_refresh_count": int(self._recent_refresh_count),
            "latest_observation_row": self._context_state_cpu_clone(
                self._latest_observation_row
            ),
            "replay_trajectory_ids": dict(self._replay_trajectory_ids),
            "next_replay_trajectory_id": int(self._next_replay_trajectory_id),
            "last_sample_composition": dict(self._last_sample_composition),
            "last_sample_valid_fraction": float(self._last_sample_valid_fraction),
        }

    def load_state_dict(self, state):
        if state.get("schema") != "m3wm-context-buffer-v1":
            raise ValueError("Unsupported Context replay state schema.")
        for key, observed, expected in (
            ("capacity", state.get("capacity"), self._capacity),
            (
                "recent_capacity",
                state.get("recent_capacity"),
                self._recent_effective_capacity,
            ),
            ("sequence_length", state.get("sequence_length"), self._sequence_length),
        ):
            if int(observed) != int(expected):
                raise ValueError(
                    f"Context replay {key} mismatch: saved={observed}, current={expected}."
                )
        if bool(state.get("transactional")) != False:
            raise ValueError("Context replay transactional mode mismatch.")
        if self._buffer is not None or self._recent_buffer is not None:
            raise RuntimeError(
                "Context replay import requires an empty Buffer instance."
            )
        self._storage_device = torch.device("cpu")
        history = state.get("history")
        if history is not None:
            self._buffer = self._reserve_buffer(
                LazyTensorStorage(self._capacity, device="cpu"),
                sampler=self._history_sampler,
                num_slices=max(1, self._history_slices),
            )
            self._buffer.load_state_dict(history)
            self._restore_torchrl_storage_metadata(
                self._buffer, self._capacity, name="history"
            )
        recent = state.get("recent")
        if recent is not None:
            self._recent_buffer = self._reserve_buffer(
                LazyTensorStorage(self._recent_effective_capacity, device="cpu"),
                sampler=self._recent_sampler,
                num_slices=max(1, self._recent_slices),
            )
            self._recent_buffer.load_state_dict(recent)
            self._restore_torchrl_storage_metadata(
                self._recent_buffer, self._recent_effective_capacity, name="recent"
            )
        self._history_trajectory_index.load_state_dict(state["history_index"])
        self._recent_trajectory_index.load_state_dict(state["recent_index"])
        self._num_eps = int(state["num_eps"])
        self._active_episode_id = state.get("active_episode_id")
        self._active_episode_length = int(state["active_episode_length"])
        self._max_episode_length = int(state["max_episode_length"])
        self._total_transitions = int(state["total_transitions"])
        self._history_episode_rows = deque(state["history_episode_rows"])
        self._history_episode_counts = Counter(state["history_episode_counts"])
        self._recent_episode_rows = deque(state["recent_episode_rows"])
        self._recent_episode_counts = Counter(state["recent_episode_counts"])
        self._recent_refresh_count = int(state["recent_refresh_count"])
        self._latest_observation_row = self._context_state_cpu_clone(
            state.get("latest_observation_row")
        )
        self._replay_trajectory_ids = {
            tuple(key): int(value)
            for (key, value) in state["replay_trajectory_ids"].items()
        }
        self._next_replay_trajectory_id = int(state["next_replay_trajectory_id"])
        self._last_sample_composition = dict(state["last_sample_composition"])
        self._last_sample_valid_fraction = float(state["last_sample_valid_fraction"])
        return self

    @staticmethod
    def _restore_torchrl_storage_metadata(replay, capacity, *, name):
        lazy_storage = getattr(replay, "_storage", None)
        storage = getattr(lazy_storage, "_storage", None)
        if not isinstance(storage, TensorDict):
            raise RuntimeError(
                f"Restored {name} replay has no TensorDict backing storage."
            )
        capacity = int(capacity)
        for key, value in storage.items(include_nested=True, leaves_only=True):
            if (
                not torch.is_tensor(value)
                or value.ndim == 0
                or int(value.shape[0]) != capacity
            ):
                raise RuntimeError(
                    f"Restored {name} replay leaf {key!r} has invalid capacity axis."
                )
        storage.batch_size = torch.Size([capacity])
        logical_length = int(len(replay))
        if logical_length:
            cursor = getattr(getattr(replay, "_writer", None), "_cursor", None)
            if torch.is_tensor(cursor):
                if cursor.numel() != 1:
                    raise RuntimeError(
                        f"Restored {name} replay has a non-scalar writer cursor."
                    )
                cursor = int(cursor.reshape(-1)[0].item())
            elif cursor is None:
                raise RuntimeError(f"Restored {name} replay has no writer cursor.")
            else:
                cursor = int(cursor)
            lazy_storage._last_cursor = torch.tensor(
                [(cursor - 1) % capacity], dtype=torch.long
            )

    def load(self, td):
        self._assert_replay_healthy("bulk replay load")
        if self.streaming_enabled:
            raise RuntimeError(
                "Bulk replay loading is forbidden in causal online replay mode."
            )
        num_new_eps = len(td)
        episode_idx = torch.arange(
            self._num_eps, self._num_eps + num_new_eps, dtype=torch.int64
        )
        td["episode"] = episode_idx.unsqueeze(-1).expand(-1, td["reward"].shape[1])
        if self._num_eps == 0:
            self._buffer = self._init(td[0])
            if self.streaming_enabled:
                self._recent_buffer = self._init_recent()
        td = td.reshape(td.shape[0] * td.shape[1])
        self._buffer.extend(td)
        if self.streaming_enabled:
            self._recent_buffer.extend(td)
        self._num_eps += num_new_eps
        if num_new_eps:
            max_length = int(td.shape[0] // num_new_eps)
            self._max_episode_length = max(self._max_episode_length, max_length)
        return self._num_eps

    def add(self, td):
        self._assert_replay_healthy("episode add")
        if self.streaming_enabled:
            raise RuntimeError(
                "Whole-episode insertion is forbidden in causal online replay mode because it would double-count streamed transitions; use begin_episode/append_transition/end_episode."
            )
        td["episode"] = torch.full_like(td["reward"], self._num_eps, dtype=torch.int64)
        if self._num_eps == 0:
            self._buffer = self._init(td)
            if self.streaming_enabled:
                self._recent_buffer = self._init_recent()
        self._buffer.extend(td)
        self._num_eps += 1
        self._max_episode_length = max(self._max_episode_length, int(len(td)))
        self._total_transitions += max(0, int(len(td)) - 1)
        return self._num_eps

    def _prepare_batch(self, td, valid_mask=None):
        origin_ids = td.get("origin_transition_id", None)
        if torch.is_tensor(origin_ids) and origin_ids.ndim >= 2:
            origin_ids = origin_ids.squeeze(-1) if origin_ids.ndim == 3 else origin_ids
            transition_ids = origin_ids[1:].to(dtype=torch.int64)
            if valid_mask is None:
                real_mask = torch.ones_like(transition_ids, dtype=torch.bool)
            else:
                mask = valid_mask.to(device=transition_ids.device, dtype=torch.bool)
                real_mask = mask[:-1] & mask[1:]
            real_mask = real_mask & (transition_ids >= 0)
            real_values = transition_ids[real_mask].detach().cpu()
            self._last_sample_unique_transition_ids = tuple(
                sorted({int(value) for value in real_values.tolist()})
            )
            self._last_sample_real_loss_rows = int(real_mask.sum().item())
        else:
            self._last_sample_unique_transition_ids = ()
            self._last_sample_real_loss_rows = 0
        td = td.select(
            "obs",
            "action",
            "reward",
            "terminated",
            "task",
            "context_severity",
            strict=False,
        ).to(self._device, non_blocking=True)
        obs = td.get("obs").contiguous()
        if valid_mask is not None:
            valid_mask = valid_mask.to(self._device, non_blocking=True)
            if valid_mask.ndim != 2 or tuple(valid_mask.shape) != tuple(obs.shape[:2]):
                raise RuntimeError(
                    f"Online replay valid mask must have shape [time, batch], received {tuple(valid_mask.shape)} for obs {tuple(obs.shape)}."
                )
            valid_mask = valid_mask.to(dtype=torch.bool).contiguous()
            self._last_sample_valid_fraction = float(valid_mask.float().mean().item())
        else:
            self._last_sample_valid_fraction = 1.0
        action = td.get("action")[1:].contiguous()
        reward = td.get("reward")[1:].unsqueeze(-1).contiguous()
        terminated = td.get("terminated", None)
        if terminated is not None:
            terminated = td.get("terminated")[1:].unsqueeze(-1).contiguous()
        else:
            terminated = torch.zeros_like(reward)
        task = td.get("task", None)
        if task is not None:
            task = task[0].contiguous()
        context_severity = td.get("context_severity", None)
        if context_severity is not None:
            context_severity = context_severity[1:].contiguous()
        if valid_mask is not None:
            transition_mask = (valid_mask[:-1] & valid_mask[1:]).unsqueeze(-1)
            action = torch.where(transition_mask, action, torch.zeros_like(action))
            reward = torch.where(transition_mask, reward, torch.zeros_like(reward))
            terminated = torch.where(
                transition_mask, terminated, torch.ones_like(terminated)
            )
            return (obs, action, reward, terminated, task, context_severity, valid_mask)
        return (obs, action, reward, terminated, task, context_severity)

    def sample(self):
        self._assert_replay_healthy("sample")
        batch = self._sample_once()
        True
        return batch

    def _sample_replay(self, replay, slices):
        count = int(slices)
        if count <= 0:
            raise ValueError("sample count must be positive")
        flat, info = replay.sample(
            batch_size=count * self._sequence_length, return_info=True
        )
        try:
            td = flat.view(count, self._sequence_length).permute(1, 0)
        except RuntimeError as error:
            raise RuntimeError(
                f"Online replay sampler returned an unexpected TensorDict shape: {tuple(flat.shape)}; expected {(count * self._sequence_length,)}."
            ) from error
        mask = None
        mask = info.get(("collector", "mask"))
        if mask is None:
            mask = None
        else:
            mask = mask.reshape(count, self._sequence_length).permute(1, 0)
        return (td, mask)

    def _sample_once(self):
        if not self.ready:
            raise RuntimeError(
                f"Replay does not yet contain one causal transition (sequence rows={self._sequence_length})."
            )
        if not self.streaming_enabled:
            td = self._buffer.sample().view(-1, self._sequence_length).permute(1, 0)
            self._last_sample_composition = {
                "history_slices": int(self.cfg.batch_size),
                "recent_slices": 0,
            }
            return self._prepare_batch(td)
        if self._recent_slices > 0 and (not self._recent_ready):
            td, valid_mask = self._sample_replay(self._buffer, int(self.cfg.batch_size))
            self._last_sample_composition = {
                "history_slices": int(self.cfg.batch_size),
                "recent_slices": 0,
            }
            return self._prepare_batch(td, valid_mask)
        parts = []
        masks = []
        if self._history_slices > 0:
            td, mask = self._sample_replay(self._buffer, self._history_slices)
            parts.append(td.permute(1, 0))
            masks.append(mask)
        if self._recent_slices > 0:
            td, mask = self._sample_replay(self._recent_buffer, self._recent_slices)
            parts.append(td.permute(1, 0))
            masks.append(mask)
        if not parts:
            raise RuntimeError("Online replay mixture produced an empty batch.")
        td = (parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)).permute(1, 0)
        valid_mask = None
        if masks:
            if any((mask is not None for mask in masks)):
                normalized_masks = [
                    (
                        mask
                        if mask is not None
                        else torch.ones(
                            self._sequence_length,
                            int(part.shape[0]),
                            dtype=torch.bool,
                            device=part.device,
                        )
                    )
                    for (part, mask) in zip(parts, masks)
                ]
                valid_mask = (
                    normalized_masks[0]
                    if len(normalized_masks) == 1
                    else torch.cat(normalized_masks, dim=1)
                )
        self._last_sample_composition = {
            "history_slices": self._history_slices,
            "recent_slices": self._recent_slices,
        }
        return self._prepare_batch(td, valid_mask)
