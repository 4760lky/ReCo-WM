from __future__ import annotations
import copy
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
import numpy as np
import torch
from common.m3wm_context_agent_bundle import ContextAgentBundle, _clone
from common.m3wm_context_replay_bank import ContextReplayBank


@dataclass(frozen=True)
class ContextSwitchReceipt:
    source_context_id: int
    target_context_id: int
    reason: str
    source_bundle_hash: str
    target_bundle_hash: str
    runtime_segment_id: int
    target_loaded_from_disk: bool
    switch_wall_ms: float


class ContextAgentBank:
    schema = "m3wm-context-agent-bank-v1"

    def __init__(
        self,
        *,
        initial_bundle: ContextAgentBundle,
        bundle_factory: (
            Callable[[int, ContextAgentBundle, bool], ContextAgentBundle] | None
        ) = None,
        storage_dir: str | Path | None = None,
        replay_bank: ContextReplayBank | None = None,
        release_inactive: bool = False,
    ) -> None:
        if int(initial_bundle.context_id) != 0:
            raise ValueError("Initial Context must have id 0")
        self._bundles: dict[int, ContextAgentBundle] = {0: initial_bundle}
        self._active_context_id = 0
        initial_bundle.active = True
        self._next_context_id = 1
        self._bundle_factory = bundle_factory
        self.storage_dir = None if storage_dir is None else Path(storage_dir)
        self.replay_bank = replay_bank or ContextReplayBank()
        self.release_inactive = bool(release_inactive)
        self._global_env_step = 0
        self._switch_receipts: list[ContextSwitchReceipt] = []
        self._pending_init_capsule: dict[str, Any] | None = None
        self._init_capsule_records: list[dict[str, Any]] = []

    @property
    def active_context_id(self) -> int:
        return self._active_context_id

    @property
    def active_bundle(self) -> ContextAgentBundle:
        return self._bundles[self._active_context_id]

    def bundle(self, context_id: int) -> ContextAgentBundle:
        try:
            return self._bundles[int(context_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown Context {context_id}") from exc

    def context_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._bundles))

    def discard_contexts_from(
        self, first_context_id: int, *, reason: str = "diagnostic_topology_reset"
    ) -> tuple[int, ...]:
        first = int(first_context_id)
        if first <= 0:
            raise ValueError("cannot discard Context 0 or use a negative Context id")
        discarded = tuple(
            (context_id for context_id in self.context_ids() if context_id >= first)
        )
        if not discarded:
            self._next_context_id = first
            return discarded
        if self._active_context_id >= first:
            self.activate(0, reason=str(reason))
        for context_id in discarded:
            bundle = self._bundles.pop(context_id)
            bundle.active = False
            if bundle.materialized:
                bundle.release_runtime()
            path = self._bundle_path(context_id)
            if path is not None and path.is_file():
                path.unlink()
        self.replay_bank.discard_contexts_from(first)
        self._next_context_id = first
        return discarded

    def _persist(self, bundle: ContextAgentBundle, *, reason: str) -> str:
        if self.storage_dir is None:
            return str(bundle.export_train_state()["bundle_hash"])
        path = self.storage_dir / f"context_{bundle.context_id:04d}.pt"
        bundle.metadata.update(
            {
                "last_persist_reason": str(reason),
                "last_persist_step": int(self._global_env_step),
            }
        )
        return bundle.save_atomic(path)

    def _bundle_path(self, context_id: int) -> Path | None:
        if self.storage_dir is None:
            return None
        return self.storage_dir / f"context_{int(context_id):04d}.pt"

    def trial_working_copy(
        self, context_id: int
    ) -> tuple[ContextAgentBundle, str, str]:
        candidate_id = int(context_id)
        if candidate_id == self._active_context_id:
            raise ValueError("trial candidate must be historical")
        if candidate_id not in self._bundles:
            raise KeyError(f"Unknown Context {candidate_id}")
        if self._bundle_factory is None:
            raise RuntimeError("trial copy requires a reconstruction factory")
        source = self.active_bundle
        source_hash = self._persist(source, reason="trial_source_fix")
        candidate_path = self._bundle_path(candidate_id)
        canonical = self._bundles[candidate_id]
        if candidate_path is not None and candidate_path.is_file():
            payload = ContextAgentBundle.load_payload(candidate_path)
        elif canonical.materialized:
            payload = canonical.export_train_state()
        else:
            raise RuntimeError("trial candidate has no valid durable bundle")
        working = self._bundle_factory(candidate_id, source, False)
        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_cpu_rng = torch.random.get_rng_state()
        torch_cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            working.import_train_state(payload, restore_rng=True)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.random.set_rng_state(torch_cpu_rng)
            if torch_cuda_rng is not None:
                torch.cuda.set_rng_state_all(torch_cuda_rng)
        working.active = False
        working.reset_activation_transients()
        return (working, str(source_hash), str(payload["bundle_hash"]))

    def set_global_env_step(self, step: int) -> None:
        if int(step) < self._global_env_step:
            raise ValueError("global environment step cannot move backwards")
        self._global_env_step = int(step)

    def _record_init_capsule(
        self, event: str, capsule: Mapping[str, Any], **extra
    ) -> None:
        record = {
            "schema": "m3wm-context-init-capsule-event-v1",
            "event": str(event),
            "global_env_step": int(self._global_env_step),
            "source_context_id": int(capsule["source_context_id"]),
            "nomination_transition_id": int(capsule["nomination_transition_id"]),
            "transaction_id": str(capsule["transaction_id"]),
            "source_revision": int(capsule["source_revision"]),
            "capsule_hash": str(capsule["capsule_hash"]),
            "component_hashes": copy.deepcopy(capsule["component_hashes"]),
            "captured_control_updates": int(capsule["captured_control_updates"]),
            "captured_theta_updates": int(capsule["captured_theta_updates"]),
            **extra,
        }
        self._init_capsule_records.append(record)
        if self.storage_dir is not None:
            path = self.storage_dir.parent / "context_agent_init_capsules.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def capture_pending_init_capsule(
        self,
        *,
        source_context_id: int,
        nomination_transition_id: int,
        transaction_id: str,
        source_revision: int,
    ) -> dict[str, Any]:
        source_id = int(source_context_id)
        if source_id != self._active_context_id:
            raise RuntimeError("NEW initialization source is no longer active")
        existing = self._pending_init_capsule
        if existing is not None:
            if (
                str(existing["transaction_id"]) == str(transaction_id)
                and int(existing["source_context_id"]) == source_id
                and (
                    int(existing["nomination_transition_id"])
                    == int(nomination_transition_id)
                )
                and (int(existing["source_revision"]) == int(source_revision))
            ):
                return _clone(existing)
            raise RuntimeError(
                "a different NEW initialization capsule is already pending"
            )
        capture = getattr(self._bundle_factory, "capture_init_capsule", None)
        if not callable(capture):
            raise RuntimeError(
                "bundle factory cannot capture a NEW initialization capsule"
            )
        capsule = capture(
            self.active_bundle,
            nomination_transition_id=int(nomination_transition_id),
            transaction_id=str(transaction_id),
            source_revision=int(source_revision),
        )
        self._pending_init_capsule = _clone(capsule)
        self._record_init_capsule("CAPTURED", capsule)
        return _clone(capsule)

    def discard_pending_init_capsule(
        self, *, reason: str, transaction_id: str | None = None
    ) -> bool:
        capsule = self._pending_init_capsule
        if capsule is None:
            return False
        if transaction_id is not None and str(capsule["transaction_id"]) != str(
            transaction_id
        ):
            raise RuntimeError(
                "cannot discard a different NEW initialization transaction"
            )
        self._record_init_capsule("DISCARDED", capsule, reason=str(reason))
        self._pending_init_capsule = None
        return True

    def pending_init_capsule_record(self) -> dict[str, Any] | None:
        capsule = self._pending_init_capsule
        if capsule is None:
            return None
        return {
            "source_context_id": int(capsule["source_context_id"]),
            "nomination_transition_id": int(capsule["nomination_transition_id"]),
            "transaction_id": str(capsule["transaction_id"]),
            "source_revision": int(capsule["source_revision"]),
            "capsule_hash": str(capsule["capsule_hash"]),
            "component_hashes": copy.deepcopy(capsule["component_hashes"]),
            "captured_control_updates": int(capsule["captured_control_updates"]),
            "captured_theta_updates": int(capsule["captured_theta_updates"]),
        }

    def restore_pending_init_capsule(self, payload: Mapping[str, Any] | None) -> None:
        if payload is None:
            self._pending_init_capsule = None
            return
        validate = getattr(self._bundle_factory, "validate_init_capsule", None)
        if not callable(validate):
            raise RuntimeError("bundle factory cannot validate a restored init capsule")
        validate(payload)
        if int(payload["source_context_id"]) != self._active_context_id:
            raise ValueError("restored initialization capsule source is not active")
        self._pending_init_capsule = _clone(payload)

    def new_context(
        self,
        *,
        reason: str = "NEW",
        init_transaction_id: str | None = None,
        source_revision: int | None = None,
    ) -> ContextAgentBundle:
        source = self.active_bundle
        context_id = self._next_context_id
        capsule = None
        if init_transaction_id is not None:
            capsule = self._pending_init_capsule
            if capsule is None:
                raise RuntimeError("confirmed NEW has no initialization capsule")
            if str(capsule["transaction_id"]) != str(init_transaction_id):
                raise RuntimeError("NEW initialization transaction mismatch")
            if int(capsule["source_context_id"]) != int(source.context_id):
                raise RuntimeError("NEW initialization source Context mismatch")
            if source_revision is None or int(capsule["source_revision"]) != int(
                source_revision
            ):
                raise RuntimeError("NEW initialization source revision mismatch")
        if self._bundle_factory is None:
            bundle = copy.deepcopy(source)
            bundle.context_id = context_id
            bundle.active = False
            bundle.runtime_segment_id = source.runtime_segment_id + 1
        elif capsule is None:
            bundle = self._bundle_factory(context_id, source, True)
        else:
            bundle = self._bundle_factory(
                context_id, source, True, init_payload=_clone(capsule)
            )
        if int(bundle.context_id) != context_id:
            raise ValueError("bundle factory returned the wrong Context id")
        self._bundles[context_id] = bundle
        self._next_context_id += 1
        return bundle

    def create_and_activate(
        self,
        *,
        reason: str = "NEW",
        init_transaction_id: str | None = None,
        source_revision: int | None = None,
    ) -> tuple[ContextAgentBundle, ContextSwitchReceipt]:
        expected_id = self._next_context_id
        bundle = self.new_context(
            reason=reason,
            init_transaction_id=init_transaction_id,
            source_revision=source_revision,
        )
        try:
            receipt = self.activate(bundle.context_id, reason=reason)
        except Exception:
            failed = self._bundles.pop(expected_id, None)
            if failed is not None and failed.materialized:
                failed.active = False
            self._next_context_id = expected_id
            raise
        if init_transaction_id is not None:
            capsule = self._pending_init_capsule
            if capsule is None:
                raise RuntimeError(
                    "NEW initialization capsule disappeared before commit"
                )
            self._record_init_capsule(
                "CONSUMED",
                capsule,
                child_context_id=int(self.active_bundle.context_id),
                context_born_step=int(self._global_env_step),
                live_parent_init_hash=self.active_bundle.metadata.get(
                    "live_parent_init_hash"
                ),
                child_schedule_offset=int(
                    getattr(
                        self.active_bundle.control_agent,
                        "_m3wm_schedule_update_offset",
                        0,
                    )
                ),
                child_main_replay_count=0,
                child_theta_replay_count=0,
            )
            self._pending_init_capsule = None
        return (self.active_bundle, receipt)

    def activate(
        self, target_context_id: int, *, reason: str = "RETURN"
    ) -> ContextSwitchReceipt:
        started = time.perf_counter()
        target = int(target_context_id)
        if target not in self._bundles:
            raise KeyError(f"Unknown Context {target}")
        source = self._active_context_id
        if target == source:
            raise ValueError("Context activation requires a different target")
        source_bundle = self._bundles[source]
        target_bundle = self._bundles[target]
        source_payload = source_bundle.export_train_state()
        target_before = (
            target_bundle.export_train_state() if target_bundle.materialized else None
        )
        source_hash = self._persist(source_bundle, reason=f"leave_{reason}")
        target_path = self._bundle_path(target)
        loaded_from_disk = bool(target_path is not None and target_path.is_file())
        if loaded_from_disk:
            target_payload = ContextAgentBundle.load_payload(target_path)
        elif target_before is not None:
            target_payload = target_before
        else:
            raise RuntimeError("Offloaded target Context has no durable bundle")
        target_hash = str(target_payload["bundle_hash"])
        replay_before = self.replay_bank.state_dict()
        restored_bundle = target_bundle
        if not target_bundle.materialized:
            if self._bundle_factory is None:
                raise RuntimeError(
                    "Offloaded Context requires a reconstruction factory"
                )
            restored_bundle = self._bundle_factory(target, source_bundle, False)
            if int(restored_bundle.context_id) != target:
                raise ValueError("bundle factory reconstructed the wrong Context id")
        try:
            restored_bundle.import_train_state(target_payload, restore_rng=True)
            restored_bundle.reset_activation_transients()
            source_bundle.active = False
            restored_bundle.active = True
            restored_bundle.runtime_segment_id += 1
            replay_receipt = self.replay_bank.switch(target)
            self._bundles[target] = restored_bundle
            self._active_context_id = target
        except Exception:
            if target_before is not None:
                target_bundle.import_train_state(target_before, restore_rng=False)
            source_bundle.import_train_state(source_payload, restore_rng=True)
            source_bundle.active = True
            if target_bundle.materialized:
                target_bundle.active = False
            self.replay_bank.load_state_dict(replay_before)
            self._active_context_id = source
            raise
        if self.release_inactive:
            source_bundle.release_runtime()
        receipt = ContextSwitchReceipt(
            source_context_id=source,
            target_context_id=target,
            reason=str(reason),
            source_bundle_hash=source_hash,
            target_bundle_hash=target_hash,
            runtime_segment_id=int(replay_receipt["segment_id"]),
            target_loaded_from_disk=loaded_from_disk,
            switch_wall_ms=(time.perf_counter() - started) * 1000.0,
        )
        self._switch_receipts.append(receipt)
        return receipt

    def append_transition(
        self,
        payload: Any,
        *,
        behavior_context_id: int,
        episode_id: int,
        assigned_context_id: int | None = None,
        terminated: bool = False,
        truncated: bool = False,
    ):
        return self.replay_bank.append(
            payload,
            behavior_context_id=behavior_context_id,
            assigned_context_id=(
                self._active_context_id
                if assigned_context_id is None
                else assigned_context_id
            ),
            episode_id=episode_id,
            terminated=terminated,
            truncated=truncated,
        )

    def receipts(self) -> tuple[ContextSwitchReceipt, ...]:
        return tuple(self._switch_receipts)

    def finalize(self) -> Path | None:
        active_hash = self._persist(self.active_bundle, reason="run_finalize")
        if self.storage_dir is None:
            return None
        entries = []
        for context_id in self.context_ids():
            path = self._bundle_path(context_id)
            if path is None or not path.is_file():
                raise RuntimeError(f"Context {context_id} has no durable final bundle")
            payload = ContextAgentBundle.load_payload(path)
            entries.append(
                {
                    "context_id": int(context_id),
                    "path": str(path),
                    "bundle_hash": str(payload["bundle_hash"]),
                    "bytes": int(path.stat().st_size),
                    "active": bool(context_id == self._active_context_id),
                    "last_persist_step": int(
                        payload.get("metadata", {}).get("last_persist_step", -1)
                    ),
                }
            )
        manifest = {
            "schema": "m3wm-context-agent-final-manifest-v1",
            "global_env_step": int(self._global_env_step),
            "active_context_id": int(self._active_context_id),
            "active_bundle_hash": str(active_hash),
            "contexts": entries,
        }
        target = self.storage_dir.parent / "final_manifest.json"
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
        return target

    def state_dict(self) -> dict[str, Any]:
        bundles = {}
        for context, bundle in self._bundles.items():
            if bundle.materialized:
                bundles[context] = bundle.export_train_state()
                continue
            path = self._bundle_path(context)
            if path is None or not path.is_file():
                raise RuntimeError(f"Offloaded Context {context} has no durable bundle")
            bundles[context] = ContextAgentBundle.load_payload(path)
        return {
            "schema": self.schema,
            "active_context_id": self._active_context_id,
            "next_context_id": self._next_context_id,
            "global_env_step": self._global_env_step,
            "bundles": bundles,
            "replay_bank": self.replay_bank.state_dict(),
            "receipts": tuple(self._switch_receipts),
            "pending_init_capsule": _clone(self._pending_init_capsule),
            "init_capsule_records": copy.deepcopy(self._init_capsule_records),
        }
