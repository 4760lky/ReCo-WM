from __future__ import annotations
import json
import pytest
import torch
from common.m3wm_context_agent_bank import ContextAgentBank
from common.m3wm_context_agent_bundle import ContextAgentBundle
from common.m3wm_context_replay_bank import ContextReplayBank


class ToyAgent(torch.nn.Module):

    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.model = torch.nn.Linear(3, 2)
        self.optim = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)
        self.pi_optim = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.scale = torch.nn.Module()
        self._prev_mean = torch.zeros(2)
        self._m3wm_train_updates = 0
        self.proposal_generator = torch.Generator().manual_seed(seed + 100)

    def step(self):
        self.optim.zero_grad()
        loss = self.model.weight.square().sum()
        loss.backward()
        self.optim.step()
        self._m3wm_train_updates += 1


class ToyReplay:

    def __init__(self):
        self.rows = []
        self._total_transitions = 0

    def state_dict(self):
        return {"rows": list(self.rows), "total": self._total_transitions}

    def load_state_dict(self, state):
        self.rows = list(state["rows"])
        self._total_transitions = int(state["total"])


def _bundle(context_id: int, seed: int) -> ContextAgentBundle:
    return ContextAgentBundle(
        context_id=context_id,
        control_agent=ToyAgent(seed),
        control_replay=ToyReplay(),
        theta_agent=ToyAgent(seed + 10),
        theta_replay=ToyReplay(),
        predictor_state={"version": seed},
    )


def test_bundle_roundtrip_restores_model_optimizer_and_private_rng(tmp_path):
    bundle = _bundle(0, 7)
    bundle.control_agent.step()
    bundle.control_replay.rows.append({"segment": 0, "owner": 0})
    bundle.control_replay._total_transitions = 1
    path = tmp_path / "context_0000.pt"
    expected = bundle.export_train_state()
    digest = bundle.save_atomic(path)
    assert digest == expected["bundle_hash"]
    restored = _bundle(0, 999)
    payload = ContextAgentBundle.load_payload(path)
    restored.import_train_state(payload)
    assert restored.export_train_state()["bundle_hash"] == digest
    assert restored.control_agent._m3wm_train_updates == 1
    assert restored.control_replay.rows == [{"segment": 0, "owner": 0}]


def test_bank_new_context_gets_distinct_bundle_and_only_one_active(tmp_path):
    initial = _bundle(0, 3)

    def factory(context_id, source, fresh_optimizer):
        target = _bundle(context_id, 100 + context_id)
        target.control_agent.model.load_state_dict(
            source.control_agent.model.state_dict()
        )
        target.theta_agent.model.load_state_dict(source.theta_agent.model.state_dict())
        target.metadata["new_optimizer_policy"] = "fresh"
        return target

    bank = ContextAgentBank(
        initial_bundle=initial, bundle_factory=factory, storage_dir=tmp_path
    )
    bank.set_global_env_step(10)
    child = bank.new_context(reason="NEW")
    assert bank.active_context_id == 0
    assert not child.active
    assert child.control_replay.rows == []
    assert child.control_agent is not initial.control_agent
    receipt = bank.activate(1, reason="RETURN")
    assert receipt.source_context_id == 0
    assert bank.active_context_id == 1
    assert bank.bundle(0).active is False
    assert bank.bundle(1).active is True
    assert bank.replay_bank.segment_id == 1


def test_atomic_new_discards_provisional_context_when_activation_fails(tmp_path):
    initial = _bundle(0, 3)

    def factory(context_id, source, fresh_optimizer):
        target = _bundle(context_id, 100 + context_id)
        target.control_replay = object()
        return target

    bank = ContextAgentBank(
        initial_bundle=initial, bundle_factory=factory, storage_dir=tmp_path
    )
    with pytest.raises(TypeError, match="replay has no state_dict codec"):
        bank.create_and_activate(reason="pnn_evidence_s0_new")
    assert bank.context_ids() == (0,)
    assert bank.active_context_id == 0
    assert bank._next_context_id == 1


def test_return_loads_persisted_target_and_failed_load_keeps_source_active(tmp_path):
    initial = _bundle(0, 3)

    def factory(context_id, source, fresh_optimizer):
        return _bundle(context_id, 100 + context_id)

    bank = ContextAgentBank(
        initial_bundle=initial, bundle_factory=factory, storage_dir=tmp_path
    )
    child = bank.new_context()
    bank.activate(child.context_id, reason="NEW")
    bank.activate(0, reason="RETURN")
    assert bank.receipts()[-1].target_loaded_from_disk is True
    path = tmp_path / "context_0001.pt"
    path.write_bytes(b"corrupt")
    before_segment = bank.replay_bank.segment_id
    with pytest.raises(Exception):
        bank.activate(1, reason="RETURN")
    assert bank.active_context_id == 0
    assert bank.bundle(0).active is True
    assert bank.bundle(1).active is False
    assert bank.replay_bank.segment_id == before_segment


def test_production_offload_reconstructs_only_the_active_bundle(tmp_path):
    initial = _bundle(0, 3)

    def factory(context_id, source, fresh_optimizer):
        return _bundle(context_id, 100 + context_id)

    bank = ContextAgentBank(
        initial_bundle=initial,
        bundle_factory=factory,
        storage_dir=tmp_path,
        release_inactive=True,
    )
    child = bank.new_context()
    bank.activate(child.context_id, reason="NEW")
    assert bank.bundle(0).materialized is False
    assert bank.bundle(1).materialized is True
    receipt = bank.activate(0, reason="RETURN")
    assert receipt.target_loaded_from_disk is True
    assert bank.bundle(0).materialized is True
    assert bank.bundle(1).materialized is False
    assert bank.state_dict()["active_context_id"] == 0


def test_replay_rows_preserve_behavior_and_never_cross_switch_segments():
    replay = ContextReplayBank()
    replay.append("n0", behavior_context_id=0, episode_id=4)
    replay.append("n1", behavior_context_id=0, episode_id=4)
    replay.switch(1)
    replay.append("jm0", behavior_context_id=0, assigned_context_id=1, episode_id=4)
    replay.append("jm1", behavior_context_id=1, episode_id=4)
    assert [row.behavior_context_id for row in replay.rows(1)] == [0, 1]
    sequences = replay.sample_sequences(1, length=2)
    assert len(sequences) == 1
    assert all((row.segment_id == 1 for row in sequences[0]))
    replay.switch(0)
    replay.append("n2", behavior_context_id=1, assigned_context_id=0, episode_id=4)
    assert all((row.segment_id == 1 for row in replay.rows(1)))
    assert replay.rows(0)[-1].segment_id == 2
    assert all((row.payload is None for row in replay.rows(0) + replay.rows(1)))


def test_finalize_persists_active_bundle_and_validates_manifest(tmp_path):
    initial = _bundle(0, 3)

    def factory(context_id, source, fresh_optimizer):
        return _bundle(context_id, 100 + context_id)

    snapshots = tmp_path / "snapshots"
    bank = ContextAgentBank(
        initial_bundle=initial, bundle_factory=factory, storage_dir=snapshots
    )
    child = bank.new_context()
    bank.set_global_env_step(40)
    bank.activate(child.context_id, reason="NEW")
    bank.bundle(1).control_agent.step()
    bank.set_global_env_step(80)
    manifest_path = bank.finalize()
    assert manifest_path == tmp_path / "final_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["active_context_id"] == 1
    assert manifest["global_env_step"] == 80
    assert len(manifest["contexts"]) == 2
    active = next((row for row in manifest["contexts"] if row["active"]))
    assert active["context_id"] == 1
    payload = ContextAgentBundle.load_payload(snapshots / "context_0001.pt")
    assert payload["control"]["runtime"]["_m3wm_train_updates"] == 1


def test_trial_working_copy_does_not_mutate_historical_bundle(tmp_path):
    initial = _bundle(0, 3)

    def factory(context_id, source, fresh_optimizer):
        return _bundle(context_id, 100 + context_id)

    bank = ContextAgentBank(
        initial_bundle=initial, bundle_factory=factory, storage_dir=tmp_path
    )
    child = bank.new_context()
    bank.activate(child.context_id, reason="NEW")
    bank.activate(0, reason="RETURN")
    canonical_before = ContextAgentBundle.load_payload(tmp_path / "context_0001.pt")
    working, source_hash, candidate_hash = bank.trial_working_copy(1)
    assert source_hash
    assert candidate_hash == canonical_before["bundle_hash"]
    working.control_agent.step()
    canonical_after = ContextAgentBundle.load_payload(tmp_path / "context_0001.pt")
    assert canonical_after["bundle_hash"] == canonical_before["bundle_hash"]
    assert bank.active_context_id == 0
    assert bank.bundle(1).control_agent._m3wm_train_updates == 0
