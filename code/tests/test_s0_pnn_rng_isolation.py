import torch
from common.m3wm_s0_pnn import PersistentPNNEnsemble


def test_pnn_initialization_keeps_process_rng_and_does_not_seed_cuda(monkeypatch):
    before = torch.random.get_rng_state().clone()

    def fail_cuda_seed(*args, **kwargs):
        raise AssertionError("CPU-only PNN initialization seeded CUDA")

    monkeypatch.setattr(torch.cuda, "manual_seed_all", fail_cuda_seed)
    PersistentPNNEnsemble(
        state_dim=3, action_dim=2, members=2, hidden=8, layers=2, seed=2001
    )
    assert torch.equal(before, torch.random.get_rng_state())


def test_private_cpu_seed_matches_expected_member_initialization():
    ensemble = PersistentPNNEnsemble(
        state_dim=2, action_dim=1, members=1, hidden=4, layers=2, seed=17
    )
    actual = [p.detach().clone() for p in ensemble.models[0].parameters()]
    with torch.random.fork_rng(devices=[], enabled=True):
        generator = torch.Generator(device="cpu").manual_seed(17)
        torch.random.set_rng_state(generator.get_state())
        expected_model = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.SiLU(),
            torch.nn.Linear(4, 4),
            torch.nn.SiLU(),
            torch.nn.Linear(4, 6),
        ).double()
    expected = [p.detach().clone() for p in expected_model.parameters()]
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)
