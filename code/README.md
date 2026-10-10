# ReCo-WM: online Context-agent experiments

Shared implementation and separate configurations for four benchmarks:

| Configuration | Environment | State / action dimensions |
| --- | --- | --- |
| `Walker2d` | Gym Walker2d-v2 | 17 / 6 |
| `Hopper` | Gym Hopper-v2 | 11 / 3 |
| `HC` | Gym HalfCheetah-v2 | 17 / 6 |
| `DMC` | DeepMind Control Walker Run | 24 / 6 |

## Algorithm

Each Context owns a TD-MPC2 controller, a separate θ-proposal learner, optimizers, replay buffers and a probabilistic neural-network ensemble. The active predictor scores incoming transitions before learning from them. Accumulated likelihood evidence, future-data confirmation and bounded candidate trials determine NEW / RETURN decisions. Context 0 is initialized from scratch.

The detector receives only `(observation, action, reward, next_observation, terminated, truncated, transition_id)`. Environment labels are used only for reporting. The environment follows Normal → Joint Malfunction → Wind → Velocity twice, with 40,000 steps per phase; that schedule is not a detector input. Transition IDs support ordering, disjoint evidence windows and cooldowns.

## Installation

The learner requires Python 3.10 and a CUDA-capable GPU. Install a CUDA-compatible PyTorch build and the dependencies:

```bash
python -m pip install -r requirements.txt
```

The requirements record the tested learner versions. Replay serialization depends on the paired TorchRL/TensorDict versions; changing them requires rerunning the tests. The supplied CUDA profile uses `cuda:0` for the PNN; `--pnn-device` is the explicit command-line override.

### Gym-v2 environments

Use a separate Python 3.7 environment with `requirements-gym-v2.txt` and a working MuJoCo 1.50 installation. The simulator and any required license are not redistributed. Point the bridge at that environment:

```bash
export M3WM_LOCOMOTION_V2_WORKER_PYTHON="/absolute/path/to/legacy-env/bin/python"
export M3WM_MJPRO_PATH="/absolute/path/to/mjpro150"
```

Do not substitute Gym-v4/v5 or a newer MuJoCo runtime when comparing these benchmarks.

### DMC environment

Install `requirements-dmc.txt` in a separate Python 3.10 environment:

```bash
export M3WM_DMC_PYTHON="/absolute/path/to/dmc-env/bin/python"
export MUJOCO_GL=egl
```

If `M3WM_DMC_PYTHON` is unset, the bridge uses the learner's interpreter.

## Run

```bash
python run.py --env Walker2d --dry-run
python run.py --env Walker2d --seed 2001
python run.py --env Hopper --seed 2001
python run.py --env DMC --seed 2001
python run.py --env HC --seed 2001
```

Default budget: 320,000 online environment steps. To check integration without changing the phase schedule:

```bash
python run.py --env HC --steps 1100 --run-id smoke
python -m pytest -q
```

`--pnn-device cpu` runs PNN computation on CPU; the TD-MPC2 learner still requires CUDA. The command-line device takes precedence over the JSON setting. `--output-dir` changes the run directory. Every invocation starts a new run; an existing run directory is rejected.

## Outputs

Runs are written under `runs/`, separately from source code:

- `manifest.json`, `run_status.json`: configuration and completion status.
- `online_steps.csv`, `online_episodes.csv`: rewards, decisions and update counts.
- `context_agent_pnn_fits.jsonl`, `context_agent_audit.json`: predictor updates and Context evidence.
- `context_agents/snapshots/`: complete per-Context training states.
- `models/final.pt`: final controller checkpoint.

## Reproducibility

The four commands above run the paper protocol from fresh initialization. Use the same seed list and hardware configuration as the paper for a matching table or curve. Paper figures, comparison plots, training logs and checkpoints are intentionally not included in this source repository; they remain in the paper's supplementary materials. The included tests cover the detector contract, autonomous NEW/RETURN decisions, Context-agent state restoration and environment wiring.

## Source layout

`run.py` is the only training entry. `configs/` contains environment and learner settings; `tdmpc2.py` and `common/world_model.py` implement learning and planning; `common/m3wm_s0_*` implements PNN evidence and Context selection; `common/m3wm_context_agent_*` stores and restores complete agents; `envs/` and `external_baselines/` provide simulator bridges; `tests/` contains regression tests.

TD-MPC2 copyright and license are preserved in `licenses/TD-MPC2-MIT.txt`. See `THIRD_PARTY.md` for attribution.

All four environment configurations use `m3wm_context_selector: pnn_evidence`, implemented by `S0PNNEvidenceContextSelector`. The CUSUM threshold is configured through `m3wm_pnn_evidence_cusum_threshold` (300.0). Context selection uses PNN prediction likelihoods, accumulated evidence and subsequent-data confirmation to decide STAY, NEW or RETURN.

ReCo-WM is the public project name. Existing `m3wm_*` configuration keys, Python module paths and internal attributes are retained to avoid an unrelated API refactor; selection logic and threshold values are unchanged. The selector now saves and accepts only the `reco-wm-s0-pnn-evidence-context-selector-v1` schema. Selector states saved by older releases are not accepted by this release; use the corresponding archived source version to load them.
