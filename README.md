<div align="center">

# ReCo-WM: Restorable and Compositional World Models in Changing Environments

<!-- Authors and affiliations: to be completed -->
**Authors TBD** · *Affiliations TBD*

<!-- Replace the `#` targets once the assets exist. Keep the order: paper, arXiv, video, project page. -->
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b)](paper/ReCo-WM.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-lightgrey)](#)
[![Video](https://img.shields.io/badge/Video-demo-ff0000)](#demo-video)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue)](code/requirements.txt)

<!-- Teaser: export Fig. 1 of the paper as media/teaser.png (≤ 1600 px wide) -->
<img src="media/teaser.png" alt="ReCo-WM overview" width="90%">

</div>

---

## Overview

Robots deployed for long-term operation in open and changing environments must continually cope with shifts in
dynamics, disturbances and task requirements. Environmental change does not always imply novelty: previously
encountered conditions may return, and related variations can often be handled by reorganising knowledge that has
already been acquired. **ReCo-WM** factorises recurrent adaptation into two coupled layers:

- an **upper restoration layer** that uses predictive evidence with temporal confirmation to *retain* the active
  learner, *restore* a complete historical learning state (world model, policy, value functions, optimizer state and
  replay), or *create* a new one; and
- a **lower composition layer** that, conditioned on the selected learner, uses multi-timescale histories to
  recompose reusable dynamics, reward and value primitives, with control proposals verified by the primary world
  model before execution.

The guiding principle is **reuse before relearning**: restore knowledge when a regime recurs, recompose it as
conditions vary, and learn anew only when existing learners no longer explain the interaction.

## Repository layout

| Path | Content |
|---|---|
| [`code/`](code/) | Implementation (TD-MPC2 backbone, PNN evidence, context selection, complete-state restoration), configs for the four benchmarks, tests. See [`code/README.md`](code/README.md). |
| [`paper/`](paper/) | Paper PDF and supplementary material (added on release). |
| [`media/`](media/) | Teaser figure and demo video (added on release). |

## Quick start (simulation)

```bash
cd code
python -m pip install -r requirements.txt          # Python 3.10, CUDA-capable PyTorch
python run.py --env HC --steps 1100 --run-id smoke  # integration check
python -m pytest -q                                 # regression tests
python run.py --env Walker2d --seed 2001            # full 320k-step protocol
```

The Gym-v2 (MuJoCo 1.50) and DeepMind Control environments run in separate interpreters that the learner talks to
through bridges; environment-specific setup, environment variables and the recurrent eight-stage protocol
(Normal → Joint Malfunction → Wind → Velocity, twice) are documented in [`code/README.md`](code/README.md).

## Results

Whole-run mean per-step reward over 320k interactions (eight 40k-step stages, five seeds; full stage-level tables in
the paper):

| Method | HalfCheetah | Walker2d | Hopper | DMC Walker Run |
|---|---:|---:|---:|---:|
| MBCD | −0.506 | −1.839 | −1.025 | 0.027 |
| SWOKS | −1.573 | −1.997 | −0.813 | 0.046 |
| 3RL | −2.110 | −0.195 | 0.195 | 0.023 |
| LCPO | −2.310 | −1.740 | −0.802 | 0.023 |
| **ReCo-WM** | **−0.271** | **0.416** | **0.428** | **0.615** |

<!-- Add the learning-curve figure once exported: media/results_curves.png -->

## Demo video

<!-- Option A (recommended): upload the MP4 as a GitHub Release asset and link it here.
     Option B: host on YouTube/Bilibili and embed a thumbnail that links to it.
     GitHub renders a short MP4 inline if it is pasted into an issue/PR and the generated URL is used here. -->
*Coming soon.* See [`media/README.md`](media/README.md) for the hosting plan.

## Citation

```bibtex
@article{recowm2026,
  title   = {ReCo-WM: Restorable and Compositional World Models in Changing Environments},
  author  = {TBD},
  journal = {TBD},
  year    = {2026}
}
```

## Acknowledgements and license

The simulation code builds on [TD-MPC2](https://github.com/nicklashansen/tdmpc2) (MIT); the upstream notice is kept
in [`code/licenses/TD-MPC2-MIT.txt`](code/licenses/TD-MPC2-MIT.txt) and attribution is listed in
[`code/THIRD_PARTY.md`](code/THIRD_PARTY.md). Simulator binaries, assets, trained weights and run data are not
redistributed. Project-specific code is released under the [MIT License](LICENSE).
