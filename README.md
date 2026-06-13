# AI_61100 Final Project — WF-TRON Locomotion

Reinforcement learning for a wheeled-footed biped robot (WF-TRON, 8 DOF) using MuJoCo.  
Compares four training strategies: Baseline PPO, Two-Phase Teacher-Student, and two variants of Concurrent Teacher-Student (CTS).

| Branch | Method | Description |
|--------|--------|-------------|
| `baseline` | Standard PPO | Plain MLP actor, no privileged info |
| `ts-2phase` | Two-Phase TS | Teacher (phase 1) → Student distillation (phase 2) |
| `ts-cts-no-vel` | CTS (no velocity) | Concurrent T/S, teacher sees height only |
| `ts-cts-vel-pred` | CTS + vel predictor | Concurrent T/S + student velocity prediction head |

---

## Quick Start (One-Click Notebook)

The notebook `notebooks/wf_tron_full_report.ipynb` is self-contained.  
It auto-clones the repo, installs dependencies, and downloads logs from HuggingFace.

```bash
# 1. Download the notebook from GitHub (pick any branch)
#    https://github.com/luongtranpt/AI_61100_Final_Project/blob/ts-cts-vel-pred/notebooks/wf_tron_full_report.ipynb

# 2. Open Jupyter and run all cells
jupyter notebook wf_tron_full_report.ipynb
# or
jupyter lab wf_tron_full_report.ipynb
```

The notebook will:
1. Clone the repo automatically (if not already present)
2. Install `uv` and all project dependencies
3. Download training logs (~420 MB) from HuggingFace: [`Luong22/tron1-rl-logs`](https://huggingface.co/datasets/Luong22/tron1-rl-logs)
4. Generate all plots and tables
5. Run survival rate evaluation (requires CUDA GPU)

---

## Manual Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/luongtranpt/AI_61100_Final_Project.git
cd AI_61100_Final_Project
git checkout ts-cts-vel-pred   # or any branch

uv sync
```

---

## Training

```bash
uv run python scripts/rsl_rl/train.py Mjlab-WF-Tron
```

Logs are saved to `logs/rsl_rl/wf_tron/<run_folder>/`.

---

## Survival Rate Evaluation

Evaluates how long robots survive on rough terrain (requires CUDA GPU).

```bash
uv run python scripts/rsl_rl/eval_survival.py Mjlab-WF-Tron \
    --checkpoint-file logs/rsl_rl/wf_tron/<run_folder>/model_10000.pt \
    --num-envs 128 \
    --eval-time 20.0 \
    --deploy-mode student   # or "teacher" for baseline
```

Example output:
```
Time (s)   Alive   Survival
     5.0     128     100.0%
    10.0     121      94.5%
    15.0     108      84.4%
    20.0      97      75.8%
```

---

## Play / Visualisation

### On a local machine (with display)

```bash
git checkout ts-cts-vel-pred
uv run python scripts/rsl_rl/play.py Mjlab-WF-Tron \
    --checkpoint-file logs/rsl_rl/wf_tron/<run_folder>/model_10000.pt \
    --deploy-mode student \
    --viewer native
```

### On a headless server (no display)

The viewer runs over **HTTP** using [Viser](https://viser.studio).  
No display or GPU display driver needed.

```bash
git checkout ts-cts-vel-pred
uv run python scripts/rsl_rl/play.py Mjlab-WF-Tron \
    --checkpoint-file logs/rsl_rl/wf_tron/<run_folder>/model_10000.pt \
    --deploy-mode student \
    --viewer viser
```

Then open a browser and go to:
```
http://localhost:8080
```

If accessing from a **remote machine**, forward the port via SSH first:
```bash
ssh -L 8080:localhost:8080 user@server-ip
```
Then open `http://localhost:8080` in your local browser.

> The `--viewer auto` flag (default) automatically picks `native` if a display is detected, otherwise falls back to `viser`.

### Deploy modes

| Flag | Description |
|------|-------------|
| `--deploy-mode student` | Uses the proprioceptive encoder (obs history) — deployment-ready |
| `--deploy-mode teacher` | Uses the privileged encoder (height map + velocity) — upper bound |

### Checkpoint shortcuts

| Branch | Checkpoint path |
|--------|----------------|
| `baseline` | `logs/rsl_rl/wf_tron/2026-06-10_16-34-45_baseline_918361d/model_10000.pt` |
| `ts-2phase` | `logs/rsl_rl/wf_tron/2026-06-10_19-29-24_ts-2phase_b3c4301/model_10000.pt` |
| `ts-cts-no-vel` | `logs/rsl_rl/wf_tron/2026-06-11_02-00-32_ts-cts-no-vel_550c7f6/model_10000.pt` |
| `ts-cts-vel-pred` | `logs/rsl_rl/wf_tron/2026-06-11_02-02-48_ts-cts-vel-pred_4adfe00/model_10000.pt` |

---

## Project Structure

```
tron1-rl-mjlab/
├── notebooks/
│   └── wf_tron_full_report.ipynb   # Main report notebook (one-click)
├── scripts/rsl_rl/
│   ├── train.py                    # Training entry point
│   ├── play.py                     # Play / visualisation (native + viser)
│   └── eval_survival.py            # Survival rate evaluation
├── src/tron1_rl_mjlab/
│   ├── tasks/                      # Task configs (env, reward, terrain)
│   └── __init__.py                 # Task registration
├── rsl_rl/                         # Modified RSL-RL (CTS PPO + TSModel)
├── logs/                           # Training logs (download from HuggingFace)
└── 2405.10830v2.pdf                # CTS reference paper
```

---

## Notebook Sections

| Section | Content |
|---------|---------|
| 0. Setup | Auto-install deps + download logs |
| 1. Motivation | Problem statement |
| 2. Robot & Environment | WF-TRON specs, observations, terrain |
| 3. Reward Function | All reward terms and weights |
| 4. Methods | Baseline / 2-phase / CTS architecture diagrams |
| 5. Code Walkthrough | Key implementation details |
| 6. Training Curves | Reward, terrain level, tracking reward |
| 7. Velocity Tracking Error | Lin vel error plot + final value table |
| 8. Survival Rate Evaluation | 128 robots × 20s on rough terrain |
| 9. Play / Visualisation | HTTP viewer instructions |
| 10. Discussion & Conclusion | Summary and future work |

---

## Reference

> *Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion* (2022) — included as `2405.10830v2.pdf`
