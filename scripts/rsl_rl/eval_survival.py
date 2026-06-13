"""Evaluate survival rate of N robots over T seconds on mixed terrain.

Survival rate(t) = fraction of robots that have NOT had a failure termination by time t.
Failure = failing_timer triggered (bad orientation / bad contact).
Timeout is excluded — robots that only hit time limit are still counted as survived.

Usage:
    uv run python scripts/rsl_rl/eval_survival.py Mjlab-WF-Tron \
        --checkpoint-file logs/rsl_rl/wf_tron/<run>/model_XXXX.pt \
        --num-envs 128 --eval-time 100.0
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from dataclasses import asdict

from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class EvalConfig:
    checkpoint_file: str
    num_envs: int = 128
    eval_time: float = 100.0
    """Total evaluation time in seconds."""
    log_interval: float = 5.0
    """Print survival rate every this many seconds."""
    device: str | None = None
    deploy_mode: Literal["teacher", "student"] = "student"
    terrain: Literal["rough", "plane"] = "rough"
    max_init_terrain_level: int = 2


def run_eval(task_id: str, cfg: EvalConfig) -> None:
    configure_torch_backends()
    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)

    # Override num_envs and terrain
    env_cfg.scene.num_envs = cfg.num_envs
    from tron1_rl_mjlab.tasks.cfg.terrain_cfg import (
        ROUGH_TERRAINS_ENTITY_CFG, PLANE_ENTITY_CFG
    )
    from copy import deepcopy
    if cfg.terrain == "rough":
        terrain = deepcopy(ROUGH_TERRAINS_ENTITY_CFG)
        terrain.max_init_terrain_level = cfg.max_init_terrain_level
        env_cfg.scene.terrain = terrain
    else:
        env_cfg.scene.terrain = PLANE_ENTITY_CFG

    # Use full episode length to not cut off early
    env_cfg.episode_length_s = cfg.eval_time + 10.0

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env.sim.sense_graph = None
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device=device)
    runner.load(cfg.checkpoint_file, load_cfg={"actor": True}, strict=True, map_location=device)

    from rsl_rl.models.ts_model import TSModel
    actor = runner.alg.actor
    if cfg.deploy_mode == "student" and isinstance(actor, TSModel):
        policy = runner.get_inference_policy_student(device=device)
    else:
        policy = runner.get_inference_policy(device=device)

    # ── Eval loop ──────────────────────────────────────────────────────────
    step_dt: float = env.unwrapped.step_dt
    num_steps = int(cfg.eval_time / step_dt)
    log_every = max(1, int(cfg.log_interval / step_dt))

    # ever_failed[i] = True if env i had a failure (non-timeout) termination
    ever_failed = torch.zeros(cfg.num_envs, dtype=torch.bool, device=device)

    obs = env.get_observations()
    obs = obs.apply(lambda t: t.clamp(-100.0, 100.0))

    print(f"\n{'='*55}")
    print(f"  Survival Eval | {cfg.num_envs} robots | {cfg.eval_time}s | {cfg.terrain} terrain ")
    print(f"  Checkpoint: {Path(cfg.checkpoint_file).name}")
    print(f"{'='*55}")
    print(f"  {'Time(s)':>8}  {'Survived':>10}  {'Survival%':>10}")
    print(f"  {'-'*35}")

    with torch.inference_mode():
        for step in range(num_steps):
            actions = policy(obs)
            obs, _, dones, extras = env.step(actions)
            obs = obs.apply(lambda t: torch.nan_to_num(t.clamp(-100.0, 100.0), nan=0.0))

            # Mark failure: done but NOT timeout
            timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool().squeeze(-1)
            failures = dones.bool().squeeze(-1) & ~timeouts
            ever_failed |= failures

            t = (step + 1) * step_dt
            if (step + 1) % log_every == 0 or step == num_steps - 1:
                n_alive = cfg.num_envs - ever_failed.sum().item()
                rate = n_alive / cfg.num_envs * 100.0
                print(f"  {t:>8.1f}  {int(n_alive):>10}  {rate:>9.1f}%")

    print(f"{'='*55}")
    final_alive = cfg.num_envs - ever_failed.sum().item()
    print(f"  Final survival rate: {final_alive}/{cfg.num_envs} = {final_alive/cfg.num_envs*100:.1f}%")
    print(f"{'='*55}\n")

    env.close()


def main():
    import mjlab.tasks  # noqa: F401

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        EvalConfig,
        args=remaining_args,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    run_eval(chosen_task, args)


if __name__ == "__main__":
    main()
