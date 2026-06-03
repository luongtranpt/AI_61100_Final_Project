"""Termination functions for the task."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def bad_orientation_timer(
        env: ManagerBasedRlEnv,
        limit_angle: float,
        fail_time: float = 0.5,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate after bad orientation persists for fail_time seconds (matches isaacgym fail_to_terminal_time_s)."""
    asset: Entity = env.scene[asset_cfg.name]
    bad = torch.acos(-asset.data.projected_gravity_b[:, 2]).abs() > limit_angle

    if not hasattr(env, "_bad_orient_timer"):
        env._bad_orient_timer = torch.zeros(env.num_envs, device=env.device)

    just_reset = env.episode_length_buf <= 1
    timer = env._bad_orient_timer
    timer = torch.where(just_reset, torch.zeros_like(timer), timer)
    timer = torch.where(bad, timer + env.step_dt, torch.zeros_like(timer))
    env._bad_orient_timer = timer

    return timer >= fail_time


def failing_timer(
        env: ManagerBasedRlEnv,
        fail_time: float = 0.5,
        contact_force_threshold: float = 10.0,
        sensor_name: str = "termination_contact",
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate after accumulating fail_time seconds of failure — matches isaacgym check_termination:
      - contact force > threshold on abad/base bodies
      - OR projected_gravity_b[:, 2] > -0.1 (robot severely tilted)

    isaacgym accumulates a CUMULATIVE bad-step counter (``fail_buf += bad``) that only
    resets at episode reset — NOT a consecutive timer. A robot that briefly recovers
    still keeps its accumulated failure count, so intermittent failures add up.
    """
    asset: Entity = env.scene[asset_cfg.name]

    # Condition 1: contact force on abad/base bodies via ContactSensor
    sensor = env.scene[sensor_name]
    force_norms = torch.norm(sensor.data.force, dim=-1)  # [B, N]
    bad_contact = torch.any(force_norms > contact_force_threshold, dim=1)

    # Condition 2: projected gravity (isaacgym: projected_gravity[:, 2] > -0.1)
    bad_orient = asset.data.projected_gravity_b[:, 2] > -0.1

    bad = bad_contact | bad_orient

    if not hasattr(env, "_fail_buf"):
        env._fail_buf = torch.zeros(env.num_envs, device=env.device)

    just_reset = env.episode_length_buf <= 1
    fail_buf = env._fail_buf
    fail_buf = torch.where(just_reset, torch.zeros_like(fail_buf), fail_buf)
    fail_buf = fail_buf + bad.float()  # cumulative count of bad steps (isaacgym fail_buf += bad)
    env._fail_buf = fail_buf

    # isaacgym: reset when fail_buf > fail_to_terminal_time_s / dt
    return fail_buf > fail_time / env.step_dt


def bad_height_timer(
        env: ManagerBasedRlEnv,
        limit_height: float,
        fail_time: float = 0.5,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate after bad height persists for fail_time seconds (matches isaacgym fail_to_terminal_time_s)."""
    asset: Entity = env.scene[asset_cfg.name]
    foot_position = asset.data.body_link_pos_w[:, env._wheels_link_ids, :]
    height = asset.data.root_link_pos_w[:, 2] - foot_position[:, :, 2].mean(dim=-1) + env._foot_radius
    bad = height < limit_height

    if not hasattr(env, "_bad_height_timer"):
        env._bad_height_timer = torch.zeros(env.num_envs, device=env.device)

    just_reset = env.episode_length_buf <= 1
    timer = env._bad_height_timer
    timer = torch.where(just_reset, torch.zeros_like(timer), timer)
    timer = torch.where(bad, timer + env.step_dt, torch.zeros_like(timer))
    env._bad_height_timer = timer

    return timer >= fail_time
