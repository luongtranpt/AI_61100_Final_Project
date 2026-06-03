"""Curriculum functions for the task."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .commands import UniformWorldPoseCommandCfg

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _terrain_type_col_ranges(env: ManagerBasedRlEnv) -> dict[str, tuple[int, int]]:
    """Map sub_terrain name → (col_start, col_end) based on proportions."""
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_generator is None:
        return {}
    cfg = terrain.cfg.terrain_generator
    num_cols = cfg.num_cols
    cumulative = 0.0
    ranges: dict[str, tuple[int, int]] = {}
    for name, sub_cfg in cfg.sub_terrains.items():
        start = round(cumulative * num_cols)
        end = round((cumulative + sub_cfg.proportion) * num_cols)
        ranges[name] = (start, end)
        cumulative += sub_cfg.proportion
    return ranges


def episode_length_by_terrain(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    terrain_type: str,
) -> torch.Tensor:
    """Return mean episode length for envs on a specific terrain type (for wandb logging)."""
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_generator is None:
        return torch.zeros(1)
    ranges = _terrain_type_col_ranges(env)
    if terrain_type not in ranges:
        return torch.zeros(1)
    start, end = ranges[terrain_type]
    mask = (terrain.terrain_types >= start) & (terrain.terrain_types < end)
    if mask.sum() == 0:
        return torch.zeros(1)
    return env.episode_length_buf[mask].float().mean().unsqueeze(0)


def terrain_levels_vel(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terrain curriculum: level up/down based on distance walked.

    Safe on plane terrain — returns 0.0 if no terrain generator is present.
    """
    terrain = env.scene.terrain
    if terrain is None or terrain.cfg.terrain_generator is None:
        return torch.zeros(1)

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    distance = torch.norm(
        asset.data.root_link_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )
    terrain_size = terrain.cfg.terrain_generator.size
    move_up = distance > 2.0
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def pos_commands_ranges_level(
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice,
        max_range: UniformWorldPoseCommandCfg.Ranges,
        update_interval: int = 80 * 24,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        command_name: str = "base_pose",
) -> torch.Tensor:
    command_cfg: UniformWorldPoseCommandCfg = env.command_manager.get_term(command_name).cfg
    x = command_cfg.ranges.pos_x[1]
    if (env.common_step_counter + 1) % update_interval == 0:
        # Update position ranges
        x = command_cfg.ranges.pos_x[1] + 0.1
        y = command_cfg.ranges.pos_y[1] + 0.1
        x = min(x, max_range.pos_x[1])
        y = min(y, max_range.pos_y[1])
        command_cfg.ranges.pos_x = (-x, x)
        command_cfg.ranges.pos_y = (-y, y)

        # Update velocity ranges if they exist
        if hasattr(command_cfg.ranges, 'vel_x') and hasattr(max_range, 'vel_x'):
            vel_x = command_cfg.ranges.vel_x[1] + 0.05
            vel_y = command_cfg.ranges.vel_y[1] + 0.05
            vel_yaw = command_cfg.ranges.vel_yaw[1] + 0.1
            vel_x = min(vel_x, max_range.vel_x[1])
            vel_y = min(vel_y, max_range.vel_y[1])
            vel_yaw = min(vel_yaw, max_range.vel_yaw[1])
            command_cfg.ranges.vel_x = (-vel_x, vel_x)
            command_cfg.ranges.vel_y = (-vel_y, vel_y)
            command_cfg.ranges.vel_yaw = (-vel_yaw, vel_yaw)

    # return the mean terrain level
    return torch.ones(1, dtype=torch.float) * x
