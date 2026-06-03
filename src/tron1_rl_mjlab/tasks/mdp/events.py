"""Event functions for the task."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def prepare_quantities(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Compute the nominal foot position in the body frame.

    This function computes the nominal foot position in the body frame. This function is only suitable for TRON robot.

    The computed nominal foot position is stored in the following attributes of env:
        - env._nominal_foot_position_b: Nominal foot positions in body frame
        - env._wheels_link_ids: Body indices of wheel links
        - env._wheels_joint_ids: Joint indices of wheel joints
        - env._foot_radius: Radius of the foot/wheel (0.127m)
    """
    asset: Entity = env.scene[asset_cfg.name]

    wheel_link_idx, _ = asset.find_bodies("wheel_[RL]_Link")
    wheel_joint_ids, _ = asset.find_joints("wheel_[RL]_Joint")
    base_idx, _ = asset.find_bodies("base_Link")

    wheels_pos_w = asset.data.body_link_pos_w[:, wheel_link_idx, :]
    base_pos_w = asset.data.body_link_pos_w[:, base_idx, :]
    base_quat = asset.data.body_link_quat_w[:, base_idx, :]

    nominal_foot_position_b = torch.zeros(len(wheel_link_idx), 3, device=env.device)

    for j in range(env.num_envs):
        if torch.any(asset.data.joint_pos[j, :] > 5e-2):
            continue
        for i in range(len(wheel_link_idx)):
            nominal_foot_position_b[i, :] = quat_apply_inverse(
                base_quat[j, 0, :], wheels_pos_w[j, i, :] - base_pos_w[j, 0, :]
            )
        break

    assert (nominal_foot_position_b != 0.0).any(), "Failed to compute nominal foot positions"

    # isaacgym: terminate_after_contacts_on = ["abad", "base"]
    terminate_idx, _ = asset.find_bodies("(abad_[LR]_Link|base_Link)")

    env._nominal_foot_position_b = nominal_foot_position_b  # type: ignore
    env._wheels_link_ids = wheel_link_idx  # type: ignore
    env._wheels_joint_ids = wheel_joint_ids  # type: ignore
    env._foot_radius = 0.127  # type: ignore
    env._terminate_link_ids = terminate_idx  # type: ignore


def randomize_default_joint_pos(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    offset_range: tuple[float, float] = (-0.05, 0.05),
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> None:
    """Add random offset to default joint positions (matches isaacgym randomize_default_dof_pos)."""
    asset: Entity = env.scene[asset_cfg.name]
    leg_joint_ids, _ = asset.find_joints("(?!wheel_).*")
    n_envs = env.num_envs if env_ids is None else len(env_ids)
    idx = slice(None) if env_ids is None else env_ids

    if not hasattr(env, "_default_dof_pos_offset"):
        env._default_dof_pos_offset = torch.zeros(  # type: ignore
            env.num_envs, asset.data.joint_pos.shape[1], device=env.device
        )

    offset = torch.empty(n_envs, len(leg_joint_ids), device=env.device).uniform_(*offset_range)
    env._default_dof_pos_offset[idx][:, leg_joint_ids] = offset  # type: ignore