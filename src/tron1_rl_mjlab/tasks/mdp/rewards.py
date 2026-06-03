"""Reward functions matching isaacgym BipedWF."""

from __future__ import annotations

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.envs.mdp import action_rate_l2 as _action_rate_l2
from mjlab.envs.mdp import joint_pos_limits as _joint_pos_limits

from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_foot_positions_b(env: ManagerBasedRlEnv, asset: Entity) -> torch.Tensor:
    """Foot (wheel) positions in base frame. Returns (N, 2, 3)."""
    foot_pos_w = asset.data.body_link_pos_w[:, env._wheels_link_ids, :]
    base_pos_w = asset.data.root_link_pos_w.unsqueeze(1).expand(-1, 2, -1)
    base_quat_w = asset.data.root_link_quat_w.unsqueeze(1).expand(-1, 2, -1)
    return quat_apply_inverse(base_quat_w, foot_pos_w - base_pos_w)


# ── Balance / Safety ──────────────────────────────────────────────────────────

def keep_balance(env: ManagerBasedRlEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def nominal_foot_position(
        env: ManagerBasedRlEnv,
        base_height_target: float = 0.7664,
        std: float = 0.005,
        std_wrt_v: float = 0.5,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
        command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward feet being at nominal height under the base (isaacgym style)."""
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_b = _get_foot_positions_b(env, asset)  # (N, 2, 3)

    nominal_foot_z = -(base_height_target - env._foot_radius)
    reward = torch.zeros(env.num_envs, device=env.device)
    for i in range(2):
        height_err = nominal_foot_z - foot_pos_b[:, i, 2]
        reward += torch.exp(-height_err ** 2 / std)
    reward /= 2.0

    vel_cmd = env.command_manager.get_command(command_name)
    vel_norm = torch.norm(vel_cmd[:, :3], dim=1)
    return reward * torch.exp(-vel_norm ** 2 / std_wrt_v)


def leg_symmetry(
        env: ManagerBasedRlEnv,
        std: float = 0.001,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward symmetric lateral foot placement."""
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_b = _get_foot_positions_b(env, asset)
    err = torch.abs(foot_pos_b[:, 0, 1]) - torch.abs(foot_pos_b[:, 1, 1])
    return torch.exp(-err ** 2 / std)


def feet_distance(
        env: ManagerBasedRlEnv,
        min_dist: float = 0.32,
        max_dist: float = 0.35,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize feet that are too close or too far apart."""
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_link_pos_w[:, env._wheels_link_ids, :2]
    dist = torch.norm(foot_pos_w[:, 0, :] - foot_pos_w[:, 1, :], dim=-1)
    raw = torch.clamp(min_dist - dist, 0.0, 1.0) + torch.clamp(dist - max_dist, 0.0, 1.0)
    # isaacgym clip_single_reward=5 applies to the dt-scaled reward (func*weight*dt),
    # so the raw-func cap is 5/(|weight|*step_dt). weight=-100.
    return raw.clamp(0.0, 5.0 / (100.0 * env.step_dt))


def base_height_penalty(
        env: ManagerBasedRlEnv,
        target: float = 0.7664,
        sensor_name: str = "terrain_scan",
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize deviation from target height above terrain (matches isaacgym measured_heights)."""
    asset: Entity = env.scene[asset_cfg.name]
    robot_z = asset.data.root_link_pos_w[:, 2]
    # mean(robot_z - terrain_z) across all raycast scan points — same as isaacgym
    sensor = env.scene[sensor_name]
    base_height = (robot_z.unsqueeze(1) - sensor.data.hit_pos_w[..., 2]).mean(dim=1)
    return torch.abs(base_height - target).clamp(0.0, 5.0 / (20.0 * env.step_dt))  # weight=-20


# ── Velocity tracking ─────────────────────────────────────────────────────────

def tracking_lin_vel(
        env: ManagerBasedRlEnv,
        std: float = 0.2,
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, :2]
    error = torch.sum(torch.square(cmd - asset.data.root_link_lin_vel_b[:, :2]), dim=1)
    return torch.exp(-error / std ** 2)


def tracking_ang_vel(
        env: ManagerBasedRlEnv,
        std: float = 0.25,
        command_name: str = "base_velocity",
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    cmd = env.command_manager.get_command(command_name)[:, 2]
    error = torch.square(cmd - asset.data.root_link_ang_vel_b[:, 2])
    return torch.exp(-error / std ** 2)


def tracking_lin_vel_pb(env: ManagerBasedRlEnv) -> torch.Tensor:
    current = tracking_lin_vel(env)
    if not hasattr(env, "_prev_tracking_lin_vel"):
        env._prev_tracking_lin_vel = current.clone()  # type: ignore
        return torch.zeros(env.num_envs, device=env.device)
    just_reset = env.episode_length_buf <= 1
    delta = torch.where(just_reset, torch.zeros_like(current), current - env._prev_tracking_lin_vel)  # type: ignore
    env._prev_tracking_lin_vel = current.clone()  # type: ignore
    # value = (delta/dt)*weight*dt = delta*weight; clip_single_reward=5 → cap delta/dt at 5/(|weight|*dt). weight=1.0
    return (delta / env.step_dt).clamp(-5.0 / (1.0 * env.step_dt), 5.0 / (1.0 * env.step_dt))


def tracking_ang_vel_pb(env: ManagerBasedRlEnv) -> torch.Tensor:
    current = tracking_ang_vel(env)
    if not hasattr(env, "_prev_tracking_ang_vel"):
        env._prev_tracking_ang_vel = current.clone()  # type: ignore
        return torch.zeros(env.num_envs, device=env.device)
    just_reset = env.episode_length_buf <= 1
    delta = torch.where(just_reset, torch.zeros_like(current), current - env._prev_tracking_ang_vel)  # type: ignore
    env._prev_tracking_ang_vel = current.clone()  # type: ignore
    # cap delta/dt at 5/(|weight|*dt). weight=0.2
    return (delta / env.step_dt).clamp(-5.0 / (0.2 * env.step_dt), 5.0 / (0.2 * env.step_dt))


# ── Penalties ─────────────────────────────────────────────────────────────────

def lin_vel_z(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_link_lin_vel_b[:, 2]).clamp(0.0, 5.0 / (0.3 * env.step_dt))


def ang_vel_xy(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1).clamp(0.0, 5.0 / (0.3 * env.step_dt))


def orientation_penalty(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1).clamp(0.0, 5.0 / (12.0 * env.step_dt))


def dof_acc(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc), dim=1)


def same_foot_x_position(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize feet having different x positions in base frame."""
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_b = _get_foot_positions_b(env, asset)
    return torch.abs(foot_pos_b[:, 0, 0] - foot_pos_b[:, 1, 0]).clamp(0.0, 5.0 / (50.0 * env.step_dt))  # weight=-50


def same_foot_z_position(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize feet being at different heights in base frame."""
    asset: Entity = env.scene[asset_cfg.name]
    foot_pos_b = _get_foot_positions_b(env, asset)
    return ((foot_pos_b[:, 0, 2] - foot_pos_b[:, 1, 2]) ** 2).clamp(0.0, 5.0 / (100.0 * env.step_dt))  # weight=-100


def collision_penalty(
        env: ManagerBasedRlEnv,
        sensor_name: str = "collision_contact",
        force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize contact force > threshold on knee/hip links (matches isaacgym penalised_contact_indices)."""
    sensor = env.scene[sensor_name]
    # force: [B, N, 3], N = num knee/hip bodies (4)
    force_norms = torch.norm(sensor.data.force, dim=-1)  # [B, N]
    contacts = (force_norms > force_threshold).float()
    # isaacgym: torch.sum(norm > 1.0) → count of bodies in contact (0-4).
    # value = count*weight*dt = count*(-50)*0.02 = -1.0/body; max -4.0 < clip_single_reward=5, never clipped.
    return contacts.sum(dim=1)


def joint_vel_l2(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def weighted_joint_torques_l2(
        env: ManagerBasedRlEnv,
        torque_weight: dict[str, float],
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    if not asset.data.is_actuated:
        return torch.zeros(env.num_envs, device=env.device)
    weighted_torque = torch.zeros_like(asset.data.actuator_force)
    for joint_name, w in torque_weight.items():
        joint_idx, _ = asset.find_joints(joint_name)
        weighted_torque[:, joint_idx] = torch.square(asset.data.actuator_force[:, joint_idx]) * w
    return torch.sum(weighted_torque, dim=1)


def action_smoothness_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    current = env.action_manager.action.clone()
    if not hasattr(env, "_smooth_prev"):
        env._smooth_prev = current  # type: ignore
        return torch.zeros(current.shape[0], device=current.device)
    if not hasattr(env, "_smooth_prev_prev"):
        env._smooth_prev_prev = env._smooth_prev  # type: ignore
        env._smooth_prev = current  # type: ignore
        return torch.zeros(current.shape[0], device=current.device)
    penalty = torch.sum(
        torch.square(current - 2 * env._smooth_prev + env._smooth_prev_prev), dim=1  # type: ignore
    )
    env._smooth_prev_prev = env._smooth_prev  # type: ignore
    env._smooth_prev = current  # type: ignore
    penalty[env.episode_length_buf < 3] = 0
    return penalty.clamp(0.0, 5.0 / (0.03 * env.step_dt))


def action_rate(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Clipped wrapper around mjlab action_rate_l2 (weight=-0.03)."""
    return _action_rate_l2(env).clamp(0.0, 5.0 / (0.03 * env.step_dt))


def dof_pos_limits(
        env: ManagerBasedRlEnv,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Clipped wrapper around mjlab joint_pos_limits (weight=-2.0)."""
    return _joint_pos_limits(env, asset_cfg=asset_cfg).clamp(0.0, 5.0 / (2.0 * env.step_dt))
