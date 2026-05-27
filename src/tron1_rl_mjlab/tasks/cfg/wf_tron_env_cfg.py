import math
from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import GaussianNoiseCfg
from mjlab.viewer import ViewerConfig

from ...assets.wf_tron.wf_tron import WF_TRON_ROBOT_CFG
from .terrain_cfg import TERRAINS_ENTITY_CFG, PLANE_ENTITY_CFG
from .. import mdp

SCENE_CFG = SceneCfg(
    num_envs=4096,
    extent=1.0,
    terrain=PLANE_ENTITY_CFG,
    entities={"robot": WF_TRON_ROBOT_CFG},
)

VIEWER_CONFIG = ViewerConfig(
    origin_type=ViewerConfig.OriginType.ASSET_BODY,
    entity_name="robot",
    body_name="base_Link",
    distance=3.0,
    elevation=10.0,
    azimuth=90.0,
)


def make_commands() -> dict[str, CommandTermCfg]:
    return {
        "base_velocity": mdp.UniformVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=(5.0, 5.0),
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0),
                lin_vel_y=(0.0, 0.0),
                ang_vel_yaw=(-0.6, 0.6),
            ),
        )
    }


def make_actions() -> dict[str, ActionTermCfg]:
    return {
        "joint_pos": mdp.JointPositionActionCfg(
            entity_name="robot",
            actuator_names=("abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint"),
            scale=0.25,
            use_default_offset=True,
        ),
        "joint_vel": mdp.JointVelocityActionCfg(
            entity_name="robot",
            actuator_names=("wheel_[RL]_Joint",),
            scale=0.5,
            use_default_offset=True,
        ),
    }


def make_observations() -> dict[str, ObservationGroupCfg]:
    # Commands (3 dims: scaled vel)
    commands_terms = {
        "base_vel_commands": ObservationTermCfg(
            func=mdp.base_vel_commands,
            params={"command_name": "base_velocity"},
        ),
    }

    # Policy obs — matches isaacgym obs_buf (28 dims) + commands (3) = 31 total
    policy_terms = {
        "base_ang_vel": ObservationTermCfg(
            func=mdp.base_ang_vel,
            noise=GaussianNoiseCfg(mean=0.0, std=0.3),
            scale=0.25,
        ),
        "proj_gravity": ObservationTermCfg(
            func=mdp.projected_gravity,
            noise=GaussianNoiseCfg(mean=0.0, std=0.075),
            scale=1.0,
        ),
        "joint_pos": ObservationTermCfg(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg(
                name="robot",
                joint_names=("abad_[RL]_Joint", "hip_[RL]_Joint", "knee_[RL]_Joint"),
            )},
            noise=GaussianNoiseCfg(mean=0.0, std=0.015),
            scale=1.0,
        ),
        "joint_vel": ObservationTermCfg(
            func=mdp.joint_vel_rel,
            noise=GaussianNoiseCfg(mean=0.0, std=2.25),
            scale=0.05,
        ),
        "last_action": ObservationTermCfg(
            func=mdp.last_action,
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
            scale=1.0,
        ),
    }

    # Critic gets base_lin_vel (privileged) + commands + policy
    critic_extra = {
        "base_lin_vel": ObservationTermCfg(func=mdp.base_lin_vel, scale=2.0),
    }

    return {
        "actor": ObservationGroupCfg(
            terms=commands_terms | policy_terms,
            enable_corruption=True,
            concatenate_terms=True,
        ),
        "history": ObservationGroupCfg(
            terms=commands_terms | policy_terms,
            enable_corruption=True,
            concatenate_terms=True,
            history_length=10,
            flatten_history_dim=True,
        ),
        "critic": ObservationGroupCfg(
            terms=commands_terms | critic_extra | policy_terms,
            enable_corruption=False,
            concatenate_terms=True,
        ),
    }


def make_events() -> dict[str, EventTermCfg]:
    return {
        # Startup
        "prepare_quantities": EventTermCfg(
            func=mdp.prepare_quantities,
            mode="startup",
            params={"asset_cfg": SceneEntityCfg("robot")},
        ),
        "add_base_mass": EventTermCfg(
            func=mdp.dr.body_mass,
            mode="startup",
            params={
                "ranges": (-0.5, 2.0),
                "operation": "add",
                "distribution": "uniform",
                "asset_cfg": SceneEntityCfg("robot", body_names="base_Link"),
            },
        ),
        "add_link_mass": EventTermCfg(
            func=mdp.dr.body_mass,
            mode="startup",
            params={
                "ranges": (0.8, 1.2),
                "operation": "scale",
                "distribution": "uniform",
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_[LR]_Link"),
            },
        ),
        "robot_physics_material": EventTermCfg(
            func=mdp.dr.geom_friction,
            mode="startup",
            params={
                "ranges": {
                    0: (0.2, 1.6),
                    1: (0.2, 0.9),
                    2: (0.0, 1.0),
                },
                "operation": "abs",
                "distribution": "uniform",
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            },
        ),
        "robot_center_of_mass": EventTermCfg(
            func=mdp.dr.body_com_offset,
            mode="startup",
            params={
                "ranges": {
                    0: (-0.03, 0.03),
                    1: (-0.02, 0.02),
                    2: (-0.03, 0.03),
                },
                "operation": "add",
                "distribution": "uniform",
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        "randomize_joint_stiffness": EventTermCfg(
            func=mdp.dr.joint_stiffness,
            mode="startup",
            params={
                "ranges": (0.8, 1.2),
                "operation": "scale",
                "distribution": "log_uniform",
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            },
        ),
        "randomize_joint_damping": EventTermCfg(
            func=mdp.dr.joint_damping,
            mode="startup",
            params={
                "ranges": (0.8, 1.2),
                "operation": "scale",
                "distribution": "log_uniform",
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            },
        ),
        # Reset
        "reset_robot_base": EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (-0.2, 0.2),
                "velocity_range": (-0.5, 0.5),
            },
        ),
        # isaacgym: randomize_default_dof_pos ±0.05
        "randomize_default_joint_pos": EventTermCfg(
            func=mdp.randomize_default_joint_pos,
            mode="startup",
            params={
                "offset_range": (-0.05, 0.05),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        ),
        # Interval
        "push_robot": EventTermCfg(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(7.0, 10.0),
            params={"velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)}},
        ),
    }


def make_rewards() -> dict[str, RewardTermCfg]:
    return {
        # Balance
        "keep_balance": RewardTermCfg(
            func=mdp.keep_balance,
            weight=1.0,
        ),
        # Tracking
        "tracking_lin_vel": RewardTermCfg(
            func=mdp.tracking_lin_vel,
            weight=4.0,
            params={"std": 0.2, "command_name": "base_velocity"},
        ),
        "tracking_ang_vel": RewardTermCfg(
            func=mdp.tracking_ang_vel,
            weight=2.0,
            params={"std": 0.25, "command_name": "base_velocity"},
        ),
        "tracking_lin_vel_pb": RewardTermCfg(
            func=mdp.tracking_lin_vel_pb,
            weight=1.0,
        ),
        "tracking_ang_vel_pb": RewardTermCfg(
            func=mdp.tracking_ang_vel_pb,
            weight=0.2,
        ),
        # Regulation
        "nominal_foot_position": RewardTermCfg(
            func=mdp.nominal_foot_position,
            weight=4.0,
            params={
                "base_height_target": 0.7664,
                "std": 0.005,
                "std_wrt_v": 0.5,
                "command_name": "base_velocity",
            },
        ),
        "leg_symmetry": RewardTermCfg(
            func=mdp.leg_symmetry,
            weight=0.5,
            params={"std": 0.001},
        ),
        "same_foot_x_position": RewardTermCfg(
            func=mdp.same_foot_x_position,
            weight=-50.0,
        ),
        "same_foot_z_position": RewardTermCfg(
            func=mdp.same_foot_z_position,
            weight=-100.0,
        ),
        "lin_vel_z": RewardTermCfg(
            func=mdp.lin_vel_z,
            weight=-0.3,
        ),
        "ang_vel_xy": RewardTermCfg(
            func=mdp.ang_vel_xy,
            weight=-0.3,
        ),
        "orientation": RewardTermCfg(
            func=mdp.orientation_penalty,
            weight=-12.0,
        ),
        "torques": RewardTermCfg(
            func=mdp.weighted_joint_torques_l2,
            weight=-0.00016,
            params={
                "torque_weight": {
                    "abad_L_Joint": 1.0, "hip_L_Joint": 1.0, "knee_L_Joint": 1.0,
                    "abad_R_Joint": 1.0, "hip_R_Joint": 1.0, "knee_R_Joint": 1.0,
                    "wheel_L_Joint": 1.0, "wheel_R_Joint": 1.0,
                }
            },
        ),
        "dof_acc": RewardTermCfg(
            func=mdp.dof_acc,
            weight=-1.5e-7,
        ),
        "action_rate": RewardTermCfg(
            func=mdp.action_rate,
            weight=-0.03,
        ),
        "action_smoothness": RewardTermCfg(
            func=mdp.action_smoothness_penalty,
            weight=-0.03,
        ),
        "dof_pos_limits": RewardTermCfg(
            func=mdp.dof_pos_limits,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names="(?!wheel_).*")},
        ),
        "collision": RewardTermCfg(
            func=mdp.collision_penalty,
            weight=-50.0,
            params={"threshold": 0.05},
        ),
        "feet_distance": RewardTermCfg(
            func=mdp.feet_distance,
            weight=-100.0,
            params={"min_dist": 0.32, "max_dist": 0.35},
        ),
        "base_height": RewardTermCfg(
            func=mdp.base_height_penalty,
            weight=-20.0,
            params={"target": 0.7664},
        ),
    }


def make_terminations() -> dict[str, TerminationTermCfg]:
    return {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "bad_orientation": TerminationTermCfg(
            func=mdp.bad_orientation_timer,
            params={"limit_angle": math.pi * 0.4, "fail_time": 0.5},
        ),
        "bad_height": TerminationTermCfg(
            func=mdp.bad_height_timer,
            params={"limit_height": 0.5, "fail_time": 0.5},
        ),
    }


def make_curriculum() -> dict[str, CurriculumTermCfg]:
    return {}


SIM_CFG = SimulationCfg(
    mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
    ),
    nconmax=256,
    njmax=512,
)


def make_wf_tron_env_cfg() -> ManagerBasedRlEnvCfg:
    return ManagerBasedRlEnvCfg(
        scene=SCENE_CFG,
        observations=make_observations(),
        actions=make_actions(),
        commands=make_commands(),
        rewards=make_rewards(),
        events=make_events(),
        terminations=make_terminations(),
        curriculum=make_curriculum(),
        sim=SIM_CFG,
        viewer=VIEWER_CONFIG,
        decimation=4,
        episode_length_s=20.0,
        seed=0,
    )


def make_wf_tron_play_env_cfg() -> ManagerBasedRlEnvCfg:
    env_cfg = deepcopy(make_wf_tron_env_cfg())
    env_cfg.scene.num_envs = 4
    env_cfg.scene.terrain = PLANE_ENTITY_CFG
    return env_cfg
