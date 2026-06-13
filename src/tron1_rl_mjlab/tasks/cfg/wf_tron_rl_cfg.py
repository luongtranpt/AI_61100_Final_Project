from dataclasses import dataclass, field
from typing import Any, List, Optional

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@dataclass
class RslRlTSActorCfg(RslRlModelCfg):
    """Model config for TSModel (actor side).

    Extends RslRlModelCfg with dual-encoder parameters.
    The actor uses the privileged encoder during training and the
    proprioceptive encoder during student-mode deployment.
    """

    class_name: str = "rsl_rl.models:TSModel"
    encoder_hidden_dims: tuple = (512, 256, 128)
    """Hidden dims shared by both proprioceptive and privileged encoders."""
    encoder_latent_dim: int = 32
    """Output latent dimension of the encoders."""
    raw_obs_key: str = "actor"
    """TensorDict key for the raw actor observations."""
    history_obs_key: Optional[str] = "history"
    """TensorDict key for the observation history (proprioceptive encoder input)."""
    privileged_obs_key: str = "critic"
    """TensorDict key for the privileged observations (privileged encoder input)."""
    commands_key: Optional[str] = None
    """TensorDict key for the command/goal vector. None if commands are already in raw_obs_key."""
    base_lin_vel_key: Optional[str] = "base_vel"
    """TensorDict key for ground-truth base linear velocity (teacher input + student supervision)."""



@dataclass
class RslRlTSPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO algorithm config for CTS (Concurrent Teacher-Student) training."""

    class_name: str = "rsl_rl.algorithms:PPO"
    """Qualified class name so resolve_callable works with the local namespace package."""
    teacher_ratio: float = 0.75
    """Fraction of envs acting as teacher group (privileged encoder). Rest is student group."""
    grad_penalty_coef_schedule: Optional[List[float]] = field(
        default_factory=lambda: [0.002, 0.002, 0, 1]
    )
    """Lipschitz gradient-penalty schedule: [start_coef, end_coef, start_step, duration].
    Set to None to disable the penalty."""
    clip_reward: Optional[float] = 100.0
    """Clip total reward per step to [-clip_reward, clip_reward]. Matches isaacgym clip_reward."""


def make_wf_tron_rl_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for WF-TRON task."""
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=1500000,
        save_interval=2000,
        wandb_project="mjlab_wf_tron",
        experiment_name="wf_tron",
        obs_groups={"actor": ("actor", "history", "critic", "base_vel"), "critic": ("critic",)},
        actor=RslRlTSActorCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            encoder_hidden_dims=(256, 128),
            encoder_latent_dim=16,
            distribution_cfg={
                "class_name": "rsl_rl.modules:GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
        ),
        algorithm=RslRlTSPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            teacher_ratio=0.75,
            grad_penalty_coef_schedule=[0.002, 0.002, 0, 1],
            clip_reward=100.0,
        ),
    )
