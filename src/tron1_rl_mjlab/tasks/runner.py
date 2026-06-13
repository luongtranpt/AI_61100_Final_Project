"""Custom runner for WF-Tron: 3-phase terrain training.

Phase 1  (0 → teacher_phase_iters): rough terrain, teacher RL
Phase 2  (teacher_phase_iters → end): rough terrain, student distillation
                                       terrain level reset to 0 at phase transition
"""

from mjlab.rl import MjlabOnPolicyRunner


class WFTronPhaseRunner(MjlabOnPolicyRunner):
    """Resets terrain curriculum level when transitioning to student distillation phase."""

    FLAT_TERRAIN_ITERS = 0  # train directly on rough terrain from the start

    def __init__(self, env, train_cfg: dict, log_dir: str | None, device: str):
        super().__init__(env, train_cfg, log_dir, device)
        self._flat_terrain_iters: int = self.FLAT_TERRAIN_ITERS
        self._terrain_swapped: bool = False
        self._student_phase_reset_done: bool = False

    # ------------------------------------------------------------------
    def _on_iteration_start(self, it: int, obs):
        if (
            self._flat_terrain_iters > 0
            and it == self._flat_terrain_iters
            and not self._terrain_swapped
        ):
            obs = self._swap_to_rough_terrain()

        # Reset terrain level to 0 at the start of student distillation phase
        teacher_phase_iters = self.alg.teacher_phase_iters
        if (
            teacher_phase_iters > 0
            and it == teacher_phase_iters
            and not self._student_phase_reset_done
        ):
            self._reset_terrain_level()

        return obs

    # ------------------------------------------------------------------
    def _reset_terrain_level(self):
        """Reset terrain curriculum to level 0 at phase 2 start."""
        terrain = getattr(self.env.unwrapped.scene, "terrain", None)
        if terrain is not None and hasattr(terrain, "terrain_levels") and terrain.terrain_origins is not None:
            terrain.terrain_levels[:] = 0
            all_ids = terrain.terrain_levels.new_tensor(range(terrain.terrain_levels.shape[0]))
            terrain.env_origins[all_ids] = terrain.terrain_origins[
                terrain.terrain_levels[all_ids], terrain.terrain_types[all_ids]
            ]
            print(
                f"[WFTronPhaseRunner] iter {self.current_learning_iteration}: "
                "student phase start — terrain levels reset to 0"
            )
        self._student_phase_reset_done = True

    # ------------------------------------------------------------------
    def _swap_to_rough_terrain(self):
        from copy import deepcopy

        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from tron1_rl_mjlab.tasks.cfg.wf_tron_env_cfg import make_wf_tron_env_cfg

        print(
            f"[WFTronPhaseRunner] iter {self.current_learning_iteration}: "
            "switching to rough terrain (flat+slope), curriculum reset to level 0"
        )

        clip_actions = getattr(self.env, "clip_actions", None)
        self.env.close()

        rough_cfg = deepcopy(make_wf_tron_env_cfg())
        # Reset curriculum: all robots spawn at level 0 on the new rough terrain
        rough_cfg.scene.terrain.max_init_terrain_level = 0
        new_env = ManagerBasedRlEnv(cfg=rough_cfg, device=self.device)
        new_env.sim.sense_graph = None
        self.env = RslRlVecEnvWrapper(new_env, clip_actions=clip_actions)

        self._terrain_swapped = True
        print("[WFTronPhaseRunner] terrain swap done, collecting fresh observations")

        clip_obs = self.cfg.get("clip_obs", 100.0)
        obs = self.env.get_observations().to(self.device)
        if clip_obs is not None:
            obs = obs.apply(lambda t: t.clamp(-clip_obs, clip_obs))
        return obs
