"""Custom runner for WF-Tron: CTS (Concurrent Teacher-Student) training.

CTS trains teacher (75% of envs) and student (25% of envs) concurrently from the start.
No phase transitions — both groups run in parallel throughout training.
"""

from mjlab.rl import MjlabOnPolicyRunner


class WFTronPhaseRunner(MjlabOnPolicyRunner):
    """Runner for CTS training: no phase transitions, optional flat→rough terrain swap."""

    FLAT_TERRAIN_ITERS = 0  # train directly on rough terrain from the start

    def __init__(self, env, train_cfg: dict, log_dir: str | None, device: str):
        super().__init__(env, train_cfg, log_dir, device)
        self._flat_terrain_iters: int = self.FLAT_TERRAIN_ITERS
        self._terrain_swapped: bool = False

    # ------------------------------------------------------------------
    def _on_iteration_start(self, it: int, obs):
        if (
            self._flat_terrain_iters > 0
            and it == self._flat_terrain_iters
            and not self._terrain_swapped
        ):
            obs = self._swap_to_rough_terrain()

        return obs

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
