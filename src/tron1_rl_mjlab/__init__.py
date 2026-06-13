from mjlab.tasks.registry import register_mjlab_task

from .tasks.cfg.wf_tron_env_cfg import (
    make_wf_tron_env_cfg,
    make_wf_tron_play_env_cfg,
)
from .tasks.cfg.wf_tron_rl_cfg import make_wf_tron_rl_cfg
from .tasks.runner import WFTronPhaseRunner

register_mjlab_task(
    task_id="Mjlab-WF-Tron",
    env_cfg=make_wf_tron_env_cfg(),
    play_env_cfg=make_wf_tron_play_env_cfg(),
    rl_cfg=make_wf_tron_rl_cfg(),
    runner_cls=WFTronPhaseRunner,
)
