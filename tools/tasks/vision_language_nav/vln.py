import numpy as np
from tools.tasks.task_template import TaskTemplate

from virl.config import cfg_from_list, cfg_from_yaml_file, cfg
from virl.utils import common_utils, pipeline, geocode_utils
from virl.actions.navigation import build_navigator


class VisionLanguageNavigation(TaskTemplate):
    def run(self, platform, agent, chatbot, messager, args, **kwargs):
        if kwargs.get('pipeline_cfg', None) is not None:
            pipeline_cfg = kwargs['pipeline_cfg']
            instruction = kwargs['instruction']
        else:
            pipeline_cfg = cfg.PIPELINE
            agent.heading = cfg.INIT_HEADING
            instruction = cfg.TASK_INFO.INSTRUCTION

        self.navigator = build_navigator(
            pipeline_cfg.NAVIGATION, platform, messager, agent.start_position, self.output_dir,
            instruction=instruction, vision_model_cfg=cfg.VISION_MODELS, agent_heading=agent.heading,
            route_info=kwargs.get('route_info', None)
        )

        # Step 1: extract landmarks from the instruction
        step_counter = 1
        while True:
            is_finish, _, _ = self.navigator.navigate(info_dict={})

            if is_finish:
                break

            if step_counter % cfg.get('SAVE_INTERVAL', 10000000) == 0:
                self.save_results()

            step_counter += 1

        self.save_results()
        results = {
            'geocode': self.navigator.get_current_geocode(),
            'action_list': self.navigator.action_list,
            'trajectory': self.navigator.trajectory,
            'heading': self.navigator.get_current_heading(),
        }
        return results
