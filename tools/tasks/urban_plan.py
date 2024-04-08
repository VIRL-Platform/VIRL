import os
from prettytable import PrettyTable

from easydict import EasyDict

from tools.tasks.task_template import TaskTemplate
from tools.tasks.robot_rx399 import RobotRX399

from virl.config import cfg, cfg_from_yaml_file
from virl.agents.agent_template import AgentTemplate


class UrbanPlanner(TaskTemplate):
    def run(self, platform, agent, chatbot, messager, args, **kwargs):
        rx399_cfg = EasyDict()
        rx399_cfg = cfg_from_yaml_file(cfg.PIPELINE.RX399_CFG, rx399_cfg)

        rx399_agent = AgentTemplate(EasyDict(rx399_cfg.AGENT))

        task_solver = RobotRX399(output_dir=self.output_dir, logger=self.logger)
        self.memory = task_solver.run(
            platform, rx399_agent, chatbot, messager, args, pipeline_cfg=rx399_cfg.PIPELINE
        )

        # get the outputs of the visual checker
        self.create_heatmap(cfg.PIPELINE.OUTPUT)

    def create_heatmap(self, output_cfg):
        all_geocode_by_cate = self.memory.get_all_geocodes_by_category()

        path = os.path.join(str(self.output_dir), output_cfg.HEATMAP_DATA)
        path_template = os.path.splitext(path)[0]
        # write all geocodes to files
        for category_name, all_geocodes in all_geocode_by_cate.items():
            with open(f"{path_template}_{category_name.replace(' ', '_')}.txt", 'w') as f:
                f.write('\n'.join(f'({geo[0]}, {geo[1]})' for geo in all_geocodes))
