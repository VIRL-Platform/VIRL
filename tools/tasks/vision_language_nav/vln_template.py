
from easydict import EasyDict

from tools.tasks.task_template import TaskTemplate
from .vln import VisionLanguageNavigation

from virl.agents.agent_template import AgentTemplate
from virl.utils import common_utils, geocode_utils, pipeline


class VLNTemplate(TaskTemplate):
    def navigate_single(self, pipeline_cfg, platform, chatbot, messager, agent, args, route):
        start_place = route['start_place']
        start_position = start_place['relocated_geocode']
        
        instruction = route['instruction']
        init_heading = route.get('init_heading', 0)
        
        # show the polyline
        polyline = geocode_utils.encode_polyline([start_place['relocated_geocode']] + route['route_results']['geocode_list'])
        pipeline.draw_planned_route(
            platform, polyline, input_way_points=None,
            path=self.output_dir / pipeline_cfg.NAVIGATION.OUTPUT.ROUTE_PATH,
            file_template=pipeline_cfg.NAVIGATION.OUTPUT.FILE_TEMPLATE
        )
        
        new_agent_cfg = {
            'NAME': agent.name,
            'CITY': agent.city,
            'BACKGROUND': agent.background,
            'START_POSITION': start_position,
        }
        new_agent = AgentTemplate(EasyDict(new_agent_cfg))
        new_agent.heading = init_heading

        task_solver = VisionLanguageNavigation(output_dir=self.output_dir, logger=self.logger)
        results = task_solver.run(
            platform, new_agent, chatbot, messager, args, pipeline_cfg=pipeline_cfg,
            instruction=instruction, route_info=route
        )

        return results
