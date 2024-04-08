import os
import json
import numpy as np

from easydict import EasyDict

from tools.tasks.task_template import TaskTemplate
from .vln import VisionLanguageNavigation
from .local import Local

from virl.agents.agent_template import AgentTemplate
from virl.config import cfg_from_list, cfg_from_yaml_file, cfg
from virl.utils import common_utils, pipeline, geocode_utils
from virl.actions.navigation import build_navigator


class Tourist(TaskTemplate):
    def __init__(self, output_dir, logger):
        super().__init__(output_dir, logger)
        self.result_path = self.output_dir / 'results.json'
        
        self.current_route = None
        self.attempt_counter = 0
        
        if os.path.exists(self.result_path):
            self.resume_results()

    def run(self, platform, agent, chatbot, messager, args):
        pipeline_cfg = cfg.PIPELINE
        
        # step 1: obtain the initial route
        import ipdb; ipdb.set_trace(context=20)
        if self.current_route is None and pipeline_cfg.DATA.ROUTE_PATH != 'None':
            init_route = json.load(open(pipeline_cfg.DATA.ROUTE_PATH, 'r'))
            self.current_route = init_route
        elif self.current_route is None:
            self.current_route = self.ask_new_route_single(
                pipeline_cfg, platform, agent, chatbot, messager, args,
                agent.start_position, cfg.INIT_HEADING, None, question_to_place=True, cfg_file_path=None
            )

        # step 2: visual language navigation
        while True:
            results = self.navigate_single(
                pipeline_cfg, platform, chatbot, messager, agent, args, self.current_route
            )

            import ipdb; ipdb.set_trace(context=20)
            if self.check_arrival(pipeline_cfg.EVAL_VLN, results, self.current_route):
                break
            
            # for potential multiple attempts
            os.rename(self.output_dir / 'plan_trajectory.html', self.output_dir / f'plan_trajectory_{self.attempt_counter}.html')
            os.rename(self.output_dir / 'trajectory.html', self.output_dir / f'trajectory_{self.attempt_counter}.html')

            print('>>> Navigation failed, try again!')
            # step 3: ask for nearby concierge
            self.current_route = self.ask_new_route_single(
                pipeline_cfg, platform, agent, chatbot, messager, args,
                results['geocode'], results['heading'], self.current_route['dest_place']
            )

            os.rename(self.output_dir / 'navigator.pkl', self.output_dir / f'navigator_{self.attempt_counter}.pkl')
            import ipdb; ipdb.set_trace(context=20)
            self.attempt_counter += 1
            print(f'>>> {self.attempt_counter} new route is: ')
            print(self.current_route['instruction'])
            self.save_results()

    def ask_new_route_single(self, pipeline_cfg, platform, agent, chatbot, messager, args, start_position,
                             heading, dest_place, question_to_place=True, cfg_file_path=None):
        new_agent_cfg = {
            'NAME': agent.name,
            'CITY': agent.city,
            'BACKGROUND': agent.background,
            'START_POSITION': start_position,
        }
        new_agent = AgentTemplate(EasyDict(new_agent_cfg))
        new_agent.heading = heading

        if cfg_file_path is not None:
            new_cfg = EasyDict()
            new_cfg = cfg_from_yaml_file(cfg_file_path, new_cfg)
            new_pipeline_cfg = new_cfg.PIPELINE
        else:
            new_pipeline_cfg = pipeline_cfg
        
        task_solver = Local(output_dir=self.output_dir, logger=self.logger)
        route_info = task_solver.run(
            platform, new_agent, chatbot, messager, args, pipeline_cfg=new_pipeline_cfg,
            question_to_place=question_to_place, dest_place=dest_place, user_question=cfg.TASK_INFO.QUESTION
        )

        return route_info

    def check_arrival(self, eval_cfg, results, route):
        geocode = results['geocode']
        dest_geocode = route['dest_place']['geocode']
        
        # evaluation success rate
        distance = geocode_utils.calculate_distance_from_geocode(geocode, dest_geocode)
        if distance < eval_cfg.SUCCESS_RADIUS:
            print('Navigation Success!')
            return True
        else:
            print('Navigation Failed!')
            return False

    def save_results(self):
        results = {
            'current_route': self.current_route,
            'attempt_counter': self.attempt_counter,
        }
        common_utils.dump_json_results(results, self.result_path)
    
    def resume_results(self):
        results = json.load(open(self.result_path, 'r'))
        self.current_route = results['current_route']
        self.attempt_counter = results['attempt_counter']
    