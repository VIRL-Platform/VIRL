import os
import time

import numpy as np

from prettytable import PrettyTable
from tools.tasks.task_template import TaskTemplate

from virl.config import cfg
from virl.actions.check_surrounding.visual_checker import VisualChecker
from virl.actions.navigation import build_navigator
from virl.platform.memory.memory import Memory
from virl.utils import common_utils, geocode_utils, vis_utils


class RobotRX399(TaskTemplate):
    def run(self, platform, agent, chatbot, messager, args,  pipeline_cfg=None, **kwargs):
        start_position = agent.start_position
        if pipeline_cfg is None:
            pipeline_cfg = cfg.PIPELINE
        
        self.navigator = build_navigator(
            pipeline_cfg.NAVIGATION, platform, messager, start_position, self.output_dir, resume=args.resume
        )

        # initialize the memory
        self.memory = Memory(self.output_dir, cfg.MEMORY, args.resume)

        # initialize the visual checker
        visual_checker = VisualChecker(pipeline_cfg.CHECK_SURROUNDING, platform, messager, memory=self.memory)

        current_geocode = self.navigator.get_current_geocode()
        current_heading = self.navigator.get_current_heading()

        step_counter = 1
        platform.mover.adjust_heading_web(current_heading)
        while True:
            if not cfg.get('DEMO', False):
                # sense surroundings
                visual_checker.visual_sensing_surroundings(current_geocode, current_heading)
            else:
                # visualization only
                time.sleep(0.5)
                vis_utils.mimic_detection_panorama(platform, current_heading)

            is_finish, current_geocode, info_dict = self.navigator.navigate(info_dict={})
            current_heading = self.navigator.get_current_heading()

            if is_finish:
                break

            if step_counter % cfg.get('SAVE_INTERVAL', 10000000) == 0:
                self.save_results()

            step_counter += 1

        # get the outputs of the visual checker
        self.formulate_output(pipeline_cfg.OUTPUT)
        
        # self.evaluate(self.memory, pipeline_cfg.EVALUATION)
        return self.memory

    def evaluate(self, memory, eval_cfg):
        gt_list = common_utils.load_points_in_txt_to_list(eval_cfg.GT_PATH)
        pred_list = np.array([False] * len(memory.memory))
        matched = np.array([False] * len(gt_list))
        for i, (view_list) in enumerate(memory.memory.values()):
            for view in view_list:
                is_matched, idx = self.calculate_matching_single(view, gt_list)
                if is_matched:
                    matched[idx] = True
                    pred_list[i] = True
                    break

        table = PrettyTable()
        table = PrettyTable()
        table.field_names = ['# GT', '# TP', '# FP', 'accuracy', 'recall']
        table.add_row([len(matched), np.sum(matched), len(pred_list) - np.sum(pred_list),
                       f"{np.sum(pred_list) / len(pred_list):.2f}", f"{np.sum(matched) / len(matched):.2f}"])
        print(table)
    
    def calculate_matching_single(self, view, gt_list, radius=20):
        box = view.box
        heading_left, heading_right = geocode_utils.get_heading_range_to_box(
            box, view.shape, view.heading, view.fov
        )
        
        for i, gt_geocode in enumerate(gt_list):
            gt_heading, gt_distance = geocode_utils.get_heading_and_distance_by_geocode(
                view.geocode, gt_geocode
            )
            if geocode_utils.is_heading_in_range((heading_left, heading_right), gt_heading) \
                    and gt_distance < radius:
                return True, i
        
        return False, None
    
    def formulate_output(self, output_cfg):
        table = self.calculate_number(self.memory)
        print(table)
    
    @staticmethod
    def calculate_number(memory):
        count_results = memory.count_category()

        table = PrettyTable()
        table.field_names = ["Category"] + list(count_results.keys())
        table.add_row(["N/A"] + list(count_results.values()))
        return table
