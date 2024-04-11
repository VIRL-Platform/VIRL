import os
import pickle
import json
import tqdm

import numpy as np
from prettytable import PrettyTable

from tools.tasks.vision_language_nav.vln_template import VLNTemplate

from virl.config import cfg
from virl.utils import common_utils, geocode_utils, pipeline


class BMVisionLanguageNavigation(VLNTemplate):
    def __init__(self, output_dir, logger):
        super().__init__(output_dir, logger)

        self.results_path = self.output_dir / 'results.json'

        # metrics
        self.num_success = 0
        self.keypoint_status = {
            'start': {
                'react': 0,
            },
            'stop': {
                'arrive': 0,
            },
            'intersection': {
                'react': 0,
                'arrive': 0,
                'total': 0,
            },
        }
        self.is_success = []

        self.step_counter = 0
        
        if os.path.exists(self.results_path):
            self.resume_results()
        
    def run(self, platform, agent, chatbot, messager, args, **kwargs):
        pipeline_cfg = cfg.PIPELINE
        
        # load data
        infos = json.load(open(pipeline_cfg.DATA.ROUTE_PATH, 'r'))
        self.route_info = infos[0]

        for route_info in tqdm.tqdm(self.route_info[self.step_counter:], total=len(self.route_info[self.step_counter:])):
            # route_info = self.route_info[9]
            results = self.navigate_single(
                pipeline_cfg, platform, chatbot, messager, agent, args, route_info
            )
            # evaluate the results
            self.evaluation_single(pipeline_cfg.EVALUATION, results, route_info)
            
            os.rename(self.output_dir / 'navigator.pkl', self.output_dir / f'navigator_{self.step_counter}.pkl')
            os.rename(self.output_dir / 'plan_trajectory.html', self.output_dir / f'plan_trajectory_{self.step_counter}.html')
            os.rename(self.output_dir / 'trajectory.html', self.output_dir / f'trajectory_{self.step_counter}.html')
            # save the results
            self.step_counter += 1
            self.save_results()
            
        # formulate output
        self.formulate_output()
    
    def formulate_output(self):
        # success rate
        num_failed = len(self.route_info) - self.num_success
        table1 = PrettyTable()
        table1.field_names = ["Total Route", "Success", "Failed", "Success Rate"]
        table1.add_row([len(self.route_info), self.num_success, num_failed, self.num_success / len(self.route_info)])

        print(table1)

        # key point accuracy
        table2 = PrettyTable()
        table2.field_names = ["", "Start position", "Intersection", "Stop"]
        table2.add_row([
            "Arrive", "-",
            f"{(self.keypoint_status['intersection']['arrive'] / (self.keypoint_status['intersection']['total'] + 1e-6)):.2f}",
            f"{(self.keypoint_status['stop']['arrive'] / (num_failed + 1e-6)):.2f}"
        ])
        table2.add_row([
            "React",
            f"{(self.keypoint_status['start']['react'] / (num_failed + 1e-6)):.2f}",
            f"{(self.keypoint_status['intersection']['react'] / (self.keypoint_status['intersection']['arrive'] + 1e-6)):.2f}",
            "0"
        ])
        print(table2)
    
    def evaluation_single(self, eval_cfg, results, route):
        geocode = results['geocode']
        trajectory = results['trajectory']
        action_list = results['action_list']
        dest_geocode = route['dest_place']['geocode']
        
        # evaluation success rate
        distance = geocode_utils.calculate_distance_from_geocode(geocode, dest_geocode)
        if distance < eval_cfg.SUCCESS_RADIUS:
            self.num_success += 1
            print('Navigation Success!')
            self.is_success.append(True)
            return
        else:
            self.is_success.append(False)
            print('Navigation Failed!')

        # check key point accuracy on the start position
        start_action = action_list[0]
        if self.check_correctness_for_start_action(route, start_action):
            self.keypoint_status['start']['react'] += 1
        
        # check key point accuracy on the stop position
        # check the trajectory, if the trajectory is close to the destination close enough, then it is arrived
        dest_geocode = route['dest_place']['geocode']
        distance = geocode_utils.cal_distance_between_two_position_list([dest_geocode], trajectory)[0]
        if min(distance) < eval_cfg.STOP_ARRIVE_RADIUS:
            self.keypoint_status['stop']['arrive'] += 1
        
        # check key point accuracy on the intersection
        intersection_list = route['route_results']['geocode_list'][:-1]
        self.keypoint_status['intersection']['total'] += len(intersection_list)
        # calculate the arrive of the intersection
        distance = geocode_utils.cal_distance_between_two_position_list(intersection_list, trajectory)
        min_dist = np.min(distance, axis=1)
        argmin_dist = np.argmin(distance, axis=1)
        arrive_mask = min_dist < eval_cfg.INTERSECT_ARRIVE_RADIUS
        self.keypoint_status['intersection']['arrive'] += np.sum(arrive_mask)
        
        # calculate the correct of the intersection
        for i, intersect_geocode in enumerate(intersection_list):
            if not arrive_mask[i]:
                continue
            
            # calculate the heading
            action = self.get_direction_for_n_turn(i+1, route)
            idx_range = eval_cfg.INTERSECT_REACT_RANGE
            min_idx = argmin_dist[i]
            nearby_action_list = action_list[max(min_idx-idx_range, 0):min_idx+idx_range]
            if action in nearby_action_list:
                self.keypoint_status['intersection']['react'] += 1
    
    def check_correctness_for_start_action(self, route, start_action):
        answer = self.get_direction_for_n_turn(0, route)
        
        return answer == start_action
    
    @staticmethod
    def get_direction_for_n_turn(n_turn, route):
        milestone_info = common_utils.parse_str_json_list_to_list(route['milestone_info'])
        direction = milestone_info[n_turn]['to_next_intersection_heading'].split('(')[1].split(')')[0]
        answer = f'turn_direction({direction})'
        
        return answer
        
    def save_results(self):
        # save metrics
        results = {
            'num_success': self.num_success,
            'keypoint_status': self.keypoint_status,
            'step_counter': self.step_counter,
            'is_success': self.is_success,
        }
        common_utils.dump_json_results(results, self.results_path)

    def resume_results(self):
        results = json.load(open(self.results_path, 'r'))

        self.num_success = results['num_success']
        self.keypoint_status = results['keypoint_status']
        self.step_counter = results['step_counter']
        self.is_success = results['is_success']
