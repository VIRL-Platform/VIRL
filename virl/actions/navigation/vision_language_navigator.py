import ast
import os
import time

import numpy as np

from .navigator_template import NavigatorTemplate
from virl.lm import UnifiedChat
from virl.lm import prompt as prompt_templates
from virl.utils import geocode_utils, common_utils
from virl.utils.geocode_utils import DIRECTION_SET_ABS
from virl.perception.recognizer.recognizer import Recognizer


class VisionLanguageNavigator(NavigatorTemplate):
    def __init__(self, cfg, platform, messager, start_location, output_dir, **kwargs):
        super().__init__(cfg, platform, messager, start_location, output_dir, no_init_mover=True, **kwargs)

        vision_model_cfg = kwargs['vision_model_cfg']
        self.instruction = kwargs['instruction']
        self.chatbot = UnifiedChat()
        
        self.route_info = kwargs.get('route_info', None)
        if self.route_info is not None:
            self.landmark_list = self.route_info['route_results']['landmark_list'] + [self.route_info['dest_place']]
            self.oracle_observation_list = []
            self.key_positions = self.route_info['route_results']['geocode_list']
        
        self.landmarks = self.extract_landmarks_in_instruction()
        self.candidates = self.get_candidates()

        self.landmark_detector = self.build_landmark_detector(cfg.LANDMARK_DETECT, vision_model_cfg)
        self.action_list = []
        self.observation_list = []
        self.step_counter = 1
        self.orientation_set = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']
        self.orientation_heading = [0, 45, 90, 135, 180, 225, 270, 315]

        self.current_heading = kwargs['agent_heading']
        self.target_heading = self.current_heading

        if os.path.exists(os.path.join(output_dir, 'navigator.pkl')):
            self.resume_navigator(output_dir)
        
        platform.initialize_mover(initial_geocode=self.current_geocode)
        time.sleep(2)  # wait for the platform to initialize the mover

    def get_candidates(self):
        if len(self.cfg.LANDMARK_DETECT.CANDIDATES) == 0:
            return ',,'.join(self.landmarks)

        candidate_list = self.cfg.LANDMARK_DETECT.CANDIDATES.split(',')
        for landmark in self.landmarks:
            if landmark in candidate_list:
                idx = candidate_list.index(landmark)
                candidate_list.pop(idx)

            candidate_list.append(landmark)

        candidates = ',,'.join(candidate_list)
        return candidates

    def build_landmark_detector(self, recognize_cfg, vision_model_cfg):
        return Recognizer(vision_model_cfg, recognize_cfg, messager=self.messager, platform=self.platform)

    def extract_landmarks_in_instruction(self):
        landmark_extract_prompt_template = getattr(prompt_templates, self.cfg.LANDMARK_EXTRACT.PROMPT)
        landmark_extract_prompt = landmark_extract_prompt_template.format(instruction=self.instruction)
        answer = self.chatbot.ask(landmark_extract_prompt, model=self.cfg.LANDMARK_EXTRACT.MODEL)
        landmarks = ast.literal_eval(answer)

        while 'next intersection' in landmarks:
            landmarks.remove('next intersection')

        while 'intersection' in landmarks:
            landmarks.remove('intersection')

        if self.cfg.LANDMARK_EXTRACT.get('PLACE_TYPE', False) and hasattr(self, 'landmark_list'):
            cared_place_type = [x.strip() for x in open(self.cfg.LANDMARK_EXTRACT.PLACE_TYPE, 'r').readlines()]
            landmarks = self.add_place_type_to_landmark(
                landmarks, cared_place_type, self.cfg.LANDMARK_EXTRACT.get('PLACE_MODE', 'parenthesis')
            )

        print(f'>>> VisionLanguageNavigator: extracted landmarks: {landmarks}')
        return landmarks

    def add_place_type_to_landmark(self, landmark, cared_place_types, mode):
        new_landmarks = landmark.copy()
        for landmark_info in self.landmark_list:
            if landmark_info is None:
                continue
            landmark_name = landmark_info['name']
            if landmark_name not in new_landmarks:
                continue
            landmark_types = landmark_info['place_types']
            landmark_type = common_utils.list_intersection(landmark_types, cared_place_types)
            if len(landmark_type) == 0:
                continue
            else:
                landmark_type = landmark_type[-1]
            landmark_idx = new_landmarks.index(landmark_name)
            if mode == 'parenthesis':
                new_landmarks[landmark_idx] = f'{landmark_name} ({landmark_type.replace("_", " ")})'
            elif mode == 'sentence':
                new_landmarks[landmark_idx] = f'a {landmark_type.replace("_", " ")} {landmark_name}'
            else:
                raise ValueError

        return new_landmarks

    def actions_before_moving(self, info_dict):
        observation, oracle_observation = self.get_vision_observation(info_dict)
        if self.cfg.ACTION_PRED.get('AUTO_REGRESSIVE', True):
            action_sequence = self.get_observation_action_sequence(self.action_list, self.observation_list)
        else:
            action_sequence = self.get_observation_action_sequence(self.action_list, self.oracle_observation_list)

        cur_action_sequence = f"O_{self.step_counter}: {observation}\n" + f"A_{self.step_counter}: "
        previous_action_sequence = action_sequence + cur_action_sequence

        # predict the next action
        action_pred_prompt_template = getattr(prompt_templates, self.cfg.ACTION_PRED.PROMPT)
        action_pred_prompt = action_pred_prompt_template.format(
            instruction=self.instruction, action_sequence=previous_action_sequence
        )
        action = self.chatbot.ask(action_pred_prompt, model=self.cfg.ACTION_PRED.MODEL)
        info_dict['action'] = action
        info_dict['observation'] = observation
        info_dict['oracle_observation'] = oracle_observation

        self.action_list.append(action)
        self.observation_list.append(observation)
        if hasattr(self, 'oracle_observation_list'):
            self.oracle_observation_list.append(oracle_observation)
        self.step_counter += 1

    @staticmethod
    def get_observation_action_sequence(action_list, observation_list):
        previous_observations = ""
        for i, (action, observation) in enumerate(zip(action_list, observation_list)):
            previous_observations += f"O_{i + 1}: {observation}\n"
            previous_observations += f"A_{i + 1}: {action}\n"
        return previous_observations
    
    def get_vision_observation(self, info_dict):
        # get intersection observation
        intersection_observation = self.get_intersection_observation(info_dict)
        
        has_oracle_landmark = False
        if self.cfg.LANDMARK_DETECT.get('MANNER', 'visual') == 'visual':
            landmark_observation = self.get_landmark_observation(info_dict)
        elif self.cfg.LANDMARK_DETECT.get('MANNER', 'visual') == 'oracle':
            landmark_observation = self.get_landmark_observation_oracle(info_dict)
            has_oracle_landmark = True
        else:
            raise ValueError
        
        observation = landmark_observation + ';' + intersection_observation
        if self.route_info is not None and has_oracle_landmark:
            oracle_landmark_observation = landmark_observation
        elif self.route_info is not None and not has_oracle_landmark:
            oracle_landmark_observation = self.get_landmark_observation_oracle(info_dict)
            oracle_observation = landmark_observation + '; ' + intersection_observation
        else:
            oracle_landmark_observation = None
        
        if oracle_landmark_observation is not None:
            oracle_observation = oracle_landmark_observation + '; ' + intersection_observation
        else:
            oracle_observation = None
        
        return observation, oracle_observation

    def get_landmark_observation(self, info_dict):
        thresh = self.cfg.LANDMARK_DETECT.THRESH
        image_list = self.platform.get_all_streetview_from_geocode(
            self.current_geocode, cur_heading=self.current_heading
        )
        visual_obs_info = {
            'score': 0,
            'landmark': None,
            'spatial': None,
            'view': None,
        }
        for i, street_image in enumerate(image_list):
            results = self.landmark_detector.check(
                street_image.image, self.candidates, self.landmarks
            )
            max_idx = np.argmax(results['scores'])
            if results['scores'][max_idx] > max(thresh, visual_obs_info['score']):
                visual_obs_info['score'] = results['scores'][max_idx]
                visual_obs_info['landmark'] = results['labels'][max_idx]
                visual_obs_info['spatial'] = geocode_utils.calculate_spatial_relationship_with_headings(
                    self.current_heading, street_image.heading
                )
                visual_obs_info['view'] = street_image

            print(f"View {i}: {results['labels'][max_idx]}: {results['scores'][max_idx]}")
        
        info_dict['visual_obs'] = visual_obs_info
        if visual_obs_info['landmark'] is not None:
            observation = f"{visual_obs_info['landmark']} is on your {visual_obs_info['spatial']}"
        else:
            observation = "No landmarks nearby"
        return observation

    def get_landmark_observation_oracle(self, info_dict):
        assert self.route_info is not None
        # calculate the distance between the current position and the destination
        dest = self.route_info['dest_place']
        dist_to_dest = geocode_utils.calculate_distance_from_geocode(self.current_geocode, dest['geocode'])
        if dist_to_dest < self.cfg.LANDMARK_DETECT.ORACLE_RADIUS:
            info_dict['dest'] = True
        else:
            info_dict['dest'] = False
        
        # search nearby landmark for intersection
        keypoint_list = self.route_info['route_results']['geocode_list'][:-1]
        if len(keypoint_list) > 0:
            dist_keypoint_list = geocode_utils.cal_distance_between_two_position_list([self.current_geocode], keypoint_list)[0]
            min_idx_keypoint = np.argmin(dist_keypoint_list)
            min_dist_keypoint = dist_keypoint_list[min_idx_keypoint]
            
            if min_dist_keypoint > self.cfg.LANDMARK_DETECT.KEYPOINT_RADIUS and not info_dict['intersection'] and \
                    not info_dict['dest']:
                return "No landmarks nearby"
        
        dist_landmark_list = []
        for landmark in self.landmark_list:
            if landmark is not None:
                distance = geocode_utils.calculate_distance_from_geocode(self.current_geocode, landmark['geocode'])
                dist_landmark_list.append(distance)
            else:
                dist_landmark_list.append(1000000)

        min_idx = np.argmin(dist_landmark_list)
        min_distance = dist_landmark_list[min_idx]

        if min_distance < self.cfg.LANDMARK_DETECT.ORACLE_RADIUS:
            landmark = self.landmark_list[min_idx]
        else:
            return "No landmarks nearby"

        # calculate the spatial relationship
        heading = geocode_utils.calculate_heading_between_geocodes(self.current_geocode, landmark['geocode'])
        spatial = geocode_utils.calculate_spatial_relationship_with_headings(self.current_heading, heading)
        return f"{landmark['name']} is on your {spatial}"
    
    def get_intersection_observation(self, info_dict):
        # since current intersection observation should be oracle, we need to avoid provide too much noise
        intersection_valid_here = True
        if self.route_info is not None and len(self.route_info['route_results']['geocode_list']) > 1:
            intersect_list = self.route_info['route_results']['geocode_list'][:-1]
            distance = geocode_utils.cal_distance_between_two_position_list([self.current_geocode], intersect_list)[0]
            min_dist = np.min(distance)
            if min_dist > self.cfg.LANDMARK_DETECT.INTERSECTION_VALID_RADIUS:
                intersection_valid_here = False
                
        print(f'intersection_valid_here: {intersection_valid_here}')
        
        # for previous detect a landmark, and the agent decide to turn direction.
        # we need to check whether the agent is at the intersection
        if len(self.action_list) > 0 and 'turn_direction' in self.action_list[-1] and \
                'No landmarks nearby' not in self.observation_list[-1]:
            intersection_valid_here = True
        
        heading_list = self.platform.mover.get_all_suitable_heading_to_path_vln(
            self.current_geocode, radius_query=intersection_valid_here
        )
        info_dict['heading_list'] = heading_list
        
        if len(heading_list) > 2 and intersection_valid_here:
            observation = f' There are {len(heading_list)}-way intersections.'
            info_dict['intersection'] = True
        else:
            observation = ""
            info_dict['intersection'] = False

        return observation

    def move(self, info_dict):
        action = info_dict['action']
        heading_list = info_dict['heading_list']
        if action in ['forward()']:
            move_idx = geocode_utils.select_argmin_heading_from_heading_list(self.target_heading, heading_list)
            heading_diff = geocode_utils.cal_min_heading_diff_between_headings(self.target_heading, heading_list[move_idx])
            if heading_diff > self.cfg.get('MAX_HEADING_DIFF', 45):
                heading_list = self.platform.mover.get_all_suitable_heading_to_path_vln(
                    self.current_geocode, radius_query=True
                )
                move_idx = geocode_utils.select_argmin_heading_from_heading_list(self.target_heading, heading_list)

            self.platform.mover.adjust_heading_web(heading_list[move_idx])
            self.current_geocode = self.platform.mover.move(move_idx)
            self.current_heading = heading_list[move_idx]
        elif 'turn_direction' in action:
            direction = action.split('(')[1].split(')')[0]
            if direction in DIRECTION_SET_ABS:
                # to change the agent heading
                orientation = action.split('(')[1].split(')')[0]
                orientation_idx = self.orientation_set.index(orientation)
                self.current_heading = self.orientation_heading[orientation_idx]
                self.target_heading = self.current_heading

                self.platform.mover.adjust_heading_web(self.current_heading)
                
                # forward
                heading_list = self.platform.mover.get_all_suitable_heading_to_path_vln(
                    self.current_geocode, radius_query=True
                )
                move_idx = geocode_utils.select_argmin_heading_from_heading_list(self.target_heading, heading_list)
                self.platform.mover.adjust_heading_web(heading_list[move_idx])
                self.current_geocode = self.platform.mover.move(move_idx)
                self.current_heading = heading_list[move_idx]
        else:
            # to handle 'stop()' or error action not in the action list
            pass

        print(f'>>> VisionLanguageNavigator: after moving, current geocode is: {self.current_geocode}')

    def check_stop(self, info_dict):
        # check to interrupt or not
        if self.cfg.get('INTERRUPT', None) and self.cfg.INTERRUPT.ENABLED:
            if self.check_interrupt(info_dict):
                return True
        
        action = info_dict['action']
        if action == 'stop()':
            return True
        else:
            return False

    def actions_before_stop(self, info_dict):
        action_sequence = self.get_observation_action_sequence(self.action_list, self.observation_list)
        info_dict['action_sequence'] = action_sequence
        print(action_sequence)

    def check_interrupt(self, info_dict):
        # case 1: the agent is at the same position for a long time
        # if all the previous trajectory are the same, then interrupt
        
        interrupt = False
        if len(self.trajectory) >= self.cfg.INTERRUPT.STATIC_COUNTER:
            interrupt = True
            idx = int(f'-{self.cfg.INTERRUPT.STATIC_COUNTER}')
            previous_actions = self.action_list[idx:]
            for i in range(len(previous_actions)):
                if 'turn_direction' not in previous_actions[i]:
                    interrupt = False
                    break

        if interrupt:
            print(f'>>> VisionLanguageNavigator: Interrupted by static trajectory {self.cfg.INTERRUPT.STATIC_COUNTER} times')
            return interrupt

        # case 2: the agent keep moving to the wrong direction for a long time
        if len(self.trajectory) >= self.cfg.INTERRUPT.OPPOSITE_COUNTER:
            interrupt = True
            idx = int(f'-{self.cfg.INTERRUPT.STATIC_COUNTER}')
            previous_trajectory = self.trajectory[idx:]
            distance_matrix = geocode_utils.cal_distance_between_two_position_list(previous_trajectory, self.key_positions)
            for i in range(len(distance_matrix) - 1):
                if (distance_matrix[i] > distance_matrix[i + 1]).any():
                    interrupt = False
                    break

        if interrupt:
            print(f'>>> VisionLanguageNavigator: Interrupted by opposite trajectory {self.cfg.INTERRUPT.OPPOSITE_COUNTER} times')
            return interrupt

        # case 3: excess the max steps:
        if self.step_counter >= self.cfg.INTERRUPT.MAX_STEPS:
            interrupt = True
            print(f'>>> VisionLanguageNavigator: Interrupted by max steps {self.cfg.INTERRUPT.MAX_STEPS}')
        
        return interrupt
