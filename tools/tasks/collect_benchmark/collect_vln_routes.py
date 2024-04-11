import tqdm
import pickle
import os
import json
import glob
import numpy as np

from shapely.geometry import Point, Polygon

from tools.tasks.task_template import TaskTemplate

from virl.config import cfg
from virl.utils import common_utils, geocode_utils, pipeline, place_utils
from virl.perception.recognizer.recognizer import Recognizer
from virl.lm import prompt as prompt_templates


class CollectVLNRoutes(TaskTemplate):
    def __init__(self, output_dir, logger):
        super().__init__(output_dir, logger)

        self.route_info_path = os.path.join(self.output_dir, 'route_infos.json')
        if os.path.exists(self.route_info_path):
            infos = json.load(open(self.route_info_path, 'r'))
            self.valid_route_infos = infos[0]
            self.dest_idx = infos[1]
        else:
            self.valid_route_infos = []
            self.dest_idx = 0
        
        self.potential_dest_path = os.path.join(self.output_dir, 'potential_dest.pkl')
        if os.path.exists(self.potential_dest_path):
            infos = pickle.load(open(self.potential_dest_path, 'rb'))
            self.potential_dest = infos[0]
            self.potential_dest_queue = infos[1]
        else:
            self.potential_dest = self.potential_dest_queue = None

        place_type_path = cfg.PIPELINE.GENERATE_PLACE_QUEUE.PLACE_TYPES
        self.place_types = [x.strip() for x in open(place_type_path).readlines()]

        self.place_infos = None
        # all places infos consist of all the places in the region, without filtering by the storefront
        self.all_places_infos = None

    def run(self, platform, agent, chatbot, messager, args, **kwargs):
        pipeline_cfg = cfg.PIPELINE
        
        # step 1: find potential destinations
        common_utils.print_stage('Step 1: Find potential destinations')
        place_infos = pickle.load(open(pipeline_cfg.GENERATE_PLACE_QUEUE.PLACE_INFO_PATH, 'rb'))
        # filter by the place region
        self.place_infos = common_utils.filter_place_by_region(place_infos, pipeline_cfg.GENERATE_PLACE_QUEUE.REGION_KEY)
        
        if self.potential_dest is None and pipeline_cfg.GENERATE_PLACE_QUEUE.ENABLED:
            self.find_potential_destinations(pipeline_cfg.GENERATE_PLACE_QUEUE, platform)

        # step 2: generate routes
        common_utils.print_stage('Step 2: Generate routes')

        if len(self.valid_route_infos) >= cfg.TASK_INFO.MAX_ROUTE_NUM:
            return

        for place_id, _ in tqdm.tqdm(self.potential_dest_queue[self.dest_idx:], total=len(self.potential_dest_queue[self.dest_idx:])):
            dest_place = self.potential_dest[place_id]
            dest_geocode = dest_place['relocated_geocode']
            
            # step 2.1: sample the start location
            candidate_start_places = self.sample_start_location(
                pipeline_cfg.SAMPLE_START_POSITION, platform, dest_geocode
            )
            
            # step 2.2: generate the route
            start_place, route_results = self.generate_route(
                pipeline_cfg.ROUTE, platform, candidate_start_places, dest_place
            )
            
            self.dest_idx += 1

            if start_place is None:
                print(f'Cannot find a valid route to {dest_place["name"]}')
                continue
            
            # visualization only
            polyline = geocode_utils.encode_polyline([start_place['relocated_geocode']] + route_results['geocode_list'])
            pipeline.draw_planned_route(
                platform, polyline, input_way_points=None,
                path=self.output_dir / f'plan_trajectory_{self.dest_idx}.html',
                file_template=cfg.PIPELINE.OUTPUT.FILE_TEMPLATE
            )

            # step 3: create the ground truth for the route
            instruction, all_milestone_info = self.create_instruction_from_route(
                pipeline_cfg.GENERATE_INSTRUCTION, chatbot, route_results,
                start_place['relocated_geocode'], dest_place
            )

            cur_route_info = {
                'dest_place': dest_place,
                'start_place': start_place,
                'route_results': route_results,
                'init_heading': pipeline_cfg.GENERATE_INSTRUCTION.INIT_HEADING,
                'instruction': instruction,
                'milestone_info': all_milestone_info
            }
            self.valid_route_infos.append(cur_route_info)

            common_utils.dump_json_results([self.valid_route_infos, self.dest_idx], self.route_info_path)
            if len(self.valid_route_infos) >= cfg.TASK_INFO.MAX_ROUTE_NUM:
                break
        
        print(f'Find {len(self.valid_route_infos)} valid routes saved to {self.route_info_path}')
            
    def find_potential_destinations(self, stage_cfg, platform):
        print(f'Find potential destinations from {len(self.place_infos)} places')
        # check the distance to the nearest street view
        # if the distance is lower than the threshold, we consider it as a potential destination
        self.potential_dest = {}
        self.potential_dest_queue = {}
        for place_id, place_info in tqdm.tqdm(self.place_infos.items()):
            # filter by the place type
            cur_place_types = place_info['place_types']
            intersection = common_utils.list_intersection(self.place_types, cur_place_types)
            if len(intersection) == 0:
                continue

            # filter by the distance to the nearest street view
            relocated_position, _ = platform.relocate_geocode_by_source(place_info['geocode'])
            if relocated_position is None:
                continue
            distance = geocode_utils.calculate_distance_from_geocode(place_info['geocode'], relocated_position)
            if distance > stage_cfg.RADIUS_TO_STREET_VIEW:
                continue
            
            place_info['relocated_geocode'] = relocated_position
            self.potential_dest[place_id] = place_info
            # add to the priority queue, the priority is the distance to the nearest street view
            n_reviews = place_info['n_reviews']
            self.potential_dest_queue[place_id] = -n_reviews

        self.potential_dest_queue = sorted(self.potential_dest_queue.items(), key=lambda x: x[1])
        
        # save the potential destinations
        pickle.dump([self.potential_dest, self.potential_dest_queue], open(self.potential_dest_path, 'wb'))
        print(f'Find {len(self.potential_dest)} potential destinations saved to {self.potential_dest_path}')        
        
    def sample_start_location(self, stage_cfg, platform, dest_geocode):
        # sample the start locations
        candidate_start_places = geocode_utils.find_places_within_geocode_and_radius(
            self.place_infos, dest_geocode, stage_cfg.MIN_RADIUS, stage_cfg.MAX_RADIUS
        )
        
        print(f'Find {len(candidate_start_places)} candidate start places in [{stage_cfg.MIN_RADIUS}, {stage_cfg.MAX_RADIUS}] meters')
        return candidate_start_places
    
    def generate_route(self, stage_cfg, platform, candidate_start_places, dest_place):
        candidate_routes = []
        n_landmark_list = []
        for start_place in tqdm.tqdm(candidate_start_places.values(), total=len(candidate_start_places)):
            if 'relocated_geocode' not in start_place:
                relocated_position, _ = platform.relocate_geocode_by_source(start_place['geocode'])
                if relocated_position is None:
                    continue
                start_place['relocated_geocode'] = relocated_position
            
            start_position = start_place['relocated_geocode']
            result_dict = platform.get_routing(
                start_position, dest_place['relocated_geocode'],
                mode=stage_cfg.ROUTE_MODE,
                stopover=stage_cfg.STOPOVER,
                modify_destination=False
            )

            route = result_dict['geocode_list']
            route_distance = result_dict['distance']

            # check whether the destination street view position is close enough to the destination
            dest_dist = geocode_utils.calculate_distance_from_geocode(route[-1], dest_place['geocode'])
            if dest_dist > stage_cfg.MAX_DISTANCE_TO_LANDMARK:
                continue
            
            if route_distance > stage_cfg.MAX_DISTANCE or len(route) - 1 < stage_cfg.MIN_INTERSECT \
                    or len(route) - 1 >= stage_cfg.MAX_INTERSECT:
                continue

            n_landmarks = 0
            cur_landmark_list = []
            for i, milestone in enumerate(route[:-1]):
                potential_landmarks = platform.get_nearby_places(
                    milestone, rankby='distance', no_next_page=True
                )
                landmark = None
                for place in potential_landmarks:
                    if place['distance'] > stage_cfg.MAX_DISTANCE_TO_LANDMARK or place['n_reviews'] < stage_cfg.MIN_REVIEWS:
                        continue
                    
                    if landmark is None:
                        landmark = place
                    elif place['n_reviews'] > landmark['n_reviews']:
                        landmark = place
                
                cur_landmark_list.append(landmark)
                if landmark is not None:
                    n_landmarks += 1

            result_dict['n_landmarks'] = n_landmarks
            result_dict['landmark_list'] = cur_landmark_list

            n_landmark_list.append(n_landmarks)
            candidate_routes.append((start_place, result_dict))

            break

        if len(n_landmark_list) > 0:
            n_landmark_list = np.array(n_landmark_list)
            idx = np.argmax(n_landmark_list)
            return candidate_routes[idx]

        return None, None

    def create_instruction_from_route(self, stage_cfg, chatbot, route_results, start_position,
                                      dest_place):
        route = route_results['geocode_list']
        route_distance = route_results['distance_list']
        landmark_list = route_results['landmark_list']
        
        milestone_info_list = []
        previous_position = start_position
        agent_heading = stage_cfg.INIT_HEADING
        for i, milestone in enumerate(route):
            agent_heading_to_milestone = geocode_utils.calculate_heading_between_geocodes(
                previous_position, milestone
            )
            
            if i == len(route) - 1:
                # if this is the last milestone, we should use the destination place as the landmark
                heading_to = geocode_utils.calculate_heading_between_geocodes(milestone, dest_place['geocode'])
                spatial_relation = geocode_utils.calculate_spatial_relationship_with_headings(
                    agent_heading_to_milestone, heading_to
                )
                landmark_info = {'expression': f"The destination {dest_place['name']} is on your {spatial_relation}"}
            else:
                landmark = landmark_list[i]
                landmark_info = place_utils.calculate_milestone_information([landmark], agent_heading_to_milestone)

            milestone_info = place_utils.fill_in_milestone_information_template(
                landmark_info, i, agent_heading, route_distance[i], agent_heading_to_milestone,
                stage_cfg.MILESTONE_TEMPLATE
            )
            milestone_info_list.append(milestone_info)

            agent_heading = agent_heading_to_milestone
            previous_position = milestone
        
        all_milestone_info = "".join(milestone_info_list)

        all_milestone_info_template = getattr(prompt_templates, stage_cfg.ALL_INFO_PROMPT)
        all_milestone_info_prompt = all_milestone_info_template.format(
            milestone_information=all_milestone_info.strip()
        )
        instruction = chatbot.ask(all_milestone_info_prompt, model=stage_cfg.MODEL)

        return instruction, all_milestone_info
