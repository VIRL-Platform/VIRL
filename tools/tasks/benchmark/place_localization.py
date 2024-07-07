import os
import pickle
import tqdm

import numpy as np

from PIL import Image
from shapely.geometry import Point, Polygon
from prettytable import PrettyTable

from tools.tasks.task_template import TaskTemplate

from virl.config import cfg
from virl.actions.check_surrounding.visual_checker import VisualChecker
from virl.actions.navigation import build_navigator
from virl.utils import common_utils, geocode_utils, vis_utils


class BMStreetLoc(TaskTemplate):
    def __init__(self, output_dir, logger):
        super().__init__(output_dir, logger)
        self.preds = []
        self.gts = []
        self.place_infos = None

        self.polygon_area = None
        self.positions = None

        self.place_position_matrix = None

        # shape: (num_positions, num_places)
        self.dist_matrix = None

        self.place_types = []
        self.step_counter = 0

        # record results for active detection
        self.has_active_detect = cfg.PIPELINE.CHECK_SURROUNDING.DETECT.get('ADJUST_CAMERA', False)
        self.active_detected_places = {}
        self.active_matched_places = {}
        
        # record results for non-active detection
        # detected places are both located and correctly recognized
        self.detected_places = {}
        # matched places are only located but not necessarily correctly recognized
        self.matched_places = {}
        
        if cfg.PIPELINE.EVALUATION.EVAL_ONLY:
            self.ckpt_path = self.output_dir / 'checkpoint.pkl'
        else:
            self.ckpt_path = self.output_dir / cfg.PIPELINE.CHECK_SURROUNDING.DETECT.NAME / 'checkpoint.pkl'
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        if os.path.exists(self.ckpt_path):
            self.resume_results()

    def run(self, platform, agent, chatbot, messager, args, **kwargs):
        if cfg.PIPELINE.EVALUATION.EVAL_ONLY:
            self.place_types = [line.strip().replace('_', ' ') for line in 
                                open(cfg.PIPELINE.CHECK_SURROUNDING.CARED_LABELS_PATH, 'r').readlines()]
            self.eval_results_all()
            return

        start_position = agent.start_position
        pipeline_cfg = cfg.PIPELINE

        common_utils.print_stage('Stage 1: Loading navigator')
        self.navigator = build_navigator(
            cfg.PIPELINE.NAVIGATION, platform, messager, start_position, self.output_dir, resume=args.resume
        )
        self.polygon_area = self.navigator.polygon_area
        self.positions = self.navigator.route

        # initialize the visual checker
        common_utils.print_stage('Stage 2: Loading detectors')
        visual_checker = VisualChecker(cfg.PIPELINE.CHECK_SURROUNDING, platform, messager)
        self.place_types = visual_checker.cared_labels
        
        # loading place info
        common_utils.print_stage('Stage 3: Prepare data and gts')
        self.prepare_data_and_gt(pipeline_cfg.PREPARE_DATA, platform)

        current_geocode = self.navigator.get_current_geocode()
        current_heading = self.navigator.get_current_heading()

        while self.step_counter < len(self.positions) - 1:
            # sense surroundings
            finish_navigation = False
            
            # search all nearby places:
            nearby_place_infos = self.find_places_nearby(
                self.step_counter, current_geocode, pipeline_cfg.PREPARE_DATA.MAX_DIST_TO_STREET_VIEW
            )
            
            if len(nearby_place_infos) > 0:
                # image_list = platform.get_all_streetview_from_geocode(current_geocode, cur_heading=current_heading)
                image_list = self.get_street_view_image_with_fov_list(
                    platform, current_geocode, current_heading, pipeline_cfg.GET_STREET_VIEW.FOV_LIST
                )

                for i, street_image in enumerate(image_list):
                    find_result, result_dict = visual_checker.visual_sensing_single(street_image)

                    # query the ground truth.
                    self.record_results(result_dict, nearby_place_infos)

            is_finish, current_geocode, info_dict = self.navigator.navigate(
                info_dict={'find_result': finish_navigation})
            current_heading = self.navigator.get_current_heading()

            if is_finish:
                break
            
            self.step_counter += 1
            if self.step_counter % cfg.get('SAVE_INTERVAL', 10000000) == 0:
                self.save_results()

        # output results
        self.save_results()
        self.formulate_output()

        # remove navigator
        if os.path.exists(self.output_dir / 'navigator.pkl'):
            os.remove(self.output_dir / 'navigator.pkl')
        
    def formulate_output(self):
        num_total_place = len(self.place_infos)
        num_detected_place = len(self.detected_places)
        num_matched_place = len(self.matched_places)
        
        table = PrettyTable()
        table.field_names = ['method', '# total_place', '# TP', '# Category-agnostic TP', 'accuracy', 'recall', 'R']
        table.add_row(['non-active', num_total_place, num_detected_place, num_matched_place,
                       f'{num_detected_place / max(num_matched_place, 1e-6):.2f}',
                       f'{num_matched_place / max(num_total_place, 1e-6):.2f}',
                       f'{num_detected_place / max(num_total_place, 1e-6):.3f}'
                       ])
        if self.has_active_detect:
            table.add_row([
                'active', num_total_place, len(self.active_detected_places), len(self.active_matched_places),
                f'{len(self.active_detected_places) / max(len(self.active_matched_places), 1e-6):.2f}',
                f'{len(self.active_matched_places) / max(num_total_place, 1e-6):.2f}'
            ])
        print(table)

        # export table to csv
        table_file = self.output_dir / 'results_all.csv'
        with open(table_file, 'w', newline='') as f:
            f.write(table.get_csv_string())

        self.formulate_category_output(self.matched_places, self.detected_places)
        if self.has_active_detect:
            print('active detection results:')
            self.formulate_category_output(self.active_matched_places, self.active_detected_places)

        if not hasattr(self, 'matched_places_city'):
            return
        
        self.formulate_specific_output('city', self.matched_places_city, self.detected_places_city, self.place_infos_city)
        self.formulate_specific_output('continent', self.matched_places_continent, self.detected_places_continent, self.place_infos_continent)

    def formulate_category_output(self, matched_places, detected_places):
        table = PrettyTable()
        table.field_names = ['category', '# total', '# TP', '# Category-agnostic TP', 'accuracy', 'recall', 'R']
        
        total = common_utils.count_place_types(self.place_infos, self.place_types)
        matched = {}
        detected = {}
        for place_id in matched_places.keys():
            gt_types = self.place_infos[place_id]['place_types']
            for type in gt_types:
                matched[type] = matched.get(type, 0) + 1

        for result_lists in detected_places.values():
            cur_pred_types = []
            for pred_type, gt_types in result_lists:
                if pred_type not in cur_pred_types:
                    detected[pred_type] = detected.get(pred_type, 0) + 1
                    cur_pred_types.append(pred_type)
        
        accuracy_list = []
        recall_list = []
        r_list = []
        for type in total.keys():
            type_acc = detected.get(type, 0) / matched.get(type, 1e-6)
            type_recall = matched.get(type, 0) / max(total[type], 1e-6)
            type_r = detected.get(type, 0) / max(total[type], 1e-6)
            
            table.add_row([
                type, total[type], detected.get(type, 0),  matched.get(type, 0),
                f'{type_acc:.3f}',
                f'{type_recall:.3f}',
                f'{type_r:.3f}'
            ])
            accuracy_list.append(type_acc)
            recall_list.append(type_recall)
            r_list.append(type_r)
        
        # add average row
        table.add_row([
            'average', np.average(list(total.values())),
            np.average(list(detected.values())), np.average(list(matched.values())),
            f'{np.average(accuracy_list):.3f}',
            f'{np.average(recall_list):.3f}',
            f'{np.average(r_list):.3f}'
        ])
        
        print(table)
        # export table to csv
        table_file = self.output_dir / 'results_category.csv'
        with open(table_file, 'w', newline='') as f:
            f.write(table.get_csv_string())

    def formulate_specific_output(self, category, matched, detected, place_types):
        table = PrettyTable()
        table.field_names = [category, '# total', '# TP', '# Category-agnostic TP', 'accuracy', 'recall']

        for type in place_types.keys():
            table.add_row([
                type, place_types[type], detected.get(type, 0),  matched.get(type, 0),
                f'{detected.get(type, 0) / (matched.get(type, 0) +  + 1e-6):.2f}',
                f'{matched.get(type, 0) / place_types[type]:.2f}'
            ])

        print(table)
        # export table to csv
        table_file = self.output_dir / f'results_{category}.csv'
        with open(table_file, 'w', newline='') as f:
            f.write(table.get_csv_string())
    
    def record_results(self, result_dict, nearby_place_infos):
        if result_dict['first_detect']['is_detected']:
            self.calculate_matching(
                result_dict['first_detect'], nearby_place_infos, self.matched_places, self.detected_places
            )

        if self.has_active_detect and result_dict['final_detect']['is_detected']:
            self.calculate_matching(
                result_dict['final_detect'], nearby_place_infos, self.active_matched_places,
                self.active_detected_places
            )
    
    def load_place_info(self, data_cfg, platform):
        """
        Load and filter place information
        Args:
            data_cfg:

        Returns:

        """
        local_place_info_path = self.output_dir / 'place_infos.pickle'
        if data_cfg.PLACE_INFO != 'None':
            place_infos = pickle.load(open(data_cfg.PLACE_INFO, 'rb'))
            print(f'>>> Load {len(place_infos)} place infos from {data_cfg.PLACE_INFO}')
        elif os.path.exists(local_place_info_path):
            place_infos = pickle.load(open(local_place_info_path, 'rb'))
            print(f'>>> Load {len(place_infos)} place infos from {local_place_info_path}')
        else:
            place_infos = {}
            for position in tqdm.tqdm(self.positions, total=len(self.positions)):
                results = platform.get_nearby_places(
                    geocode=position, cal_distance=True,
                    type_custom=data_cfg.PLACE_TYPE, polygon_filter=Polygon(self.polygon_area),
                    no_next_page=data_cfg.NO_NEXT_PAGE, rankby='distance',
                )
                for result in results:
                    if result['place_id'] not in place_infos:
                        place_infos[result['place_id']] = result
            
            print(f'>>> Search {len(place_infos)} place infos from platform.')
            pickle.dump(place_infos, open(local_place_info_path, 'wb'))

        # filter place infos by place types and number of reviews
        filtered_place_infos = {}
        for place_id, place_info in place_infos.items():
            place_types = [type.replace('_', ' ') for type in place_info['place_types']]
            intersect_types = common_utils.list_intersection(place_types, self.place_types)
            if len(intersect_types) > 0 and place_info['n_reviews'] > 0:
                place_info['place_types'] = intersect_types
                filtered_place_infos[place_id] = place_info
        
        # filter place infos to remove those not near the street views.
        self.place_infos = self.filter_place_info_by_distance(filtered_place_infos, data_cfg)
        # filter place infos to remove those with less than MIN_REVIEWS
        self.place_infos = self.filter_by_n_reivews(self.place_infos, data_cfg)
        
    def filter_by_n_reivews(self, place_infos, data_cfg):
        new_place_infos = {}

        for i, (place_id, place) in enumerate(place_infos.items()):
            if place['n_reviews'] >= data_cfg.MIN_REVIEWS:
                new_place_infos[place_id] = place
        
        return new_place_infos

    def prepare_data_and_gt(self, data_cfg, platform):
        self.load_place_info(data_cfg, platform)
    
    def filter_place_info_by_distance(self, place_infos, data_cfg):
        """
        Filter place infos by distance to positions.
        Args:
            place_infos (dict): place infos

        Returns:
            filtered_place_infos (dict): filtered place infos
        """
        all_place_positions = [place_info['geocode'] for place_info in place_infos.values()]

        distance_matrix = geocode_utils.cal_distance_between_two_position_list(self.positions, all_place_positions)
        
        # min_dist_to_nearest_street_view
        min_dist = np.min(distance_matrix, axis=0)
        
        valid_mask = min_dist < data_cfg.MAX_DIST_TO_STREET_VIEW
        
        self.dist_matrix = distance_matrix[:, valid_mask]
        
        valid_place_infos = {}
        for place_id, valid in zip(place_infos.keys(), valid_mask):
            if valid:
                valid_place_infos[place_id] = place_infos[place_id]
        print(f'>>> Filtered {len(valid_place_infos)} place infos by distance to street views.')
        
        return valid_place_infos

    def find_places_nearby(self, position_idx, cur_geocode, radius):
        assert self.positions[position_idx] == cur_geocode
        nearby_place_idx = (self.dist_matrix[position_idx] < radius).nonzero()[0]
        
        all_place_infos = list(self.place_infos.values())
        nearby_place_infos = []
        for idx in nearby_place_idx:
            place_info = all_place_infos[idx]
            # update the place info with distance
            place_info['distance'] = self.dist_matrix[position_idx, idx]
            nearby_place_infos.append(place_info)

        # calculate the heading between current position and nearby place candidates
        nearby_place_geocodes = [place_info['geocode'] for place_info in nearby_place_infos]
        if len(nearby_place_geocodes) == 0:
            return []
        headings = geocode_utils.calculate_headings_between_geocode_lists([cur_geocode], nearby_place_geocodes)[0]

        for place_info, heading in zip(nearby_place_infos, headings):
            place_info['heading'] = heading

        return nearby_place_infos

    def calculate_matching(self, result_dict, nearby_place_infos, matched_places, detected_places):
        """
        Calculate the matching between the detected objects and the nearby places.
        Args:
            result_dict (dict): detection results
            nearby_place_infos (list): nearby place infos

        Returns:

        """
        detect_results = result_dict['result']
        detect_views = result_dict['view']
        
        for i, box in enumerate(detect_results['boxes']):
            if isinstance(detect_views, list):
                detect_view = detect_views[i]
            else:
                detect_view = detect_views
                
            is_matched, matched_place_info = self.calculate_matching_single(
                box, detect_view, nearby_place_infos
            )

            if not is_matched:
                continue
            
            if cfg.PIPELINE.get('DEBUG_IMAGE', False):
                # draw the image with box
                draw_results = {
                    'boxes': np.array([box]),
                    'labels': [detect_results['labels'][i]],
                    'scores': np.array([detect_results['scores'][i]]),
                    'class_idx': np.array([detect_results['class_idx'][i]])
                }
                result_image = vis_utils.draw_with_results(detect_view.image, draw_results)
                path = self.output_dir / 'debug_image' / f"{matched_place_info['place_id']}" / f'{self.step_counter}_{i}.jpg'
                path.parent.mkdir(parents=True, exist_ok=True)
                result_image.save(path)
            
            # print(matched_place_info)
            # print(detect_results['labels'][i], detect_results['scores'][i])
            
            # for matched cases
            label = detect_results['labels'][i]
            gt_categories = matched_place_info['place_types']
            
            place_id = matched_place_info['place_id']
            matched_places[place_id] = matched_places.get(place_id, []) + [(label, gt_categories)]

            if label in matched_place_info['place_types']:
                detected_places[place_id] = detected_places.get(place_id, []) + [(label, gt_categories)]

    def calculate_matching_single(self, box, view, nearby_place_infos):
        heading_left, heading_right = geocode_utils.get_heading_range_to_box(
            box, view.shape, view.heading, view.fov
        )
        
        candidate_place_info = None
        for place_info in nearby_place_infos:
            heading = place_info['heading']
            if geocode_utils.is_heading_in_range((heading_left, heading_right), heading):
                if candidate_place_info is None or place_info['distance'] < candidate_place_info['distance']:
                    candidate_place_info = place_info
        
        is_matched = candidate_place_info is not None
        return is_matched, candidate_place_info
    
    def get_street_view_image_with_fov_list(self, platform, current_geocode, current_heading, fov_list):
        image_list = []
        for fov in fov_list:
            images = platform.get_all_streetview_from_geocode(
                current_geocode, cur_heading=current_heading, fov=fov
            )
            image_list.extend(images)
        
        return image_list
    
    def save_results(self):
        if self.navigator is not None:
            self.navigator.save_navigator()
        # ckpt_path = self.output_dir / 'checkpoint.pkl'
        
        result_dict = {
            'step_counter': self.step_counter,
            'detected_places': self.detected_places,
            'matched_places': self.matched_places,
            'place_infos': self.place_infos,
        }
        with open(self.ckpt_path, 'wb') as f:
            pickle.dump(result_dict, f)
        
    def resume_results(self):
        # ckpt_path = self.output_dir / 'checkpoint.pkl'
        result_dict = pickle.load(open(self.ckpt_path, 'rb'))
        
        self.step_counter = result_dict['step_counter']
        self.detected_places = result_dict['detected_places']
        self.matched_places = result_dict['matched_places']
        self.place_infos = result_dict['place_infos']

    def eval_results_all(self):
        with open(cfg.PIPELINE.EVALUATION.REGION_FILE, 'r') as f:
            regions = f.readlines()

        self.detected_places, self.matched_places, self.place_infos = {}, {}, {}
        self.detected_places_city, self.matched_places_city, self.place_infos_city = {}, {}, {}
        self.detected_places_continent, self.matched_places_continent, self.place_infos_continent = {}, {}, {}
        for region in regions:
            region_name = region.strip().split('/')[-1].split('.')[0][13:]
            ckpt_path = self.output_dir.parent / region_name / \
                cfg.PIPELINE.CHECK_SURROUNDING.DETECT.NAME / 'checkpoint.pkl'

            result_dict = pickle.load(open(ckpt_path, 'rb'))
            self.detected_places.update(result_dict['detected_places'])
            self.matched_places.update(result_dict['matched_places'])
            self.place_infos.update(result_dict['place_infos'])

            # eval city and continent statistics
            continent, city = common_utils.map_region_to_continent_city(
                region_name[:region_name.find('_mini')] + '_' + region.split('/')[0])
            self.detected_places_city[city] = self.detected_places_city.get(city, 0) + len(result_dict['detected_places'])
            self.matched_places_city[city] = self.matched_places_city.get(city, 0) + len(result_dict['matched_places'])
            self.place_infos_city[city] = self.place_infos_city.get(city, 0) + len(result_dict['place_infos'])
            self.detected_places_continent[continent] = self.detected_places_continent.get(continent, 0) + len(result_dict['detected_places'])
            self.matched_places_continent[continent] = self.matched_places_continent.get(continent, 0) + len(result_dict['matched_places'])
            self.place_infos_continent[continent] = self.place_infos_continent.get(continent, 0) + len(result_dict['place_infos'])

        self.formulate_output()
        self.save_results()
