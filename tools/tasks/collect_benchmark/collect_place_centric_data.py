import tqdm
import pickle
import os
import glob

from shapely.geometry import Point, Polygon

from tools.tasks.task_template import TaskTemplate

from virl.config import cfg
from virl.utils import common_utils, geocode_utils
from virl.perception.recognizer.recognizer import Recognizer


class CollectPlaceCentricData(TaskTemplate):
    def __init__(self, output_dir, logger):
        super().__init__(output_dir, logger)

        self.candidates = ""
        self.cared_labels = []
        self.check_thresh = None

        self.max_width = cfg.PIPELINE.GET_PHOTO.MAX_WIDTH
        self.max_height = cfg.PIPELINE.GET_PHOTO.MAX_HEIGHT

        # record all unique places
        self.place_info_path = os.path.join(self.output_dir, 'place_infos.pickle')
        if os.path.exists(self.place_info_path):
            print(f'Loading place infos from {self.place_info_path}')
            self.place_dict = pickle.load(open(self.place_info_path, 'rb'))
        else:
            self.place_dict = {}
                
        # get all saved photo image list
        self.image_dir = os.path.join(self.output_dir, 'images')
        os.makedirs(self.image_dir, exist_ok=True)
        saved_photo_list = glob.glob(self.image_dir + '/*.jpg')
        self.saved_photo_list = [os.path.basename(path).split('.')[0] for path in saved_photo_list]
        
        # save invalid place info during filtering
        if os.path.exists(self.output_dir / 'invalid_place_id.pickle'):
            self.invalid_place_id = pickle.load(open(self.output_dir / 'invalid_place_id.pickle', 'rb'))
        else:
            self.invalid_place_id = []

        self.place_types = []
        with open(cfg.PIPELINE.SAMPLE_PLACES_IN_REGION.NEARBY_SEARCH.PLACE_TYPES, 'r') as f:
            for line in f:
                self.place_types.append(line.strip())

    def set_candidates_for_vision_model(self, pipeline_cfg):
        self.cared_labels = pipeline_cfg.CARED_LABELS

        candidates_list = pipeline_cfg.CANDIDATES.split(',') + self.cared_labels
        self.candidates = ',,'.join(candidates_list)
        self.check_thresh = pipeline_cfg.THRESH

    def run(self, platform, agent, chatbot, messager, args, **kwargs):
        pipeline_cfg = cfg.PIPELINE

        # prepare vision models
        model = Recognizer(cfg.VISION_MODELS, pipeline_cfg.GET_PHOTO.PHOTO_FILTER)
        self.set_candidates_for_vision_model(pipeline_cfg.GET_PHOTO.PHOTO_FILTER)

        # sample places info in a given region.
        if pipeline_cfg.SAMPLE_PLACES_IN_REGION.ENABLED:
            self.sample_places(platform, pipeline_cfg.SAMPLE_PLACES_IN_REGION)
        
        import ipdb; ipdb.set_trace(context=20)

        # get all place photo images
        common_utils.print_stage('Get all place photo images')
        self.get_place_photo_from_place_info(
            platform, model, cfg.PIPELINE.GET_PHOTO
        )

        # save filtered place infos
        self.filter_place_info_by_saved_photo()
        path = os.path.join(self.output_dir, 'place_infos_valid.pickle')
        pickle.dump(self.place_dict, open(path, 'wb'))
        print(f'Totally {len(self.place_dict)} valid place infos are saved to {path}')

    def get_place_photo_from_place_info(self, platform, clip, photo_cfg):
        """
        Get list of photo from place results list.

        Args:
            results (list): list of result dict

        Returns:
            photos: list of photos
            filtered_results: list of filtered results
        """
        for i, (place_id, place_info) in tqdm.tqdm(enumerate(self.place_dict.items()), total=len(self.place_dict)):
            photo_refer = place_info['photo_reference']
            if photo_refer is None or place_id in self.saved_photo_list or place_id in self.invalid_place_id:
                continue

            try:
                photo = platform.get_place_photo(photo_refer, self.max_width, self.max_height)
            except Exception:
                print(Exception)
                photo = None

            valid, photo = self.filter_with_vision_model(platform, clip, photo, place_info)

            if not valid:
                self.invalid_place_id.append(place_id)
                continue

            if photo_cfg.SAVE_IMAGE:
                photo_path = os.path.join(self.image_dir, f'{place_id}.jpg')
                try:
                    photo.save(photo_path)
                except OSError:
                    photo.convert('RGB').save(photo_path)
                    
                self.saved_photo_list.append(place_id)

            # save intermediate results
            if i % cfg.SAVE_INTERVAL == 0:
                path = os.path.join(self.output_dir, 'invalid_place_id.pickle')
                pickle.dump(self.invalid_place_id, open(path, 'wb'))

    def filter_with_vision_model(self, platform, model, photo, place_result):
        if photo is not None:
            valid = self.check_with_vision_model(model, photo)
        else:
            valid = False

        if valid:
            return valid, photo

        photo_refer_list = platform.get_photo_references_from_place_details(
            place_result['place_id'],
        )

        if len(photo_refer_list) <= 1:
            return False, None

        max_counter = 0
        for photo_refer in photo_refer_list:
            try:
                photo = platform.get_place_photo(photo_refer, self.max_width, self.max_height)
            except Exception:
                print(Exception)
                max_counter += 1
                continue
            valid = self.check_with_vision_model(model, photo)
            max_counter += 1
            if valid or max_counter >= cfg.PIPELINE.GET_PHOTO.PHOTO_FILTER.MAX_COUNT:
                break

        return valid, photo if valid else None

    def check_with_vision_model(self, model, photo):
        result = model.check(photo, self.candidates, self.cared_labels)
        max_score = max(result['scores'])
        return max_score >= self.check_thresh

    def sampling_seed_position_list(self, polygon_path, spacing):
        polygon = common_utils.load_points_in_txt_to_list(polygon_path)
                
        points = geocode_utils.grid_sample_quadrangle(polygon, spacing)

        # for visualization debug
        with open(os.path.join(self.output_dir, 'seed_positions.txt'), 'w') as f:
            f.write('\n'.join(f'({geo[0]}, {geo[1]})' for geo in points))

        return points, polygon

    def sample_places(self, platform, sample_place_cfg):
        # grid sampling
        common_utils.print_stage('Grid sampling')
        seed_positions, polygon = self.sampling_seed_position_list(
            sample_place_cfg.GRID_SAMPLE.POLYGON_PATH, sample_place_cfg.GRID_SAMPLE.SPACING
        )

        print(f'Number of seed positions: {len(seed_positions)}')

        import ipdb; ipdb.set_trace(context=20)
        relocate_point_save_path = os.path.join(self.output_dir, 'relocated_points.txt')
        if sample_place_cfg.RELOCATE:
            common_utils.print_stage('Relocate positions')
            if sample_place_cfg.RELOCATED_POINTS != 'None':
                seed_positions = common_utils.load_points_in_txt_to_list(
                    sample_place_cfg.RELOCATED_POINTS
                )
            elif os.path.exists(relocate_point_save_path):
                seed_positions = common_utils.load_points_in_txt_to_list(relocate_point_save_path)
            else:
                seed_positions, _ = geocode_utils.relocate_point_list_in_polygon(
                    platform, seed_positions, polygon
                )
                common_utils.save_points_to_txt(relocate_point_save_path, seed_positions)
                print(f'Relocated points are saved to {relocate_point_save_path}')
                
            print(f'{len(seed_positions)} seed positions after relocation')
        
        import ipdb; ipdb.set_trace(context=20)
        # search nearby places
        if not os.path.exists(self.place_info_path):
            common_utils.print_stage('Search nearby places')
            for position in tqdm.tqdm(seed_positions, total=len(seed_positions)):
                # results = platform.get_nearby_places(
                #     geocode=position, radius=sample_place_cfg.NEARBY_SEARCH.RADIUS, cal_distance=True,
                #     type_custom=sample_place_cfg.NEARBY_SEARCH.TYPE, polygon_filter=Polygon(polygon)
                # )
                results = platform.get_nearby_places(
                    geocode=position, cal_distance=True, rankby='distance',
                    no_next_page=sample_place_cfg.NEARBY_SEARCH.NO_NEXT_PAGE,
                    type_custom=sample_place_cfg.NEARBY_SEARCH.TYPE, polygon_filter=Polygon(polygon)
                )

                for result in results:
                    place_types = result['place_types']
                    intersect_types = common_utils.list_intersection(place_types, self.place_types)
                    if result['place_id'] not in self.place_dict and len(intersect_types) > 0:
                        self.place_dict[result['place_id']] = result
            
            pickle.dump(self.place_dict, open(self.place_info_path, 'wb'))
            print(f'Place infos is saved to {self.place_info_path}')

        print(f'Number of unique places: {len(self.place_dict)}')

    def filter_place_info_by_saved_photo(self):
        keys = list(self.place_dict.keys())
        for place_id in keys:
            if place_id not in self.saved_photo_list:
                self.place_dict.pop(place_id)
