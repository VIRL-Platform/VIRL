import pickle
import os

from virl.utils import geocode_utils, pipeline, common_utils
from .navigator_template import NavigatorTemplate


class RouteNavigator(NavigatorTemplate):
    def __init__(self, cfg, platform, messager, start_location, output_dir, **kwargs):
        super().__init__(cfg, platform, messager, start_location, output_dir, **kwargs)
        
        if os.path.exists(os.path.join(output_dir, 'navigator.pkl')):
            self.resume_navigator(output_dir)
            
            self.platform.initialize_mover(initial_geocode=self.current_geocode)
            return

        self.polygon_area = common_utils.load_points_in_txt_to_list(cfg.POLYGON_PATH)

        local_point_path = os.path.join(output_dir, 'points.txt')
        if cfg.POINT_PATH != 'None':
            self.points = common_utils.load_points_in_txt_to_list(cfg.POINT_PATH)
        elif os.path.exists(local_point_path):
            print('>>> RouteNavigator: load points from local file {}'.format(local_point_path))
            self.points = common_utils.load_points_in_txt_to_list(local_point_path)
        else:
            print('>>> RouteNavigator: sampling and relocate points in polygon area')
            seed_points = geocode_utils.grid_sample_quadrangle(self.polygon_area, cfg.SPACING)
            relocated_points, _ = geocode_utils.relocate_point_list_in_polygon(
                platform, seed_points, self.polygon_area
            )
            self.points = relocated_points
            common_utils.save_points_to_txt(local_point_path, self.points)
            print('>>> RouteNavigator: save points to local file {}'.format(local_point_path))

        local_route_path = os.path.join(output_dir, 'route.pkl')
        if cfg.ROUTE_PATH != 'None':
            route = pickle.load(open(cfg.ROUTE_PATH, 'rb'))
        elif os.path.exists(local_route_path):
            print('>>> RouteNavigator: load route from local file {}'.format(local_route_path))
            route = pickle.load(open(local_route_path, 'rb'))
        else:
            print(f'>>> RouteNavigator: calculate route with {cfg.TSP_ALGO}')
            route = geocode_utils.calculate_tsp_route_with_points(
                self.points, cfg.TSP_ALGO
            )
            pickle.dump(route, open(local_route_path, 'wb'))
            print('>>> RouteNavigator: save route to local file {}'.format(local_route_path))

        self.route = [self.points[idx] for idx in route]

        self.routing_idx = 1
        self.next_geocode = self.route[self.routing_idx]
        self.current_geocode = self.route[self.routing_idx - 1]

        self.platform.initialize_mover(initial_geocode=self.current_geocode)
        self.set_initial_heading()

    def set_initial_heading(self):
        expected_heading, _ = geocode_utils.get_heading_and_distance_by_geocode(
            self.current_geocode, self.next_geocode
        )
        self.current_heading = expected_heading

    def actions_before_moving(self, info_dict):
        pass

    def check_stop(self, info_dict):
        if self.routing_idx >= len(self.route):
            return True
        else:
            return False

    def move(self, info_dict):
        self.current_heading = geocode_utils.calculate_heading_between_geocodes(self.current_geocode, self.next_geocode)
        self.platform.mover.adjust_heading_web(self.current_heading)
        self.platform.mover._move_by_geocode(self.next_geocode)
        self.current_geocode = self.next_geocode
        
        # set the next geocode
        self.routing_idx += 1
        if self.routing_idx < len(self.route):
            self.next_geocode = self.route[self.routing_idx]

        print(f'>>> RouteNavigator: after moving, current geocode is: {self.current_geocode}, '
              f'next goal is : {self.next_geocode}')
        print(f'>>> RouteNavigator Moving Progress: {self.routing_idx}/{len(self.route)}')
