import time
import os
import ast

from virl.config import cfg as gcfg
from virl.utils import geocode_utils, common_utils, pipeline
from virl.lm import UnifiedChat
from .navigator_template import NavigatorTemplate
from virl.lm import prompt as prompt_templates


class PointNavigator(NavigatorTemplate):
    def __init__(self, cfg, platform, messager, start_location, output_dir, **kwargs):
        super().__init__(cfg, platform, messager, start_location, output_dir, **kwargs)

        self.way_points = None
        self.end_geocode = None
        self.stop_radius = self.cfg.STOP_RADIUS
        self.chatbot = UnifiedChat()

        # check existing navigation checkpoints
        if os.path.exists(os.path.join(output_dir, 'navigator.pkl')):
            self.resume_navigator(output_dir)
            self.platform.initialize_mover(initial_geocode=self.current_geocode)
            return
        
        if self.cfg.get('ROUTE_PATH', None):
            point_list = common_utils.load_points_in_txt_to_list(self.cfg.ROUTE_PATH)
            self.cfg.ROUTE = point_list
        
        if self.cfg.get('ROUTE', None):
            self.route = self.cfg.ROUTE
            polyline = geocode_utils.encode_polyline(self.route)
        else:
            self.route, polyline = self.calculate_route()

        # draw the planned route
        pipeline.draw_planned_route(
            self.platform, polyline, input_way_points=self.way_points if self.way_points is not None else self.route,
            path=self.output_dir / self.cfg.OUTPUT.ROUTE_PATH,
            file_template=self.cfg.OUTPUT.FILE_TEMPLATE,
            cur_position=self.current_geocode
        )

        self.routing_idx = 0
        self.next_geocode = self.route[0]

        self.platform.initialize_mover(initial_geocode=self.current_geocode)
        self.set_initial_heading()

    def calculate_route(self):
        if self.cfg.get('WAY_POINTS', None) is not None or self.cfg.get('WAY_POINTS_LANGUAGE', None) is not None:
            # parse the way points to geocode
            if self.cfg.get('WAY_POINTS_LANGUAGE', None) is not None:
                waypoint_list = self.parse_way_points_language(self.cfg['WAY_POINTS_LANGUAGE'])
            else:
                waypoint_list = self.cfg['WAY_POINTS']
            self.way_points = self.parse_waypoints(waypoint_list)

        no_last_leg = False
        if self.cfg.get('END_POSITION', None) is not None:
            end_position = self.cfg['END_POSITION']
            self.end_geocode = pipeline.position_to_geocode(self.platform, end_position)
        else:
            self.end_geocode = self.current_geocode
            no_last_leg = self.cfg.get('STOPOVER', False) or self.cfg.get('OPTIMIZED', False)

        print(f'Initialize Point Navigator with start point: {self.current_geocode}, end geocode: {self.end_geocode}')

        result_dict = self.platform.get_routing(
            self.current_geocode, self.end_geocode, self.cfg.ROUTE_MODE, way_points=self.way_points,
            polyline=self.cfg.get('POLYLINE', False), stopover=self.cfg.get('STOPOVER', False),
            optimized=self.cfg.get('OPTIMIZED', False), no_last_leg=no_last_leg
        )
        route = result_dict['geocode_list']
        print(f'>>> PointNavigator: estimated time {result_dict["time"] / 60:.3f} minutes, '
              f'distance {result_dict["distance"]} meters')
        polyline = result_dict['polyline']

        return route, polyline

    def parse_way_points_language(self, way_points_language):
        prompt_template = getattr(prompt_templates, self.cfg.WAY_POINT_PARSE_TEMPLATE)
        prompt = prompt_template.format(waypoint_query=way_points_language)
        answer = self.chatbot.ask(prompt, model=self.cfg.MODEL)
        waypoint_list = ast.literal_eval(answer)
        return waypoint_list

    def set_initial_heading(self):
        expected_heading, _ = geocode_utils.get_heading_and_distance_by_geocode(
            self.current_geocode, self.next_geocode
        )
        self.current_heading = expected_heading

    def actions_before_moving(self, info_dict):
        self.count_go_back()
        
        expected_heading, expected_dist = geocode_utils.get_heading_and_distance_by_geocode(
            self.current_geocode, self.next_geocode
        )
        if expected_dist <= self.stop_radius:
            self.switch_to_next_geocode()
            expected_heading, expected_dist = geocode_utils.get_heading_and_distance_by_geocode(
                self.current_geocode, self.next_geocode
            )

        info_dict['expected_heading'] = expected_heading
        info_dict['expected_dist'] = expected_dist

    def check_stop(self, info_dict):
        if self.routing_idx >= len(self.route):
            return True
        else:
            return False

    def switch_to_next_geocode(self):
        self.routing_idx += 1
        if self.routing_idx < len(self.route):
            self.next_geocode = self.route[self.routing_idx]
            print(f'>>> PointNavigator: Switch to next sub-geocode: {self.next_geocode}')

    def move(self, info_dict):
        heading_list = self.platform.mover.get_all_suitable_heading_to_path(
            self.current_geocode, info_dict
        )
        print(f'>>> PointNavigator: movable direction: {heading_list}')

        road_idx = geocode_utils.select_argmin_heading_from_heading_list(
            info_dict['expected_heading'], heading_list
        )

        heading_diff = geocode_utils.cal_min_heading_diff_between_headings(self.current_heading, heading_list[road_idx])
        if heading_diff > 10:
            self.platform.mover.adjust_heading_web(heading_list[road_idx])

        # visualization demo only
        if gcfg.get('DEMO', False):
            time.sleep(heading_diff / 60 + 1)

        self.current_geocode = self.platform.mover.move(road_idx)
        self.current_heading = heading_list[road_idx]

        print(f'>>> PointNavigator: after moving, current geocode is: {self.current_geocode}, '
              f'next goal is : {self.next_geocode}')
        print(f'>>> PointNavigator: Route: {self.route}, idx: {self.routing_idx}')
        print(f'################# PointNavigator: Finish Moving #################')

    def count_go_back(self):
        if len(self.trajectory) > 2 and self.current_geocode == self.trajectory[-3]:
            self.go_back_counter += 1
            print('>>> PointNavigator: Go back counter + 1')
        else:
            self.go_back_counter = 0

        # self rescue by set the switch to next geocode
        if self.go_back_counter >= self.cfg.MAX_GO_BACK:
            print('>>> PointNavigator: Dead locked, self rescue')
            self.switch_to_next_geocode()
            self.go_back_counter = 0
