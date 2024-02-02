import pickle

from virl.utils import geocode_utils, pipeline


class NavigatorTemplate(object):
    def __init__(self, cfg, platform, messager, start_location, output_dir, **kwargs):
        self.cfg = cfg
        self.platform = platform
        self.messager = messager
        self.output_dir = output_dir

        self.current_geocode = pipeline.position_to_geocode(self.platform, start_location)
        self.current_heading = None

        if not kwargs.get('no_init_mover', False):
            platform.initialize_mover(initial_geocode=self.current_geocode)
        self.trajectory = [self.current_geocode]

        # the counter for going back
        self.go_back_counter = 0

    def navigate(self, info_dict):
        """

        Args:
            info_dict:

        Returns:
            is_finish: bool. indicting whether the navigation is finished
            current_geocode: tuple. the current geocode after moving
        """
        # actions before moving, for example, check the next geocode objective
        self.actions_before_moving(info_dict)

        if self.check_stop(info_dict):
            self.show_trajectory_on_the_map(self.trajectory, self.output_dir / self.cfg.OUTPUT.TRAJ_PATH)
            print('>>> PointNavigator: Finish navigation')
            self.actions_before_stop(info_dict)
            return True, self.current_geocode, info_dict

        # move
        self.move(info_dict)
        # actions after move
        self.actions_after_move(info_dict)
        # record into the trajectory
        self.trajectory.append(self.current_geocode)
        return False, self.current_geocode, info_dict

    def actions_before_stop(self, info_dict):
        pass

    def actions_before_moving(self, info_dict):
        raise NotImplementedError

    def actions_after_move(self, info_dict):
        self.messager.clear({})

    def move(self, info_dict):
        raise NotImplementedError

    def check_stop(self, info_dict):
        raise NotImplementedError

    def get_current_geocode(self):
        return self.current_geocode

    def get_current_heading(self):
        return self.current_heading

    def parse_waypoints(self, way_points):
        parsed_way_points = []
        for position in way_points:
            if isinstance(position, str):
                geocode = pipeline.location_to_geocode(
                    self.platform, position, version='v2', relocate=True
                )
            elif isinstance(position, tuple) or isinstance(position, list):
                geocode, _ = self.platform.relocate_geocode_by_source(position, source='outdoor')
            else:
                raise NotImplementedError

            parsed_way_points.append(geocode)

        return parsed_way_points

    def get_destination_from_waypoints(self, way_points):
        # make the farthest waypoint as the destination
        max_dist = 0
        max_dist_idx = 0
        for i, geocode in enumerate(way_points):
            distance = geocode_utils.calculate_distance_from_geocode(
                self.current_geocode, geocode
            )
            if distance > max_dist:
                max_dist = distance
                max_dist_idx = i
        end_geocode = way_points[max_dist_idx]
        del way_points[max_dist_idx]

        return end_geocode, way_points

    def show_trajectory_on_the_map(self, trajectory, path):
        polyline = geocode_utils.encode_polyline(trajectory)
        pipeline.show_polyline_on_the_map(
            self.platform, polyline, path, self.cfg.OUTPUT.FILE_TEMPLATE
        )

    def save_navigator(self):
        """
        Save the navigator to the output directory, for resume
        Returns:

        """
        result_dict = {
            'geocode': self.current_geocode,
            'heading': self.current_heading,
            'trajectory': self.trajectory,
            'routing_idx': getattr(self, 'routing_idx', None),
            'next_geocode': getattr(self, 'next_geocode', None),
            'go_back_counter': getattr(self, 'go_back_counter', None),
            'end_geocode': getattr(self, 'end_geocode', None),
            'route': getattr(self, 'route', None),
            'way_points': getattr(self, 'way_points', None),
            'points': getattr(self, 'points', None),
            'polygon_area': getattr(self, 'polygon_area', None),
            # vln
            'step_counter': getattr(self, 'step_counter', None),
            'action_list': getattr(self, 'action_list', None),
            'observation_list': getattr(self, 'observation_list', None),
            'oracle_observation_list': getattr(self, 'oracle_observation_list', None),
            'target_heading': getattr(self, 'target_heading', None),
        }
        output_path = self.output_dir / 'navigator.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(result_dict, f)

    def resume_navigator(self, output_dir):
        """
        Resume the navigator from the output directory
        Returns:

        """
        print('>>> Resuming navigator')
        input_path = output_dir / 'navigator.pkl'
        result_dict = pickle.load(open(input_path, 'rb'))

        self.current_geocode = result_dict.get('geocode', None)
        self.current_heading = result_dict.get('heading', None)
        self.go_back_counter = result_dict.get('go_back_counter', None)
        self.trajectory = result_dict.get('trajectory', None)
        self.end_geocode = result_dict.get('end_geocode', None)
        self.routing_idx = result_dict.get('routing_idx', None)
        self.next_geocode = result_dict.get('next_geocode', None)
        self.route = result_dict.get('route', None)
        self.way_points = result_dict.get('way_points', None)

        # route navigator
        self.points = result_dict.get('points', None)
        self.polygon_area = result_dict.get('polygon_area', None)

        # vln
        self.step_counter = result_dict.get('step_counter', None)
        self.observation_list = result_dict.get('observation_list', None)
        self.action_list = result_dict.get('action_list', None)
        self.oracle_observation_list = result_dict.get('oracle_observation_list', None)
        self.target_heading = result_dict.get('target_heading', None)
