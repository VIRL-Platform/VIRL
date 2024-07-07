import os
import time
import requests
import pickle
import warnings
import cv2
import json

import PIL.Image as Image

from io import BytesIO
from queue import PriorityQueue
from shapely.geometry import Point

from virl.utils.common_utils import ComparableObj
from virl.utils import geocode_utils, common_utils
from virl.platform.street_view import StreetViewImage, get_perspective_from_panorama


class GoogleMapAPI(object):
    def __init__(self, **kwargs):
        self.key = os.environ.get('GOOGLE_MAP_API_KEY', None)
        self.base_urls = {
            'nearby_search': "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            'geocode': "https://maps.googleapis.com/maps/api/geocode/json",
            'streetview': "https://maps.googleapis.com/maps/api/streetview",
            'streetview_meta': "https://maps.googleapis.com/maps/api/streetview/metadata",
            'findplacefromtext': "https://maps.googleapis.com/maps/api/place/findplacefromtext/json",
            'textsearch': 'https://maps.googleapis.com/maps/api/place/textsearch/json',
            'directions': 'https://maps.googleapis.com/maps/api/directions/json',
            'place_details': 'https://maps.googleapis.com/maps/api/place/details/json',
            'place_photos': 'https://maps.googleapis.com/maps/api/place/photo'
        }

        offline_cfg = kwargs['offline_cfg']
        if offline_cfg.ENABLED:
            self.offline_cfg = offline_cfg
            self.init_offline(offline_cfg)
    
    def init_offline(self, offline_cfg):
        self.offline_mode = True
        
        self.panorama_dir = offline_cfg.PANORAMA_DIR
        self.mapping_path = offline_cfg.GPS_TO_PANO_PATH
        
        if self.panorama_dir != 'None':
            self.offline_pano = True
            print(f'Offline panorama mode is enabled. Panorama dir: {self.panorama_dir}')
        
        if self.mapping_path != 'None':
            self.offline_mapping = True
            print(f'Offline mapping mode is enabled. Mapping path: {self.mapping_path}')
            self.gps_to_pano_mapping = pickle.load(open(self.mapping_path, 'rb'))

        # TODO: add place information

    def get_geocode_from_address(self, address: str, language='en') -> tuple:
        """
        Parse natural language address to geocode, a.k.a latitude and longitude

        Examples:
            https://maps.googleapis.com/maps/api/geocode/json?address=the%20university%20of%20hong%20kong&key=APIKEY
        Args:
            address: text address

        Returns:
            (latitude, longitude)
        """
        base_url = self.base_urls['geocode']

        params = {
            'address': address,
            'language': language,
            'key': self.key
        }

        response_json = requests.get(base_url, params=params).json()

        latitude = response_json['results'][0]['geometry']['location']['lat']
        longitude = response_json['results'][0]['geometry']['location']['lng']

        return latitude, longitude

    def get_geocode_from_address_v2(self, address: str, language='en',
                                    query_info: list = ('formatted_address', 'name', 'rating',
                                                        'opening_hours', 'geometry')) -> tuple:
        """
        Parse natural language address to geocode -- latitude and longitude
        use another api: findplacefromtext

        Compared to get_geocode_from_address, the geocode provided
        by this function is more accurate (guess this is caused by the different data source)

        reference:
        https://developers.google.com/maps/documentation/places/web-service/search-find-place#inputtype
        
        Args:
            address:
            language:
            query_info:

        Returns:
            geocode (tuple): latitude and longitude
        """
        base_url = self.base_urls['findplacefromtext']
        params = {
            'fields': ','.join(query_info),
            'input': address,
            'inputtype': 'textquery',
            'language': language,
            'key': self.key
        }

        response_json = requests.get(base_url, params=params).json()

        if response_json['status'] != 'OK':
            print(response_json)
            raise ValueError(f'Cannot find geocode for {address}')

        try:
            latitude = response_json['candidates'][0]['geometry']['location']['lat']
            longitude = response_json['candidates'][0]['geometry']['location']['lng']
        except:
            raise ValueError(f'Cannot find geocode for {address}')

        return latitude, longitude

    def get_nearby_places(self, geocode: tuple, ranking_type: str = 'distance', relocated=False,
                          language='en', radius_custom=None, cal_distance=True, type_custom=None,
                          polygon_filter=None, no_next_page=False, **kwargs) -> list:
        """https://developers.google.com/maps/documentation/places/web-service/search-nearby

        Args:
            language: [Official API param] language for the search result
            relocated: whether to relocate the geocode by the streetview meta api
            ranking_type:
            geocode (tuple): latitude and longitude
            cal_distance: whether to calculate the distance between the place and the current geocode
            radius_custom: the radius for the search, unit: meter.
                            Compared to radius in official api, this can be more natural,
                            because the radius in official api will conflict with some other parameters such as
                            ranked by distance.
            type_custom: the official type filtering cannot support some types, e.g., 'establishment'
            polygon_filter: a shapely polygon to filter the places
            no_next_page: whether to get the next page of the search result
            **kwargs: other parameters to feed in the official api

        Returns:
            final_list: a list of place information ranked by ranking_type or rankby
        """
        pri_queue = PriorityQueue()
        base_url = self.base_urls['nearby_search']
        # url = "location=-33.8670522%2C151.1957362&radius=1500&type=restaurant&key=YOUR_API_KEY"        
        params = {
            'location': f'{geocode[0]},{geocode[1]}',
            'language': language,
            'key': self.key
        }
        params.update(kwargs)

        response_json = requests.get(base_url, params=params).json()
        pri_queue = self.parse_nearby_json(
            response_json, geocode, pri_queue, ranking_type, relocated=relocated, radius=radius_custom,
            cal_distance=cal_distance, type_custom=type_custom, polygon_filter=polygon_filter
        )

        # if there is next page
        # https://maps.googleapis.com/maps/api/place/nearbysearch/json?pagetoken=NEXT_PAGE_TOKEN&key=YOUR_API_KEY
        while not no_next_page and response_json.get('next_page_token', None) and pri_queue.qsize() > 0:
            next_page_token = response_json['next_page_token']
            next_url = f'{base_url}?pagetoken={next_page_token}&key={self.key}'
            # sleep to wait the preparation of next page
            response_json = self.loop_to_get_next_page(next_url)

            pri_queue = self.parse_nearby_json(
                response_json, geocode, pri_queue, ranking_type, relocated=relocated, radius=radius_custom,
                cal_distance=cal_distance, type_custom=type_custom, polygon_filter=polygon_filter
            )

        final_list = [x.data for x in pri_queue.queue]

        return final_list

    @staticmethod
    def loop_to_get_next_page(next_url):
        while True:
            time.sleep(2)  # sleep to wait the preparation of next page
            response_json = requests.get(next_url).json()
            if response_json['status'] == 'OK':
                return response_json

    def get_nearby_places_v2(self, geocode: tuple, query='near', ranking_type='distance', radius_custom=None,
                             language='en', rankby='distance', deduplicate_with_name=False, min_reviews=0,
                             **kwargs) -> list:
        """https://developers.google.com/maps/documentation/textsearch/web-service/search-nearby

        Similar to get_nearby_places, but this is built upon another database and can provide
        more accurate geolocation in our test.

        Args:
            language:
            geocode (tuple): [Official API param] latitude and longitude
            query (int): [Official API param] description for the searched place
            ranking_type: distance or rating
            radius_custom: the radius for the search, unit: meter.
                            Compared to radius in official api, this can be more natural,
                            because the radius in official api will conflict with some other parameters such as
                            ranked by distance.
            language: [Official API param] language for the search result
            rankby: [Official API param] distance or prominence (default). This is a
            deduplicate_with_name: whether to remove the results with the same name.
                                    (In our test, google place api will sometimes return places with
                                    the same name, maybe because of the unremoved out-dated data.)
            min_reviews: the minimum number of reviews for the place

        Returns:
            list: _description_
        """
        pri_queue = PriorityQueue()
        base_url = self.base_urls['textsearch']
        params = {
            'location': f'{geocode[0]},{geocode[1]}',
            'query': query,
            'language': language,
            'rankby': rankby,
            'key': self.key
        }
        params.update(kwargs)

        response_json = requests.get(base_url, params=params).json()
        pri_queue = self.parse_nearby_json(
            response_json, geocode, pri_queue, ranking_type, radius=radius_custom, min_reviews=min_reviews
        )

        # if there is next page
        # https://maps.googleapis.com/maps/api/place/nearbysearch/json?pagetoken=NEXT_PAGE_TOKEN&key=YOUR_API_KEY
        while response_json.get('next_page_token', None):
            next_page_token = response_json['next_page_token']
            next_url = f'{base_url}?pagetoken={next_page_token}&key={self.key}'
            # sleep to wait the preparation of next page
            response_json = self.loop_to_get_next_page(next_url)
            pri_queue = self.parse_nearby_json(
                response_json, geocode, pri_queue, ranking_type, radius=radius_custom, min_reviews=min_reviews
            )
            # result_dict.update(new_result_dict)

        final_list = []
        while not pri_queue.empty():
            final_list.append(pri_queue.get().data)

        if deduplicate_with_name:
            name_list = []
            final_list = [x for x in final_list if x['name'] not in name_list and not name_list.append(x['name'])]
        
        return final_list

    def parse_nearby_json(self, response_json, cur_location, pri_queue, ranking_type,
                          radius=None, relocated=False, cal_distance=True, type_custom=None,
                          polygon_filter=None, min_reviews=0):
        for result in response_json['results']:
            name = result['name']
            types = result['types']

            # filter unwanted types
            if type_custom is not None:
                intersect_list = common_utils.list_intersection(types, type_custom)
                if len(intersect_list) == 0:
                    continue

            if relocated:
                # the geocode from this source is more accurate than geocode api.
                geocode = self.get_geocode_from_address_v2(address=name)
            else:
                geocode = (result['geometry']['location']['lat'], result['geometry']['location']['lng'])

            point = Point(geocode)
            if polygon_filter is not None and (not polygon_filter.contains(point)):
                continue

            rating = result['rating'] if 'rating' in result else 0
            distance = geocode_utils.calculate_distance_from_geocode(cur_location, geocode) if cal_distance else None
            heading = geocode_utils.calculate_heading_between_geocodes(cur_location, geocode) if cal_distance else None

            if radius is not None and distance > radius:
                return pri_queue

            n_reviews = result['user_ratings_total'] if 'user_ratings_total' in result else 0
            if n_reviews < min_reviews:
                continue

            cur_dict = {
                'name': name,
                'geocode': geocode,
                'rating': rating,
                'place_id': result['place_id'],
                'distance': distance,
                'n_reviews': n_reviews,
                'heading': heading,
                'photo_reference': result['photos'][0]['photo_reference'] if 'photos' in result else None,
                'place_types': result['types'],
                'plus_code': result['plus_code'] if 'plus_code' in result else None,
                'vicinity': result['vicinity'] if 'vicinity' in result else None,
                # 'address': name + ', ' + result['formatted_address']
            }

            if ranking_type == 'rating':
                priority = -rating
            elif ranking_type == 'distance' and cal_distance:
                priority = distance
            else:
                priority = 0

            pri_queue.put(ComparableObj(priority, cur_dict))

        return pri_queue

    def get_streetview_from_geocode(self, geocode: tuple, size: tuple, heading: int,
                                    pitch: int, fov: int, source: str = 'outdoor',
                                    idx=None):
        """
        Args:
            geocode (tuple): latitude and longitude
            size (tuple): (width, height). 640x640 is the max size for free user.
            heading (int): The compass heading of the camera.
            pitch (int): Angle of camera's vertical axis. Value range in [-90, 90]
            fov (int): Field of view of the camera in degrees, which must be between 10 and 120.
            source (str): default or outdoor
            idx (int): the index of the image in the check around process

        Returns:
            image: google street view image
        """
        # Check the offline cache
        if self.offline_mode and self.offline_pano:
            is_success, street_image = self.get_streetview_from_geocode_offline(
                geocode, size, heading, pitch, fov, source, idx
            )
            if is_success:
                return street_image
            else:
                warnings.warn(f'Cannot find the streetview image for {geocode} in offline database. Call online api.')

        base_url = self.base_urls['streetview']
        # example url:
        # https://maps.googleapis.com/maps/api/streetview?size=400x400&location=47.5763831,-122.4211769&fov=80&heading=0&pitch=0&key=YOUR_API_KEY
        params = {
            'size': f'{size[0]}x{size[1]}',
            'location': f'{geocode[0]},{geocode[1]}',
            'fov': fov,
            'heading': heading,
            'pitch': pitch,
            'source': source,
            'key': self.key
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
        elif response.status_code == 404:
            raise Exception(f"Streetview not found: {params}")
        else:
            raise Exception(f"Unexpected status code: {response.status_code}")

        street_image = StreetViewImage(
            image, heading, pitch, fov, geocode, i=idx
        )

        return street_image

    def get_streetview_from_geocode_offline(self, geocode: tuple, size: tuple, heading: int,
                                            pitch: int, fov: int, source: str = 'outdoor',
                                            idx=None, **kwargs):
        """
        Args:
            geocode (tuple): latitude and longitude
            size (tuple): (width, height). 640x640 is the max size for free user.
            heading (int): The compass heading of the camera.
            pitch (int): Angle of camera's vertical axis. Value range in [-90, 90]
            fov (int): Field of view of the camera in degrees, which must be between 10 and 120.
            source (str): default or outdoor
            idx (int): the index of the image in the check around process

        Returns:
            image: google street view image
        """
        # Step 1: load the panorama image accrodnig to the geocode
        # map the geocode to the panorama id
        if geocode in self.gps_to_pano_mapping:
            pano_id = self.gps_to_pano_mapping[geocode]
        else:
            geocode, pano_id = self.relocate_geocode_by_source(geocode, source=source)
        
        # Step 2: load the panorama image
        pano_img_path = os.path.join(self.panorama_dir, f'{pano_id}.jpg')
        pano_image_metadata_path = os.path.join(self.panorama_dir, f'{pano_id}.metadata.json')
        if not os.path.exists(pano_img_path):
            warnings.warn(f'Cannot find the panorama image for {pano_id} in {self.panorama_dir}')
            return False, None

        img = cv2.imread(pano_img_path, cv2.IMREAD_COLOR)
        img_metadata = json.load(open(pano_image_metadata_path, 'r'))
        north_rotation = img_metadata['rotation']
        image = get_perspective_from_panorama(img, fov, heading, pitch, size[1], size[0], north_rotation)
        
        street_image = StreetViewImage(
            image, heading, pitch, fov, geocode, i=idx
        )

        return True, street_image
        
    def relocate_geocode_by_source(self, geocode: tuple, source: str = 'outdoor'):
        if self.offline_mode and self.offline_mapping:
            is_success, new_geocode, pano_id = self._relocate_geocode_by_source_offline(geocode)
            if is_success:
                return new_geocode, pano_id
            else:
                warnings.warn(f'Cannot find the nearest geocode within {self.offline_cfg.MAPPING_RADIUS} '
                              f'for {geocode}. Call online api.')
        
        base_url = self.base_urls['streetview_meta']

        params = {
            'location': f'{geocode[0]},{geocode[1]}',
            'source': source,
            'key': self.key
        }
        
        response_json = requests.get(base_url, params=params).json()
        if response_json['status'] == 'OK':
            new_geocode = (response_json['location']['lat'], response_json['location']['lng'])
            pano_id = response_json['pano_id']
            return new_geocode, pano_id
        elif response_json['status'] in ['ZERO_RESULTS', 'UNKNOWN_ERROR']:
            return None, None
        else:
            print(f'Relocated geocode error: {response_json}')

    def _relocate_geocode_by_source_offline(self, geocode: tuple):
        # for the case that geocode is in the mapping file, directly get the pano id
        geocode = tuple(geocode)
        if geocode in self.gps_to_pano_mapping:
            pano_id = self.gps_to_pano_mapping[geocode]
            return True, geocode, pano_id
        
        # for the case that geocode is not in the mapping file,
        # calculate the disatnce to find the nearest geocode.
        gps_list = list(self.gps_to_pano_mapping.keys())
        distance_matrix = geocode_utils.cal_distance_between_two_position_list(
            [geocode], gps_list
        )[0]
        
        if distance_matrix.min() < self.offline_cfg.MAPPING_RADIUS:
            nearest_geocode = gps_list[distance_matrix.argmin()]
            pano_id = self.gps_to_pano_mapping[nearest_geocode]
            return True, nearest_geocode, pano_id
        else:
            return False, None, None

    def get_routing(self, origin, destination, mode='walking', avoid='indoor', language='en',
                    way_points=None, polyline=False, stopover=False, optimized=False,
                    no_last_leg=False, modify_destination=True, **kwargs):
        """
        Follow https://developers.google.com/maps/documentation/directions/get-directions#ExampleRequests

        Args:
            no_last_leg: remove the last leg of the route, for the case that destination is really cared
            optimized:
            stopover:
            origin: address or geocode of the start point
            destination: address or geocode of the end point
            mode: transportation mode in [driving, walking, bicycling, transit]
            avoid: features to avoid in [tolls, highways, ferries, indoor]
            language: language for return results
            way_points: a list of geocodes that the route need to pass through
            polyline: whether to use polyline as the route representation
            modify_destination: whether to modify the last geocode to the destination geocode in the returned route
            **kwargs:

        Returns:

        """
        params, origin, destination = self._get_route_params(origin, destination, mode, avoid, language)
        params.update(kwargs)

        if way_points is not None:
            way_points_str = self.formulate_waypoints(way_points, stopover, optimized)
            params['waypoints'] = f'{way_points_str}'

        base_url = self.base_urls['directions']
        response_json = requests.get(base_url, params=params).json()

        if stopover or optimized:
            legs = response_json['routes'][0]['legs']  # [:-1]
            if no_last_leg:
                legs = legs[:-1]
            result_dict = self.parse_routes_results_stopover(legs)
        else:
            result_dict = self.parse_routes_results_via(
                response_json, origin, polyline
            )

        # set proper destination geocode
        geocode_list = result_dict['geocode_list']
        distance = geocode_utils.calculate_distance_from_geocode(geocode_list[-1], destination)
        if modify_destination and not no_last_leg:
            if distance < 15:
                geocode_list[-1] = destination
            else:
                geocode_list.append(destination)
            result_dict['geocode_list'] = geocode_list

        return result_dict

    @staticmethod
    def formulate_waypoints(way_points, stopover, optimized):
        prefix = 'optimize:true|' if optimized else ''
        if stopover or optimized:
            way_points_str = '|'.join([f'{p[0]},{p[1]}' for p in way_points])
        else:
            way_points_str = '|'.join([f'via:{p[0]},{p[1]}' for p in way_points])

        return prefix + way_points_str

    def parse_routes_results_stopover(self, legs):
        result_dict = {
            'geocode_list': [],
            'polyline': [],
            'time': 0,
            'distance': 0,
        }

        for leg in legs:
            steps = leg['steps']
            cur_result_dict = self.get_route_single_step(steps)
            geocode_list = cur_result_dict['geocode_list']
            polyline_list = cur_result_dict['polyline']
            result_dict['geocode_list'].extend(geocode_list)
            result_dict['polyline'].extend(polyline_list)
            result_dict['distance'] += leg['distance']['value']
            result_dict['time'] += leg['duration']['value']

        return result_dict

    def parse_routes_results_via(self, response_json, origin, polyline):
        legs = response_json['routes'][0]['legs'][0]
        steps = legs['steps']

        if polyline:
            result_dict = self.get_route_single_step_polyline(steps, origin)
        else:
            result_dict = self.get_route_single_step(steps)

        result_dict.update({
            'time': legs['duration']['value'],
            'distance': legs['distance']['value'],
        })
        return result_dict

    def get_route_single_step(self, steps):
        geocode_list = []
        polyline_list = []
        distance_list = []
        time_list = []
        instruction_list = []
        for i, step in enumerate(steps):
            end_geocode = (step['end_location']['lat'], step['end_location']['lng'])
            polyline_str = step['polyline']['points']
            polyline_list.append(polyline_str)
            instruction_list.append(step['html_instructions'])
            distance_list.append(step['distance']['value'])
            time_list.append(step['duration']['value'])
            end_geocode, _ = self.relocate_geocode_by_source(end_geocode)
            geocode_list.append(end_geocode)

        result_dict = {
            'geocode_list': geocode_list,
            'polyline': polyline_list,
            'instructions': instruction_list,
            'distance_list': distance_list,
            'time_list': time_list,
        }

        return result_dict

    def get_route_single_step_polyline(self, steps, start_geocode):
        geocode_list = []
        polyline_list = []
        refer_geocode = start_geocode
        for i, step in enumerate(steps):
            polyline_str = step['polyline']['points']
            polyline_list.append(polyline_str)
            polyline_geocode = geocode_utils.decode_polyline(polyline_str)
            subsampled_geocode_list = self.subsample_geocode_by_distance(
                refer_geocode, polyline_geocode
            )
            refer_geocode = subsampled_geocode_list[-1]
            geocode_list.extend(subsampled_geocode_list)

        return geocode_list, polyline_list

    def subsample_geocode_by_distance(self, previous_geocode, geocode_list, distance=15):
        """Subsample a list of geocodes

        Args:
            previous_geocode:
            distance:
            geocode_list (list): a list of geocodes


        Returns:
            list: a list of subsampled geocodes
        """
        refer_geocode = previous_geocode
        subsampled_geocode_list = []
        for i, geocode in enumerate(geocode_list):
            if geocode_utils.calculate_distance_from_geocode(geocode, refer_geocode) > distance:
                relocated_geocode, _ = self.relocate_geocode_by_source(geocode, source='outdoor')
                if geocode_utils.calculate_distance_from_geocode(geocode, refer_geocode) > distance:
                    refer_geocode = relocated_geocode
                    subsampled_geocode_list.append(relocated_geocode)

        return subsampled_geocode_list

    def get_place_reviews(self, place_id, language='en', fields='reviews'):
        base_url = self.base_urls['place_details']

        params = {
            'place_id': place_id,
            'language': language,
            'fields': fields,
            'key': self.key,
        }

        response_json = requests.get(base_url, params=params).json()
        try:
            reviews = response_json['result']['reviews']
        except:
            reviews = []

        return reviews

    def _get_route_params(self, origin, destination, mode='walking', avoid='indoor', language='en'):
        """
        Follow https://developers.google.com/maps/documentation/directions/get-directions
        """
        if isinstance(origin, str):
            origin = self.get_geocode_from_address_v2(origin)

        if isinstance(destination, str):
            destination = self.get_geocode_from_address_v2(destination)

        params = {
            'origin': f'{origin[0]},{origin[1]}',
            'destination': f'{destination[0]},{destination[1]}',
            'mode': mode,
            'avoid': avoid,
            'language': language,
            'key': self.key,
        }

        return params, origin, destination

    def get_transportation_time(self, origin, destination, mode='walking', avoid='indoor', language='en'):
        params, _, _ = self._get_route_params(origin, destination, mode, avoid, language)

        base_url = self.base_urls['directions']
        response_json = requests.get(base_url, params=params).json()

        legs = response_json['routes'][0]['legs']
        all_time = 0
        for leg in legs:
            all_time += leg['duration']['value']

        return all_time

    def get_place_photo(self, photo_reference, max_width=400, max_height=400):
        base_url = self.base_urls['place_photos']

        params = {
            'maxwidth': max_width,  # ranging in 1-1600.
            'maxheight': max_height,
            'photo_reference': photo_reference,
            'key': self.key,
        }

        response = requests.get(base_url, params=params)

        if response.status_code == 200:
            image = Image.open(BytesIO(response.content))
        elif response.status_code == 404:
            raise Exception(f"Photo not found: {params}")
        else:
            raise Exception(f"Unexpected status code: {response.status_code}")

        return image

    def get_photo_references_from_place_details(self, place_id, fields='photos'):
        """
        Compared to nearby search, place details provide more information about the place,
        Returns:

        """
        base_url = self.base_urls['place_details']

        params = {
            'place_id': place_id,
            'fields': fields,
            'key': self.key,
        }

        response_json = requests.get(base_url, params=params).json()
        photo_info_list = response_json['result']['photos']
        photo_refer_list = [x['photo_reference'] for x in photo_info_list]

        return photo_refer_list
