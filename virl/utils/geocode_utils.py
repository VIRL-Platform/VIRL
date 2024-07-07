import math
import torch
import polyline
import folium
import tqdm

import numpy as np

from folium.plugins import FastMarkerCluster
from geopy.distance import geodesic
from geopy.point import Point
from shapely.geometry import Point, Polygon


DIRECTION_HEADING = [0, 45, 90, 135, 180, 225, 270, 315]
DIRECTION_SET = ['front', 'right front', 'right', 'right behind', 'behind', 'left behind', 'left', 'left front']
DIRECTION_SET_ABS = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest']


def calculate_distance_from_geocode(geo_code_a, geo_code_b):
    return geodesic(geo_code_a, geo_code_b).meters


def get_geocode_by_heading_and_distance(geocode: tuple, heading: float, distance: float):
    """

    Args:
        geocode (tuple): latitude and longitude
        heading (float): the heading angle
        distance (float): distance in meters
    """
    current_point = (geocode[0], geocode[1])
    next_geocode = geodesic(kilometers=distance * 0.001).destination(
        current_point, heading
    )
    return next_geocode.latitude, next_geocode.longitude


def calculate_heading_between_geocodes(point_a: tuple, point_b: tuple):
    """
    Compute the distance and heading angle from geocode1 to geocode2.

    Args:
        point_a (tuple): _description_
        point_b (tuple): _description_
    """
    lat1 = math.radians(point_a[0])
    lat2 = math.radians(point_b[0])
    
    diff_long = math.radians(point_b[1] - point_a[1])
    
    x = math.sin(diff_long) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diff_long))
    
    initial_bearing = math.atan2(x, y)
    
    # Now we have the initial bearing but math.atan2() returns values from -π to + π 
    # So we need to normalize the result by converting it to a compass bearing as measured in degrees, 
    # i.e. the range 0° ... 360°
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360
    
    return compass_bearing


def calculate_headings_between_geocode_lists(points_a: list, points_b: list):
    """
    Compute the heading angle from each point in points_a to every point in points_b.

    Args:
        points_a (list): List of tuples, where each tuple represents a GPS location.
        points_b (list): List of tuples, where each tuple represents a GPS location.
    """
    points_a, points_b = np.array(points_a), np.array(points_b)

    lat1 = np.radians(points_a[:, 0])[:, np.newaxis]
    lat2 = np.radians(points_b[:, 0])

    diff_long = np.radians(points_b[:, 1] - points_a[:, 1][:, np.newaxis])

    x = np.sin(diff_long) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_long))

    initial_bearing = np.arctan2(x, y)

    # Now we have the initial bearing but np.arctan2() returns values from -π to + π 
    # So we need to normalize the result by converting it to a compass bearing as measured in degrees, 
    # i.e. the range 0° ... 360°
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


def get_heading_pitch_fov_to_box(bbox, image_shape, heading, pitch, fov=90,
                                 enlarge_factor=1.5, min_fov=10):
    # image dimensions
    img_width, img_height = image_shape

    # calculate image center
    img_center_x = img_width / 2
    img_center_y = img_height / 2

    # bounding box coordinates
    x1, y1, x2, y2 = bbox

    # calculate bounding box center
    bbox_center_x = (x1 + x2) / 2
    bbox_center_y = (y1 + y2) / 2

    # calculate pixel offsets
    offset_x = bbox_center_x - img_center_x
    offset_y = bbox_center_y - img_center_y

    # convert pixel offsets to changes in heading and pitch
    # Note: the scale factors here are hypothetical and may need adjustment
    delta_heading = offset_x / img_width * fov
    delta_pitch = offset_y / img_height * fov

    # apply changes to original heading and pitch
    new_heading = heading + delta_heading
    # positive pitch is downward, negative pitch is upward
    new_pitch = pitch - delta_pitch

    # make sure heading is within [0, 360) and pitch is within [-90, 90]
    new_heading = new_heading % 360
    new_pitch = max(-90, min(90, new_pitch))

    if isinstance(new_heading, torch.Tensor):
        new_heading = new_heading.item()
        new_pitch = new_pitch.item()

    # adjust fov
    if x2 - x1 > y2 - y1:
        long_edge = x2 - x1
        final_fov_region = long_edge * enlarge_factor
        new_fov = fov * (final_fov_region / img_width)
    else:
        long_edge = y2 - y1
        final_fov_region = long_edge * enlarge_factor
        new_fov = fov * (final_fov_region / img_height)

    new_fov = max(min_fov, new_fov)

    new_bbox_center_x = img_center_x
    new_bbox_center_y = img_center_y
    new_box_width = (x2 - x1) * fov / new_fov
    new_box_height = (y2 - y1) * fov / new_fov
    new_box = (new_bbox_center_x - new_box_width / 2.0, new_bbox_center_y - new_box_height / 2.0,
               new_bbox_center_x + new_box_width / 2.0, new_bbox_center_y + new_box_height / 2.0)

    return new_heading, new_pitch, new_fov, new_box


def get_heading_range_to_box(bbox, image_shape, heading, fov):
    # image dimensions
    img_width, img_height = image_shape

    # calculate image center
    img_center_x = img_width / 2

    # bounding box coordinates
    x1, y1, x2, y2 = bbox

    # calculate pixel offsets
    offset_x1 = x1 - img_center_x
    offset_x2 = x2 - img_center_x

    # convert pixel offsets to changes in heading and pitch
    # Note: the scale factors here are hypothetical and may need adjustment
    delta_heading1 = offset_x1 / img_width * fov
    delta_heading2 = offset_x2 / img_width * fov

    # apply changes to original heading and pitch
    new_heading_left = (heading + delta_heading1) % 360
    new_heading_right = (heading + delta_heading2) % 360

    if isinstance(new_heading_left, torch.Tensor):
        new_heading_left = new_heading_left.item()
        new_heading_right = new_heading_right.item()

    return (new_heading_left, new_heading_right)


def show_geocodes_on_map(geocode_list):
    # Create a map centered at the first geocode
    m = folium.Map(location=geocode_list[0], zoom_start=5)

    # Add a marker cluster
    FastMarkerCluster(geocode_list).add_to(m)

    # Save the map to an HTML file
    m.save('../temp_files/map.html')


def get_heading_and_distance_by_geocode(geocode1, geocode2):
    distance = calculate_distance_from_geocode(geocode1, geocode2)
    heading = calculate_heading_between_geocodes(geocode1, geocode2)
    return heading, distance


def get_intersect_from_geocodes_and_heading(p1, heading1, p2, heading2):
    # Convert heading angles to standard mathematical angles
    theta1 = np.radians((90 - heading1) % 360)
    theta2 = np.radians((90 - heading2) % 360)

    # Define line functions
    # Line equation format: y = mx + b
    m1 = np.tan(theta1)
    m2 = np.tan(theta2)
    b1 = p1[0] - m1 * p1[1]
    b2 = p2[0] - m2 * p2[1]

    if m1 == m2:  # The lines are parallel, hence they don't intersect
        return False
    else:
        # Calculate intersection point
        x_intersect = (b2 - b1) / (m1 - m2)
        y_intersect = m1 * x_intersect + b1

        # Convert intersection coordinates back to latitude and longitude
        intersect_geocode = (y_intersect, x_intersect)

        return intersect_geocode


def decode_polyline(polyline_str):
    decoded_polyline = polyline.decode(polyline_str)
    return decoded_polyline


def encode_polyline(geocode_list):
    encoded_polyline = polyline.encode(geocode_list)
    return encoded_polyline


def merge_polylines(polyline_list):
    geocode_list = []
    for cur_polyline in polyline_list:
        geocode_list.extend(polyline.decode(cur_polyline))
    merged_polyline = polyline.encode(geocode_list)
    return merged_polyline


def calculate_spatial_relationship_with_headings(agent_heading, heading_to):
    heading_diff = (heading_to - agent_heading) % 360
    # min_heading_idx = np.argmin(abs(heading_diff - np.array(DIRECTION_HEADING)))
    min_heading_idx = select_argmin_heading_from_heading_list(heading_diff, DIRECTION_HEADING)
    spatial_relation = DIRECTION_SET[min_heading_idx]
    return spatial_relation


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


def grid_sample_quadrangle(quadrangle, spacing):
    """
    Function to sample points inside a quadrangle
    Args:
    quadrangle (list): a list of four tuples representing the quadrangle (lat, lng)
    spacing (float): the spacing between grid points

    Returns:
    list: a list of tuples representing the sampled points (lat, lng)
    """
    # Create a polygon object
    polygon = Polygon(quadrangle)

    # Get the bounding box of the polygon
    minx, miny, maxx, maxy = polygon.bounds

    # Convert the spacing from meters to degrees
    spacing_lat = spacing / 111000
    spacing_lng = spacing / 111000

    # Create a grid of points inside the bounding box
    points = []
    for x in frange(minx, maxx, spacing_lat):
        for y in frange(miny, maxy, spacing_lng):
            point = Point(x, y)
            if polygon.contains(point):
                points.append((x, y))

    return points


def is_point_in_quadrangle(point, quadrangle):
    """
    Function to check if a point is inside a quadrangle
    Args:
    point (tuple): a tuple representing the point to check (lat, lng)
    quadrangle (list): a list of four tuples representing the quadrangle (lat, lng)

    Returns:
    bool: True if the point is in the quadrangle, False otherwise
    """
    point = Point(point)
    polygon = Polygon(quadrangle)
    return polygon.contains(point)


def relocate_point_list_in_polygon(platform, point_list, polygon):
    relocated_points = {}
    for point in tqdm.tqdm(point_list, total=len(point_list)):
        relocated_position, pano_id = platform.relocate_geocode_by_source(point, source='outdoor')
        if relocated_position is None or not is_point_in_quadrangle(relocated_position, polygon):
            continue

        if relocated_position not in relocated_points:
            relocated_points[relocated_position] = pano_id

    return list(relocated_points.keys()), list(relocated_points.values())


def euclidean_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)


def nearest_neighbor_algorithm(distances):
    """
    Nearest Neighbor Algorithm for tsp
    Args:
        distances:

    Returns:

    """
    n = len(distances)
    start = 0
    path = [start]
    unvisited = list(range(1, n))

    while unvisited:
        current = path[-1]
        next = min(unvisited, key=lambda x: distances[current, x])
        path.append(next)
        unvisited.remove(next)

    return path


def two_opt_algorithm(distances):
    """
    2-opt Algorithm for TSP
    Args:
        distances:

    Returns:

    """
    n = len(distances)
    # Start with a random tour
    tour = list(range(n))
    improved = True

    while improved:
        improved = False
        for i in range(n):
            for j in range(i + 2, n):
                old_distance = distances[tour[i - 1], tour[i]] + distances[tour[j - 1], tour[j]]
                new_distance = distances[tour[i - 1], tour[j - 1]] + distances[tour[i], tour[j]]
                if new_distance < old_distance:
                    tour[i:j] = reversed(tour[i:j])
                    improved = True

    return tour


def calculate_tsp_route_with_points(point_list, opt_algo='2opt'):
    """
    Calculate the tsp route with a list of points
    Args:
        point_list:
        opt_algo

    Returns:

    """
    # calculate the distance
    distances = np.zeros((len(point_list), len(point_list)))

    for i, point1 in enumerate(point_list):
        for j, point2 in enumerate(point_list):
            distances[i, j] = euclidean_distance(point1, point2)

    print(f"Start to run {opt_algo} algorithm")
    if opt_algo == 'nn':
        route = nearest_neighbor_algorithm(distances)
    elif opt_algo == '2opt':
        route = two_opt_algorithm(distances)
    else:
        raise ValueError()

    return route


def haversine_distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371.0 * c
    m = km * 1000
    return m


def cal_distance_between_two_position_list(position_list_a, position_list_b):
    """
    A naive implementation to calculate the distance between two position list.
    The error between this implementation to the geopy.distance.geodesic is about 0.x meters.

    Args:
        position_list_a (list): _description_
        position_list_b (list): _description_

    Returns:
        dist_matrix (numpy.array): _description_
    """
    positions_a = np.array(position_list_a)
    positions_b = np.array(position_list_b)

    dist_matrix = haversine_distance(
        positions_a[:, 0, np.newaxis], positions_a[:, 1, np.newaxis], positions_b[:, 0], positions_b[:, 1]
    )

    return dist_matrix


def create_polygon_around_geocode(geocode, max_distance):
    # convert meters to degree
    max_distance = max_distance / 111300
    left_top = (geocode[0] + max_distance, geocode[1] - max_distance)
    left_bottom = (geocode[0] + max_distance, geocode[1] + max_distance)
    right_top = (geocode[0] - max_distance, geocode[1] - max_distance)
    right_bottom = (geocode[0] - max_distance, geocode[1] + max_distance)

    return [left_top, left_bottom, right_bottom, right_top]


def find_places_within_geocode_and_radius(place_infos, geocode, min_radius=None, max_radius=None):
    positions = [place_info['geocode'] for place_info in place_infos.values()]
    distance_matrix = cal_distance_between_two_position_list([geocode], positions)[0]
    
    valid_mask = np.ones(len(distance_matrix)).astype(np.bool_)
    if max_radius is not None:
        valid_mask = valid_mask & (distance_matrix <= max_radius)
    if min_radius is not None:
        valid_mask = valid_mask & (distance_matrix >= min_radius)
    
    filtered_place_infos = {}
    for i, (place_id, place_info) in enumerate(place_infos.items()):
        if valid_mask[i]:
            place_info['distance'] = distance_matrix[i]
            filtered_place_infos[place_id] = place_info

    return filtered_place_infos


def get_direction_abs_by_heading(heading):
    min_heading_idx = select_argmin_heading_from_heading_list(heading, DIRECTION_HEADING)
    return DIRECTION_SET_ABS[min_heading_idx]


def select_argmin_heading_from_heading_list(expected_heading, heading_list):
    all_headings = np.array(heading_list)
    heading_diff = np.abs(all_headings - expected_heading)
    heading_diff = np.where(heading_diff > 180, 360 - heading_diff, heading_diff)
    heading_idx = np.argmin(heading_diff)
    return heading_idx


def cal_min_heading_diff_between_headings(heading1, heading2):
    heading_diff = np.abs(heading1 - heading2)
    if heading_diff > 180:
        heading_diff = 360 - heading_diff
    return heading_diff


def is_heading_in_range(heading_range, heading, heading_epsilon=0.0):
    heading_left, heading_right = heading_range
    heading_left = (heading_left - heading_epsilon) % 360
    heading_right = (heading_right + heading_epsilon) % 360
    
    if heading_left < heading_right:
        return heading_left <= heading <= heading_right
    else:
        # wraps around 0/360
        return heading >= heading_left or heading <= heading_right


def get_heading_list_by_range_and_fov(cur_heading, heading_range, fov):
    num_image = heading_range // fov
    if num_image % 2 == 0:
        start_heading = (cur_heading - num_image // 2 * fov + fov / 2) % 360
    else:
        start_heading = (cur_heading - num_image // 2 * fov) % 360
    heading_list = []
    for i in range(num_image):
        heading_list.append((start_heading + i * fov) % 360)

    return heading_list


def calculate_square_region(point1, point2, buffer=10):
    lat1, lng1 = point1
    lat2, lng2 = point2
    # Constants
    R = 6378137  # Radius of the Earth in meters

    def offset_point(lat, lon, d, angle):
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)
        brng = math.radians(angle)

        lat2 = math.asin(math.sin(lat1) * math.cos(d / R) + math.cos(lat1) * math.sin(d / R) * math.cos(brng))
        lon2 = lon1 + math.atan2(math.sin(brng) * math.sin(d / R) * math.cos(lat1),
                                 math.cos(d / R) - math.sin(lat1) * math.sin(lat2))
        return math.degrees(lat2), math.degrees(lon2)

    # Calculate the angle of the line between the two points
    angle = math.atan2(lng2 - lng1, lat2 - lat1) * 180 / math.pi

    # Calculate the internal buffer points
    lat1_inner_offset1, lng1_inner_offset1 = offset_point(lat1, lng1, buffer, angle + 90)
    lat1_inner_offset2, lng1_inner_offset2 = offset_point(lat1, lng1, buffer, angle - 90)
    lat2_inner_offset1, lng2_inner_offset1 = offset_point(lat2, lng2, buffer, angle + 90)
    lat2_inner_offset2, lng2_inner_offset2 = offset_point(lat2, lng2, buffer, angle - 90)

    # Inner square region coordinates
    inner_square_region = [
        (lat1_inner_offset1, lng1_inner_offset1),
        (lat1_inner_offset2, lng1_inner_offset2),
        (lat2_inner_offset1, lng2_inner_offset1),
        (lat2_inner_offset2, lng2_inner_offset2)
    ]

    return inner_square_region


def extend_line(point1, point2, extension_distance):
    def calculate_new_point(lat, lon, bearing, distance):
        R = 6378.1  # Radius of the Earth in km
        bearing = math.radians(bearing)

        lat1 = math.radians(lat)
        lon1 = math.radians(lon)

        lat2 = math.asin(math.sin(lat1) * math.cos(distance / R) + 
                         math.cos(lat1) * math.sin(distance / R) * math.cos(bearing))
        lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance / R) * math.cos(lat1),
                            math.cos(distance / R) - math.sin(lat1) * math.sin(lat2))

        return math.degrees(lat2), math.degrees(lon2)

    def calculate_bearing(pointA, pointB):
        lat1 = math.radians(pointA[0])
        lat2 = math.radians(pointB[0])
        diff_long = math.radians(pointB[1] - pointA[1])

        x = math.sin(diff_long) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diff_long))

        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360

        return compass_bearing

    distance_km = extension_distance / 1000  # convert meters to kilometers

    bearing = calculate_bearing(point1, point2)

    # Extend in the direction of point1 to point2
    extended_point2 = calculate_new_point(point2[0], point2[1], bearing, distance_km)

    # Extend in the opposite direction
    opposite_bearing = (bearing + 180) % 360
    extended_point1 = calculate_new_point(point1[0], point1[1], opposite_bearing, distance_km)

    return extended_point1, extended_point2


def calculate_square_region_with_extend(point1, point2, extend_dis=10):
    extended_point1, extended_point2 = extend_line(
        point1, point2, extension_distance=extend_dis
    )
    square_region = calculate_square_region(
        extended_point1, extended_point2, buffer=extend_dis
    )
    return square_region
