import time
import os
import numpy as np

from concurrent.futures import ProcessPoolExecutor

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager

from virl.platform.file_template import get_file_template_by_name
from virl.utils import geocode_utils


class StreetViewMover(object):
    def __init__(self, key, cfg, initial_geocode, platform, output_dir):
        self.cfg = cfg
        self.street_view_query = cfg.get('STREET_VIEW_QUERY', True)
        self.radius_query = cfg.get('RADIUS_QUERY', False) and cfg.RADIUS_QUERY.ENABLED
        self.platform = platform

        self.current_path_elements = []
        # tuple: (geocode, heading, distance)
        self.current_possible_geocodes = []

        driver_path = cfg.get('WEB_DRIVER_PATH', None)
        file_path = output_dir / 'mover.html'
        self.create_file(file_path, initial_geocode, key)

        self.driver = None
        self.init_driver(driver_path, file_path)

    def init_driver(self, driver_path, file_path):
        options = Options()
        if self.cfg.get('HEADLESS', False):
            options.add_argument("--headless")
        try:
            self.driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
        except:
            self.driver = webdriver.Chrome(driver_path, options=options)
        
        abs_file_path = 'file://' + os.path.abspath(file_path)
        self.driver.get(abs_file_path)

    def create_file(self, file_path, geocode, key):
        with open(file_path, 'w') as file_html:
            content_template = get_file_template_by_name(self.cfg.FILE_TEMPLATE)
            _, pano_id = self.platform.relocate_geocode_by_source(geocode)
            file_content = content_template.format(
                lat=geocode[0],
                lng=geocode[1],
                pano_id=pano_id,
                heading=0.0,
                pitch=0.0,
                key=key
            )
            file_html.write(file_content)

    def get_all_possible_paths(self):
        attempts = 0
        while attempts < 3:
            try:
                return self._get_all_possible_paths()
            except StaleElementReferenceException:
                attempts += 1
                time.sleep(1)
        raise StaleElementReferenceException("Max number of attempts exceeded.")

    def _get_all_possible_paths(self):
        self.current_path_elements = self.driver.execute_script("""
        return document.querySelector('svg').querySelectorAll('path[role="button"]')
        """)

    def move(self, idx, max_waited_time=5):
        old_geocode = self.get_current_geocode()
        self._move(idx)
        wait_time = 0
        while True:
            time.sleep(0.2)
            wait_time += 0.2
            new_geocode = self.get_current_geocode()
            if new_geocode != old_geocode or wait_time > max_waited_time:
                current_geocode = new_geocode
                break

        current_geocode, _ = self.platform.relocate_geocode_by_source(current_geocode, 'outdoor')
        self.current_path_elements = []
        self.current_possible_geocodes = []
        return current_geocode

    def _move(self, idx):
        if self.street_view_query and idx < len(self.current_path_elements):
            self.move_by_elements(idx)
        elif self.radius_query and idx >= len(self.current_path_elements):
            self.move_by_geocode(idx - len(self.current_path_elements))
        else:
            raise ValueError('Invalid moving direction index: {idx}')

    def move_by_elements(self, idx):
        action = """
        var buttons = document.querySelector('svg').querySelectorAll('path[role="button"]');
        var event = new MouseEvent('click', {{
            'view': window,
            'bubbles': true,
            'cancelable': true
        }});
        buttons[{idx}].dispatchEvent(event);
        """.format(idx=idx)
        self.driver.execute_script(action)

    def move_by_geocode(self, idx):
        geocode = self.current_possible_geocodes[idx][0]
        self._move_by_geocode(geocode)

    def _move_by_geocode(self, geocode):
        lat_input = self.driver.find_element(By.ID, 'lat')
        lng_input = self.driver.find_element(By.ID, 'lng')
        update_button = self.driver.find_element(By.ID, 'update-button')

        lat_input.clear()
        lng_input.clear()

        lat_input.send_keys(f'{geocode[0]}')
        lng_input.send_keys(f'{geocode[1]}')

        update_button.click()
        lat_input.clear()
        lng_input.clear()

    def adjust_heading_web(self, heading):
        head_input = self.driver.find_element(By.ID, 'heading')
        update_button = self.driver.find_element(By.ID, 'head-update-button')

        head_input.clear()

        head_input.send_keys(f'{heading}')

        update_button.click()
        head_input.clear()
    
    def close(self):
        self.driver.close()

    def get_current_geocode(self):
        element = self.driver.find_element(By.ID, "panorama-coordinates")
        content = element.text
        lat_info, lng_info = content.split(',')
        latitude = float(lat_info.split(' ')[2].strip())
        longitude = float(lng_info.split(' ')[2].strip())
        return latitude, longitude

    def get_current_heading_and_pitch(self):
        element = self.driver.find_element(By.ID, "panorama-pov")
        content = element.text
        heading_info, pitch_info = content.split(',')
        heading = float(heading_info.split(' ')[2].strip())
        pitch = float(pitch_info.split(' ')[2].strip())
        return heading, pitch

    def get_suitable_heading_to_path(self, idx):
        """

        Args:
            idx: the index of the path element

        Returns:
            heading: the heading that is suitable to the specific path
        """
        path_element = self.current_path_elements[idx]
        transform = path_element.get_attribute('transform')
        if 'rotate' in transform:
            rotate = float(transform.split('rotate(')[1].split(')')[0])
        else:
            rotate = 0.0
        current_heading = self.get_current_heading_and_pitch()[0]
        heading = (rotate + current_heading) % 360
        return heading

    def get_all_suitable_heading_to_path(self, geocode, info_dict=None):
        heading_list = []
        if self.street_view_query:
            heading_list1 = []
            self.get_all_possible_paths()
            for idx in range(len(self.current_path_elements)):
                heading_list1.append(self.get_suitable_heading_to_path(idx))
            heading_list += heading_list1

            # consider to query by radius or not.
            # This can help to reduce the number of radius query.
            if self.radius_query:
                heading_diff = np.abs(np.array(heading_list) - info_dict['expected_heading'])
                min_heading_diff = np.min(heading_diff)
                if min_heading_diff < self.cfg.RADIUS_QUERY.CONDITION_HEADING_RANGE:
                    return heading_list

        if self.radius_query:
            radius_cfg = self.cfg.RADIUS_QUERY
            possible_geocode_results = self.query_nearby_area(
                geocode, max_radius=radius_cfg.MAX_RADIUS, delta_radius=radius_cfg.DELTA_RADIUS,
                delta_heading=radius_cfg.DELTA_HEADING, heading_range=radius_cfg.HEADING_RANGE,
                existing_heading=heading_list
            )
            heading_list2 = []
            self.current_possible_geocodes = []
            for geocode, (heading, distance) in possible_geocode_results.items():
                heading_list2.append(heading)
                self.current_possible_geocodes.append((geocode, heading, distance))
            heading_list += heading_list2

        return heading_list

    def get_all_suitable_heading_to_path_vln(self, geocode, radius_query=False, info_dict=None):
        """
        Compared to the naive version, this will only use radius query in the intersection
        TODO: merge with function get_all_suitable_heading_to_path

        Args:
            geocode (tuple): _description_
            info_dict (dict, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        heading_list = []
        if self.street_view_query:
            heading_list1 = []
            self.get_all_possible_paths()
            for idx in range(len(self.current_path_elements)):
                heading_list1.append(self.get_suitable_heading_to_path(idx))
            heading_list += heading_list1

        # if there is no intersection, then only use the street view query
        # if there is intersection, then use both the street view and the radius query
        if len(heading_list) < 2 or radius_query or (len(heading_list) > 2 and self.radius_query):
            radius_cfg = self.cfg.RADIUS_QUERY
            possible_geocode_results = self.query_nearby_area(
                geocode, max_radius=radius_cfg.MAX_RADIUS, delta_radius=radius_cfg.DELTA_RADIUS,
                delta_heading=radius_cfg.DELTA_HEADING, heading_range=radius_cfg.HEADING_RANGE,
                existing_heading=heading_list
            )
            heading_list2 = []
            self.current_possible_geocodes = []
            for geocode, (heading, distance) in possible_geocode_results.items():
                heading_list2.append(heading)
                self.current_possible_geocodes.append((geocode, heading, distance))
            heading_list += heading_list2

        return heading_list

    def query_nearby_area(self, geocode, max_radius=10, delta_radius=2, delta_heading=30,
                          heading_range=10, existing_heading=()):
        possible_geocode_results = {}
        current_geocode, _ = self.platform.relocate_geocode_by_source(geocode, source='outdoor')
        existing_heading = [heading % 360 for heading in existing_heading]

        possible_radius = np.arange(delta_radius, max_radius, delta_radius)
        possible_headings = np.arange(0, 360, delta_heading)
        for heading in possible_headings:
            for radius in possible_radius:
                heading_to, distance_to, query_geocode = self.query_nearby_walkable_position_single(
                    current_geocode, heading, radius, existing_heading, possible_geocode_results, heading_range
                )
                if heading_to is not None:
                    possible_geocode_results[query_geocode] = (heading_to, distance_to)
                    existing_heading.append(heading_to)

        return possible_geocode_results

    def query_nearby_walkable_position_single(self, current_geocode, heading, radius, existing_heading,
                                              possible_geocode_results, heading_range):
        if not self.check_valid_of_heading(heading, existing_heading, heading_range=heading_range):
            return None, None, None
        possible_geocode = geocode_utils.get_geocode_by_heading_and_distance(current_geocode, heading, radius)

        relocated_geocode, _ = self.platform.relocate_geocode_by_source(possible_geocode, source='outdoor')
        if relocated_geocode is not None and relocated_geocode != current_geocode and \
                relocated_geocode not in possible_geocode_results:
            heading_to_B = geocode_utils.calculate_heading_between_geocodes(current_geocode, relocated_geocode)
            distance_to_B = geocode_utils.calculate_distance_from_geocode(current_geocode, relocated_geocode)

            # for each heading, only consider its nearest walkable geocode
            if self.check_valid_of_heading(heading_to_B, existing_heading, heading_range=heading_range):
                return heading_to_B, distance_to_B, relocated_geocode
            else:
                return None, None, None
        else:
            return None, None, None

    @staticmethod
    def check_valid_of_heading(heading, existing_heading, heading_range=10):
        for heading_to_B in existing_heading:
            if (abs(heading - heading_to_B) % 360) < heading_range:
                return False
        return True
