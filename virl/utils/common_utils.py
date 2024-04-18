import json
import io
import base64
import logging
import re
import random
import string
import os

import numpy as np

from termcolor import colored
from PIL import Image


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ComparableObj(object):
    def __init__(self, priority, data):
        self.data = data
        self.priority = priority

    def __lt__(self, other):
        return self.priority < other.priority

    def __le__(self, other):
        return self.priority <= other.priority


def parse_answer_to_json(answer):
    start_idx = answer.index('{')
    end_idx = answer.index('}')
    return json.loads(answer[start_idx:end_idx+1])


def print_stage(content, c='#'):
    c = c[0]  # ensure c is a single character
    content_length = len(content)
    print('\n' + c * (content_length + 4))
    print(f'{c} {content} {c}')
    print(c * (content_length + 4) + '\n')


def print_prompt(prompt):
    print(colored('Prompt:', 'blue', attrs=['bold']))
    print(prompt)


def print_answer(answer):
    print(colored('Answer:', 'red', attrs=['bold']))
    print(answer)


def encode_image_to_string(image, show=False):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(np.uint8(image), 'RGB')
    img_format = image.format if image.format is not None else 'PNG'
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=img_format.upper())
    base64_string = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    if show:
        base64_string = f'data:image/{img_format.lower()};base64,{base64_string}'
    return base64_string


def decode_string_to_image(image_str: str):
    base64_decoded = base64.b64decode(image_str.split('base64,')[1])
    image = Image.open(io.BytesIO(base64_decoded))
    image = np.array(image)
    return image


def extract_numbers(s):
    match = re.search(r'\b\d+(\.\d+)?\b', s)
    if match:
        # Convert the matched number to a float if it has a decimal point, otherwise to an int
        number_str = match.group(0)
        return float(number_str) if '.' in number_str else int(number_str)
    else:
        # If no number is found, return None or raise an error
        return None


def ordinal(n: int):
    # https://stackoverflow.com/a/20007730
    if n < 0:
        return "-" + ordinal(-n)
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix


def dict_to_str_with_newline(data):
    formatted_items = [f"\"{key}\": \"{value}\"" for key, value in data.items()]
    formatted_string = "{\n" + ",\n".join(formatted_items) + "\n}"
    return formatted_string


def list_intersection(list1, list2):
    list3 = [value for value in list1 if value in list2]
    return list3


def load_points_in_txt_to_list(path):
    points = []
    with open(path, 'r') as f:
        for line in f:
            split_line = line.replace(')', '').replace('(', '').strip().split(',')
            points.append((float(split_line[0].strip()), float(split_line[1].strip())))

    return points


def save_points_to_txt(path, points):
    with open(path, 'w') as f:
        f.write('\n'.join(f'{geo[0]}, {geo[1]}' for geo in points))


def count_place_types(place_infos, cared_types):
    types = {}
    for place_info in place_infos.values():
        place_types = place_info['place_types']
        for place_type in place_types:
            if place_type in cared_types:
                types[place_type] = types.get(place_type, 0) + 1

    return types


def filter_place_by_region(place_infos, region_key):
    filtered_place_infos = {}
    for place_id, place_info in place_infos.items():
        if place_info['region'] == region_key:
            filtered_place_infos[place_id] = place_info

    return filtered_place_infos


def parse_str_json_list_to_list(plan):
    reply = plan.strip().split('\n\n')

    # Parse each dictionary string into a dictionary object
    parsed_reply = [eval(entry) for entry in reply]
    # Print the parsed reply
    initial_plan_list = [entry for entry in parsed_reply]
    return initial_plan_list


def dump_json_results(content, path):
    def convert(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError

    with open(path, 'w') as f:
        json.dump(content, f, default=convert, indent=4)


def generate_name(length=20):
    # Generate a random string of upper and lowercase letters and digits
    letters_and_digits = string.ascii_letters + string.digits
    random_string = ''.join(random.choice(letters_and_digits) for _ in range(length))
    return random_string


def save_tmp_image_to_file(img, output_dir, img_format='PNG'):
    img_name = generate_name() + f".{img_format.lower()}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    img_path = os.path.join(output_dir, img_name)
    img.save(img_path)
    
    return img_path


def map_region_to_continent_city(region):

    mapper = {
        'Rosebank_Johannesburg_Africa': ['Africa', 'Johannesburg'],
        'Surulere_Lagos_Africa': ['Africa', 'Lagos'],
        'Khar_Mumbai_Asia': ['Asia', 'Mumbai'],
        'Lajpat_Nagar_New_Delhi_Asia': ['Asia', 'New_Delhi'],
        'Prince_Edward_HK_Asia': ['Asia', 'HK'],
        'Shinjuku_Tokyo_Asia': ['Asia', 'Tokyo'],
        'CBD_Melbourne_Australia': ['Australia', 'Melbourne'],
        'SouthBank_Melbourne_Australia': ['Australia', 'Melbourne'],
        'Brera_Milan_Europe': ['Europe', 'Milan'],
        'Oxford_St_London_Europe': ['Europe', 'London'],
        'Chinatown_Manhattan_NY_North_America': ['North_America', 'NY'],
        'SoHo_NY_North_America': ['North_America', 'NY'],
        'Union_Square_SF_North_America': ['North_America', 'SF'],
        'Monserrat_Buenos_South_America': ['South_America', 'Buenos'],
    }

    return mapper[region]
