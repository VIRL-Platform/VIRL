import time
import cv2
import numpy as np
from PIL import Image

from virl.config import cfg
from virl.utils import common_utils, geocode_utils


def mimic_detection_panorama(platform, current_heading):
    heading_range = cfg.PLATFORM.STREET_VIEW.HEADING_RANGE
    fov = cfg.PLATFORM.STREET_VIEW.FOV
    heading_list = geocode_utils.get_heading_list_by_range_and_fov(
        current_heading, heading_range, fov
    )

    for heading in heading_list:
        platform.mover.adjust_heading_web(heading)
        time.sleep(0.8)
    
    platform.mover.adjust_heading_web(current_heading)
    time.sleep(0.5)


def compute_colors_for_labels(labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = (30 * (labels[:, None] - 1) + 1) * palette
    colors = (colors % 255).astype("uint8")
    return colors


def draw_with_results(image, results):
    if isinstance(image, Image.Image):
        image = np.array(image)

    boxes = results['boxes']
    class_idx = results['class_idx']
    scores = results['scores']
    labels = results['labels']

    colors = compute_colors_for_labels(class_idx).tolist()

    # template = "{}: {:.2f}"
    result_image = image.copy()
    for box, score, label, color in zip(boxes, scores, labels, colors):
        box = box.astype(np.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        result_image = cv2.rectangle(
            result_image, tuple(top_left), tuple(bottom_right), tuple(color), 2
        )

        x, y = box[:2]
        if isinstance(score, list) or isinstance(score, np.ndarray):
            s = "{}: {:.2f}, {:.2f}".format(label, score[0], score[1])
        else:
            s = "{}: {:.2f}".format(label, score)
        result_image = cv2.putText(
            result_image, s, (int(x), int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
        )

    return Image.fromarray(result_image)
