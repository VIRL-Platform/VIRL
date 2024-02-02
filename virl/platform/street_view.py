import cv2
import numpy as np


class StreetViewImage(object):
    def __init__(self, image, heading, pitch, fov, geocode, i=None):
        self.heading = heading
        self.pitch = pitch
        self.shape = image.size
        self.image = image
        self.fov = fov
        self.geocode = geocode
        self.box = None
        # the i-th view in current agent pose
        self.i = i

        # for visual memory
        self.obj_id = None
        self.category = None
        self.box_score = None

    def set_detect_result(self, detect_result):
        self.box = detect_result['boxes']
        self.category = detect_result['labels']
        self.box_score = detect_result['scores']

    def set_obj_id(self, obj_id):
        self.obj_id = obj_id

    def show(self):
        self.image.show()

    def __repr__(self):
        return f"StreetViewImage(geocode={self.geocode}, " \
               f"heading={self.heading}, pitch={self.pitch}, " \
               f"fov={self.fov})"


def xyz2lonlat(xyz):
    atan2 = np.arctan2
    asin = np.arcsin

    norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
    xyz_norm = xyz / norm
    x = xyz_norm[..., 0:1]
    y = xyz_norm[..., 1:2]
    z = xyz_norm[..., 2:]

    lon = atan2(x, z)
    lat = asin(y)
    lst = [lon, lat]

    out = np.concatenate(lst, axis=-1)
    return out


def lonlat2XY(lonlat, shape):
    X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
    Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
    lst = [X, Y]
    out = np.concatenate(lst, axis=-1)

    return out


def get_perspective_from_panorama(img_name, FOV, heading, pitch, height, width, north_rotation):
    """
    Modified from https://github.com/fuenwang/Equirec2Perspec
    heading is left/right angle, pitch is up/down angle, both in degree
    Args:
        img_name:
        FOV:
        heading:
        pitch:
        height:
        width:
        north_rotation:

    Returns:

    """
    # adjust heading to match the Google street view format
    heading = ((heading - 180) % 360 + north_rotation) % 360

    _img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    [_height, _width, _] = _img.shape

    f = 0.5 * width * 1 / np.tan(0.5 * FOV / 180.0 * np.pi)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1],
    ], np.float32)
    K_inv = np.linalg.inv(K)

    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
    xyz = xyz @ K_inv.T

    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(heading))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(pitch))
    R = R2 @ R1
    xyz = xyz @ R.T
    lonlat = xyz2lonlat(xyz)
    XY = lonlat2XY(lonlat, shape=_img.shape).astype(np.float32)
    persp = cv2.remap(_img, XY[..., 0], XY[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)

    return persp

