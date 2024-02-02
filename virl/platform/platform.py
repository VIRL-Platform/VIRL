import tqdm

from .google_map_apis import GoogleMapAPI
from .mover import StreetViewMover
from virl.utils import geocode_utils


class Platform(GoogleMapAPI):
    def __init__(self, platform_cfg, output_dir):
        super().__init__()
        self.platform_cfg = platform_cfg
        self.mover = None
        self.output_dir = output_dir
        self.street_view_cfg = platform_cfg.STREET_VIEW

    def initialize_mover(self, initial_geocode):
        self.mover = StreetViewMover(
            self.key, self.platform_cfg.MOVER, initial_geocode, self,
            output_dir=self.output_dir
        )

    def get_all_streetview_from_geocode(self, geocode: tuple, size: tuple = None,
                                        pitch: int = None, fov: int = None, source: str = None,
                                        cur_heading=0, heading_list=None, all_around=False):
        """Get all streetview images from a geocode.

        Args:
            all_around: check 360 degree
            cur_heading:
            geocode (tuple): latitude and longitude
            size (tuple): (width, height). 640x640 is the max size for free user.
            pitch (int): Angle of camera's vertical axis. Value range in [-90, 90]
            fov (int): Field of view of the camera in degrees, which must be between 10 and 120.
            source (str): default or outdoor
            heading_list (list): a list of heading angles

        Returns:
            list: a list of images
        """
        size = size if size is not None else self.street_view_cfg.get('SIZE', (640, 640))
        pitch = pitch if pitch is not None else self.street_view_cfg.get('PITCH', 0)
        fov = fov if fov is not None else self.street_view_cfg.get('FOV', 90)
        source = source if source is not None else self.street_view_cfg.get('SOURCE', 'outdoor')
        images = []

        if heading_list is None and not all_around:
            heading_range = self.street_view_cfg.get('HEADING_RANGE', 360)
            heading_list = geocode_utils.get_heading_list_by_range_and_fov(
                cur_heading, heading_range, fov
            )
        elif heading_list is None and all_around:
            heading_list = range(0, 360, fov)

        for i, heading in enumerate(heading_list):
            images.append(self.get_streetview_from_geocode(
                geocode, size, heading, pitch, fov, source=source, idx=i
            ))

        return images

    def get_place_photo_from_list(self, place_results_list, max_width=400, max_height=400, save_debug=False):
        """Get a list of photos from a place.

        Args:
            place_results_list (list): a list of photo refer
            max_width (int): max width of the photo
            max_height (int): max height of the photo
            save_debug (bool): save the debug images

        Returns:
            list: a list of photos
        """
        photos = []
        for result in tqdm.tqdm(place_results_list, totoal=len(place_results_list)):
            photo_refer = result['photo_reference']
            if photo_refer is not None:
                photo = self.get_place_photo(photo_refer, max_width, max_height)

                if save_debug:
                    photo.save(f"../output/debug/{result['name']}.jpg")

                photos.append(photo)

        return photos
