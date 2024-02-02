from .visual_explore import panorama_street_view_template, panorama_no_street_view_template
from .polyline import polyline_template, polyline_with_waypoints_template, \
    polyline_with_waypoints_startpoint_template
from .heatmap import heatmap_template

__all__ = {
    'panorama_street_view_template': panorama_street_view_template,
    'panorama_no_street_view_template': panorama_no_street_view_template,
    'polyline_template': polyline_template,
    'polyline_with_waypoints_template': polyline_with_waypoints_template,
    'heatmap_template': heatmap_template,
    'polyline_with_waypoints_startpoint_template': polyline_with_waypoints_startpoint_template
}


def get_file_template_by_name(name):
    return __all__[name]
