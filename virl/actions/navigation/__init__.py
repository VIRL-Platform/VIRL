from .point_navigator import PointNavigator
from .navigator_template import NavigatorTemplate
from .intention_navigator import IntentionNavigator
from .route_navigator import RouteNavigator
from .vision_language_navigator import VisionLanguageNavigator


__all__ = {
    'PointNavigator': PointNavigator,
    'NavigatorTemplate': NavigatorTemplate,
    'IntentionNavigator': IntentionNavigator,
    'RouteNavigator': RouteNavigator,
    'VisionLanguageNavigator': VisionLanguageNavigator
}


def build_navigator(cfg, platform, messager, current_location, output_dir, **kwargs):
    return __all__[cfg.NAME](cfg, platform, messager, current_location, output_dir, **kwargs)
