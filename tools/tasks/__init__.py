from tools.tasks.route_optimizer import RouteOptimizer

from tools.tasks.place_recommender import PlaceRecommender
from tools.tasks.estate_recommender import EstateRecommender

from tools.tasks.robot_rx399 import RobotRX399
from tools.tasks.urban_plan import UrbanPlanner
from tools.tasks.intentional_explorer import IntentionalExplorer

from tools.tasks.vision_language_nav.local import Local
from tools.tasks.vision_language_nav.tourist import Tourist

from .interactive_concierge import InteractiveConcierge


__all__ = {
    # Agents
    'RouteOptimizer': RouteOptimizer,
    'PlaceRecommender': PlaceRecommender,
    'EstateRecommender': EstateRecommender,
    'RobotRX399': RobotRX399,
    'UrbanPlanner': UrbanPlanner,
    'IntentionalExplorer': IntentionalExplorer,
    'Local': Local,
    'Tourist': Tourist,
    'InteractiveConcierge': InteractiveConcierge,
}


def build_task_solver(task_name, **kwargs):
    return __all__[task_name](**kwargs)
