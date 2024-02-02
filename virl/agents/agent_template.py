
from collections import defaultdict


def default_status_value():
    # this func is required to make the agent picklable, w/ defaultdict status
    return 50  # default to 50% status [0, 100]


class AgentTemplate(object):
    def __init__(self, cfg):
        self.intention = cfg.get('INTENTION', 'None')
        self.intention_suppl = cfg.get('INTENTION_SUPPL', 'None')
        self.full_intention = self.intention + " " + self.intention_suppl
        self.city = cfg.CITY
        self.start_position = cfg.START_POSITION

        self.background = cfg.get('BACKGROUND', 'None')
        self.name = cfg.get('NAME', 'None')

        self.status = defaultdict(default_status_value)
        if cfg.get('STATUS', None):
            # [0, 100]
            for key, value in cfg.STATUS.items():
                self.status[key.lower()] = value

    def update_intention(self, intention):
        self.intention = intention
