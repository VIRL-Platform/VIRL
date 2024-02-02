import time

from virl.utils.common_utils import AverageMeter, parse_answer_to_json


class ChatBotTemplate(object):
    def __init__(self, cfg, **kwargs):
        self.cfg = cfg
        self.ask_counter = 0
        self.timer = AverageMeter()

    def ask(self, content, json=False, **kwargs):
        end = time.time()
        answer = self._ask(content, **kwargs)
        self.timer.update(time.time() - end)
        self.ask_counter += 1

        if json:
            answer = parse_answer_to_json(answer)

        return answer

    def _ask(self, content, **kwargs):
        pass

    def get_time(self):
        return self.timer.val, self.timer.avg
