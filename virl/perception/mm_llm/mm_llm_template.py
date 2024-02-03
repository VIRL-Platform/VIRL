import matplotlib.pyplot as plt

from virl.lm import UnifiedChat
from virl.lm.prompt import vision_templates


class MultiModalLLMTemplate(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def predict(self, image, question):
        raise NotImplementedError

    def ask(self, image, question, return_json=False):
        answer = self.predict(image, question)
        if return_json:
            prompt_template = vision_templates.PARSE_ANSWER_TO_JSON
            prompt = prompt_template.format(question=question, answer=answer)
            answer = UnifiedChat.ask(prompt, json=True)

        return answer

    @staticmethod
    def visualize(image, question, answer_json, image_name):
        plt.figure()
        plt.imshow(image)
        plt.axis('off')

        answer = answer_json['answer'] + ' ' + answer_json['explanation']
        plt.figtext(0.03, 0.9, question, wrap=True, horizontalalignment='left', fontsize=9)
        plt.figtext(0.03, 0.03, answer, wrap=True, horizontalalignment='left', fontsize=9)
        plt.savefig(image_name)
