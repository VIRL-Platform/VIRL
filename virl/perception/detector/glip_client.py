import os
import json
import numpy as np
import matplotlib.pyplot as plt
from gradio_client import Client

from virl.utils import common_utils
from virl.config import cfg


class GLIPClient(object):
    def __init__(self, cfg, **kwargs):
        self.server_url = cfg.SERVER
        self.client = Client(self.server_url)

    # @staticmethod
    def inference(self, img, caption, score_thresh=0.7, need_draw=False):
        img_format = img.format

        output_dir = os.path.join(cfg.get('OUTPUT_DIR', 'output'), 'tmp')
        img_path = common_utils.save_tmp_image_to_file(img, output_dir, img_format)

        while True:
            flag = True
            try:
                response = self.client.predict(
                    img_path,
                    caption,
                    score_thresh,
                    need_draw,
                    api_name="/predict"
                )
            except:
                print('Timeout! Resend the message.')
                flag = False
            if flag:
                break

        answer, image_string = response[0], response[1]
        # if answer is a json path
        if '.json' in answer:
            answer = json.load(open(answer, 'r'))
        
        answer['boxes'] = np.array(answer['boxes'])
        answer['class_idx'] = np.array(answer['class_idx'])
        answer['scores'] = np.array(answer['scores'])
        answer['labels'] = np.array(answer['labels'])
        if need_draw:
            image_out = common_utils.decode_string_to_image(image_string)
            self.imshow(image_out, caption)

        return answer, image_string

    @staticmethod
    def imshow(img, caption, image_name='debug_output.png'):
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.figtext(0.5, 0.9, caption, wrap=True, horizontalalignment='center', fontsize=20)
        plt.savefig(image_name)


def main():
    from easydict import EasyDict
    from PIL import Image
    cfg = {
        'SCORE_THRESH': 0.5,
    }
    client = GLIPClient(EasyDict(cfg))
    img = Image.open('/xxx/mcdonald.png')
    prompt = 'a restaurant'
    answer, grouned_img = client.inference(img, prompt, need_draw=True)

    print('answer: ', answer)


if __name__ == '__main__':
    main()
