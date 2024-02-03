import os
import base64
import requests
import numpy as np
import matplotlib.pyplot as plt

from virl.utils.common_utils import encode_image_to_string, decode_string_to_image


class GLIPClient(object):
    def __init__(self, cfg, **kwargs):
        self.server_url = cfg.SERVER

    # @staticmethod
    def predict_from_path(self, path, caption, score_thresh=0.5):
        assert os.path.exists(path)
        with open(path, "rb") as img_file:
            base64_string = base64.b64encode(img_file.read()).decode('utf-8')
        img_type = path.split('.')[-1]
        response = requests.post(self.server_url + "/predict", json={
            "data": [
                f'data:image/{img_type.lower()};base64,{base64_string}',
                caption,
                score_thresh,
            ]
        }).json()
        assert 'data' in response, f'predict failed: {response}'

    # @staticmethod
    def inference(self, img, caption, score_thresh=0.7, need_draw=False):
        img_format = img.format
        base64_string = encode_image_to_string(img)
        while True:
            flag = True
            try:
                response = requests.post(self.server_url + "/run/predict", json={
                    "data": [
                        f'data:image/{img_format.lower()};base64,{base64_string}',
                        caption,
                        score_thresh,
                        need_draw,
                    ]},
                    timeout=10
                ).json()
            except:
                print('Timeout! Resend the message.')
                flag = False
            if flag:
                break
        assert 'data' in response, f'predict failed: {response}'
        answer, image_string = response['data'][0], response['data'][1]
        answer['boxes'] = np.array(answer['boxes'])
        answer['class_idx'] = np.array(answer['class_idx'])
        answer['scores'] = np.array(answer['scores'])
        answer['labels'] = np.array(answer['labels'])
        if need_draw:
            image_out = decode_string_to_image(image_string)
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
        'SCORE_THRESH': 0.7,
    }
    client = GLIPClient(EasyDict(cfg))
    img = Image.open('/xxx/GLIP/mcdonald.png')
    prompt = 'a restaurant'
    answer, grouned_img = client.inference(img, prompt, need_draw=True)

    print('answer: ', answer)


if __name__ == '__main__':
    main()
