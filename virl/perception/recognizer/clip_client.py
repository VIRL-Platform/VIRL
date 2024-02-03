import ast

import cv2
import base64
import requests
import numpy as np

from virl.utils.common_utils import encode_image_to_string, decode_string_to_image


class CLIPClient(object):
    def __init__(self, cfg):
        self.server_url = cfg.SERVER

    def inference(self, img, text, temperature=100.0, img_format=None):
        """

        Args:
            img: PIL.Image format
            text: classification candidates in string format, separated by ',,',
                  for example: 'restaurant,,bar,,cafe,,hotel'.
            temperature: only works for CLIP model, default: 100.0

        Returns:
            results: dict, {'scores': list of scores for each candidate in the text in the same order}
        """
        if img_format is None:
            img_format = img.format
        img_base64_string = encode_image_to_string(img)

        while True:
            flag = True
            try:
                response = requests.post(self.server_url + "/run/predict", json={
                    "data": [
                        f'data:image/{img_format.lower()};base64,{img_base64_string}',
                        text,
                        temperature
                    ]
                    },
                    timeout=100
                ).json()
            except requests.Timeout:
                print('Timeout! Resend the message.')
                flag = False
            if flag:
                break
        assert 'data' in response, f'predict failed: {response}'
        answer = response['data'][0]
        
        logit_list, score_list = ast.literal_eval(answer)
        logit_list = ast.literal_eval(logit_list)
        score_list = ast.literal_eval(score_list)
        
        results = {
            'logits': logit_list,
            'scores': score_list,
        }
        return results


if __name__ == '__main__':
    from easydict import EasyDict
    from PIL import Image

    cfg = {
        'SERVER': 'http://147.8.183.198:22380',
    }

    image = Image.open('/Users/dingry/Downloads/000200.jpeg')
    text = 'a beautiful street,,a beautiful building'

    clip_client = CLIPClient(EasyDict(cfg))
    print(clip_client.check(image, text, text, 100.0))
    print(clip_client.check(image, text, text, 10.0))
