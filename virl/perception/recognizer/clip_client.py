import os
import ast
import requests

from gradio_client import Client

from virl.utils import common_utils
from virl.config import cfg


class CLIPClient(object):
    def __init__(self, cfg):
        self.server_url = cfg.SERVER
        self.client = Client(self.server_url)

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

        output_dir = os.path.join(cfg.get('OUTPUT_DIR', 'output'), 'tmp')
        img_path = common_utils.save_tmp_image_to_file(img, output_dir, img_format)

        while True:
            flag = True
            try:
                answer = self.client.predict(
                    img_path,
                    text,  # str in 'Text' Textbox component
                    temperature,  # float (numeric value between 1 and 100) in 'Temperature' Slider component
                    api_name="/predict"
                )
            except requests.Timeout:
                print('Timeout! Resend the message.')
                flag = False
            if flag:
                break
        
        logit_list, score_list = ast.literal_eval(answer)
        logit_list = ast.literal_eval(logit_list)
        score_list = ast.literal_eval(score_list)
        
        results = {
            'logits': logit_list,
            'scores': score_list,
        }
        os.remove(img_path)

        return results


if __name__ == '__main__':
    from easydict import EasyDict
    from PIL import Image

    cur_cfg = {
        'SERVER': 'http://147.8.181.77:22411/',
    }

    image = Image.open('/home/jihan/Desktop/glip_img/jhotel.png')
    text = 'a beautiful street,,a beautiful building'

    clip_client = CLIPClient(EasyDict(cur_cfg))
    print(clip_client.inference(image, text, 100.0))
    print(clip_client.inference(image, text, 10.0))
