import os
import requests

from gradio_client import Client
from virl.utils import common_utils


class LightGlueClient(object):
    def __init__(self, cfg):
        self.server_url = cfg.SERVER
        self.client = Client(self.server_url)

    def inference(self, img0, img1):
        img0_format = img0.format
        img1_format = img1.format
        
        output_dir = os.path.join(cfg.get('OUTPUT_DIR', 'output'), 'tmp')
        img0_path = common_utils.save_tmp_image_to_file(img0, output_dir, img0_format)
        img1_path = common_utils.save_tmp_image_to_file(img1, output_dir, img1_format)

        while True:
            flag = True
            try:
                response = self.client.predict(
				    img0_path,	 # str (filepath or URL to image) in 'Image1' Image component
				    img1_path,	 # str (filepath or URL to image) in 'Image2' Image component
				    api_name="/predict"
                )
            except requests.Timeout:
                print('Timeout! Resend the message.')
                flag = False
            if flag:
                break
        
        answer = int(response)
        
        os.remove(img0_path)
        os.remove(img1_path)
        return answer


if __name__ == '__main__':
    from easydict import EasyDict
    from PIL import Image

    cfg = {
        'SERVER': 'http://147.8.181.77:22411/',
    }

    image0 = Image.open("/home/jihan/Desktop/glip_img/jhotel.png")
    image1 = Image.open("/home/jihan/Desktop/glip_img/jhotel.png")

    lightglue_client = LightGlueClient(EasyDict(cfg))
    print(lightglue_client.inference(image0, image1))
