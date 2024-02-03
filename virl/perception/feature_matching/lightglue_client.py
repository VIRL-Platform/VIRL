import requests

from virl.utils.common_utils import encode_image_to_string


class LightGlueClient(object):
    def __init__(self, cfg):
        self.server_url = cfg.SERVER

    def inference(self, img0, img1):
        img0_format = img0.format
        img0_base64_string = encode_image_to_string(img0)
        img1_format = img1.format
        img1_base64_string = encode_image_to_string(img1)

        while True:
            flag = True
            try:
                response = requests.post(self.server_url + "/run/predict", json={
                    "data": [
                        f'data:image/{img0_format.lower()};base64,{img0_base64_string}',
                        f'data:image/{img1_format.lower()};base64,{img1_base64_string}',
                    ]
                    },
                    timeout=10
                ).json()
            except requests.Timeout:
                print('Timeout! Resend the message.')
                flag = False
            if flag:
                break
        assert 'data' in response, f'predict failed: {response}'
        answer = response['data'][0]
        return answer


if __name__ == '__main__':
    from easydict import EasyDict
    from PIL import Image

    cfg = {
        'SERVER': 'http://xxx',
    }

    image0 = Image.open('xxx.png')
    image1 = Image.open('xxx.png')

    lightglue_client = LightGlueClient(EasyDict(cfg))
    print(lightglue_client.inference(image0, image1))
