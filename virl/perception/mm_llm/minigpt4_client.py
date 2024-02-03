import os
import io
import base64
import requests

from PIL import Image

from virl.perception.mm_llm.mm_llm_template import MultiModalLLMTemplate


class MiniGPT4Client(MultiModalLLMTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cfg = cfg
        self.previous_chat = []
        self.server = cfg.SERVER
        self.beam_search = cfg.BEAM_SEARCH
        self.temperature = cfg.TEMPERATURE
    
    def set_beam_search(self, beam):
        self.beam_search = beam

    def set_temperature(self, temp):
        self.temperature = temp

    def clear(self):
        response = requests.post(f"{self.server}/run/clear", json={
            "data": [
                None,
                None,
            ]
        }).json()
        assert 'data' in response, f'clear failed: {response}'
        self.previous_chat = []

    def upload_image_from_path(self, path):
        assert os.path.exists(path)
        with open(path, "rb") as img_file:
            base64_string = base64.b64encode(img_file.read()).decode('utf-8')
        img_type = path.split('.')[-1]
        response = requests.post(f"{self.server}/run/upload_img", json={
            "data": [
                f'data:image/{img_type};base64,{base64_string}',
                None,
                None,
            ]
        }).json()
        assert 'data' in response, f'upload failed: {response}'

    def upload_image(self, img):
        img = img.convert('RGB')
        img_format = img.format if img.format is not None else 'PNG'
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img_format.upper())
        base64_string = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        response = requests.post(f"{self.server}/run/upload_img", json={
            "data": [
                f'data:image/{img_format.lower()};base64,{base64_string}',
                None,
                None,
            ]
        }).json()
        assert 'data' in response, f'upload failed: {response}'

    def input_text(self, text):
        response = requests.post(f"{self.server}/run/return_text", json={
            "data": [
                self.previous_chat + [[text, None]],
                None,
                None,
                self.beam_search,
                self.temperature,
            ]
        }).json()
        assert 'data' in response
        self.previous_chat = response['data'][0]

    def print_all_chat_history(self):
        print(self.previous_chat)

    def print_latest_chat(self):
        print(self.previous_chat[-1])

    def predict(self, img, prompt):
        self.clear()
        self.upload_image(img)
        self.input_text(prompt)
        return self.previous_chat[-1][1]


def main():
    from easydict import EasyDict
    cfg = {
        'SERVER': "xxx",
        'BEAM_SEARCH': 1,
        'TEMPERATURE': 1.0,
    }
    client = MiniGPT4Client(EasyDict(cfg))
    client.clear()
    path = 'xxx.jpg'
    # client.upload_image_from_path(path)
    image = Image.open(path).convert('RGB')
    client.upload_image(image)
    client.input_text('Can I go to the bathroom in any places of the image?')
    client.print_latest_chat()

    client.clear()


if __name__ == '__main__':
    main()
