import io
import base64
import requests
from PIL import Image

from virl.perception.mm_llm.mm_llm_template import MultiModalLLMTemplate


class InstructBLIPClient(MultiModalLLMTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.min_length = cfg.MIN_LENGTH
        self.max_length = cfg.MAX_LENGTH
        self.beam_size = cfg.BEAM_SIZE
        self.length_penalty = cfg.LENGTH_PENALTY
        self.repetition_penalty = cfg.REPETITION_PENALTY
        self.top_p = cfg.TOP_P
        self.server = cfg.SERVER
        print('>>> Initialize InstructBLIP Client....')

    def set_min_length(self, min_length):
        self.min_length = min_length

    def set_max_length(self, max_length):
        self.max_length = max_length

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def set_length_penalty(self, length_penalty):
        self.length_penalty = length_penalty

    def set_repetition_penalty(self, repetition_penalty):
        self.repetition_penalty = repetition_penalty

    def set_top_p(self, top_p):
        self.top_p = top_p

    def predict(self, img, prompt):
        img_format = img.format
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img_format)
        base64_string = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
        while True:
            flag = True
            try:
                response = requests.post(f"{self.server}/run/predict", json={
                    "data": [
                        f'data:image/{img_format.lower()};base64,{base64_string}',
                        prompt,
                        self.min_length,
                        self.max_length,
                        self.beam_size,
                        self.length_penalty,
                        self.repetition_penalty,
                        self.top_p,
                        "Beam Search"
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
        return response['data'][0]


def main():
    server = 'xxx'
    client = InstructBLIPClient(server)

    img = Image.open('/xxx.png')
    prompt = 'Can I go to the bathroom in any places of the image? Why?'
    answer = client.predict(img, prompt)

    print('question:', prompt)
    print('answer: ', answer)


if __name__ == '__main__':
    main()
