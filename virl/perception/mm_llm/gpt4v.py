import os
import requests
from openai import OpenAI

from virl.perception.mm_llm.mm_llm_template import MultiModalLLMTemplate
from virl.utils import common_utils


class GPT4V(MultiModalLLMTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.client = OpenAI()
        self.api_key = os.getenv("OPENAI_API_KEY")

    def predict(self, img, prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        content = [{"type": "text", "text": f"{prompt}"}]

        content = self.add_image_to_content(img, content)

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        answer = response.json()['choices'][0]['message']['content']
        return answer

    def add_image_to_content(self, image, content):
        if isinstance(image, list):
            for img in image:
                content = self.add_single_image_to_content(img, content)
        else:
            content = self.add_single_image_to_content(image, content)

        return content

    @staticmethod
    def add_single_image_to_content(image, content):
        base64_image = common_utils.encode_image_to_string(image, show=True)
        content.append({
            "type": "image_url",
            "image_url": {
                "url": base64_image
            }
        })
        return content
