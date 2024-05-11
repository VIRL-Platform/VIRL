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
        self.model_name = cfg.MODEL_NAME

    def predict(self, img, prompt):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        content = [{"type": "text", "text": f"{prompt}"}]

        content = self.add_image_to_content(img, content)

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ],
            "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        if response.status_code != 200:
            import ipdb; ipdb.set_trace()
            type = response.json()['type']
            if type in ['server_error']:
                print(f"Retry due to {type}")
                return self.predict(img, prompt)
        
        try:
            answer = response.json()['choices'][0]['message']['content']
        except:
            print(response.json())
            raise ValueError("Failed to get answer from GPT-4 Vision API")
            
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
