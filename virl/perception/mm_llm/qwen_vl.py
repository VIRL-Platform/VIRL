import os
import time

from virl.perception.mm_llm.mm_llm_template import MultiModalLLMTemplate
from virl.utils import common_utils
from virl.config import cfg

try:
    from dashscope import MultiModalConversation
except:
    pass


class QwenVL(MultiModalLLMTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.model_name = cfg.MODEL_NAME

    def predict(self, img, prompt, img_path=None):
        img_format = img.format if img.format is not None else 'PNG'
        
        if img_path is None:
            output_dir = os.path.join(cfg.get('OUTPUT_DIR', 'output'), 'tmp')
            img_path = common_utils.save_tmp_image_to_file(img, output_dir, img_format)

        messages = [{
            'role': 'user',
            'content': [
                {
                    'image': img_path
                },
                {
                    'text': prompt
                },
            ]
        }]
        response = MultiModalConversation.call(model=self.model_name, messages=messages)
        
        if response.status_code == 200:
            answer = response.output.choices[0].message.content[0]['text']
        elif response.status_code == 429:
            print(response)
            time.sleep(2)
            return self.predict(img, prompt, img_path=img_path)
        elif response.status_code == 400:
            answer = response.message
            import ipdb; ipdb.set_trace(context=20)
        else:
            print(response)
            raise ValueError("Failed to get answer from Qwen Vision API")
        
        if img_path is not None:
            os.remove(img_path)
        return answer
