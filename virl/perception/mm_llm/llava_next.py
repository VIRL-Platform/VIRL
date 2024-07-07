import torch
from PIL import Image
import torch
from PIL import Image
import requests

from virl.perception.mm_llm.mm_llm_template import MultiModalLLMTemplate


class LLaVANext(MultiModalLLMTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
        
        model_name = cfg.MODEL_NAME
        
        print('>>> Initialize LLaVA Next model....')
        self.processor = LlavaNextProcessor.from_pretrained(model_name)

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ) 

        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    @torch.no_grad()
    def predict(self, image, question):
        raw_image = image
        
        inputs = self.processor(question, raw_image, return_tensors="pt").to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=100)

        answer = self.processor.decode(output[0], skip_special_tokens=True)

        answer = answer.split('ASSISTANT:')[-1].strip()
        return answer
