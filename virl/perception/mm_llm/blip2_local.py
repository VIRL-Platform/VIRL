import torch
from PIL import Image

from virl.perception.mm_llm.mm_llm_template import MultiModalLLMTemplate

from lavis.models import load_model_and_preprocess


class BLIP2(MultiModalLLMTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        model_name = cfg.MODEL_NAME
        model_type = cfg.MODEL_TYPE
        
        print('>>> Initialize BLIP2 model....')
        self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name=model_name, model_type=model_type, is_eval=True, device=self.device
        )

    @torch.no_grad()
    def predict(self, image: Image, prompt: str) -> str:
        """
        image: PIL.Image
        question: str
        """
        raw_image = image.convert('RGB')
        image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.device)

        prompt = prompt.replace('Answer:', '')
        answer = self.model.generate({"image": image, "prompt": f"{prompt} Answer:"})[0]
        return answer
