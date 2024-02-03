import torch
from PIL import Image

from virl.perception.mm_llm.mm_llm_template import MultiModalLLMTemplate

from lavis.models import load_model_and_preprocess


class InstructBLIPLocal(MultiModalLLMTemplate):
    def __init__(self, cfg):
        super().__init__(cfg)
        
        self.min_length = cfg.MIN_LENGTH
        self.max_length = cfg.MAX_LENGTH
        self.beam_size = cfg.BEAM_SIZE
        self.length_penalty = cfg.LENGTH_PENALTY
        self.repetition_penalty = cfg.REPETITION_PENALTY
        self.top_p = cfg.TOP_P
        self.sampling = cfg.SAMPLING
        
        model_name = cfg.MODEL_NAME
        model_type = cfg.MODEL_TYPE
        
        print('>>> Initialize InstructBLIP model....')
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
        image = image.convert('RGB')
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)

        use_nucleus_sampling = self.sampling == "Nucleus sampling"
        samples = {
            "image": image,
            "prompt": prompt,
        }

        output = self.model.generate(
            samples,
            length_penalty=float(self.length_penalty),
            repetition_penalty=float(self.repetition_penalty),
            num_beams=self.beam_size,
            max_length=self.max_length,
            min_length=self.min_length,
            top_p=self.top_p,
            use_nucleus_sampling=use_nucleus_sampling,
        )

        return output[0]
