import torch


class CLIPLocal(object):
    def __init__(self, cfg):
        import clip  # importing here so only tries to import if used

        self.model_name = cfg.NAME
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load(self.model_name, device=self.device)
        print(f"Loaded CLIP model: {self.model_name}")
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = clip.tokenize
        self.temperature = cfg.get('TEMPERATURE', 100.0)

    def inference(self, img, text, temperature=None):
        """
        Args:
            img: PIL.Image format
            text: classification candidates in string format, separated by ',,',
                  for example: 'restaurant,,bar,,cafe,,hotel'.
            temperature: only works for CLIP model, default: 100.0

        Returns:
            results: dict, {'scores': list of scores for each candidate in the text in the same order}
        """

        if temperature is None:
            temperature = self.temperature

        temperature /= 100.0  # the default temperature is 100.0

        image = self.preprocess(img).unsqueeze(0).to(self.device)

        candidates = text.split(',,')
        tok = self.tokenizer(candidates).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            text_logits, _ = self.model(image, tok)

            # scale by temperature
            text_logits *= temperature

            text_probs = torch.nn.functional.softmax(text_logits, dim=-1)

        return {
            'logits': text_logits.cpu().numpy().tolist()[0],
            'scores': text_probs.cpu().numpy().tolist()[0],
        }
