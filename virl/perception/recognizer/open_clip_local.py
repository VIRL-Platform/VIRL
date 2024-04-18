import torch


class OpenCLIPLocal(object):
    def __init__(self, cfg):
        import open_clip  # importing here so only tries to import if used

        self.model_name = cfg.NAME
        self.pretrained = cfg.PRETRAINED
        self.temperature = cfg.get('TEMPERATURE', 100.0)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(self.model_name, pretrained=self.pretrained, device=self.device)
        print(f"Loaded OpenCLIP model: {self.model_name}")
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

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
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(tok)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_logits = temperature * image_features @ text_features.T
            text_probs = torch.nn.functional.softmax(text_logits, dim=-1)

        return {
            'logits': text_logits.cpu().numpy().tolist()[0],
            'scores': text_probs.cpu().numpy().tolist()[0],
        }


if __name__ == '__main__':
    from easydict import EasyDict
    from PIL import Image

    cfg = {
        "NAME": "ViT-B-32",
        "PRETRAINED": "laion2b_s34b_b79k",
    }

    image = Image.open('../data/place_centric_data/place_centric_images/ChIJ_____-Bn1moRWGI1BpiLeN8.jpg')
    text = 'a beautiful street,,a beautiful building,,a shop,,watches'

    clip_client = OpenCLIPLocal(EasyDict(cfg))
    print(text.split(',,'))
    print(clip_client.inference(image, text, 100.0))
    print(clip_client.inference(image, text, 50.0))
    print(clip_client.inference(image, text, 25.0))
    print(clip_client.inference(image, text, 10.0))
    print(clip_client.inference(image, text, 1.0))

"""
acc: 0.2708116002503651
"""
