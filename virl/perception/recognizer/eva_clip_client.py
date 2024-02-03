import torch
import sys


class EvaCLIPClient(object):
    def __init__(self, cfg):
        model_name = cfg.MODEL_NAME
        pretrained = "eva_clip"  # or "/path/to/EVA02_CLIP_B_psz16_s8B.pt"

        sys.path.append(cfg.MODEL_PATH)
        from eva_clip import create_model_and_transforms, get_tokenizer

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = create_model_and_transforms(model_name, pretrained, force_custom_clip=True)
        self.tokenizer = get_tokenizer(model_name)
        self.model = self.model.to(self.device)

    def inference(self, img, text, temperature=100.0):
        """
        Args:
            img: PIL.Image format
            text: classification candidates in string format, separated by ',,',
                  for example: 'restaurant,,bar,,cafe,,hotel'.
            temperature: default: 100.0

        Returns:
            results: dict, {'scores': list of scores for each candidate in the text in the same order}
        """
        text_list = text.split(",,")
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        text = self.tokenizer(text_list).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            logits = (temperature * image_features @ text_features.T)
            scores = logits.softmax(dim=-1)
            
        results = {
            'logits': logits[0].cpu().numpy().tolist(),
            'scores': scores[0].cpu().numpy().tolist()
        }
        return results


if __name__ == '__main__':
    from easydict import EasyDict
    from PIL import Image

    cfg = {
        'MODEL_NAME': 'EVA02-CLIP-bigE-14-plus',
        'MODEL_PATH': 'xxx/rei'
    }

    image = Image.open('xxx.jpg')
    text = 'a beautiful street,,a beautiful building'

    clip_client = EvaCLIPClient(EasyDict(cfg))
    print(clip_client.inference(image, text, 100.0))
    print(clip_client.inference(image, text, 10.0))
