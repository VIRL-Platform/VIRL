from PIL import Image
import numpy as np

import torch
from torchvision.ops import box_convert
from groundingdino.util.inference import load_model, predict, annotate
import groundingdino.datasets.transforms as T

from virl.utils import common_utils, vis_utils


class GroundingDINO(object):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.model = load_model(cfg.CFG_FILE, cfg.CKPT_FILE, device='cuda')
        self.BOX_THRESHOLD = cfg.BOX_THRESHOLD
        self.TEXT_THRESHOLD = cfg.TEXT_THRESHOLD
    
    def load_image(self, image_pil):
        # load image
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def inference(self, image, caption, _, need_draw=False):
        classes = caption.split(',')
        text_prompt = (' . ').join(classes)  # "chair . person . dog ."
        # print(classes)
        image_transformed = self.load_image(image)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=self.BOX_THRESHOLD,
            text_threshold=self.TEXT_THRESHOLD,
            remove_combined=True
        )
        # print(phrases)
        h, w, _ = np.asarray(image).shape
        boxes = boxes * torch.Tensor([w, h, w, h])

        valid_idx = []
        for i in range(len(boxes)):
            if phrases[i] in classes:
                valid_idx.append(i)

        answers = {
            'boxes': box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()[valid_idx],
            'scores': logits.cpu().numpy()[valid_idx],
            'labels': np.array(phrases)[valid_idx],
            'class_idx': np.array([classes.index(phrase) for phrase in phrases])[valid_idx]
        }

        annotated_frame = vis_utils.draw_with_results(np.asarray(image), answers)
        if need_draw:
            annotated_frame.save('draw.png', 'PNG')
        
        return answers, annotated_frame


if __name__ == '__main__':
    from easydict import EasyDict as edict
    cfg = edict({'CFG_FILE': 'xxx.py',
           'CKPT_FILE': 'xxx.pth',
           'BOX_THRESHOLD': 0.35,
           'TEXT_THRESHOLD': 0.25})
    gd = GroundingDINODemo(cfg)
    img = Image.open('xx.jpg')
    prompt = 'stone'
    answers, _ = gd.inference(
        img, prompt, None, need_draw=True)
    print(answers)
