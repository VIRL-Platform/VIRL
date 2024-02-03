import torch
import cv2
import numpy as np
from PIL import Image


class Recognizer(object):
    def __init__(self, vision_model_cfg, recognize_cfg, messager=None, platform=None) -> None:
        self.recognize_cfg = recognize_cfg
        self.messager = messager
        self.platform = platform

        self.model = self.build_model(vision_model_cfg)

    def build_model(self, vision_model_cfg):
        model_cfg = getattr(vision_model_cfg, self.recognize_cfg.NAME)
        if self.recognize_cfg.NAME == 'CLIP':
            from virl.perception.recognizer.clip_client import CLIPClient
            model = CLIPClient(model_cfg)
        elif self.recognize_cfg.NAME == 'EvaCLIP':
            from virl.perception.recognizer.eva_clip_client import EvaCLIPClient
            model = EvaCLIPClient(model_cfg)
        elif self.recognize_cfg.NAME == 'LLaVA':
            from virl.perception.recognizer.llava_client import LLaVAClient
            model = LLaVAClient(model_cfg)
        elif self.recognize_cfg.NAME == 'PaddleOCR':
            from virl.perception.recognizer.paddle_ocr import PaddleOCR
            model = PaddleOCR(model_cfg)
        else:
            raise NotImplementedError

        return model

    def check(self, img, candidates, cared_labels):
        """


        Args:
            img: image in PIL.Image format
            candidates: candidates in string format, separated by ',,',
                        for example: 'restaurant,,bar,,cafe,,hotel'.
                        Candidates can include potential categories that are not cared,
                        for example: others, background.
            cared_labels: list of cared labels, for example: ['restaurant', 'bar', 'cafe', 'hotel'].
                          The final classification result will be the highest score among all cared labels.


        Returns:

        """
        answer = self.model.inference(img, candidates)
        score_list = answer['scores']

        results = {
            'labels': [],
            'scores': [],
        }
        for i, category in enumerate(candidates.split(',,')):
            if category in cared_labels:
                results['labels'].append(category)
                results['scores'].append(float(score_list[i]))

        results['labels'] = np.array(results['labels'])
        results['scores'] = np.array(results['scores'])
        return results
