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
        recognizer_mapping = {
            'CLIP': ('virl.perception.recognizer.clip_client', 'CLIPClient'),
            'EvaCLIP': ('virl.perception.recognizer.eva_clip_client', 'EvaCLIPClient'),
            'LLaVA': ('virl.perception.recognizer.llava_client', 'LLaVAClient'),
            'PaddleOCR': ('virl.perception.recognizer.paddle_ocr', 'PaddleOCR'),
            'CLIPLocal': ('virl.perception.recognizer.clip_local', 'CLIPLocal'),
            'OpenCLIP': ('virl.perception.recognizer.open_clip_local', 'OpenCLIPLocal'),
        }

        if self.recognize_cfg.NAME in recognizer_mapping:
            module_name, class_name = recognizer_mapping[self.recognize_cfg.NAME]
            module = __import__(module_name, fromlist=[class_name])
            recognizer_class = getattr(module, class_name)
            model = recognizer_class(model_cfg)
        else:
            raise NotImplementedError(f"Recognizer {self.recognize_cfg.NAME} is not implemented.")

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
