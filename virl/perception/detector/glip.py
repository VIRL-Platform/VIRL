"""
Modified from ViperGPT and GLIP.
https://github.com/cvlab-columbia/viper/blob/main/vision_models.py
Modified by Jihan YANG
Copyright reserved from 2023 - present
"""
import copy
import os
import cv2
import contextlib
import torch
import tqdm

import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np

from virl.utils import common_utils

try:
    with contextlib.redirect_stderr(open(os.devnull, "w")):  # Do not print nltk_data messages when importing
        from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo, to_image_list, create_positive_map, \
            create_positive_map_label_to_token_from_positive_map
except:
    pass

from virl.utils import common_utils, vis_utils


class CustomGLIPDemo(GLIPDemo):
    def get_results_from_predictions(self, predictions):
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels")
        boxes = predictions.bbox
        
        new_labels = []
        if self.entities and self.plus:
            for i in labels:
                if i <= len(self.entities):
                    new_labels.append(self.entities[i - self.plus])
                else:
                    new_labels.append('object')
        else:
            new_labels = ['object' for i in labels]
        
        results = {
            'boxes': boxes,
            'labels': new_labels,
            'scores': scores,
            'class_idx': labels,
        }
        return results
            
    @torch.no_grad()
    def run(self, image, caption, thresh=0.5):
        predictions = self.compute_prediction(image, caption)
        top_predictions = self._post_process(predictions, thresh)
        
        results = self.get_results_from_predictions(top_predictions)
        return results


class GLIPModel(object):
    def __init__(self, glip_cfg, devices='cuda'):
        print('>>> Initialize GLIP....')
        self.model_size = glip_cfg.MODEL_SIZE
        self.root_path = glip_cfg.ROOT_PATH
        self.score_thresh = glip_cfg.SCORE_THRESH

        if self.model_size == 'tiny':
            config_file = os.path.join(self.root_path, 'configs/pretrain/glip_Swin_T_O365_GoldG.yaml')
            ckpt_file = os.path.join(self.root_path, "MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth")
        elif self.model_size == 'large':
            config_file = os.path.join(self.root_path, 'configs/pretrain/glip_Swin_L.yaml')
            ckpt_file = os.path.join(self.root_path, 'MODEL/glip_large_model.pth')
        else:
            raise NotImplementedError

        from maskrcnn_benchmark.config import cfg
        cfg.local_rank = 0
        cfg.num_gpus = 1
        cfg.merge_from_file(config_file)
        cfg.merge_from_list(["MODEL.WEIGHT", ckpt_file])
        cfg.merge_from_list(["MODEL.DEVICE", devices])

        self.glip = CustomGLIPDemo(
            cfg,
            min_image_size=800,
            confidence_threshold=0.7,
            show_mask_heatmaps=False
        )

    @staticmethod
    def load(pil_image):
        pil_image = pil_image.convert('RGB')
        # convert to BGR format
        image = np.array(pil_image)[:, :, [2, 1, 0]]
        return image

    def inference(self, image, caption, score_thresh, need_draw=False):
        image = self.load(image)
        results = self.glip.run(image, caption, thresh=score_thresh)

        if need_draw:
            result_image = vis_utils.draw_with_results(image, results)
            results['result_image'] = result_image
            return results, result_image[..., ::-1]
        else:
            return results, None

    @staticmethod
    def imshow(img, caption, image_name='debug_output.png'):
        plt.figure()
        plt.imshow(img)
        plt.axis("off")
        plt.figtext(0.5, 0.9, caption, wrap=True, horizontalalignment='center', fontsize=20)
        plt.savefig(image_name)

    def detect_surroundings(self, image_list, place, debug=False):
        detected_mask = np.zeros(len(image_list), dtype=np.bool_)
        for i, image in tqdm.tqdm(enumerate(image_list)):
            results = self.inference(image, place, need_draw=debug)
            labels = results['labels']
            if place in labels:
                detected_mask[i] = True
                print(f'>>> Find the place in view {i}.')
                if 'result_image' in results:
                    result_image = results['result_image']
                    self.imshow(
                        result_image, place, image_name=f'../visual_output/glip_results_view_{i}.png'
                    )

        if detected_mask.sum() == 0:
            print('>>> No suitable candidate found by GLIP in current location.')
        else:
            print('>>> GLIP find suitable candidates in current location.')

