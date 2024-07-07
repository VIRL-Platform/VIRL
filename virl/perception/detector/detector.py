import torch
import copy
import numpy as np

from virl.utils import common_utils, geocode_utils, vis_utils


class Detector(object):
    def __init__(self, vision_model_cfg, detect_cfg, messager=None, platform=None):
        self.detect_cfg = detect_cfg
        self.messager = messager
        self.platform = platform

        self.proposal_thresh = self.detect_cfg.PROPOSAL_SCORES
        if self.detect_cfg.get('DOUBLE_CHECK', None):
            self.check_thresh = self.detect_cfg.DOUBLE_CHECK.CHECK_SCORES

        self.need_double_check = self.detect_cfg.get('DOUBLE_CHECK', None) and self.detect_cfg.DOUBLE_CHECK.ENABLED

        self.model = None
        self.build_model(vision_model_cfg)

    def build_model(self, vision_model_cfg):
        """
        Build the detector for detection.
        
        For each detector, it should at least have the following functions:
        - detect: detect the objects in the image.

        Args:
            vision_model_cfg (_type_): _description_
        """
        detector_mapping = {
            'GLIP': ('virl.perception.detector.glip_client', 'GLIPClient'),
            'GLIP_CLIP': ('virl.perception.detector.glip_clip_client', 'GLIPCLIPClient'),
            'GroundingDINO': ('virl.perception.detector.grounding_dino', 'GroundingDINO'),
            'OWL_VIT': ('virl.perception.detector.owl_vit', 'OWLVIT'),
            'OWL_VITv2': ('virl.perception.detector.owl_vitv2', 'OWLVITv2'),
            'OpenSeeD': ('virl.perception.detector.openseed', 'OpenSeeD'),
        }

        if self.detect_cfg.NAME in detector_mapping:
            module_name, class_name = detector_mapping[self.detect_cfg.NAME]
            module = __import__(module_name, fromlist=[class_name])
            detector_class = getattr(module, class_name)

            devices = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = detector_class(getattr(vision_model_cfg, self.detect_cfg.NAME), devices=devices)
        else:
            raise NotImplementedError(f"Detector {self.detect_cfg.NAME} is not implemented.")

    def detect(self, image, candidates, cared_labels, score_thresh, need_draw):
        results, _ = self.model.inference(image, candidates, score_thresh, need_draw)
        filtered_results = self.filter_unrelated_labels(results, cared_labels)
        # this is optional
        result_image = vis_utils.draw_with_results(image, filtered_results)

        return filtered_results, result_image

    def filter_unrelated_labels(self, results, cared_labels):
        filtered_results = {}
        mask = self.filter_by_cared_labels(results, cared_labels)

        for key, value in results.items():
            filtered_results[key] = value[mask]
        return filtered_results

    @staticmethod
    def filter_by_cared_labels(results, cared_labels):
        mask = np.zeros(len(results['labels']), dtype=bool)
        for i, label in enumerate(results['labels']):
            if label in cared_labels:
                mask[i] = True

        return mask

    def check(self, street_image, candidates, cared_labels):
        print(f'>>> Check view {street_image.i} with detector.')

        results = {}
        detect_results, result_image = self.detect(
            street_image.image, candidates, cared_labels, self.proposal_thresh, need_draw=False
        )

        result_image_str = common_utils.encode_image_to_string(result_image, show=True)
        # send the detected images to HTML
        self.messager.send_image(
            'send_image', f'detectImages{street_image.i + 1}', result_image_str
        )
        results['first_detect'] = {
            'is_detected': detect_results['labels'].shape[0] > 0,
            'view': street_image,
            'result': detect_results
        }

        # adjust camera to capture object-centered view
        is_detected = detect_results['labels'].shape[0] > 0
        if is_detected:
            object_views, is_detected_final, detect_results = self.adjust_camera_to_recheck_detection_result(
                street_image, detect_results, candidates,
                adjust_camera=self.detect_cfg.get('ADJUST_CAMERA', None) and self.detect_cfg.ADJUST_CAMERA.ENABLED
            )
            is_detected = is_detected_final and is_detected
        else:
            object_views = is_detected_final = None

        results['final_detect'] = {
            'is_detected': is_detected_final,
            'view': object_views,
            'result': detect_results
        }

        return results, is_detected, object_views

    def double_check(self, new_image, candidates, refer_label):
        new_results, result_image = self.detect(
            new_image.image, candidates, [refer_label], self.check_thresh, need_draw=False
        )
        result_image_str = common_utils.encode_image_to_string(result_image, show=True)

        if new_results['labels'].shape[0] > 0:
            idx = (new_results['labels'] == refer_label).nonzero()[0][0]
            label = refer_label
            detect_result = {
                'boxes': new_results['boxes'][idx],
                'labels': label,
                'scores': new_results['scores'][idx],
                'class_idx': new_results['class_idx'][idx]
            }
        else:
            detect_result = None

        return detect_result, result_image_str

    def adjust_camera_to_recheck_detection_result(self, street_image, previous_results,
                                                  candidates, adjust_camera=False):
        boxes = previous_results['boxes']
        labels = previous_results['labels']
        adjusted_views = []
        detection_results = {
            'boxes': [],
            'labels': [],
            'scores': [],
            'class_idx': []
        }

        for i, (box, label) in enumerate(zip(boxes, labels)):
            if adjust_camera:
                adjusted_views, detection_results = self.adjust_camera_and_recheck(
                    box, label, street_image, candidates, adjusted_views, detection_results, previous_results, i
                )
            else:
                detect_result = {
                    'boxes': box,
                    'labels': label,
                    'scores': previous_results['scores'][i],
                    'class_idx': previous_results['class_idx'][i]
                }
                for key in detection_results.keys():
                    detection_results[key].append(detect_result[key])

                object_view = copy.deepcopy(street_image)
                object_view.set_detect_result(detect_result)
                adjusted_views.append(object_view)

        return adjusted_views, len(adjusted_views) > 0, detection_results

    def adjust_camera_and_recheck(self, box, label, street_image, candidates, adjusted_views,
                                  detection_results, previous_results, i):
        new_heading, new_pitch, new_fov, new_box = geocode_utils.get_heading_pitch_fov_to_box(
            box, street_image.shape, street_image.heading, street_image.pitch, street_image.fov,
            enlarge_factor=self.detect_cfg.ADJUST_CAMERA.ENLARGE_RATIO,
            min_fov=self.detect_cfg.ADJUST_CAMERA.MIN_FOV
        )

        new_image = self.platform.get_streetview_from_geocode(
            street_image.geocode, street_image.shape, new_heading, new_pitch, new_fov, source='outdoor',
            idx=street_image.i
        )

        # double check
        # import ipdb; ipdb.set_trace(context=20)
        if self.need_double_check:
            detect_result, result_image_str = self.double_check(new_image, candidates, refer_label=label)
            # send the detected images to HTML
            self.messager.send_image(
                'send_image', f'detectImages{new_image.i + 1}', result_image_str
            )
            # import ipdb; ipdb.set_trace(context=20)
        else:
            detect_result = {
                'boxes': box,
                'labels': label,
                'scores': previous_results['scores'][i],
                'class_idx': previous_results['class_idx'][i]
            }

        if detect_result is not None:
            new_image.set_detect_result(detect_result)
            adjusted_views.append(new_image)
            for key in detection_results.keys():
                detection_results[key].append(detect_result[key])

        return adjusted_views, detection_results
