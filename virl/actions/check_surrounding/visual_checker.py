import numpy as np

from virl.config import cfg
from virl.utils import geocode_utils
from virl.perception.feature_matching.lightglue_client import LightGlueClient
from virl.perception.detector import Detector
from virl.perception.mm_llm import MultiModalLLM
from virl.lm import prompt as prompt_templates


class VisualChecker(object):
    def __init__(self, checker_cfg, platform, messager, memory=None):
        self.checker_cfg = checker_cfg
        self.memory = memory

        if checker_cfg.get('CARED_LABELS', None):
            self.cared_labels = checker_cfg.CARED_LABELS
        elif checker_cfg.get('CARED_LABELS_PATH', None):
            self.cared_labels = [line.strip().replace('_', ' ') for line in open(checker_cfg.CARED_LABELS_PATH, 'r').readlines()]
        else:
            raise NotImplementedError

        candidates = checker_cfg.CANDIDATES
        self.candidates = candidates + ',' + ','.join(self.cared_labels)

        self.need_check_duplicate = self.checker_cfg.get('CHECK_DUPLICATE', None)
        self.need_check_duplicate_with_fm = self.checker_cfg.get('CHECK_DUPLICATE', None) and \
                                            self.checker_cfg.CHECK_DUPLICATE.CHECK_WITH_FM

        self.platform = platform
        self.messager = messager
        self.models = {}

        self.create_visual_models(checker_cfg.USED_MODELS)

    def create_visual_models(self, used_models):
        if 'DETECT' in used_models:
            self.models['detector'] = Detector(
                cfg.VISION_MODELS, self.checker_cfg.DETECT, self.messager, self.platform
            )

        if 'MM_LLM' in used_models:
            self.models['mm_llm'] = MultiModalLLM(
                cfg.VISION_MODELS, self.checker_cfg.MM_LLM.NAME
            )

        if 'LIGHTGLUE' in used_models:
            self.models['matcher'] = LightGlueClient(cfg.VISION_MODELS.LIGHTGLUE)

    def add_cared_categories(self, categories):
        """

        Args:
            categories: list. new cared categories

        Returns:

        """
        self.cared_labels += categories
        self.candidates += ',' + ','.join(self.cared_labels)

    def set_mm_llm_question(self, intention):
        prompt_template = getattr(prompt_templates, self.checker_cfg.MM_LLM.PROMPT)
        self.question = prompt_template.format(intention=intention)

        # send the answer to HTML
        self.messager.send_text('send_text', 'VLPrompt', self.question)

    def visual_sensing_surroundings(self, current_geocode, current_heading):
        print(f'>>> Visual Checker: Current geocode is: {current_geocode}')
        image_list = self.platform.get_all_streetview_from_geocode(
            current_geocode, cur_heading=current_heading
        )

        # run detection for each image
        is_detected = False
        adjust_pano_heading = self.checker_cfg.get('ADJUST_PANO_HEADING', False)
        for i, street_image in enumerate(image_list):
            if adjust_pano_heading:
                self.platform.mover.adjust_heading_web(street_image.heading)
            
            # check with detector
            if 'detector' in self.models:
                is_detected_view, detect_results = self.check_with_detector(street_image)
                is_detected = is_detected or is_detected_view

            # check with multi-modal models
            if 'mm_llm' in self.models:
                is_detected_view = self.check_with_mm_llm(street_image)
                is_detected = is_detected or is_detected_view

        return is_detected

    def visual_sensing_single(self, street_image):
        # run detection for each image
        is_detected = False
        result_dict = {}
        # check with detector
        if 'detector' in self.models:
            is_detected_model, detect_results = self.check_with_detector(street_image)
            is_detected = is_detected or is_detected_model
            result_dict.update(detect_results)

        # check with multi-modal models
        if 'mm_llm' in self.models:
            is_detected_model = self.check_with_mm_llm(street_image)
            is_detected = is_detected or is_detected_model

        return is_detected, result_dict

    def check_with_detector(self, street_image):
        detect_results, is_detected, object_views = self.models['detector'].check(
            street_image, self.candidates, self.cared_labels
        )
        # check the duplication of the detected objects
        if self.need_check_duplicate and is_detected and object_views is not None:
            self.deduplicate_and_record_to_memory(detect_results['final_detect'])

        return is_detected, detect_results

    def check_with_mm_llm(self, street_image):
        answer = self.models['mm_llm'].check(street_image, self.question, return_json=True)

        # send the answer to HTML
        text = "{}. {}".format(answer['answer'], answer['explanation'])
        self.messager.send_text('send_text', f'VLText{street_image.i + 1}', text)
        return answer['answer']

    def deduplicate_and_record_to_memory(self, detect_results):
        adjusted_views = detect_results['view']
        results = detect_results['result']
        
        for i, view in enumerate(adjusted_views):
            # de-duplicate the results
            is_duplicate, obj_id = self.check_duplication_single(view)
            
            cur_results = {
                'boxes': np.array([results['boxes'][i]]),
                'labels': [results['labels'][i]],
                'scores': np.array([results['scores'][i]]),
                'class_idx': np.array([results['class_idx'][i]])
            }
            
            # record the view to memory
            if not is_duplicate:
                print('>>> This is a novel obj, add to visual memory.')
                self.memory.add(view, cur_results)
            else:
                self.memory.add_new_view_to_exist_memory(view, obj_id, cur_results)
                print(f'>>> This is a duplicate obj with {obj_id}, add it as a novel view.')

    def check_duplication_single(self, view):
        # get candidates in the visual memory
        candidates = self.memory.retrieve_by_geocode(
            view, radius=self.checker_cfg.CHECK_DUPLICATE.RETRIEVE_RADIUS
        )

        is_duplicate = False
        obj_id = None
        for candidate in candidates:
            if candidate.geocode == view.geocode:
                is_duplicate = self.fast_duplication_check_in_same_geocode(view, candidate)
            elif self.checker_cfg.CHECK_DUPLICATE.CHECK_WITH_GEO_HEADING:
                is_duplicate = self.check_duplication_by_geocode_and_heading(
                    view, candidate, noise_radius=self.checker_cfg.CHECK_DUPLICATE.HEADING_NOISE_RADIUS
                )

            if self.checker_cfg.CHECK_DUPLICATE.CHECK_WITH_FM:
                is_duplicate_fm = self.check_duplicate_with_feature_matching(
                    view, candidate, self.checker_cfg.CHECK_DUPLICATE.MATCH_THRESHOLD
                )
                is_duplicate = is_duplicate_fm or is_duplicate

            if is_duplicate:
                obj_id = candidate.obj_id
                break
            else:
                obj_id = None

        return is_duplicate, obj_id

    def fast_duplication_check_in_same_geocode(self, view, candidate):
        if abs(view.heading - candidate.heading) > self.checker_cfg.CHECK_DUPLICATE.FAST_CHECK_RADIUS:
            return False
        else:
            return True

    @staticmethod
    def check_duplication_by_geocode_and_heading(view, candidate, noise_radius=10):
        """
        Check if the view is duplicate with the candidate by geocode and heading.
        If duplicate, return True and the obj_id of the candidate.
        If not duplicate, return False and None.
        Args:
            view:
            candidate:
            noise_radius:

        Returns:

        """
        intersect_geocode = geocode_utils.get_intersect_from_geocodes_and_heading(
            view.geocode, view.heading, candidate.geocode, candidate.heading
        )
        candidate_to_intersect = geocode_utils.calculate_heading_between_geocodes(
            candidate.geocode, intersect_geocode
        )
        view_to_intersect = geocode_utils.calculate_heading_between_geocodes(view.geocode, intersect_geocode)
        if abs(candidate_to_intersect - candidate.heading) < noise_radius and \
                abs(view_to_intersect - view.heading) < noise_radius:
            return True
        else:
            return False

    def check_duplicate_with_feature_matching(self, view, candidate, thresh):
        matches = self.models['matcher'].inference(view.image, candidate.image)
        is_match = int(matches) > thresh
        return is_match
