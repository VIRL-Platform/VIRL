# Modified from https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.7/tools/infer/predict_system.py

import os
import sys

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import cv2
import copy
import numpy as np
import json
import time
import logging
from PIL import Image


paddle_ocr_path = '/xxx/PaddleOCR'
paddle_ocr_parent = '/xxx'
sys.path.insert(0, paddle_ocr_parent)

# print(sys.path)
import PaddleOCR.tools.infer.utility as utility
import PaddleOCR.tools.infer.predict_rec as predict_rec
import PaddleOCR.tools.infer.predict_det as predict_det
import PaddleOCR.tools.infer.predict_cls as predict_cls
from PaddleOCR.tools.infer.utility import draw_ocr_box_txt, get_rotate_crop_image, get_minarea_rect_crop

sys.path = sys.path[1:]

from virl.lm import UnifiedChat
from virl.lm import prompt as prompt_templates


class TextSystem(object):
    def __init__(self, args):
        # if not args.show_log:
        #     logger.setLevel(logging.INFO)

        self.text_detector = predict_det.TextDetector(args)
        self.text_recognizer = predict_rec.TextRecognizer(args)
        self.use_angle_cls = args.use_angle_cls
        self.drop_score = args.drop_score
        if self.use_angle_cls:
            self.text_classifier = predict_cls.TextClassifier(args)

        self.args = args
        self.crop_image_res_index = 0

    def draw_crop_rec_res(self, output_dir, img_crop_list, rec_res):
        os.makedirs(output_dir, exist_ok=True)
        bbox_num = len(img_crop_list)
        for bno in range(bbox_num):
            cv2.imwrite(
                os.path.join(output_dir,
                             f"mg_crop_{bno+self.crop_image_res_index}.jpg"),
                img_crop_list[bno])
            # logger.debug(f"{bno}, {rec_res[bno]}")
        self.crop_image_res_index += bbox_num

    def __call__(self, img, cls=True):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}

        if img is None:
            return None, None, time_dict

        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.text_detector(img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            end = time.time()
            time_dict['all'] = end - start
            return None, None, time_dict

        img_crop_list = []

        dt_boxes = sorted_boxes(dt_boxes)

        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            if self.args.det_box_type == "quad":
                img_crop = get_rotate_crop_image(ori_im, tmp_box)
            else:
                img_crop = get_minarea_rect_crop(ori_im, tmp_box)
            img_crop_list.append(img_crop)
        if self.use_angle_cls and cls:
            img_crop_list, angle_list, elapse = self.text_classifier(
                img_crop_list)
            time_dict['cls'] = elapse

        rec_res, elapse = self.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse

        if self.args.save_crop_res:
            self.draw_crop_rec_res(self.args.crop_res_save_dir, img_crop_list,
                                   rec_res)
        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)
        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


class PaddleOCR(object):
    def __init__(self, cfg):
        self.cfg = cfg
        args = utility.init_args().parse_known_args()[0]
        args.rec_char_dict_path = os.path.join(paddle_ocr_path, args.rec_char_dict_path)
        args.vis_font_path = os.path.join(paddle_ocr_path, args.vis_font_path)
        args.det_model_dir = cfg.DET_MODEL_DIR
        args.rec_model_dir = cfg.REC_MODEL_DIR
        args.use_angle_cls = cfg.USE_ANGLE_CLS
        if args.use_angle_cls:
            args.cls_model_dir = cfg.CLS_MODEL_DIR

        self.args = args
        self.text_sys = TextSystem(args)
        self.is_visualize = False
        self.font_path = args.vis_font_path
        self.drop_score = args.drop_score
        self.draw_img_save_dir = args.draw_img_save_dir
        os.makedirs(self.draw_img_save_dir, exist_ok=True)
        self.save_results = []
        
        self.chatbot = UnifiedChat()

        # logger.info(
        #     "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', "
        #     "if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320"
        # )

        # warm up 10 times
        if args.warmup:
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                res = self.text_sys(img)

        self.total_time = 0
        self.cpu_mem, self.gpu_mem, self.gpu_util = 0, 0, 0
        self._st = time.time()
        self.count = 0

    def call_text_system(self, img):
        font_path = self.args.vis_font_path
        drop_score = self.args.drop_score
        draw_img_save_dir = self.args.draw_img_save_dir
        os.makedirs(draw_img_save_dir, exist_ok=True)
        save_results = []

        # warm up 10 times
        if self.args.warmup:
            img = np.random.uniform(0, 255, [640, 640, 3]).astype(np.uint8)
            for i in range(10):
                res = self.text_sys(img)

        # starttime = time.time()
        dt_boxes, rec_res, time_dict = self.text_sys(img)

        res = [{
            "transcription": rec_res[i],
            "points": np.array(dt_boxes[i]).astype(np.int32).tolist(),
        } for i in range(len(dt_boxes))]

        if self.is_visualize:
            image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            boxes = dt_boxes
            txts = [rec_res[i][0] for i in range(len(rec_res))]
            scores = [rec_res[i][1] for i in range(len(rec_res))]

            draw_img = draw_ocr_box_txt(
                image,
                boxes,
                txts,
                scores,
                drop_score=drop_score,
                font_path=font_path)

            save_file = 'temp.png'

            cv2.imwrite(
                os.path.join(draw_img_save_dir,
                                os.path.basename(save_file)),
                draw_img[:, :, ::-1])

        if self.args.benchmark:
            self.text_sys.text_detector.autolog.report()
            self.text_sys.text_recognizer.autolog.report()

        with open(
                os.path.join(draw_img_save_dir, "system_results.txt"),
                'w',
                encoding='utf-8') as f:
            f.writelines(save_results)
        return rec_res, dt_boxes

    def post_process(self, rec_res, det_boxes, img_shape):
        # delete google watermark in predictions
        new_res = []
        for res_item, box_item in zip(rec_res, det_boxes):
            if res_item[0].lower() == "google" and box_item[..., -1].min() > 0.92 * img_shape[1]:
                continue
            else:
                new_res.append(res_item)
        return new_res

    def inference(self, img, candidate_categories):
        # transform PIL image to numpy array
        img = np.asarray(img)[..., ::-1]

        # crop bottom pixels to remove google watermark
        img = img[:-25, :, :]
        category_list = candidate_categories.split(",,")

        # paddle OCR
        rec_res, det_boxes = self.call_text_system(img)

        if len(rec_res) == 0:
            return {
                'scores': np.array([0.0] *  len(category_list)),
            }
        
        # call GPT to generate answer
        category_score_list = []
        rec_text_list = [text[0].strip() for text in rec_res]
        rec_score_list = [float(text[1]) for text in rec_res]
        all_text = ", ".join(rec_text_list)
        prompt_template = getattr(prompt_templates, self.cfg.PROMPT)

        for category in category_list:
            prompt = prompt_template.format(
                text_list=all_text,
                place_name=category
            )
            answer_json = self.chatbot.ask(prompt, model=self.cfg.MODEL, json=True)
            if answer_json['answer'] == 'Yes':
                refer_name = answer_json['refer']
                cur_score = self.refer_name_to_score(refer_name, rec_text_list, rec_score_list)
                category_score_list.append(cur_score)
            else:
                category_score_list.append(0.0)

        results = {
            'scores': category_score_list,
        }
        return results

    @staticmethod
    def refer_name_to_score(refer_name, rec_text_list, rec_score_list):
        refer_name_list = refer_name.split(',')
        refer_name_list = [name.strip() for name in refer_name_list]
        score_list = []
        for cur_refer_name in refer_name_list:
            if cur_refer_name in rec_text_list:
                score = rec_score_list[rec_text_list.index(cur_refer_name)]
                score_list.append(score)

        if len(score_list) == 0:
            return 0.0
        else:
            return np.mean(score_list)


if __name__ == "__main__":
    from easydict import EasyDict
    cfg = EasyDict({'DET_MODEL_DIR': '/xxx/PaddleOCR/ckpt/ch_PP-OCRv4_det_server_infer',
                    'REC_MODEL_DIR': '/xxx/PaddleOCR/ckpt/ch_PP-OCRv4_rec_server_infer',
                    'USE_ANGLE_CLS': True,
                    'CLS_MODEL_DIR': '/xxx/ckpt/ch_ppocr_mobile_v2.0_cls_slim_infer',
                    'PROMPT': 'ocr_result_to_recognition_template',
                    'MODEL': 'gpt-3.5-turbo-0613'
                    })

    paddle_ocr = PaddleOCR(cfg)
    image_file = "xxx.jpg"
    image = Image.open(image_file)
    text = "a beautiful street,,a beautiful building"
    paddle_ocr.inference(image, text)
