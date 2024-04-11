import tqdm
import pickle
import os
import glob
import json
from tqdm import tqdm
import csv
import numpy as np

import PIL.Image as Image

from tools.tasks.task_template import TaskTemplate

from virl.config import cfg
from virl.lm import prompt as prompt_templates
from virl.perception.mm_llm.mm_llm import MultiModalLLM
from virl.utils import common_utils


class BMPlaceCentricVQA(TaskTemplate):
    def __init__(self, output_dir, logger):
        super().__init__(output_dir, logger)
        self.image_paths = None
        self.qa_pairs = None

        self.tp = 0
        self.chatbot = None

        self.ckpt_path = self.output_dir / cfg.PIPELINE.VQA.MM_LLM / 'prediction_results.pkl'
        self.ckpt_path.parent.parent.mkdir(parents=True, exist_ok=True)
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        self.prediction_results = {}
        if os.path.exists(self.ckpt_path):
            self.resume_results()

    def run(self, platform, agent, chatbot, messager, args, **kwargs):
        pipeline_cfg = cfg.PIPELINE
        self.chatbot = chatbot

        # step 1: prepare data
        self.load_data_and_gt(pipeline_cfg.PREPARE_DATA)

        if cfg.EVALUATION.EVAL_ONLY:
            self.eval_results()
            return

        # step 2: load multi-modal large language model
        mm_llm = MultiModalLLM(cfg.VISION_MODELS, pipeline_cfg.VQA.MM_LLM)

        # step 3: run vqa
        for i, image_path in tqdm(enumerate(self.image_paths), total=len(self.image_paths)):
            place_id = os.path.basename(image_path).split('.')[0]

            if place_id in self.prediction_results:
                continue

            image = Image.open(image_path)
            is_correct, model_answers, model_answers_raw = self.circular_eval(
                mm_llm, image, self.qa_pairs[place_id], pipeline_cfg.VQA)
            self.prediction_results[place_id] = {
                'is_correct': is_correct,
                'model_answers': model_answers,
                'model_answers_raw': model_answers_raw
            }
            # save temperary results
            self.save_results()

        # step 4: calculate accuracy
        self.calculate_accuracy()
        
        self.save_results()

    def load_data_and_gt(self, data_cfg):
        self.image_paths = glob.glob(data_cfg.IMAGE_DIR + '/*.jpg')
        self.qa_pairs = json.load(open(data_cfg.QA_PAIRS, 'r'))
        
        assert len(self.image_paths) == len(self.qa_pairs)

    def circular_eval(self, mm_llm, image, qa_pair, eval_cfg):
        question = qa_pair['question']
        answer = qa_pair['answer']
        answer_choice_list = ['A', 'B', 'C', 'D']
        answer_idx = answer_choice_list.index(answer)
        candidate_answer_list = [qa_pair['answer_a'], qa_pair['answer_b'], qa_pair['answer_c'], qa_pair['answer_d']]

        # get answer from multi-modal large language model
        is_correct = False
        model_answers, model_answers_raw = [], []
        for i in tqdm(range(len(answer_choice_list))):
            full_question = self.create_full_question(eval_cfg, candidate_answer_list, question)
            model_answer_raw = mm_llm.check(image, full_question, return_json=False)
            model_answers_raw.append(model_answer_raw)

            is_correct, model_answer = self.parse_model_answer(eval_cfg, full_question, answer, model_answer_raw)
            model_answers.append(model_answer)
            if not is_correct:
                break

            # shift answer choice
            answer_idx = (answer_idx - 1) % len(answer_choice_list)
            answer = answer_choice_list[answer_idx]
            candidate_answer_list = candidate_answer_list[1:] + [candidate_answer_list[0]]

        if is_correct:
            self.tp += 1
        return is_correct, model_answers, model_answers_raw

    @staticmethod
    def create_full_question(eval_cfg, candidate_answer_list, question):
        question_template = getattr(prompt_templates, eval_cfg.FULL_QUESTION_TEMPLATE)
        question = question_template.format(
            question=question,
            answer_a=candidate_answer_list[0],
            answer_b=candidate_answer_list[1],
            answer_c=candidate_answer_list[2],
            answer_d=candidate_answer_list[3],
        )
        return question.strip()

    def parse_model_answer(self, eval_cfg, full_question, answer, model_answer):
        if model_answer == '':
            return False, model_answer

        # Step 1: match prediction
        match_candidate_list = [
            f"{answer}", f"{answer}.", f"{answer})", f"{answer},", f"{answer})."
        ]
        if model_answer in match_candidate_list:
            return True, model_answer

        # Step 2: chatgpt to extract,
        candidate_answer = full_question.split(' Choices: ')[1].strip()
        question = full_question.split(' Choices: ')[0].strip().split('Q: ')[1].strip()
        gpt_match_template = getattr(prompt_templates, eval_cfg.GPT_MATCH_PROMPT)
        prompt = gpt_match_template.format(
            question=question,
            options=candidate_answer,
            prediction=model_answer
        )
        parsed_answer = self.chatbot.ask(prompt, model=eval_cfg.MODEL, json=False)

        return answer == parsed_answer, parsed_answer

    def save_results(self):
        pickle.dump(self.prediction_results, open(self.ckpt_path, 'wb'))

    def resume_results(self):
        self.prediction_results = pickle.load(open(self.ckpt_path, 'rb'))
        self.tp = sum([v['is_correct'] for k, v in self.prediction_results.items()])

    def calculate_accuracy(self, prediction_results):
        mean_acc = {'tp': {}, 'total': {}}
        overall_acc = 0
        for place_id, v in prediction_results.items():
            place_types = self.qa_pairs[place_id]['attributes']
            correct = v['is_correct']
            overall_acc += int(correct)

            for place_type in place_types:
                mean_acc['tp'][place_type] = mean_acc['tp'].get(place_type, 0) + correct / len(place_types)
                mean_acc['total'][place_type] = mean_acc['total'].get(place_type, 0) + 1 / len(place_types)

        overall_accuracy = overall_acc / (len(prediction_results) + 1e-6)
        mean_accuracy, accuracy_dict, count_dict = self.calculate_mean_acc_from_dict(mean_acc)
        return {'acc': overall_accuracy, 'macc': mean_accuracy, 'total': len(prediction_results),
                'acc_dict': accuracy_dict, 'count_duct': count_dict}

    @staticmethod
    def calculate_mean_acc_from_dict(mean_acc_dict):
        acc_dict = {}
        count_dict = {}
        for place_type, place_tp in mean_acc_dict['tp'].items():
            acc_dict[place_type] = place_tp / mean_acc_dict['total'][place_type]
            count_dict[place_type] = mean_acc_dict['total'][place_type]

        return np.mean(list(acc_dict.values())), acc_dict, count_dict

    def eval_city(self, place_info):
        city_prediction_results = {}
        for place_id, result in self.prediction_results.items():
            region = place_info[place_id]['region']
            continent, city = common_utils.map_region_to_continent_city(region)
            if city not in city_prediction_results:
                city_prediction_results[city] = {}
            city_prediction_results[city][place_id] = result

        city_accuracy = {}
        for city, city_results in city_prediction_results.items():
            city_accuracy[city] = self.calculate_accuracy(city_results)
        city_accuracy = dict(sorted(city_accuracy.items(), key=lambda item: item[0], reverse=False))
        return city_accuracy

    def eval_continent(self, place_info):
        continent_prediction_results = {}
        for place_id, result in self.prediction_results.items():
            region = place_info[place_id]['region']
            continent, city = common_utils.map_region_to_continent_city(region)
            if continent not in continent_prediction_results:
                continent_prediction_results[continent] = {}
            continent_prediction_results[continent][place_id] = result

        continent_accuracy = {}
        for continent, continent_results in continent_prediction_results.items():
            continent_accuracy[continent] = self.calculate_accuracy(continent_results)
        continent_accuracy = dict(sorted(continent_accuracy.items(), key=lambda item: item[0], reverse=False))
        return continent_accuracy

    def eval_place_type(self, qa_pairs):
        place_type_prediction_results = {}
        for place_id, result in self.prediction_results.items():
            place_types = qa_pairs[place_id]['attributes']
            for place_type in place_types:
                if place_type not in place_type_prediction_results:
                    place_type_prediction_results[place_type] = {}
                place_type_prediction_results[place_type][place_id] = result

        # sort dict
        place_type_accuracy = {}
        for place_type, place_type_results in place_type_prediction_results.items():
            place_type_accuracy[place_type] = self.calculate_accuracy(place_type_results)
        place_type_accuracy = dict(sorted(place_type_accuracy.items(), key=lambda item: item[0], reverse=False))
        return place_type_accuracy

    def write_to_file(self, filename, accuracies, eval_type):
        print('=====================================')
        with open(self.output_dir / cfg.PIPELINE.VQA.MM_LLM / filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([eval_type, 'accuracy', 'mean_accuracy', 'number of samples'])
            print(eval_type, 'accuracy', 'mean_accuracy', 'number of samples')
            print('--------------------------------')
            for item, accuracy in accuracies.items():
                writer.writerow([item, '{:.2f}%'.format(accuracy['acc'] * 100),
                                 '{:.2f}%'.format(accuracy['macc'] * 100), accuracy['total']])
                print(item, '{:.2f}%'.format(accuracy['acc'] * 100),
                      '{:.2f}%'.format(accuracy['macc'] * 100), accuracy['total'])
        print('=====================================')

    def eval_results(self):
        accuracy = self.calculate_accuracy(self.prediction_results)
        print('=====================================')
        print('Evaluation Results')
        print('Acc: {:.2f}%, mAcc: {:.2f}%, total: {}'.format(
            accuracy['acc'] * 100, accuracy['macc'] * 100, accuracy['total']))

        place_info = pickle.load(open(cfg.EVALUATION.REGION_FILE, 'rb'))
        assert len(self.qa_pairs) == len(self.prediction_results)

        # ===== place type accuracy ======
        accuracy['acc_dict'] = dict(sorted(accuracy['acc_dict'].items(), key=lambda item: item[0], reverse=False))
        print('=====================================')
        with open(self.output_dir / cfg.PIPELINE.VQA.MM_LLM / 'place_type_accuracy.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['place type', 'accuracy', 'count'])
            print('place type', 'accuracy', 'count')
            print('--------------------------------')
            for item, acc in accuracy['acc_dict'].items():
                writer.writerow([item, '{:.2f}%'.format(acc * 100), accuracy['count_duct'][item]])
                print(item, '{:.2f}%'.format(acc * 100), accuracy['count_duct'][item])
        print('=====================================')

        # ===== city accuracy ======
        city_accuracy = self.eval_city(place_info)
        if 'city' in cfg.EVALUATION.EVAL_CONTENT:
            self.write_to_file('city_accuracy.csv', city_accuracy, 'city')

        # ===== continent accuracy ======
        continent_accuracy = self.eval_continent(place_info)
        if 'continent' in cfg.EVALUATION.EVAL_CONTENT:
            self.write_to_file('continent_accuracy.csv', continent_accuracy, 'continent')
