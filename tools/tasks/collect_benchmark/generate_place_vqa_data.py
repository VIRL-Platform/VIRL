import tqdm
import pickle
import os
import json

from tools.tasks.task_template import TaskTemplate

from virl.config import cfg
from virl.utils import common_utils, geocode_utils
from virl.lm import prompt as prompt_templates


class GeneratePlaceVQAData(TaskTemplate):
    def __init__(self, output_dir, logger):
        super().__init__(output_dir, logger)
        self.qa_pair_path = os.path.join(self.output_dir, 'qa_pairs.json')
        
        if os.path.exists(cfg.PIPELINE.QA_PAIR_PATH):
            self.qa_pairs = json.load(open(cfg.PIPELINE.QA_PAIR_PATH, 'r'))
        elif os.path.exists(self.qa_pair_path):
            self.qa_pairs = json.load(open(self.qa_pair_path, 'r'))
        else:
            self.qa_pairs = {}
        print(f'Loaded {len(self.qa_pairs)} QA pairs')
        
        self.valid_place_type_list = []
        with open(cfg.PIPELINE.PLACE_TYPES, 'r') as f:
            for line in f:
                self.valid_place_type_list.append(line.strip())
    
    def run(self, platform, agent, chatbot, messager, args, **kwargs):
        pipeline_cfg = cfg.PIPELINE

        # load place infos
        place_info_dict = pickle.load(open(pipeline_cfg.PLACE_INFO, 'rb'))

        # load place types
        place_types = []
        with open(pipeline_cfg.PLACE_TYPES, 'r') as f:
            for line in f:
                place_types.append(line.strip())

        # prompt LLM to generate candidate answers
        for i, (place_id, place_info) in tqdm.tqdm(enumerate(place_info_dict.items()), total=len(place_info_dict)):
            if place_id in self.qa_pairs:
                continue

            place_name = place_info['name']
            raw_place_types = place_info['place_types']
            intersect_types = common_utils.list_intersection(raw_place_types, self.valid_place_type_list)
            place_types = [type.replace('_', ' ') for type in intersect_types]
            place_types_str = ', '.join(place_types)

            question = pipeline_cfg.QUESTION

            # generate answer and full question
            answer_json = self.generate_answer(
                pipeline_cfg, chatbot, question, place_name, place_types_str
            )

            qa_pair = {
                'question': question,
                'place': place_name,
                'attributes': place_types,
                'answer': answer_json['true'],
                'answer_a': answer_json['A'],
                'answer_b': answer_json['B'],
                'answer_c': answer_json['C'],
                'answer_d': answer_json['D'],
            }

            # add qa pair to qa pairs
            self.qa_pairs[place_id] = qa_pair

            if i % cfg.SAVE_INTERVAL == 0:
                self.save_qa_pairs()

        # save qa pairs
        self.save_qa_pairs()

    def save_qa_pairs(self):
        json.dump(self.qa_pairs, open(self.qa_pair_path, 'w'))

    @staticmethod
    def generate_answer(pipeline_cfg, chatbot, question, place_name, place_type):
        prompt_template = getattr(prompt_templates, pipeline_cfg.ANSWER_PROMPT)
        prompt = prompt_template.format(
            question=question,
            place_name=place_name,
            attributes=place_type
        )
        answer_json = chatbot.ask(prompt, model=pipeline_cfg.MODEL, json=True)

        return answer_json
