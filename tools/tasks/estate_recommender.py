import os
import tqdm
import json
import ast

from tools.tasks.task_template import TaskTemplate
from virl.config import cfg, cfg_from_list, cfg_from_yaml_file
from virl.lm import prompt as prompt_templates
from virl.platform.estate_apis import EstateAPIs
from virl.utils import common_utils, pipeline


class EstateRecommender(TaskTemplate):
    def __init__(self, output_dir, logger):
        super().__init__(output_dir, logger)
        
        self.estate_candidates_path = output_dir / 'estate_candidates.json'
        self.estate = None
        
    def run(self, platform, agent, chatbot, messager, args, custom_agent=None, **kwargs):
        if custom_agent is not None:
            self_agent = agent
            agent = custom_agent

        pipeline_cfg = cfg.PIPELINE
        
        # Step 1: Get the suitable estate candidates from estate API
        common_utils.print_stage('Step 1: Get the suitable estate candidates from estate API')
        estate_apis = EstateAPIs()
        self.get_estate_candidates(pipeline_cfg.FIND_ESTATE_CANDIDATES, estate_apis)
        print(f'>>> There are {len(self.estate)} estate candidates.')
        
        place_position_list = [(estate['latitude'], estate['longitude']) for estate in self.estate]
        common_utils.save_points_to_txt(self.output_dir / 'estate_list.txt', place_position_list)
        
        # Step 2: For each candidate, find its cared nearby places
        common_utils.print_stage('Step 2: For each candidate, find its cared nearby places')
        self.search_estate_nearby(pipeline_cfg.NEARBY_INFO, platform, chatbot, agent)
        
        # Step 3: LLM rate the estate candidates and shorten the candidates list to a maximum length
        common_utils.print_stage('Step 3: LLM rate the estate candidates and shorten the '
                                 'candidates list to a maximum length')
        self.rate_estate_candidates(pipeline_cfg.RATING, chatbot, agent)
        
        # Step 4: LLM select the final estate candidate
        # self.final_recommend(pipeline_cfg.FINAL_RECOMMEND, lm, agent)

        # Step 4: Give the top K results and the reason from LLM
        self.formulate_output()

    def formulate_output(self):
        print(f"According to your requirement, the top {len(self.estate)} estate candidates are:")
        for i, estate in enumerate(self.estate):
            print(f">>> The {i+1}th candidate estate is:")
            print(f">>> {estate['estate_info'].strip()}")
            print(f">>> Rating: {estate['rating']}. Reason: {estate['rating_expression']} \n")

    def final_recommend(self, stage_cfg, chatbot, agent):
        all_estate_info = ""
        single_estate_candidate_template = getattr(prompt_templates, stage_cfg.CANDIDATE_PROMPT)
        for i, estate in enumerate(self.estate):
            cur_estate_info = single_estate_candidate_template.format(
                idx=i,
                estate_info=estate['estate_info'].strip()
            )
            all_estate_info += cur_estate_info
        
        final_recommend_prompt_template = getattr(prompt_templates, stage_cfg.PROMPT)
        final_recommend_prompt = final_recommend_prompt_template.format(
            background=agent.background, requirement=agent.intention,
            all_estate_info=all_estate_info.strip(),
        )
        answer_json = chatbot.ask(final_recommend_prompt, model=stage_cfg.MODEL, json=True)
        final_idx = answer_json['idx']
        print(f">>> The final recommended estate is: {self.estate[final_idx]['estate_info'].strip()}")
        print(f">>> Reason: {answer_json['reason']}")

    def rate_estate_candidates(self, stage_cfg, chatbot, agent):
        estate_rating_prompt_template = getattr(prompt_templates, stage_cfg.PROMPT)
        for estate in tqdm.tqdm(self.estate):
            if 'rating' in estate:
                continue
            estate_rating_prompt = estate_rating_prompt_template.format(
                background=agent.background, requirement=agent.intention,
                estate_info=estate['estate_info'].strip(),
            )
            answer_json = chatbot.ask(
                estate_rating_prompt, model=stage_cfg.MODEL, json=True
            )
            estate['rating'] = float(answer_json['rating'])
            estate['rating_expression'] = answer_json['explanation']

        sorted_estate = sorted(self.estate, key=lambda x: x["rating"])
        self.estate = sorted_estate[:stage_cfg.MAX_NUM]
        common_utils.dump_json_results(self.estate, self.estate_candidates_path)

    def get_estate_candidates(self, stage_cfg, estate_apis):
        if os.path.exists(self.estate_candidates_path):
            self.estate = json.load(open(self.estate_candidates_path, 'r'))
            print(f'>>> Load estate candidates from {self.estate_candidates_path}')
        elif stage_cfg.ESTATE_INFO != 'None':
            estate_path = stage_cfg.ESTATE_INFO
            self.estate = json.load(open(estate_path, 'r'))
            print(f'>>> Load estate candidates from {estate_path}')
        elif stage_cfg.QUERY_API:
            query_configs = stage_cfg.PARAMS
            if stage_cfg.QUERY_MODE == 'sale':
                self.estate = estate_apis.query_sale(query_configs)
            elif stage_cfg.QUERY_MODE == 'rent':
                self.estate = estate_apis.query_rent(query_configs)
            else:
                raise NotImplementedError
            common_utils.dump_json_results(self.estate, self.estate_candidates_path)

        # Step 1.2: Filter by price
        self.estate = estate_apis.filter_by_price(self.estate, stage_cfg.MIN_PRICE, stage_cfg.MAX_PRICE)

    def search_estate_nearby(self, stage_cfg, platform, chatbot, agent):
        print(f'>>> Searching estate candidates nearby among {len(self.estate)}...')
        print("Parse the cared nearby places...")
        
        prompt_template = getattr(prompt_templates, stage_cfg.PARSE_TYPES.PROMPT)
        prompt = prompt_template.format(
            estate_requirement=agent.intention,
        )
        answer = chatbot.ask(prompt, model=stage_cfg.PARSE_TYPES.MODEL)
        cared_place_list = ast.literal_eval(answer)

        for estate in tqdm.tqdm(self.estate):
            if 'nearby_places' in estate:
                continue
            cur_geocode = (estate['latitude'], estate['longitude'])
            cur_nearby_places_dict = {}
            express_list = []
            expression_dict = {}
            for place_type in cared_place_list:
                nearby_places = platform.get_nearby_places_v2(
                    cur_geocode, query=place_type, radius_custom=stage_cfg.SEARCH.RADIUS,
                    deduplicate_with_name=stage_cfg.SEARCH.DEDUPLICATE, min_reviews=stage_cfg.SEARCH.MIN_REVIEWS,
                )
                cur_nearby_places_dict[place_type] = nearby_places
                express_list.append(f'{len(nearby_places)} {place_type}')
                expression_dict[place_type] = {}
                expression_dict[place_type]['number'] = len(nearby_places)
                if len(nearby_places) > 0:
                    expression_dict[place_type]['closest'] = f"{int(nearby_places[0]['distance'])} meters"
                else:
                    expression_dict[place_type]['closest'] = 'None'
            estate['nearby_places'] = cur_nearby_places_dict
            estate['nearby_places_expression_dict'] = expression_dict

            # formulate the nearby places by prompt template
            expression = ', '.join(express_list)
            expression = "There are " + expression + f" within {stage_cfg.SEARCH.RADIUS} meters."
            print(expression)
            estate['nearby_places_expression'] = expression
        
        common_utils.dump_json_results(self.estate, self.estate_candidates_path)
        self.fill_estate_info(stage_cfg)
        print('>>> Finish searching estate candidates nearby.')

    def fill_estate_info(self, stage_cfg):
        estate_info_template = getattr(prompt_templates, stage_cfg.INFO_PROMPT)
        for estate in self.estate:
            cur_estate_info = estate_info_template.format(
                address=estate['formattedAddress'],
                price=estate['price'],
                property_type=estate['propertyType'],
                size=estate['squareFootage'] if 'squareFootage' in estate else 'Unknown',
                bedrooms=estate['bedrooms'] if 'bedrooms' in estate else 0,
                bathrooms=estate['bathrooms'] if 'bathrooms' in estate else 0,
                year_built=estate['yearBuilt'] if 'yearBuilt' in estate else 'Unknown',
                nearby_radius=stage_cfg.SEARCH.RADIUS,
                nearby_info=estate['nearby_places_expression_dict'],
            )
            estate['estate_info'] = cur_estate_info
