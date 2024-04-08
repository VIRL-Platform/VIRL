from queue import PriorityQueue

from tools.tasks.task_template import TaskTemplate
from virl.config import cfg
from virl.lm import prompt as prompt_templates
from virl.utils import common_utils, pipeline


class PlaceRecommender(TaskTemplate):
    def run(self, platform, agent, chatbot, messager, args, pipeline_cfg=None, custom_agent=None):
        if custom_agent is not None:
            self_agent = agent
            agent = custom_agent

        geocode = pipeline.position_to_geocode(platform, agent.start_position)
        print(f'>>> Current geocode is: {geocode}')

        if pipeline_cfg is None:
            pipeline_cfg = cfg.PIPELINE

        # from user intention to place type
        places = pipeline.intention_to_place(pipeline_cfg.INTENT_TO_PLACE, agent, chatbot, args)

        print(f'>>> Searching {places} nearby {geocode} via Google Map APIs...')

        # Get nearby places
        common_utils.print_stage('Step 1: obtain nearby proposals')
        candidates = self.search_nearby_places(
            pipeline_cfg.SEARCH_NEARBY, platform, geocode, places
        )

        common_utils.print_stage('Step 2: Search/Summarize intro of place proposals')
        if pipeline_cfg.PLACE_INTRO.MANNER == 'websearch':
            candidates = pipeline.search_intro(pipeline_cfg.PLACE_INTRO, agent, chatbot, candidates)
        elif pipeline_cfg.PLACE_INTRO.MANNER == 'summarize_reviews':
            candidates = pipeline.review_to_intro(pipeline_cfg.PLACE_INTRO, platform, chatbot, candidates)
        else:
            raise ValueError

        # agent rate place candidates according to background and intention
        common_utils.print_stage('Step 3: Rating places candidates')
        final_list = self.rate_place_candidates(pipeline_cfg.RATING, agent, chatbot, candidates)
        print('>>> Finish rating.')

        # Output all rated candidates
        common_utils.print_stage(f'Output: candidates place rating and reasons')
        for i, candidate in enumerate(final_list):
            print(f'>>> {i}-th: {candidate["name"]}; {candidate["intro"]}')
            print(f">>> Agent rating: {candidate['bot_rating']}, Agent Reason: {candidate['bot_explain']}\n")

        final_selection = final_list[0]
        return final_selection

    @staticmethod
    def search_nearby_places(search_cfg, platform, geocode, places):
        nearby_places = platform.get_nearby_places_v2(
            geocode, query=places, ranking_type=search_cfg.RANKING,
            radius_custom=search_cfg.get('RADIUS_CUSTOM', None),
        )
        print(f'>>> Finish Searching with {len(nearby_places)} results.')
        candidates = nearby_places[:search_cfg.TOPK]

        return candidates

    @staticmethod
    def rate_place_candidates(rating_cfg, agent, chatbot, candidates):
        place_rating_prompt_template = getattr(prompt_templates, rating_cfg.PROMPT)
        rating_queue = PriorityQueue()

        for i, candidate in enumerate(candidates):
            place_rating_prompt = place_rating_prompt_template.format(
                background=agent.background, intention=agent.intention,
                place_intro=candidate['intro']
            )
            answer_json = chatbot.ask(
                place_rating_prompt, model=rating_cfg.MODEL, json=True
            )
            candidate['bot_rating'] = float(answer_json['rating'])
            candidate['bot_explain'] = answer_json['explanation']
            rating_queue.put(common_utils.ComparableObj(-candidate['bot_rating'], candidate))

        final_list = []
        while not rating_queue.empty():
            final_list.append(rating_queue.get().data)

        return final_list
