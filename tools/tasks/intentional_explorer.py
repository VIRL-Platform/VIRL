from .task_template import TaskTemplate
from virl.config import cfg
from virl.utils import pipeline
from virl.lm import prompt as prompt_templates
from virl.utils.common_utils import ordinal, print_stage
from virl.actions.check_surrounding.visual_checker import VisualChecker
from virl.actions.navigation import build_navigator
from virl.platform.memory.memory import Memory


class IntentionalExplorer(TaskTemplate):
    def __init__(self, output_dir, logger):
        super().__init__(output_dir, logger)
        self.place_counter = 0

    def run(self, platform, agent, chatbot, messager, args, pipeline_cfg=None, **kwargs):
        start_position = agent.start_position
        if pipeline_cfg is None:
            pipeline_cfg = cfg.PIPELINE

        self.navigator = build_navigator(
            pipeline_cfg.NAVIGATION, platform, messager, start_position, output_dir=self.output_dir,
            intention=agent.full_intention, resume=args.resume
        )

        self.memory = Memory(self.output_dir, cfg.MEMORY, args.resume)

        visual_checker = VisualChecker(pipeline_cfg.CHECK_SURROUNDING, platform, messager, memory=self.memory)

        # Step 1: From intention to place that need to find
        print_stage('Step 1: From intention to place that needs to be found')
        place = pipeline.intention_to_place(pipeline_cfg.INTENT_TO_PLACE, agent, chatbot, args)
        print(f'>>> Place that needs to be found: {place}')
        visual_checker.add_cared_categories([place])
        self.navigator.place_type = place

        current_geocode = self.navigator.get_current_geocode()
        current_heading = self.navigator.get_current_heading()
        step_counter = 1
        while True:
            print('>>> Find the place from google street view in current location.')
            finish_navigation = False
            image_list = platform.get_all_streetview_from_geocode(
                current_geocode, cur_heading=current_heading, all_around=True if current_heading is None else False
            )
            for i, street_image in enumerate(image_list):
                find_result, result_dict = visual_checker.visual_sensing_single(street_image)
                if find_result:
                    # action = self.select_actions_decision(pipeline_cfg.SELECT_ACTIONS, agent, chatbot, place)
                    action = 'search_info()'
                    finish_navigation = self.execute_actions(
                        action, agent, platform, chatbot, place, pipeline_cfg, result_dict
                    )
                    break
            
            is_finish, current_geocode, info_dict = self.navigator.navigate(info_dict={'find_result': finish_navigation})
            current_heading = self.navigator.get_current_heading()
            if is_finish:
                break
                
            if step_counter % cfg.get('SAVE_INTERVAL', 10000000) == 0:
                self.save_results()

            step_counter += 1

    def select_actions_decision(self, pipeline_cfg, agent, chatbot, place):
        self.place_counter += 1
        question_template = getattr(prompt_templates, pipeline_cfg.PROMPT)
        
        condition = f"{agent.name} sees {place} in the street. " \
                    f"This is the {ordinal(self.place_counter)} place " \
                    "he/she has seen."

        question = question_template.format(
            background=agent.background,
            intention=agent.full_intention,
            condition=condition
        )
        answer = chatbot.ask(question, json=True)
        action = answer['action']

        return action

    def execute_actions(self, action, agent, platform, chatbot, place, pipeline_cfg, result_dict=None):
        if action == 'enter_place()':
            print(f'>>> {agent.name} enters the found {place}.')
            return True
        elif action == 'continual_find()':
            return False
        elif action == 'search_info()' and result_dict is not None:
            place_results = self.search_info(result_dict, platform, place, pipeline_cfg)
            if len(place_results) == 0:
                print('>>> The found place is not valid.')
                return False
            place_results = pipeline.review_to_intro(pipeline_cfg.PLACE_INTRO_REVIEW, platform, chatbot, place_results)
            for place_result in place_results:
                action = self.intro_to_action(pipeline_cfg.INTRO_TO_ACTION, agent, chatbot, place_result['intro'])
                is_select = self.execute_actions(action, agent, platform, chatbot, place, pipeline_cfg, None)
                if is_select:
                    return True

            return False
        else:
            raise ValueError(f'Unknown action: {action}')

    @staticmethod
    def search_info(result_dict, platform, place, pipeline_cfg):
        final_detect_views = result_dict['final_detect']['view']
        final_detect_results = result_dict['final_detect']['result']

        place_results = []
        for i, view in enumerate(final_detect_views):
            box = final_detect_results['boxes'][i]
            place_result = pipeline.query_place_in_the_google_map_single(
                pipeline_cfg.SEARCH_INFO, platform, box, view, place
            )
            if place_result is not None:
                place_results.append(place_result)

        return place_results

    @staticmethod
    def intro_to_action(pipeline_cfg, agent, chatbot, intro):
        question_template = getattr(prompt_templates, pipeline_cfg.PROMPT)

        question = question_template.format(
            background=agent.background,
            intention=agent.full_intention,
            intro=intro
        )
        answer = chatbot.ask(question, json=True)
        action = answer['action']

        return action
