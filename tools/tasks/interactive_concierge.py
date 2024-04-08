import pickle
import os
import copy

from prettytable import PrettyTable
from datetime import datetime, timedelta
from easydict import EasyDict

from tools.tasks.task_template import TaskTemplate
from tools.tasks.place_recommender import PlaceRecommender

from virl.config import cfg, cfg_from_yaml_file
from virl.lm import prompt as prompt_templates
from virl.utils import common_utils, pipeline
from virl.agents.agent_template import AgentTemplate


class InteractiveConcierge(TaskTemplate):
    def __init__(self, output_dir, logger):
        super(InteractiveConcierge, self).__init__(output_dir, logger)
        self.remain_budget = None
        self.all_plan_list = []
        self.activity_counter = 1
        self.all_plan_list_no_review = []
        self.all_status_list = []
        self.all_remain_budget_list = []

    def run(self, platform, agent, chatbot, messager, args, pipeline_cfg=None, custom_agent=None):
        self.agent = custom_agent
        self_agent = agent
        agent = custom_agent
        
        # resume if there is a checkpoint
        checkpoint_path = self.output_dir / 'checkpoint.pkl'
        if os.path.exists(checkpoint_path):
            self.resume_checkpoint(agent)
        else:
            self.remain_budget = cfg.TASK_INFO.BUDGET

        start_position = agent.start_position
        start_geocode = pipeline.position_to_geocode(platform, start_position)
        print(f'>>> Current geocode is: {start_geocode}')

        initial_plan_template = getattr(prompt_templates, cfg.PIPELINE.INITIAL_PLAN.PROMPT)
        while True:
            common_utils.print_stage(f'Plan activity #{self.activity_counter}', c='*')

            # Step 1: get the initial plan
            common_utils.print_stage('Step 1: get the initial plan')
            cur_plan, cur_plan_json = self.obtain_draft_plan(agent, chatbot, start_position, initial_plan_template)
            
            if cur_plan_json['type'] == 'end':
                self.all_plan_list.append(cur_plan)
                self.all_plan_list_no_review.append(self.remove_review_status_in_plan(agent, cur_plan))
                break
            
            common_utils.print_stage(f'Step 2: Interactive revise the plan {self.activity_counter} by a loop')
            adjust_plan_json = cur_plan_json
            # the internal revise loop: Hierarchical Coordinator -> Interoceptive Estimator -> Supervisor
            while True:
                adjust_plan_json, is_revised = self.adjust_plan(
                    chatbot, platform, agent, messager, args, adjust_plan_json
                )

                if not is_revised:
                    cur_status_modified_plan_json = adjust_plan_json
                    break

            common_utils.print_stage(f'Step 3: store plan {self.activity_counter} to working memory and update the status')            
            # after the plan is confirmed, we need to update the agent status and budget

            cur_status_modified_plan = common_utils.dict_to_str_with_newline(cur_status_modified_plan_json)
            self.remain_budget = self.adjust_agent_status_and_budget_by_plan(
                agent, cur_status_modified_plan_json, self.remain_budget
            )

            # store existing plan to working memory
            self.all_status_list.append(copy.deepcopy(agent.status))
            self.all_remain_budget_list.append(self.remain_budget)

            self.all_plan_list.append(cur_status_modified_plan)
            self.all_plan_list_no_review.append(self.remove_review_status_in_plan(agent, cur_status_modified_plan))
            self.activity_counter += 1

            self.save_checkpoint(agent)

        print('\n'.join(self.all_plan_list))
        self.count_status_to_table(agent, self.remain_budget)
        self.formulate_simplified_version()

    def adjust_plan(self, chatbot, platform, agent, messager, args, cur_plan_json):
        common_utils.print_stage('Step 2.1: Hierarchical Coordinator ground plan to real information')
        geo_revised_plan_json = self.revise_according_to_geospatial_information(
            chatbot, platform, agent, messager, args, cur_plan_json
        )
        geo_revised_plan = common_utils.dict_to_str_with_newline(geo_revised_plan_json)

        common_utils.print_stage('Step 2.2: Interoceptive Estimator review the plan for status change and cost')
        status_review_template = getattr(prompt_templates, cfg.PIPELINE.REVIEW_STATUS.MEASURE_PROMPT)
        status_review_prompt = status_review_template.format(
            background=agent.background, intention=agent.intention, activity=geo_revised_plan
        )
        cur_review = chatbot.ask(status_review_prompt, model=cfg.PIPELINE.REVIEW_STATUS.MODEL)
        cur_review_json = common_utils.parse_answer_to_json(cur_review)

        cur_plan_json.update(cur_review_json)
        cur_plan_with_status_change = common_utils.dict_to_str_with_newline(geo_revised_plan_json)

        common_utils.print_stage('Step 2.3: revise the plan considering status change and cost')
        cur_status_modified_plan_json, is_revised = self.revise_according_to_review_and_user_feedback(
            agent, chatbot, platform, self.remain_budget, cur_plan_with_status_change
        )

        return cur_status_modified_plan_json, is_revised
    
    @staticmethod
    def count_status_to_table(agent, remain_budget):
        initial_status = [cfg.agent.STATUS[key.upper()] for key in agent.status.keys()]

        table = PrettyTable()
        table.field_names = [""] + list(agent.status.keys()) + ['Budget']
        table.add_row(["Before the day"] + initial_status + [cfg.TASK_INFO.BUDGET])
        table.add_row(["After the day"] + [str(value) for value in agent.status.values()] + [str(remain_budget)])

        print(table)

    @staticmethod
    def adjust_agent_status_and_budget_by_plan(agent, plan, remain_budget):
        # adjust agent status according to the plan
        for key, value in agent.status.items():
            if (key + '_change') in plan:
                change_value_str = plan[key + '_change'].split(' ')[0]
                change_equation = f"{str(value)} + ({change_value_str})"
                new_value = eval(change_equation)
                agent.status[key] = max(min(new_value, 100), 0)
            else:
                raise ValueError(f"Key {key} is not in agent status.")

        # update the budget according to the plan
        extracted_cost = common_utils.extract_numbers(plan['cost'].split(' ')[0])
        if isinstance(extracted_cost, str):
            cost = float(eval(extracted_cost))
        else:
            cost = extracted_cost

        remain_budget -= cost
        return remain_budget

    def revise_according_to_review_and_user_feedback(self, agent, chatbot, platform, remain_budget,
                                                     current_plan_with_status_change):
        print(f"Current planned activity is {current_plan_with_status_change}")
                        
        cur_plan_json = common_utils.parse_answer_to_json(current_plan_with_status_change)

        # supervisor -> take human input to revise the plan
        # take language input
        if cfg.PIPELINE.REVISE_PLAN.USER_INPUT:
            user_input = input("Please enter your feedback (empty means no feedback): ")
        else:
            user_input = ''

        # take user status modification input
        if cfg.PIPELINE.REVISE_PLAN.get('USER_STATUS', None):
            print(f"Current status: {agent.status}")
            print(f"Current budget: {remain_budget}")
            user_input_status = input("Please enter status in dict style (empty means no feedback): ")
        else:
            user_input_status = ''
        
        if user_input_status != '':
            user_input_status = eval(user_input_status)
            # user input status is list
            for key, value in agent.status.items():
                if key in user_input_status:
                    agent.status[key] = user_input_status[key]
            
        has_user_input = user_input != '' or user_input_status != ''

        if not has_user_input and cur_plan_json['type'] == 'eat':
            print("The current activity is eating, so we don't need to revise the plan.")
            return cur_plan_json, False

        if len(self.all_plan_list_no_review) == 0:
            previous_activity = "No"
        elif len(self.all_plan_list_no_review) < cfg.PIPELINE.REVISE_PLAN.get('N_PREVIOUS', 100):
            previous_activity = '\n'.join(self.all_plan_list_no_review)
        else:
            start_idx = 0 - cfg.PIPELINE.REVISE_PLAN.N_PREVIOUS
            previous_activity = '\n'.join(self.all_plan_list_no_review[start_idx:])
        
        # supervisor -> audit: judge whether the current activity need to be revise
        # if there is user feedback, concierge should revise the plan
        if not has_user_input:
            judge_prompt_template = getattr(prompt_templates, cfg.PIPELINE.REVISE_PLAN.MODIFY_JUDGE_PROMPT)
            judge_prompt = judge_prompt_template.format(
                background=agent.background, intention=agent.intention,
                activity=current_plan_with_status_change,
                stress=agent.status['stress'], joy=agent.status['joy'], hunger=agent.status['hunger'],
                energy=agent.status['energy'], sadness=agent.status['sadness'], pain=agent.status['pain'], 
                budget=remain_budget, previous_activities=previous_activity
            )
            judge_json = chatbot.ask(judge_prompt, model=cfg.PIPELINE.REVISE_PLAN.MODEL, json=True)
            judge = judge_json['judge']
            # if supervisor approve the plan, then we don't need to revise the plan
            if judge.lower() == 'no':
                print("The current activity doesn't need to be revised.")
                return cur_plan_json, False

        # supervisor -> revise: revise the plan
        status_modify_template = getattr(prompt_templates, cfg.PIPELINE.REVISE_PLAN.MODIFY_PROMPT)
        status_modify_prompt = status_modify_template.format(
            background=agent.background, intention=agent.intention,
            activity=current_plan_with_status_change,
            stress=agent.status['stress'], joy=agent.status['joy'], hunger=agent.status['hunger'],
            energy=agent.status['energy'], sadness=agent.status['sadness'], pain=agent.status['pain'],
            budget=remain_budget, previous_activities=previous_activity,
            user_requirements=user_input
        )
        cur_status_modified_plan = chatbot.ask(status_modify_prompt, model=cfg.PIPELINE.REVISE_PLAN.MODEL)
        cur_status_modified_plan_json = common_utils.parse_answer_to_json(cur_status_modified_plan)

        return cur_status_modified_plan_json, True

    def revise_according_to_geospatial_information(self, chatbot, platform, agent, messager, args,
                                                   cur_status_modified_plan_json):
        activity_type = cur_status_modified_plan_json['type']
        if activity_type == 'transport':
            revised_plan = self.add_concrete_transportation_time(chatbot, platform, cur_status_modified_plan_json)
            return revised_plan
        elif activity_type == 'eat':
            revised_plan = self.add_concrete_restaurant(
                platform, chatbot, agent, messager, args, cur_status_modified_plan_json
            )
            return revised_plan

        return cur_status_modified_plan_json

    def add_concrete_transportation_time(self, chatbot, platform, cur_status_modified_plan_json):
        # obtain the transportation mode
        cur_plan = common_utils.dict_to_str_with_newline(cur_status_modified_plan_json)
        transport_mode_prompt_template = getattr(
            prompt_templates, cfg.PIPELINE.TRANSPORT_CHECK.TRANSPORT_MODE_PROMPT
        )
        transport_mode_prompt = transport_mode_prompt_template.format(activity=cur_plan)
        transport_mode = chatbot.ask(transport_mode_prompt, model=cfg.PIPELINE.TRANSPORT_CHECK.MODEL)

        start_position = cur_status_modified_plan_json['start_place'] + ', ' + self.agent.city
        end_position = cur_status_modified_plan_json['end_place'] + ', ' + self.agent.city

        duration = platform.get_transportation_time(
            start_position, end_position, transport_mode
        )
        start_time = cur_status_modified_plan_json['time'].split('-')[0].strip()
        end_time = self.calculate_time_with_duration(start_time, int(duration / 60))

        new_time = f"{start_time} - {end_time}"
        cur_status_modified_plan_json['time'] = new_time

        return cur_status_modified_plan_json

    def add_concrete_restaurant(self, platform, chatbot, agent, messager, args, cur_status_modified_plan_json):
        # obtain the restaurant
        new_agent_cfg = {
            'NAME': agent.name,
            'CITY': agent.city,
            'BACKGROUND': agent.background,
            'INTENTION': cur_status_modified_plan_json['content'],
            'START_POSITION': cur_status_modified_plan_json['start_place'] + ', ' + agent.city
        }
        new_agent = AgentTemplate(EasyDict(new_agent_cfg))

        new_cfg = EasyDict()
        new_cfg = cfg_from_yaml_file(cfg.PIPELINE.FIND_RESTAURANT.CFG_PATH, new_cfg)

        task_solver = PlaceRecommender(output_dir=self.output_dir, logger=self.logger)
        place_info = task_solver.run(
            platform, None, chatbot, messager, args, pipeline_cfg=new_cfg.PIPELINE, custom_agent=new_agent
        )

        # modify the plan
        # cur_status_modified_plan_json['content'] = f"Having food near the {cur_status_modified_plan_json['start_place']}."
        cur_status_modified_plan_json['restaurant_reason'] = f" {place_info['bot_explain']}"
        cur_status_modified_plan_json['start_place'] = place_info['name'] + ', ' + agent.city
        return cur_status_modified_plan_json

    @staticmethod
    def calculate_time_with_duration(start_time, duration):
        start_time_obj = datetime.strptime(start_time, "%H:%M")
        end_time_obj = start_time_obj + timedelta(minutes=duration)
        end_time = end_time_obj.strftime("%H:%M")

        return end_time

    def save_checkpoint(self, agent):
        checkpoint = {
            'all_plan_list': self.all_plan_list,
            'agent_status': agent.status,
            'remain_budget': self.remain_budget,
            'activity_counter': self.activity_counter,
            'all_plan_list_no_review': self.all_plan_list_no_review,
            'all_status_list': self.all_status_list,
            'all_remain_budget_list': self.all_remain_budget_list
        }

        output_path = self.output_dir / 'checkpoint_temp.pkl'
        with open(output_path, 'wb') as f:
            pickle.dump(checkpoint, f)

        if os.path.exists(self.output_dir / 'checkpoint.pkl'):
            os.remove(self.output_dir / 'checkpoint.pkl')

        os.rename(output_path, self.output_dir / 'checkpoint.pkl')

    def resume_checkpoint(self, agent):
        checkpoint_path = self.output_dir / 'checkpoint.pkl'
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)

        self.all_plan_list = checkpoint['all_plan_list']
        self.remain_budget = checkpoint['remain_budget']
        self.activity_counter = checkpoint['activity_counter']
        agent.status = checkpoint['agent_status']
        self.all_plan_list_no_review = checkpoint['all_plan_list_no_review']
        self.all_status_list = checkpoint['all_status_list']
        self.all_remain_budget_list = checkpoint['all_remain_budget_list']

    @staticmethod
    def remove_review_status_in_plan(agent, cur_plan_with_status_change, keys_to_remove=['cost', 'restaurant_reason']):
        new_plan = copy.deepcopy(common_utils.parse_answer_to_json(cur_plan_with_status_change))
        status_keys = [key + '_change' for key in agent.status.keys()]
        keys_to_remove = keys_to_remove + status_keys
        for key in keys_to_remove:
            new_plan.pop(key, None)

        return common_utils.dict_to_str_with_newline(new_plan)

    def formulate_simplified_version(self):
        simple_plan_list = []
        for i, cur_plan in enumerate(self.all_plan_list):
            plan_json = common_utils.parse_answer_to_json(cur_plan)
            info = f"{plan_json['time']}: [{plan_json['type']}] {plan_json['content']}. ({plan_json['start_place']})"
            print(info)
            simple_plan_list.append(info)
            if i < len(self.all_plan_list) - 1:
                print(self.all_status_list[i], self.all_remain_budget_list[i])
        
        print('\n'.join(simple_plan_list))

    def obtain_draft_plan(self, agent, chatbot, start_position, initial_plan_template):
        initial_plan_prompt = initial_plan_template.format(
            background=agent.background, intention=agent.intention, start_location=start_position,
            previous_plan='\n'.join(self.all_plan_list_no_review)
        )
        cur_plan = chatbot.ask(initial_plan_prompt, model=cfg.PIPELINE.INITIAL_PLAN.MODEL)

        cur_plan_json = common_utils.parse_answer_to_json(cur_plan)
        
        return cur_plan, cur_plan_json
    