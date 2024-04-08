import numpy as np
from tools.tasks.task_template import TaskTemplate

from virl.config import cfg
from virl.lm import prompt as prompt_templates
from virl.utils import pipeline, geocode_utils, place_utils


class Local(TaskTemplate):
    def run(self, platform, agent, chatbot, messager, args, **kwargs):
        question_to_place = True
        if kwargs.get('pipeline_cfg', None) is not None:
            pipeline_cfg = kwargs['pipeline_cfg']
            question_to_place = kwargs.get('question_to_place', True)
        else:
            pipeline_cfg = cfg.PIPELINE
            agent.heading = cfg.INIT_HEADING

        start_position = agent.start_position
        geocode = pipeline.position_to_geocode(platform, start_position)
        print(f'>>> Current geocode is: {geocode}')

        # Step 1: search the target place
        if question_to_place:
            user_question = kwargs.get('user_question', cfg.TASK_INFO.QUESTION)
            dest_place = self.search_destination(
                pipeline_cfg.QUESTION_TO_PLACE, chatbot, platform, agent, start_position, user_question
            )
        else:
            dest_place = kwargs['dest_place']

        dest_name = dest_place['name']
        print(f">>> I recommend that you go to {dest_name}.")

        # Step 2: obtain the route to the place
        if pipeline_cfg.ROUTE.get('RELOCATE', True):
            destination, _ = platform.relocate_geocode_by_source(dest_place['geocode'], source='outdoor')
        else:
            destination = dest_place['geocode']

        result_dict = platform.get_routing(
            start_position, destination,
            mode=pipeline_cfg.ROUTE.ROUTE_MODE,
            stopover=pipeline_cfg.ROUTE.STOPOVER,
            modify_destination=False
        )

        route = result_dict['geocode_list']
        route_distance = result_dict['distance_list']
        polyline = geocode_utils.encode_polyline([start_position] + route)
        pipeline.draw_planned_route(
            platform, polyline, input_way_points=None,
            path=self.output_dir / pipeline_cfg.OUTPUT.ROUTE_PATH,
            file_template=pipeline_cfg.OUTPUT.FILE_TEMPLATE
        )

        # Step 3: obtain the nearby landmarks of the route
        milestone_info_list, landmark_list = self.obtain_landmarks(
            pipeline_cfg, platform, chatbot, agent, route, start_position, dest_name, dest_place, route_distance
        )

        # Step 4: formulate the prompt
        all_milestone_info = "".join(milestone_info_list)

        all_milestone_info_template = getattr(prompt_templates, pipeline_cfg.MILESTONE_INFO.ALL_INFO_PROMPT)
        all_milestone_info_prompt = all_milestone_info_template.format(
            milestone_information=all_milestone_info.strip()
        )
        instruction = chatbot.ask(all_milestone_info_prompt, model=pipeline_cfg.MILESTONE_INFO.MODEL)

        start_place = {
            'geocode': start_position,
            'relocated_geocode': start_position,
        }
        result_dict['landmark_list'] = landmark_list
        route_info = {
            'dest_place': dest_place,
            'start_place': start_place,
            'route_results': result_dict,
            'init_heading': agent.heading,
            'instruction': instruction,
            'milestone_info': all_milestone_info
        }

        return route_info

    def search_destination(self, stage_cfg, chatbot, platform, agent, start_position, user_question):
        question_place_template = getattr(prompt_templates, stage_cfg.PROMPT)
        question_place_prompt = question_place_template.format(question=user_question)
        answer_json = chatbot.ask(question_place_prompt, json=True, model=stage_cfg.MODEL)
        place = answer_json['answer']

        place_candidates = platform.get_nearby_places_v2(
            start_position, query=place, radius_custom=stage_cfg.RADIUS
        )

        if len(place_candidates) == 0:
            print(f'>>> The {place} is too far away from the current position.')
            return

        if stage_cfg.get('CANDIDATE_SELECT', 'distance') == 'distance':
            dest_place = place_candidates[0]
        elif stage_cfg.get('CANDIDATE_SELECT') == 'llm':
            dest_place = self.llm_to_select_destination(stage_cfg, chatbot, agent, place_candidates)
        else:
            raise ValueError(f"Unknown destination selection manner: {stage_cfg.CANDIDATE_SELECT}")

        return dest_place
    
    def llm_to_select_destination(self, stage_cfg, chatbot, agent, place_candidates):
        all_place_candidates_info = ", ".join([f"<{place['name']}>" for place in place_candidates])
        
        all_place_candidates_info = "[" + all_place_candidates_info + "]"
        
        select_dest_template = getattr(prompt_templates, stage_cfg.SELECT_PROMPT)
        select_dest_prompt = select_dest_template.format(
            background=agent.background, place_candidates=all_place_candidates_info
        )

        answer_json = chatbot.ask(select_dest_prompt, json=True, model=stage_cfg.MODEL)
        place_name = answer_json['name']
        
        for place in place_candidates:
            if place['name'] == place_name:
                dest_place = place
                break
        
        return dest_place
    
    @staticmethod
    def select_suitable_landmark_by_llm(pipeline_cfg, chatbot, landmarks):
        if len(landmarks) == 0:
            return []

        candidate_list = []
        candidate_name_list = []
        for landmark in landmarks:
            candidate_list.append(f"<{landmark['name']}>")
            candidate_name_list.append(landmark['name'])

        candidate_strs = ", ".join(candidate_list)

        select_landmarks_template = getattr(prompt_templates, pipeline_cfg.PROMPT)
        select_landmarks_prompt = select_landmarks_template.format(candidates=candidate_strs)
        answer_json = chatbot.ask(select_landmarks_prompt, json=True, model=pipeline_cfg.MODEL)

        idx = candidate_name_list.index(answer_json['name'])

        return [landmarks[idx]]

    @staticmethod
    def select_suitable_landmark_by_reviews(pipeline_cfg, landmarks):
        if len(landmarks) == 0:
            return []

        n_reviews_list = [landmark['n_reviews'] for landmark in landmarks]
        max_idx = np.argmax(n_reviews_list)
        max_reviews = n_reviews_list[max_idx]
        if max_reviews < pipeline_cfg.MIN_REVIEWS:
            return []
        else:
            return [landmarks[max_idx]]

    def obtain_landmarks(self, pipeline_cfg, platform, chatbot, agent, route, start_position, 
                         dest_name, dest_place, route_distance):
        milestone_info_list = []
        landmark_list = []
        previous_position = start_position

        agent_heading = agent.heading
        for i, milestone in enumerate(route):
            agent_heading_to_milestone = geocode_utils.calculate_heading_between_geocodes(
                previous_position, milestone
            )
            
            if i == len(route) - 1:
                # if this is the last milestone, we should use the destination place as the landmark
                heading_to = geocode_utils.calculate_heading_between_geocodes(milestone, dest_place['geocode'])
                spatial_relation = geocode_utils.calculate_spatial_relationship_with_headings(
                    agent_heading_to_milestone, heading_to
                )
                landmark_info = {'expression': f"The destination {dest_name} is on your {spatial_relation}"}
            else:
                landmarks = platform.get_nearby_places(
                    milestone, rankby='distance', radius_custom=pipeline_cfg.SELECT_LANDMARKS.RADIUS,
                    no_next_page=True
                )

                if pipeline_cfg.SELECT_LANDMARKS.MANNER == 'llm_select':
                    landmarks = self.select_suitable_landmark_by_llm(pipeline_cfg.SELECT_LANDMARKS, chatbot, landmarks)
                elif pipeline_cfg.SELECT_LANDMARKS.MANNER == 'reviews_select':
                    landmarks = self.select_suitable_landmark_by_reviews(pipeline_cfg.SELECT_LANDMARKS, landmarks)
                else:
                    raise ValueError(f"Unknown manner: {pipeline_cfg.SELECT_LANDMARKS.MANNER}")
                
                if len(landmarks) == 0:
                    landmark_list.append(None)
                else:
                    landmark_list.append(landmarks[0])
                
                landmark_info = place_utils.calculate_milestone_information(landmarks, agent_heading_to_milestone)
            
            milestone_info = place_utils.fill_in_milestone_information_template(
                landmark_info, i, agent_heading, route_distance[i], agent_heading_to_milestone,
                pipeline_cfg.MILESTONE_INFO.SINGLE_PROMPT
            )
            milestone_info_list.append(milestone_info)

            agent_heading = agent_heading_to_milestone
            previous_position = milestone
            
        return milestone_info_list, landmark_list
    