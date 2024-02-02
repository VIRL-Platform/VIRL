import numpy as np

from virl.config import cfg
from virl.utils import common_utils, pipeline
from virl.lm import UnifiedChat
from virl.lm import prompt as prompt_templates
from virl.perception.mm_llm import MultiModalLLM
from .navigator_template import NavigatorTemplate


class IntentionNavigator(NavigatorTemplate):
    def __init__(self, nav_cfg, platform, messager, start_location, output_dir, intention, **kwargs):
        super().__init__(nav_cfg, platform, messager, start_location, output_dir, **kwargs)

        self.from_road_idx = 'N/A'
        self.from_road_heading = None

        if nav_cfg.get('MM_LLM', None):
            self.mm_llm = MultiModalLLM(cfg.VISION_MODELS, nav_cfg.MM_LLM)
        else:
            self.mm_llm = None
        self.intention = intention

    def move(self, info_dict):
        direction_image_list, heading_list = self.get_suitable_images_for_direction()
        # calculate the from_road_idx
        if self.from_road_heading is not None:
            all_headings = np.array(heading_list)
            # minus 180 degree since the incoming heading is the opposite direction of the road
            heading_diff = np.abs(((all_headings - 180) % 360) - self.from_road_heading)
            heading_diff = np.where(heading_diff > 180, 360 - heading_diff, heading_diff)
            self.from_road_idx = np.argmin(heading_diff)

        moving_direction_idx = self.get_moving_direction_idx_by_llm(direction_image_list)

        import ipdb; ipdb.set_trace(context=20)
        print(moving_direction_idx)
        self.from_road_heading = heading_list[moving_direction_idx]

        # Step 5.2: Move to the next location
        self.current_geocode = self.platform.mover.move(idx=moving_direction_idx)
        self.current_heading = heading_list[moving_direction_idx]
        print(f'>>> IntentionNavigator: after moving, the geocode is: {self.current_geocode}')

    def check_stop(self, info_dict):
        return info_dict['find_result']

    def actions_before_moving(self, info_dict):
        pass

    def get_suitable_images_for_direction(self, save_image=False):
        heading_list = self.platform.mover.get_all_suitable_heading_to_path(self.current_geocode)
        direction_image_list = self.platform.get_all_streetview_from_geocode(
             self.current_geocode, heading_list=heading_list
        )

        # send the road image to HTML
        image_str_list = []
        for img in direction_image_list:
            base64_string = common_utils.encode_image_to_string(img.image)
            image_str_list.append(base64_string)
        element_id_list = [f'Road{i + 1}' for i in range(len(image_str_list))]
        self.messager.send_image_list('send_image_list', element_id_list, image_str_list)

        if save_image:
            for i, street_image in enumerate(direction_image_list):
                street_image.image.save(f'../visual_output/direction_{i}.png')

        return direction_image_list, heading_list

    def get_moving_direction_idx_by_llm(self, direction_image_list):
        all_road_descriptions = ""
        road_description_template = getattr(prompt_templates, self.cfg.DESCRIPTION.PROMPT)
        caption_prompt_template = getattr(prompt_templates, self.cfg.CAPTION.PROMPT)
        caption_prompt = caption_prompt_template.format(
            place_type=self.place_type
        ).strip()
        for i, street_image in enumerate(direction_image_list):
            # caption the image
            caption = self.mm_llm.check(street_image.image, caption_prompt, return_json=False)

            all_road_descriptions += road_description_template.format(idx=i, description=caption)

        # ask which direction to go
        decision_template = getattr(prompt_templates, self.cfg.DECISION.PROMPT)
        prompt = decision_template.format(
            n_paths=len(direction_image_list), max_idx=len(direction_image_list) - 1,
            from_road_idx=self.from_road_idx, intention=self.intention, road_descriptions=all_road_descriptions
        )
        common_utils.print_prompt(prompt)

        answer_json = UnifiedChat.ask(
            prompt, json=True, chatbot=self.cfg.DECISION.CHATBOT,
            model=self.cfg.DECISION.MODEL
        )
        common_utils.print_answer(answer_json)

        # send text to HTML
        answer = 'idx:{}, reason: {}'.format(answer_json['idx'], answer_json['reason'])
        self.messager.send_text(
            'send_text', 'RoadText', answer
        )

        idx = int(answer_json['idx'])
        return idx

    def set_mm_llm(self, mm_llm):
        self.mm_llm = mm_llm
