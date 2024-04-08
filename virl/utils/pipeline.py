from random import sample

from virl.platform.platform import Platform
from virl.lm import UnifiedChat
from virl.ui.messager import Messager
from virl.agents import build_agent
from virl.utils import common_utils, geocode_utils
from virl.lm import prompt as prompt_templates

from virl.platform.file_template import get_file_template_by_name


def init_world_and_agent(cfg, output_dir):
    platform = Platform(cfg.PLATFORM, output_dir)

    agent = build_agent(cfg.AGENT)

    chatbot = UnifiedChat()

    messager = Messager(cfg.UI)

    if cfg.get('TASK_INFO', None) and cfg.TASK_INFO.get('AGENT', None):
        custom_agent = build_agent(cfg.TASK_INFO.AGENT)
    else:
        custom_agent = None

    return platform, agent, chatbot, messager, custom_agent


def location_to_geocode(platform, address, version='v2', relocate=False):
    if version == 'v1':
        geocode = platform.get_geocode_from_address(address)
    elif version == 'v2':
        geocode = platform.get_geocode_from_address_v2(address)
    else:
        raise NotImplementedError

    if relocate:
        geocode, _ = platform.relocate_geocode_by_source(geocode, source='outdoor')
    return geocode


def position_to_geocode(platform, position):
    if isinstance(position, str):
        geocode = location_to_geocode(
            platform, position, version='v2', relocate=True
        )
        if geocode is None:
            geocode = location_to_geocode(platform, position, version='v2', relocate=False)
    elif isinstance(position, tuple) or isinstance(position, list):
        geocode, _ = platform.relocate_geocode_by_source(
            position, source='outdoor'
        )
    else:
        raise NotImplementedError

    return geocode


def intention_to_place(pipeline_cfg, agent, chatbot, args):
    if args.custom_place != 'None':
        place = args.custom_place
    else:
        intention_prompt_template = getattr(
            prompt_templates, pipeline_cfg.PROMPT
        )
        prompt = intention_prompt_template.format(intention=agent.intention)
        answer = chatbot.ask(prompt, json=True)
        place = answer['place']

    return place


def search_intro(pipeline_cfg, agent, chatbot, candidates):
    need_distance = pipeline_cfg.get('NEED_DISTANCE', True)
    place_prompt_template = getattr(prompt_templates, pipeline_cfg.PROMPT)
    for i, candidate in enumerate(candidates):
        print(f'>>> Search intro of the {i}th candidate:')
        place_name = candidate['name']
        place_prompt = place_prompt_template.format(
            intention=agent.intention, place_name=place_name, city_name=agent.city
        )
        answer_json = chatbot.search(place_prompt, json=True)
        intro = answer_json['intro']
        candidate['intro'] = intro
        if need_distance:
            candidate['intro'] += f" (Distance: {int(candidate['distance'])} meters)."
        common_utils.print_answer(candidate['intro'])

    print(f'>>> Finish Searching introduction.')
    return candidates


def query_place_in_the_google_map_single(pipeline_cfg, platform, box, street_image, place):
    radius = pipeline_cfg.RADIUS
    heading_epsilon = pipeline_cfg.get('HEADING_EPS', 0.0)

    heading_left, heading_right = geocode_utils.get_heading_range_to_box(
        box, street_image.shape, street_image.heading, street_image.fov
    )
    
    # heading, pitch, _, _ = geocode_utils.get_heading_pitch_fov_to_box(
    #     box, street_image.shape, street_image.heading, street_image.pitch, street_image.fov
    # )

    geocode, _ = platform.relocate_geocode_by_source(street_image.geocode, source='outdoor')
    place_list = platform.get_nearby_places(
        geocode, rankting_type='distance', relocated=False, rankby='distance', type=place, language='en',
    )

    # filter place list
    final_result = None
    for place in place_list:
        if place['distance'] < radius and geocode_utils.is_heading_in_range((heading_left, heading_right), place['heading'], heading_epsilon):
            final_result = place
            break
    
    return final_result


def review_to_intro(pipeline_cfg, platform, chatbot, candidates):
    summarization_template = getattr(prompt_templates, pipeline_cfg.SUMMARIZE_PROMPT)
    single_review_template = getattr(prompt_templates, pipeline_cfg.REVIEW_PROMPT)
    need_distance = pipeline_cfg.get('NEED_DISTANCE', True)

    for candidate in candidates:
        place_id = candidate['place_id']
        review_list = platform.get_place_reviews(place_id)

        # sample reviews from the review list
        if len(review_list) > pipeline_cfg.N_REVIEW:
            review_list = sample(review_list, pipeline_cfg.N_REVIEW)

        # fill in all reviews
        all_reviews = ""
        for i, review in enumerate(review_list):
            review_prompt = single_review_template.format(
                idx=i, review=review['text'], rating=review['rating']
            )
            all_reviews += review_prompt

        # ask LLM to summarize
        summarization_prompt = summarization_template.format(all_reviews=all_reviews)
        print(f'>>> Summarize the reviews of {candidate["name"]}...')
        answer_json = chatbot.ask(summarization_prompt, json=True)
        intro = answer_json['summarization']
        candidate['intro'] = intro
        if need_distance:
            candidate['intro'] += f" (Distance: {int(candidate['distance'])} meters)."

        candidate['intro'] = intro

    return candidates


def draw_planned_route(platform, polyline_list, input_way_points, path, 
                       file_template, cur_position=None):
    if input_way_points is not None:
        show_polyline_and_waypoints_on_the_map(
            platform, polyline_list, input_way_points, path, file_template, cur_position=cur_position
        )
    else:
        show_polyline_on_the_map(platform, polyline_list, path, file_template)


def show_polyline_and_waypoints_on_the_map(platform, polyline, waypoints, path,
                                           file_template, cur_position=None):
    if isinstance(polyline, list):
        polyline = geocode_utils.merge_polylines(polyline)

    waypoints = geocode_utils.encode_polyline(waypoints)
    with open(path, 'w') as file_html:
        content_template = get_file_template_by_name(file_template)
        file_content = content_template.format(
            polyline=polyline.replace("\\", "\\\\"),
            waypoints=waypoints.replace("\\", "\\\\"),
            lat=cur_position[0] if cur_position is not None else 0.0,
            lng=cur_position[1] if cur_position is not None else 0.0,
            key=platform.key
        )
        file_html.write(file_content)


def show_polyline_on_the_map(platform, polyline, path, file_template):
    if isinstance(polyline, list):
        polyline = geocode_utils.merge_polylines(polyline)

    with open(path, 'w') as file_html:
        content_template = get_file_template_by_name(file_template)
        file_content = content_template.format(
            polyline=polyline.replace("\\", "\\\\"),
            key=platform.key
        )
        file_html.write(file_content)
