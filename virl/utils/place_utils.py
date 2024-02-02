from virl.utils import geocode_utils
from virl.lm import prompt as prompt_templates


def calculate_milestone_information(landmarks, agent_heading):
    if len(landmarks) > 0 and landmarks[0] is not None:
        landmark_info = landmarks[0]
        heading_to = landmark_info['heading']
        landmark_info['spatial_relation'] = geocode_utils.calculate_spatial_relationship_with_headings(
            agent_heading, heading_to
        )
        landmark_info['expression'] = f"{landmark_info['name']} is on your {landmark_info['spatial_relation']}"
        return landmark_info
    else:
        return {'expression': 'No landmark nearby'}


def fill_in_milestone_information_template(landmark_info, milestone_idx, agent_heading, distance,
                                           heading_to_milestone, prompt_name):
    milestone_info_template = getattr(prompt_templates, prompt_name)
    
    to_milestone_direction = geocode_utils.get_direction_abs_by_heading(int(heading_to_milestone))
    
    milestone_info = milestone_info_template.format(
        idx=milestone_idx + 1, landmarks=landmark_info['expression'], distance=distance,
        human_heading=int(agent_heading), to_milestone_heading=int(heading_to_milestone),
        to_milestone_direction=to_milestone_direction
    )

    return milestone_info
