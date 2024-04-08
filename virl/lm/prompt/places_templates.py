PLACE_RATING_TEMPLATE = """
[Task Description]
Your task is to rate the places candidates delimited by triple backticks according to the human background and intention delimited by <>.

[Input]
Human background: <{background}>
Human intention: <{intention}>
Place candidates: ```{place_intro}```

[Output]
You should only respond in JSON format as described below:
{{"rating": "A value range in [0, 10], larger is better", "explanation": ""}}
"""


PLACE_INTRO_TEMPLATE = """
Your task is to give the introduction of the place candidates considering the provided human intention delimited by <>.

Human intention: <{intention}>

Place name: ```{place_name}```

You should only respond in JSON format as described below:
{{"intro": "Text within 50 words."}}
"""

QUESTION_TO_PLACE_TEMPLATE = """
You are QueryGPT. Your task is to identify the specific location or type of place that should be searched for on Google Maps based on the user-provided question.

Question: <{question}>

Please respond only in the JSON format as outlined below:
{{"answer": "A type of place or a specific location"}}
"""


SINGLE_PLACE_REVIEW_TEMPLATE = """
Review {idx}: <{review} (Rating: {rating})>.
"""

SINGLE_INTERSECTION_GIVEN_DIRECTION_TEMPLATE = """
{{
"intersection_idx": "{idx}",
"landmarks": "{landmarks}",
"distance": "{distance} meters",
"human_heading": "{human_heading}",
"to_next_intersection_heading": "{to_milestone_heading} ({to_milestone_direction})",
}}
"""

REAL_ESTATE_INTENTION_TO_PLACE_TEMPLATE = """
[Role]
You are an estate agent Vivek. Your task is to parse the user's estate requirements to specified place type that the user cared.

[Input]
User's estate requirements: <{estate_requirement}>

[Output]
You should answer with a list a place type: ["place type 1", "place type 2", ...]
"""


RATE_ESTATE_TEMPLATE = """
[Role]
You are an estate agent Vivek. Your task is to rate the estate candidates according to the provided human background and intention delimited by <>.

[Input]
Human background: <{background}>

Human requirement: <{requirement}>

Estate information: {estate_info}

[Output]
You should only respond in JSON format as described below:
{{"rating": "A single value in [0, 10], higher is better", "explanation": "Text explanation within 50 words."}}
"""

ESTATE_FINAL_RECOMMEND_TEMPLATE = """
[Role]
You are an estate agent Vivek. Your task is to recommend a the estate among all candidates according to the provided human background and intention delimited by <>.

[Input]
Human background: <{background}>

Human requirement: <{requirement}>

Estate candidate information:
{all_estate_info}

[Output]
You should select the most suitable one and answer in JSON format:
{{"idx": "the index of final ", "reason": "explanation."}}
"""

ESTATE_INFO_TEMPLATE = """
{{
   "address": {address},
   "price": {price},
   "property type": {property_type},
   "square footage": {size},
   "bedrooms": {bedrooms},
   "bathrooms": {bathrooms},
   "year built": {year_built},
   "nearby places with {nearby_radius} meters": {nearby_info},
}}
"""

SINGLE_ESTATE_CANDIDATE_TEMPLATE = """
Estate candidate {idx}: {estate_info}
"""


LANGUAGE_INSTRUCTION_NAVIGATION_GIVEN_DIRECTION_TEMPLATE = """
[Role]
You are DirectionGPT. Your mission is to provide clear, natural language directions to guide someone through a series of intersections based on provided details about each crossroad and the person's orientation. You should mention direction in 

[Input Format]
The input will be a list in JSON format containing information about each intersection:
{{
"intersection_idx": "Index of the intersection",
"landmarks": "Landmarks near this intersection described in relation to its position",
"distance": "Distance from this intersection to the previous one",
"human_heading": "Current orientation of the person, ranging from 0--360 (0 is north, 90 is east, 180 is south, 270 is west). Turning right increases the heading, and turning left decreases it.",
"to_next_intersection_heading": "Similar to `human_heading`. The direction from the person's current position toward this intersection"
}}

[Task Description]
You should provide clear, natural language directions. Avoid mentioning specific degree values or exact distances.

[Examples]
Example Input:
{{
"intersection_idx": "1",
"landmarks": "No landmarks nearby",
"distance": "72 meters",
"human_heading": "90",
"to_next_intersection_heading": "25 (northeast)"
}}

{{
"intersection_idx": "2",
"landmarks": "Starbucks is on your right",
"distance": "33 meters",
"human_heading": "25",
"to_next_intersection_heading": "0 (north)"
}}

{{
"intersection_idx": "3",
"landmarks": "Walmart is on your left front",
"distance": "20 meters",
"human_heading": "0",
"to_next_intersection_heading": "315 (northwest)"
}}

Example Output:
First, turn slightly right towards the northeast, and walk straight until you reach the next intersection.
Next, turn left and continue due north until the next intersection with a Starbucks on the right.
Finally, turn left to face northwest and walk straight just a little further until you see the destination Walmart on your left front.

[Input]
{milestone_information}

[Output]
"""


NAVIGATION_LANGUAGE_TO_WAYPOINT_LIST = """
[Role]
You are ParseGPT. Your task is to parse the user's natural language to all place addresses that the user want to go.

[Input]
User query: <{waypoint_query}>

[Output]
You should answer with a list a address: ["address 1", "address 2", ...]
"""


INTENTION_TO_PLACE_TEMPLATE = """
[Role]
You are PlaceSuggesterGPT, an expert in recommending types of places based on user-specified intentions.

[Task Description]
Given a specified intention, determine the type of "place" one should seek. For instance, for the intention to eat lunch, the recommended type is "restaurant". You must respond in the following JSON format:
{{"place": "Desired Place Type"}}

[Example]
For the intention "buy a book", the output might be:
{{"place": "bookstore"}}

[Input]
Intention: {intention}

[Output]
Your recommended place type based on the user-specified intention, in the specified JSON format:
"""

RANDOM_EXPLORE_ACTION_TEMPLATE = """
[Role]
You are ActionSelectorGPT, proficient in choosing the most appropriate action based on a user's background, intention, and current condition.

[Task Description]
Given the user's background, intention, and condition, you need to determine the most suitable action from a predefined list. Respond in the following format:
{{"action": "Selected Action", "reason": "Justification for the chosen action"}}

Possible actions:
  - enter_place(): Enter the designated place. This should only be selected if you are sure the place is a great match for the user.
  - search_info(): Retrieve information about the place. This is a great option to help learn more about the place before committing to entering it.
  - continual_find(): Continue searching for the next fitting place. This is a good option if you think the current place is not a good match.

[Example]
For the background "coffee afficionado", intention "find a good espresso", and condition "standing in front of a coffee shop", the output might be:
{{"action": "search_info()", "reason": "The user is a coffee afficionado, so it's worth making sure the coffee shop has good reviews on their espresso before entering."}}

[Input]
Background: {background}

Intention: {intention}

Condition: {condition}

[Output]
Your chosen action and the reasoning behind your decision in the prescribed JSON format:
"""

PLACE_REVIEW_SUMMARIZE_TEMPLATE = """
[Role]
You are SummarizeGPT, skilled at condensing multiple reviews into a concise introduction for a location.

[Task Description]
Given multiple reviews with ratings, craft a brief overview of the place. Your response should be in the following JSON format:
{{"summarization": "Concise description (limited to 80 words)"}}

[Example]
For reviews "Great ambiance but average food (Rating: 3)" and "Loved the decor, food could be better (Rating: 3.5)", the output might be:
{{"summarization": "The place boasts great ambiance and decor, but the food quality receives mixed reviews."}}

[Input]
Reviews: {all_reviews}

[Output]
Your concise overview (max 80 words) based on the provided reviews, in the prescribed JSON format:
"""

RANDOM_EXPLORE_ACTIONS_WITH_INTRO_TEMPLATE = """
[Role]
You are ActionSelectorGPT, proficient in choosing the most appropriate action based on a user's background, intention, and an overview of a place.

[Task Description]
Evaluate the provided user background, intention, and place overview to select the most suitable action from the list. Your response should be in the following JSON format:
{{"action": "Selected Action", "reason": "Justification for your choice"}}

Possible actions:
   - enter_place(): Enter the designated place.
   - continual_find(): Continue searching for another appropriate place.

[Example]
For the background "loves historical sites", intention "discover local history", and place overview "This is a 200-year-old preserved mansion", the output might be:
{{"action": "enter_place()", "reason": "The historical mansion aligns with the user's interest in historical sites."}}

[Input]
Background: <{background}>
Intention: {intention}
Place Overview: <{intro}>

[Output]
Your chosen action and the rationale behind your decision in the prescribed JSON format:
"""

LOCAL_SELECT_PLACE_TEMPLATE = """
[Role]
You are PlaceSelectorGPT, an expert in selecting the most suitable place based on personal knowledge.

[Tasks Description]
Your mission is to select the most suitable place according to your background knowledge and massive place informations.
For each place candidate, you will be provided with the place name.

[Input]
Background knowledge: <{background}>

Place candidates: {place_candidates}

[Output]
You should recommend according to the background knowledge and place information and answer in JSON format as described below:
{{"name": "The name of the selected place", "reason": "Explanation for your choice"}}
"""
