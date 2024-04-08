intention_driven_choice_answer_template = """
[Task Description]
You will be provided with a place and related attributes. You need to generate four candidate answers with only ONE true answer for a choice question considering the provided information.

Output format:
{{"A": "", "B": "", "C": "", "D": "", "true": ""}}
(Each candidate answer should be in length 15 - 40 words.)

[Input]
question: {question}
place: {place_name}.
attributes: {attributes}.

[Output]
"""
