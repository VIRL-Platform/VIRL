intention_driven_qa_template = """
Q: {question} Choices: A. {answer_a}; B. {answer_b}; C. {answer_c}; D. {answer_d}
"""


intention_driven_qa_yesno_template = """
Dose following human intention can be achieved in this image? Intention: {intention}. Answer yes or no:
"""


match_mm_llm_answer_template = """
You are an AI assistant to help me matching an answer with several options of a multiple choice question.
You are provided with a question, several options, and an answer,
and you need to find which option is most similar to the answer.
If the meaning of all options are significantly different from the answer, output X.
Your should output a single uppercase character in A, B, C, D (if they are valid options), and X.
Example 1:
Question: What is the main object in image? Options: A. teddy bear B. rabbit C. cat D. dog
Answer: a cute teddy bear
Your output: A
Example 2:
Question: What is the main object in image? Options: A. teddy bear B. rabbit C. cat D. dog
Answer: Spider
Your output: X
Example 3:
Question: {question}? Options: {options}
Answer: {prediction}
Your output:
"""


match_mm_llm_answer_yesno_template = """
[Role]
You are ParseGPT. Your task is to parse the raw answer to yes, no or unknown.

[Input]
Question: {question}
Raw Answer: {prediction}

[Ouput]
Your should answer me in json: {{"ans": "parsed answer in yes/no/unknown"}}
"""


ocr_result_to_recognition_template = """
[Role]
You are DecisionGPT. Your should decide whether a place is in the image according to the recognized text from OCR model.

[Input]
Recognized texts list:
{text_list}

The place name:
{place_name}

[Output]:
You should answer in json format: {{"answer": "Yes/No", "refer", "the text you refer to"}}
"""
