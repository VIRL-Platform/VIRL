PARSE_ANSWER_TO_JSON = """
Please parse the following answer according to the question:

Question: {question}

Answer: {answer}

You should answer in the following JSON format: 
{{"answer": "Yes or No or Uncertain", "explanation": "reason on the answer"}}
"""
