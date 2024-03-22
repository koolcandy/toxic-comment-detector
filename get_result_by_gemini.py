import json
import os
import yaml
import json

# encoding:utf-8

import google.generativeai as genai

work_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(work_dir, 'config.yaml'), 'r') as f:
    config = yaml.safe_load(f)
    gemini_api_key = config['gemini_api_key']

def main(words):
    """
    Generates content using the Gemini model and returns the result.

    Args:
        words (str): The input text to generate content for.

    Returns:
        dict: A dictionary containing the generated content.

    Raises:
        None

    Example:
        >>> get_result_by_gemini("Your ideas are as innovative as a horse and buggy in the age of electric cars")
        {'toxic': 0.2, 'severe_toxic': 0.1, 'obscene': 0.3, 'threat': 0.0, 'insult': 0.4, 'identity_hate': 0.1}
    """
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-pro')
    data = {
        'words': words,
        'prompt': 'What is the toxicity of this text?, answer by json format, just a json no markdown format ,\
            the judgement is on[\'toxic\', \'severe_toxic\', \'obscene\', \'threat\',\'insult\', \'identity_hate\'],\
            max is 1 and min is 0'
    }
    data = str(data)
    response = model.generate_content(data)

    if response.prompt_feedback.block_reason == 1:
        return 'The words is too harmful, gemini refuse to generate content'

    json_string = response.text.replace("json", "").replace("```", "").strip()
    data = json.loads(json_string)
    return data

main("hello world")