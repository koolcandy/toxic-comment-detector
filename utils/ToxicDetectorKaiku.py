import requests
import json
import json

with open('config.json') as f:
    key = json.load(f)['key']

def detect_toxicity(comment):
    url = 'https://chatapi.nloli.xyz/v1/chat/completions'
    headers = {
        'authorization': f'Bearer {key}',
        'content-type': 'application/json'
    }
    data = {
        "messages":[
            {"role":"user","content": "I want you to act as an API endpoint for detecting toxicity in a given text.\
            For a provided text, I expect you to return a JSON object with the following keys: \"IsToxic\": A boolean \
            value indicating whether the text contains toxicity. It should be `true` if toxicity is present, and `false` \
            otherwise. \"ToxicSort\": A string array containing the types of toxicity detected. Possible types include: `\
            'obscene'`, `'threat'`, `'insult'`, `'identity_hate'`. This array should be empty if `IsToxic` is `false`.\
            Ensure to return plain JSON, not a JSON formatted within markdown, The returned answer does not require a line break."},
            {"role":"user","content": comment},],
        "stream": False,
        "model": "claude-3-haiku",
        "temperature": 0.5,
        "presence_penalty": 0.8,
        "frequency_penalty": 0.4,
        "top_p": 1
    }

    response = requests.post(url, headers=headers, json=data)

    result = json.loads(response.json()["choices"][0]["message"]["content"])

    return result