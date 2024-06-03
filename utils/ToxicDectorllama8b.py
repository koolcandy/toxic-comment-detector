import requests

def main(comment):
    url = "http://localhost:11434/api/generate"

    data = {
        "model": "llama3:8b",
        "prompt": (
            "I want you to act as an API endpoint for detecting toxicity in a given text in <<<>>>. For a provided text, "
            "I expect you to return a JSON object with the following keys: \"isToxic\": A boolean value indicating whether "
            "the text contains toxicity. It should be `true` if toxicity is present, and `false` otherwise. \"ToxicSort\": "
            "A string array containing the types of toxicity detected. Possible types include: `\'obscene\'`, `\'threat\'`, "
            "`\'insult\'`, `\'identity_hate\'`. This array should be empty if `IsToxic` is `false`. Ensure to return plain "
            f"JSON <<<{comment}>>>"
        ),
        "format": "json",
        "stream": False
    }

    response = requests.post(url, json=data)

    return(response.json()['response']).replace(" ", "").replace("\n", "")

if __name__ == "__main__":
    print(main("大ばか、死ね、地獄に落ちろ"))
    print(main("大笨蛋,去死吧,下地狱吧"))
    print(main("바보야, 죽어라, 지옥에 가라"))


