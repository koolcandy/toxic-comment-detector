import os
from utils import ShowToxicLevel, ShowIsToxic, ToxicDetectorKaiku
from textblob import TextBlob
from flask import Flask, render_template, request
# encoding:utf-8

base_dir = os.path.dirname(os.path.abspath(__file__))

def get_result_en(words):
    isToxic = ShowIsToxic.getResult(TextBlob(words))
    toxicLevel = ShowToxicLevel.getResult(TextBlob(words))
    result = {**isToxic, **toxicLevel}
    if result['isToxic'] == 1:
        result['isToxic'] = 'yes'
    else:
        result['isToxic'] = 'no'
    return result
app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    if  (request.args.get("apiEndpoint") == "localmodel"):
        comment = request.args.get("comment")
        result = get_result_en(comment)
        print(result)
        return render_template('ShowEnResult.html', result=result, comment=comment)
    else:
        comment = request.args.get("comment")
        result = ToxicDetectorKaiku.detect_toxicity(comment)
        print(result)
        return render_template('Resulthaiku.html', result=result)

@app.route("/")
def home():
    return render_template("Main.html")

if __name__ == '__main__':
    if not os.path.exists(os.path.join(base_dir, 'model')):
        print('Local models not founded, start raining model...')
        ShowToxicLevel.train()
        print('Local models trained successfully.')
    app.run(debug=True)