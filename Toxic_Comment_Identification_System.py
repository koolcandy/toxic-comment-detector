import os
from utils import ShowToxicLevel, ShowIsToxic, GetResultbybert, ToxicDectorllama8b
from textblob import TextBlob
from flask import Flask, render_template, request
# encoding:utf-8

base_dir = os.path.dirname(os.path.abspath(__file__))

def get_result_en(words):
    isToxic = ShowIsToxic.getResult(TextBlob(words))
    toxicLevel = ShowToxicLevel.getResult(TextBlob(words))
    result = {**isToxic, **toxicLevel}
    result['isToxic'] = False if result['isToxic'] == 1 else True
    return result
app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    if  (request.args.get("apiEndpoint") == "localmodel(LSTM)"):
        comment = request.args.get("comment")
        result = get_result_en(comment)
        print(result)
        return render_template('ShowEnResult.html', result=result, comment=comment)
    if  (request.args.get("apiEndpoint") == "localmodel(BERT)"):
        comment = request.args.get("comment")
        result = GetResultbybert.getResult(comment)
        result['isToxic'] = False if result['isToxic'] == 1 else True
        print(result)
        return result
    if  (request.args.get("apiEndpoint") == "llama"):
        comment = request.args.get("comment")
        result = ToxicDectorllama8b.main(comment)
        print(result)
        return render_template('Resultllama.html', result=result)
    else:
        return "Error method!!!"

@app.route("/")
def home():
    return render_template("Main.html")

if __name__ == '__main__':
    if not os.path.exists(os.path.join(base_dir, 'model')):
        print('Local models not founded, start raining model...')
        ShowToxicLevel.train()
        print('Local models trained successfully.')
    app.run(debug=True)