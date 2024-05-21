import os
from utils import ShowToxicLevel, ShowIsToxic, ColdChinese
from textblob import TextBlob
from flask import Flask, render_template, request
from langdetect import detect
# encoding:utf-8

base_dir = os.path.dirname(os.path.abspath(__file__))

def get_result_en(words):
    isToxic = ShowIsToxic.getResult(TextBlob(words))
    toxicLevel = ShowToxicLevel.getResult(TextBlob(words))
    result = {**isToxic, **toxicLevel}
    return result

def get_result_cn(words):
    return ColdChinese.getResult(words) #直接用huggingface的模型, 不需要训练, I'm lazy ~~~

app = Flask(__name__)

@app.route("/predict", methods=["GET"])
def predict():
    comment = request.args.get("comment")
    lang = detect(comment)
    if lang == 'en':
        result = get_result_en(comment)
        return render_template('ShowEnResult.html', result=result, comment=comment)
    elif lang == 'zh-cn':
        result = get_result_cn(comment)
        return render_template('ShowCnResult.html', result=result['isToxic'], comment=comment)
    else:
        return 'Error: Language detection failed'

@app.route("/")
def home():
    return render_template("Main.html")

if __name__ == '__main__':
    if not os.path.exists(os.path.join(base_dir, 'model')):
        print('Local models not founded, start raining model...')
        ShowToxicLevel.train()
        print('Local models trained successfully.')
    app.run(debug=True)
    # print(get_result("This is a test comment."))