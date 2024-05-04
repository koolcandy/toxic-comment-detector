import os
from utils import train_model
from utils import get_result_by_local_model
from textblob import TextBlob
from flask import Flask, render_template, request
# encoding:utf-8

base_dir = os.path.dirname(os.path.abspath(__file__))

def get_result(words):

    print('Local bad words not matched, start to use local model to predict...')
    result = get_result_by_local_model.main(TextBlob(words))
    # print('Local model result: %s' % result)

    return result

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    comment = request.form["comment"]
    result = get_result(comment)  # 使用您现有的 get_result 函数
    return render_template('showresult.html', result=result, comment=comment)

@app.route("/")
def home():
    return render_template("main.html")

if __name__ == '__main__':
    if not os.path.exists(os.path.join(base_dir, 'model')):
        print('Local models not founded, start raining model...')
        train_model.main()
        print('Local models trained successfully.')
    app.run(debug=True)
#驱动代码