import joblib
import pandas as pd
from cleantext import clean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

def clean_text(text):
    return clean(text,
        fix_unicode=True,               
        to_ascii=True,                  
        lower=True,                     
        no_line_breaks=False,           
        no_urls=False,                  
        no_emails=False,                
        no_phone_numbers=False,         
        no_numbers=False,               
        no_digits=False,                
        no_currency_symbols=False,      
        no_punct=False,                 
        replace_with_punct="",          
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang="en"                       
    )

def main():
    df_train = pd.read_csv('/Users/hitt/Documents/crtical skill project/datas/train.csv')
    df_train['comment_text'] = df_train['comment_text'].apply(lambda text : clean_text(text))

    cols = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']

    X = df_train['comment_text']
    tfd = TfidfVectorizer(stop_words='english',max_features=5000)
    X_data = tfd.fit_transform(X)

    lr = LogisticRegression(C=12,max_iter=500)

    # 创建/model目录，如果它不存在
    if not os.path.exists('model'):
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model'))

    # 训练模型并保存
    for label in cols:
        y_train = df_train[label]
        lr.fit(X_data,y_train)
        joblib.dump(lr, os.path.join(os.path.dirname(os.path.abspath(__file__)),'model', f'{label}_model.pkl'))

    # 保存 TfidfVectorizer
    joblib.dump(tfd, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model', 'tfidf_vectorizer.pkl'))

