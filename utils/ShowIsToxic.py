import joblib
import pandas as pd
from cleantext import clean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

from sklearn.model_selection import train_test_split

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def train():
    df_train = pd.read_csv(os.path.join(base_dir, 'datas', 'toxic_content.csv'))
    df_train['comment_text'] = df_train['comment_text'].apply(lambda text : clean_text(text))

    X = df_train['comment_text']
    y = df_train['toxic']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 创建一个具有指定参数的TfidfVectorizer对象
    tfd = TfidfVectorizer(stop_words='english', max_features=5000)

    # 将训练数据转换为TF-IDF矩阵
    X_train_tfd = tfd.fit_transform(X_train)
    X_test_tfd = tfd.transform(X_test)

    # 创建一个具有指定参数的LogisticRegression对象
    lr = LogisticRegression(C=12, max_iter=500)

    lr.fit(X_train_tfd, y_train)

    accuracy = lr.score(X_test_tfd, y_test)
    print(f"dataset accuracy is: {accuracy:.2f}") 

    joblib.dump(lr, os.path.join(base_dir,'model', 'istoxic', 'toxic_model.pkl'))
    joblib.dump(tfd, os.path.join(base_dir,'model', 'istoxic', 'tfidf_vectorizer.pkl'))

def getResult(text):
    result = {}
    text = clean_text(text)
    tfd = joblib.load(os.path.join(base_dir,'model', 'istoxic', 'tfidf_vectorizer.pkl'))
    text = tfd.transform([text])
    model = joblib.load(os.path.join(base_dir,'model', 'istoxic', 'toxic_model.pkl'))
    result['isToxic'] = model.predict(text)[0]
    result['probability'] = round(max(model.predict_proba(text).tolist()[0]), 2)
    return result

if __name__ == '__main__':
    train()