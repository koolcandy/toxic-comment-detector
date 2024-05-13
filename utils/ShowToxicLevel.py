import joblib
import pandas as pd
from cleantext import clean
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os


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
    df_train = pd.read_csv(os.path.join(base_dir, 'datas', 'train.csv'))
    df_train['comment_text'] = df_train['comment_text'].apply(lambda text : clean_text(text))

    cols = ['severe_toxic', 'obscene', 'threat','insult', 'identity_hate']

    X = df_train['comment_text']
    
    # 创建一个具有指定参数的TfidfVectorizer对象
    tfd = TfidfVectorizer(stop_words='english',max_features=5000)
    
    # 将输入数据转换为TF-IDF矩阵
    X_data = tfd.fit_transform(X)

    # 创建一个具有指定参数的LogisticRegression对象
    lr = LogisticRegression(C=12,max_iter=500)

    for label in cols:
        y = df_train[label]
        X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=42)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        print(f'dataset accuracy for {label} is {accuracy_score(y_test, y_pred):.2f}')

        joblib.dump(lr, os.path.join(base_dir,'model', 'toxiclevel', f'{label}_model.pkl'))
        joblib.dump(tfd, os.path.join(base_dir,'model', 'toxiclevel', 'tfidf_vectorizer.pkl'))

def getResult(comment):
    
    tfd = joblib.load(os.path.join(base_dir, 'model', 'toxiclevel', 'tfidf_vectorizer.pkl'))

    cols = ['severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    cleaned_comment = clean_text(comment)
    comment_data = tfd.transform([cleaned_comment])

    predictions = {}
    for label in cols:
        lr = joblib.load(os.path.join(base_dir, 'model', 'toxiclevel', f'{label}_model.pkl'))
        predictions[label] = round(lr.predict_proba(comment_data)[:,1].tolist()[0], 2)
    return predictions

if __name__ == '__main__':
    train()