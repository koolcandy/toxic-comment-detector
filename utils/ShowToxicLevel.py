import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

tqdm.pandas()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = word_tokenize(text)
    words = [word for word in words if word not in stopwords.words('english')]
    text = ' '.join(words)
    return text

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def train():
    df_train = pd.read_csv(os.path.join(base_dir, 'datas', 'train.csv'))

    df_train['comment_text'] = df_train['comment_text'].progress_apply(lambda text : clean_text(text))

    cols = ['obscene', 'threat','insult', 'identity_hate']

    X = df_train['comment_text']
    
    # 创建一个具有指定参数的TfidfVectorizer对象
    tfd = TfidfVectorizer(stop_words='english',max_features=5000)

    accuracies = []
    
    # 将输入数据转换为TF-IDF矩阵
    X_data = tfd.fit_transform(X)

    # 创建一个具有指定参数的LogisticRegression对象
    lr = LogisticRegression(C=12,max_iter=500)

    for label in cols:
        y = df_train[label]
        X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.2, random_state=42)
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy*100)
        print(f'dataset accuracy for {label} is {accuracy_score(y_test, y_pred):.2f}')


        joblib.dump(lr, os.path.join(base_dir,'model', 'toxiclevel', f'{label}_model.pkl'))
        joblib.dump(tfd, os.path.join(base_dir,'model', 'toxiclevel', 'tfidf_vectorizer.pkl'))
    

    bars = plt.bar(cols, accuracies)
    plt.ylim(0, 100)
    plt.bar(cols, accuracies)
    plt.xlabel('Labels')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy for each label', y=1.05)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom')

    plt.savefig('accuracy.png')

def getResult(comment):
    
    tfd = joblib.load(os.path.join(base_dir, 'model', 'toxiclevel', 'tfidf_vectorizer.pkl'))

    cols = ['obscene', 'threat', 'insult', 'identity_hate']

    cleaned_comment = clean_text(comment)
    comment_data = tfd.transform([cleaned_comment])

    predictions = {}
    for label in cols:
        lr = joblib.load(os.path.join(base_dir, 'model', 'toxiclevel', f'{label}_model.pkl'))
        predictions[label] = round(lr.predict_proba(comment_data)[:,1].tolist()[0], 2)
    return predictions

if __name__ == '__main__':
    train()