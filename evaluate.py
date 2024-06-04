import random
from utils import ShowIsToxic, GetResultbybert, ToxicDectorllama8b
import pandas
from tqdm import tqdm
import json

def main():
    data = pandas.read_csv('datas/toxic_content.csv')
    comment = data['comment_text']
    score = data['toxic']

    LSTM_score = 0
    BERT_score = 0
    cout = 0
    for i in tqdm(range(1, data.shape[0], 100)):
        cout += 1
        if ShowIsToxic.getResult(comment[i])['isToxic'] == score[i]:
            LSTM_score += 1           
            
        if GetResultbybert.getResult(comment[i])['isToxic'] == score[i]:
            BERT_score += 1
               
    print(f'\rLSTM model accuracy: {LSTM_score/cout}, BERT model accuercy: {BERT_score/cout} ', end='')

    import matplotlib.pyplot as plt

    lstm_accuracy = LSTM_score / cout * 100
    bert_accuracy = BERT_score / cout * 100

    labels = ['LSTM', 'BERT']
    accuracy = [lstm_accuracy, bert_accuracy]
    bars = plt.bar(labels, accuracy)
    plt.ylim(0, 100)
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 2), va='bottom')  # va: vertical alignment

    plt.savefig('accuracy_plot.png')
    plt.show()

if __name__ == '__main__':
    main()