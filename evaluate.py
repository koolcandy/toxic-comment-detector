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
    llama_score = 0
    for i in range(1, data.shape[0]):
        # IsBertRight = False
        # IsLstmRight = False
        if ShowIsToxic.getResult(comment[i])['isToxic'] == score[i]:
            LSTM_score += 1
            # IsLstmRight = True
            
            
        if GetResultbybert.getResult(comment[i])['isToxic'] == score[i]:
            BERT_score += 1
            # IsBertRight = True

        try:
            if json.loads(ToxicDectorllama8b.main(comment[i]))['isToxic'] == (False if score[i] == 1 else True):
                llama_score += 1
        except Exception as e:
            print(f"An error occurred: {e}")
            continue
            
               
        print(f'\rLSTM model accuracy: {LSTM_score/i}, BERT model accuercy: {BERT_score/i}, llama model accuercy: {llama_score/i} ', end='')
        
if __name__ == '__main__':
    main()