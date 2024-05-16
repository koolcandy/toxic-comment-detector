import torch
from transformers.models.bert import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('thu-coai/roberta-base-cold')
model = BertForSequenceClassification.from_pretrained('thu-coai/roberta-base-cold')
model.eval()

def getResult(comment):
    model_input = tokenizer(comment,return_tensors="pt",padding=False)
    model_output = model(**model_input, return_dict=True)
    prediction = torch.argmax(model_output[0].cpu(), dim=-1)
    prediction = [p.item() for p in prediction]
    print(prediction[0]) 

if __name__ == '__main__':
    getResult("你是个傻逼")