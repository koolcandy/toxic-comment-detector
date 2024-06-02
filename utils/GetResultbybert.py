import os
from cleantext import clean
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = ''
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
    
device = torch.device(device)
print(f'Using device: {device}')

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def train():
    df_train = pd.read_csv(os.path.join('toxic_content.csv'))
    df_train['comment_text'] = df_train['comment_text'].apply(lambda text: clean(text))

    X = df_train['comment_text']
    y = df_train['toxic']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 加载预训练的BERT模型和tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(device)  # 将模型移动到GPU上

    # 对数据进行tokenization
    X_train_encoded = tokenizer.batch_encode_plus(list(X_train.values), padding=True, truncation=True, max_length=512, return_tensors='pt')
    X_test_encoded = tokenizer.batch_encode_plus(list(X_test.values), padding=True, truncation=True, max_length=512, return_tensors='pt')

    # 将数据移动到GPU上
    X_train_encoded = {k: v.to(device) for k, v in X_train_encoded.items()}
    X_test_encoded = {k: v.to(device) for k, v in X_test_encoded.items()}
    y_train = torch.tensor(y_train.values, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.long).to(device)

    # 创建数据集
    train_dataset = TextDataset(X_train_encoded, y_train)
    test_dataset = TextDataset(X_test_encoded, y_test)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    # 训练模型
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    model.train()
    
    loss_dict = {}
    epoch = 0

    while ( 1 ):
        for batch in tqdm(train_loader):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch += 1
        loss_dict[epoch+1] = loss.item()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                outputs = model(**batch)
                accuracy = (outputs.logits.argmax(dim=-1) == batch['labels']).float().mean()
            print(f'Accuracy for Epoch {epoch+1}: {accuracy.item()}')

            if (accuracy.item() >= 0.99) :
                print("accuracy is ok now, save the model")
                torch.save(model.state_dict(), 'model.pth')
                break



    epochs = list(loss_dict.keys())
    loss_values = list(loss_dict.values())

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_values, marker='o')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('loss_plot.png')

def getResult(text):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.load_state_dict(torch.load('model.pth'))
    model.to(device)

    model.eval()
    with torch.no_grad():
        text = clean(text)
        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        probability = torch.nn.functional.softmax(outputs.logits, dim=-1).max().item()
        isToxic = outputs.logits.argmax(dim=-1).item()
        return {'isToxic': isToxic, 'probability': round(probability, 2)}

if __name__ == "__main__":
    train()