# ✅ Import libraries
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.tokenize import word_tokenize
from collections import Counter
import nltk
nltk.download('punkt')
print("✅ Libraries loaded.")

# ✅ Load GloVe
glove_path = 'glove_data/glove.6B.100d.txt'
embedding_dim = 100
glove_embeddings = {}
with open(glove_path, 'r', encoding='utf8') as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        vec = np.array(values[1:], dtype='float32')
        glove_embeddings[word] = vec
print(f"✅ Loaded {len(glove_embeddings)} GloVe vectors.")

# ✅ Best parameters
hidden_dim = 64
dropout = 0.1
lr = 1e-2
epochs = 10
patience = 2

# ✅ Load CSVs
df_train = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_train.csv')
df_valid = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_valid.csv')
df_test = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_test.csv')
print(f"✅ Train shape: {df_train.shape}, Valid shape: {df_valid.shape}, Test shape: {df_test.shape}")

# ✅ Tokenize plain sentences (no XML)
df_train['tokens'] = df_train['statement'].apply(word_tokenize)
df_valid['tokens'] = df_valid['statement'].apply(word_tokenize)
df_test['tokens'] = df_test['statement'].apply(word_tokenize)

# ✅ Combine train + valid
df_final_train = pd.concat([df_train, df_valid], ignore_index=True)

# ✅ Build vocab
all_tokens = [token for sent in df_final_train['tokens'] for token in sent]
vocab = {'<PAD>': 0, '<UNK>': 1}
vocab.update({word: idx + 2 for idx, word in enumerate(Counter(all_tokens))})
print(f"✅ Vocab size: {len(vocab)}")

# ✅ Encode tokens
def encode(tokens, vocab):
    return [vocab.get(t, vocab['<UNK>']) for t in tokens]

df_final_train['input_ids'] = df_final_train['tokens'].apply(lambda x: encode(x, vocab))
df_test['input_ids'] = df_test['tokens'].apply(lambda x: encode(x, vocab))

# ✅ Embedding matrix
embedding_matrix = np.zeros((len(vocab), embedding_dim))
for word, idx in vocab.items():
    vec = glove_embeddings.get(word)
    if vec is not None:
        embedding_matrix[idx] = vec
    else:
        embedding_matrix[idx] = np.random.normal(scale=0.6, size=(embedding_dim,))
print("✅ Embedding matrix ready")

# ✅ Dataset & DataLoader
class SimpleDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.data = list(zip(input_ids, labels))
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def collate_fn(batch):
    input_ids, labels = zip(*batch)
    padded = pad_sequence([torch.tensor(x) for x in input_ids], batch_first=True, padding_value=0)
    return padded, torch.tensor(labels)

train_dataset = SimpleDataset(df_final_train['input_ids'].tolist(), df_final_train['label_binary'].tolist())
test_dataset = SimpleDataset(df_test['input_ids'].tolist(), df_test['label_binary'].tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
print("✅ Final DataLoaders ready")

# ✅ BiLSTM Model
class BiLSTMClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim=128, num_classes=2, dropout=0.3):
        super().__init__()
        num_embeddings, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float), freeze=False, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = torch.mean(lstm_out, dim=1)
        out = self.dropout(out)
        return self.fc(out)

# ✅ Final train + test evaluation
def train_final_model(model, train_loader, test_loader, epochs, lr, device, patience=2):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"✅ Epoch {epoch+1}/{epochs} complete.")

    # Evaluation on test
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)

    print(f"\n🧪 Test Results — Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")
    return acc, f1, prec, rec

# ✅ Run final training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

final_model = BiLSTMClassifier(embedding_matrix=embedding_matrix, hidden_dim=hidden_dim, dropout=dropout)
acc, f1, prec, rec = train_final_model(final_model, train_loader, test_loader, epochs, lr, device, patience)

# ✅ Save results
pd.DataFrame([{
    "accuracy": acc,
    "f1": f1,
    "precision": prec,
    "recall": rec
}]).to_csv("BiLSTM_GloVe_base_train.csv", index=False)