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
from tqdm import tqdm
import ast
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
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
hidden_dim = 128
dropout = 0.3
lr = 1e-3
epochs = 3
patience = 2

# ✅ Load CSVs
df_train = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_train.csv')
df_valid = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_valid.csv')
df_test = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_test.csv')
print(f"✅ Train shape: {df_train.shape}, Valid shape: {df_valid.shape}, Test shape: {df_test.shape}")

# ✅ XML tagging
def merge_adjacent_entities(entities):
    if not entities: return []
    entities = sorted(entities, key=lambda x: x['start'])
    merged = [entities[0]]
    for current in entities[1:]:
        last = merged[-1]
        if current['entity'] == last['entity'] and current['start'] <= last['end'] + 1:
            last['end'] = current['end']
        else:
            merged.append(current)
    return merged

def insert_xml_tags(text, entities):
    if not entities: return text
    if isinstance(entities, str):
        try: entities = ast.literal_eval(entities)
        except: return text
    entities = merge_adjacent_entities(entities)
    entities.sort(key=lambda x: x['start'])
    offset = 0
    for ent in entities:
        open_tag = f"<{ent['entity']}>"
        close_tag = f"</{ent['entity']}>"
        start = ent['start'] + offset
        end = ent['end'] + offset
        text = text[:start] + open_tag + text[start:end] + close_tag + text[end:]
        offset += len(open_tag) + len(close_tag)
    return text

tqdm.pandas()
df_train['A_XML_statement'] = df_train.progress_apply(lambda row: insert_xml_tags(row['statement'], row['A_raw_entities']), axis=1)
df_valid['A_XML_statement'] = df_valid.progress_apply(lambda row: insert_xml_tags(row['statement'], row['A_raw_entities']), axis=1)
df_test['A_XML_statement'] = df_test.progress_apply(lambda row: insert_xml_tags(row['statement'], row['A_raw_entities']), axis=1)
print("✅ XML tags added")

# ✅ Tokenize
df_train['tokens'] = df_train['A_XML_statement'].apply(word_tokenize)
df_valid['tokens'] = df_valid['A_XML_statement'].apply(word_tokenize)
df_test['tokens'] = df_test['A_XML_statement'].apply(word_tokenize)

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

# ✅ BiLSTM with GloVe
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

# ✅ Train function
def train_final_model(model, train_loader, test_loader, epochs, lr, device, patience=2):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = 0
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # Optional: Print training progress
        print(f"✅ Epoch {epoch+1}/{epochs} complete.")

    # Evaluation on test set
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

# ✅ Run Final Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

final_model = BiLSTMClassifier(embedding_matrix=embedding_matrix, hidden_dim=hidden_dim, dropout=dropout)
acc, f1, prec, rec = train_final_model(final_model, train_loader, test_loader, epochs, lr, device, patience)

# ✅ Save test evaluation results
pd.DataFrame([{
    "accuracy": acc,
    "f1": f1,
    "precision": prec,
    "recall": rec
}]).to_csv("BiLSTM_GloVe_test_results.csv", index=False)
print("✅ Test results saved to BiLSTM_GloVe_test_results.csv")