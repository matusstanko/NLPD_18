# ✅ Import libraries
from sklearn.metrics import precision_score, recall_score
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score
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

# Grid setup
param_grid = {
    "hidden_dim": [64, 128, 256],
    "dropout": [0.1, 0.3, 0.5],
    "lr": [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
}
epochs = 10
patience = 2



# ✅ Load CSVs
df_train = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_train.csv')
df_valid = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_valid.csv')
print(f"✅ Train shape: {df_train.shape}, Valid shape: {df_valid.shape}")

# ✅ XML tagging functions
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

# ✅ Apply XML
tqdm.pandas()
df_train['A_XML_statement'] = df_train.progress_apply(lambda row: insert_xml_tags(row['statement'], row['A_raw_entities']), axis=1)
df_valid['A_XML_statement'] = df_valid.progress_apply(lambda row: insert_xml_tags(row['statement'], row['A_raw_entities']), axis=1)
print("✅ XML tags added")

# ✅ Tokenize
df_train['tokens'] = df_train['A_XML_statement'].apply(word_tokenize)
df_valid['tokens'] = df_valid['A_XML_statement'].apply(word_tokenize)



# ✅ Build vocab
all_tokens = [token for sent in df_train['tokens'] for token in sent]
vocab = {'<PAD>': 0, '<UNK>': 1}
vocab.update({word: idx + 2 for idx, word in enumerate(Counter(all_tokens))})
print(f"✅ Vocab size: {len(vocab)}")

# ✅ Encode tokens
def encode(tokens, vocab):
    return [vocab.get(t, vocab['<UNK>']) for t in tokens]

df_train['input_ids'] = df_train['tokens'].apply(lambda x: encode(x, vocab))
df_valid['input_ids'] = df_valid['tokens'].apply(lambda x: encode(x, vocab))



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

train_dataset = SimpleDataset(df_train['input_ids'].tolist(), df_train['label_binary'].tolist())
valid_dataset = SimpleDataset(df_valid['input_ids'].tolist(), df_valid['label_binary'].tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
print("✅ DataLoaders ready")

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
        out = torch.mean(lstm_out, dim=1)  # mean pooling instead of last timestep
        out = self.dropout(out)
        return self.fc(out)








device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

results = []
try_id = 1

def train_model_return_scores(model, train_loader, valid_loader, epochs, lr, device, patience=2):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = 0
    best_epoch = 0
    epochs_no_improve = 0

    all_preds, all_labels = [], []
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds)
        rec = recall_score(all_labels, all_preds)

        print(f"📚 Epoch {epoch+1}/{epochs} | Val Acc: {acc:.4f} | F1: {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("🛑 Early stopping")
                break

    return acc, f1, prec, rec, epoch + 1, best_epoch


# 🔁 Grid Search Loop
for hd in param_grid['hidden_dim']:
    for do in param_grid['dropout']:
        for lr in param_grid['lr']:
            print(f"\n🔍 Try {try_id} | hidden_dim={hd}, dropout={do}, lr={lr}")
            model = BiLSTMClassifier(embedding_matrix=embedding_matrix, hidden_dim=hd, dropout=do)
            acc, f1, prec, rec, total_epochs, best_epoch = train_model_return_scores(
                model, train_loader, valid_loader, epochs, lr, device, patience
            )
            results.append({
                "try_id": try_id,
                "hidden_dim": hd,
                "dropout": do,
                "lr": lr,
                "accuracy": acc,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "total_epochs": total_epochs,
                "best_epoch": best_epoch
            })
            try_id += 1

# 💾 Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="f1", ascending=False)
results_df.to_csv("BilSTM_GloVe_eval.csv", index=False)
print("✅ Results saved to bilstm_gridsearch_results.csv")










