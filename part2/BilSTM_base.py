import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm
import ast
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
print("✅ Libraries loaded.")


# Load CSVs
df_train = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_train.csv')
df_valid = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_valid.csv')
print(f"✅ Train shape: {df_train.shape}, Valid shape: {df_valid.shape}")


# Functions for XML tagging (same as before)
def merge_adjacent_entities(entities):
    if not entities:
        return []
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
    if not entities:
        return text
    if isinstance(entities, str):
        try:
            entities = ast.literal_eval(entities)
        except:
            return text
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

# Apply XML insertion
tqdm.pandas()
df_train['A_XML_statement'] = df_train.progress_apply(
    lambda row: insert_xml_tags(row['statement'], row['A_raw_entities']), axis=1)
df_valid['A_XML_statement'] = df_valid.progress_apply(
    lambda row: insert_xml_tags(row['statement'], row['A_raw_entities']), axis=1)

print("✅ XML tags added to train & valid")



# Tokenize XML-tagged sentences
df_train['tokens'] = df_train['A_XML_statement'].apply(word_tokenize)
df_valid['tokens'] = df_valid['A_XML_statement'].apply(word_tokenize)

# Build vocabulary from training tokens
all_tokens = [token for sentence in df_train['tokens'] for token in sentence]
vocab = {'<PAD>': 0, '<UNK>': 1}
vocab.update({word: idx + 2 for idx, word in enumerate(Counter(all_tokens))})

print(f"✅ Vocabulary size: {len(vocab)}")

# Convert tokens to indices
def encode(tokens, vocab):
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

df_train['input_ids'] = df_train['tokens'].apply(lambda x: encode(x, vocab))
df_valid['input_ids'] = df_valid['tokens'].apply(lambda x: encode(x, vocab))
print("✅ Tokens encoded")


class SimpleDataset(Dataset):
    def __init__(self, input_ids, labels):
        self.data = list(zip(input_ids, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    input_ids, labels = zip(*batch)
    padded = pad_sequence([torch.tensor(x) for x in input_ids], batch_first=True, padding_value=0)
    return padded, torch.tensor(labels)

# Wrap your data
train_dataset = SimpleDataset(df_train['input_ids'].tolist(), df_train['label_binary'].tolist())
valid_dataset = SimpleDataset(df_valid['input_ids'].tolist(), df_valid['label_binary'].tolist())

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

print("✅ DataLoaders ready")


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_classes=2, dropout=0.3):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # last timestep
        out = self.dropout(out)
        return self.fc(out)


def train_model(model, train_loader, valid_loader, epochs, lr, device):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss:.4f}")

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())

        f1 = f1_score(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        print(f"🧪 Val Accuracy: {acc:.4f} | F1: {f1:.4f}\n")




# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# Initialize the model
model = BiLSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=100,
    hidden_dim=128,
    num_classes=2,
    dropout=0.3
)

# Train it!
train_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    epochs=5,
    lr=1e-3,
    device=device
)










