import os
import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, get_scheduler
from torch.optim import AdamW

# ✅ Step 1: Load & Preprocess Data
def load_ner_data(file_path):
    """Loads data and returns a list of dictionaries with 'tokens' and 'ner_tags'."""
    data = [] 
    tokens, ner_tags = [], []

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if line == "":  # Sentence boundary
                if tokens:
                    data.append({"tokens": tokens, "ner_tags": ner_tags})
                    tokens, ner_tags = [], []
                continue

            if line.startswith("#"):  # Ignore metadata
                continue

            parts = line.split("\t")  
            if len(parts) >= 3:
                tokens.append(parts[1])  
                ner_tags.append(parts[2])  

    return data

# Load dataset
file_path = "en_ewt-ud-train.iob2"  
ner_dataset = load_ner_data(file_path)

print(f"✅ Loaded {len(ner_dataset)} sentences from dataset.")

# ✅ Step 2: Tokenization with BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_and_align_labels(sample):
    """Tokenizes words while aligning the NER labels with subword splits."""
    tokens, labels = [], []

    for word, label in zip(sample["tokens"], sample["ner_tags"]):
        tokenized_word = tokenizer.tokenize(word)
        subword_labels = [label] + [f"I-{label[2:]}" if label.startswith("B-") else label] * (len(tokenized_word) - 1)
        tokens.extend(tokenized_word)
        labels.extend(subword_labels)

    return tokens, labels

print("✅ Tokenization function ready.")

# ✅ Step 3: Convert Data for Training
# Create NER tag mappings
unique_tags = list(set(tag for sample in ner_dataset for tag in sample["ner_tags"]))
tag_to_index = {tag: i for i, tag in enumerate(unique_tags)}
index_to_tag = {i: tag for tag, i in tag_to_index.items()}

# Define max sequence length
MAX_LENGTH = 32  

def encode_sample(sample):
    """Encodes tokens and labels into numerical IDs."""
    tokenized_tokens, tokenized_labels = tokenize_and_align_labels(sample)
    token_ids = tokenizer.convert_tokens_to_ids(tokenized_tokens)
    label_ids = [tag_to_index[tag] for tag in tokenized_labels]

    # Padding
    token_ids = token_ids[:MAX_LENGTH] + [tokenizer.pad_token_id] * (MAX_LENGTH - len(token_ids))
    label_ids = label_ids[:MAX_LENGTH] + [tag_to_index["O"]] * (MAX_LENGTH - len(label_ids))

    return {"input_ids": torch.tensor(token_ids), "labels": torch.tensor(label_ids)}

# Encode dataset
encoded_dataset = [encode_sample(sample) for sample in ner_dataset]

print(f"✅ Data encoded. Total sentences: {len(encoded_dataset)}")

# ✅ Step 4: Define Model & Training Setup
model = BertForTokenClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(tag_to_index)
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ✅ Step 5: Prepare DataLoader
batch_size = 16
data_collator = DataCollatorForTokenClassification(tokenizer)
train_dataloader = DataLoader(encoded_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)

# ✅ Step 6: Define Training Parameters
optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
num_epochs = 3  
num_training_steps = len(train_dataloader) * num_epochs
lr_scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# ✅ Step 7: Train the Model
model.train()  
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tag_to_index["O"])  

for epoch in range(num_epochs):
    print(f"🚀 Training Epoch {epoch+1}/{num_epochs}...")

    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}  # Move batch to GPU

        optimizer.zero_grad()  
        outputs = model(**batch)  # Forward pass
        loss = outputs.loss  

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

    print(f"✅ Epoch {epoch+1} completed. Loss: {loss.item():.4f}")

# ✅ Step 8: Save Trained Model
if not os.path.exists("ner_model"):
    os.makedirs("ner_model")

model.save_pretrained("ner_model")
tokenizer.save_pretrained("ner_model")

print("🎉 Training completed! Model saved in 'ner_model/'.")