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



# Load training data
train_data = load_ner_data("en_ewt-ud-train.iob2")

print(f"✅ Loaded {len(train_data)} sentences from training dataset.")

# Load dev data
dev_data = load_ner_data("en_ewt-ud-dev.iob2")

print(f"✅ Loaded {len(dev_data)} sentences from dev dataset.")

# Get unique labels
unique_tags = set()
for example in train_data + dev_data:
    for tag in example["ner_tags"]:
        unique_tags.add(tag)
unique_tags = sorted(list(unique_tags))  # Sort for consistency
print(unique_tags)  # e.g. ["B-LOC", "B-ORG", "B-PER", "I-LOC", ...]

# tag to id
tag2id = {tag: idx for idx, tag in enumerate(unique_tags)}
id2tag = {idx: tag for idx, tag in enumerate(unique_tags)}

import torch
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

model_name = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    """
    Tokenizes the input tokens using a tokenizer and aligns the NER tags with the new subword tokens.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens (CLS, SEP, PAD, etc.)
                label_ids.append(-100)
            else:
                label_ids.append(tag2id[labels[word_idx]])
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs

from datasets import Dataset

# Convert our Python lists of dicts into HuggingFace Dataset
train_dataset = Dataset.from_list(train_data)
dev_dataset = Dataset.from_list(dev_data)

# Now apply our tokenize_and_align_labels function
train_dataset = train_dataset.map(tokenize_and_align_labels, batched=True)
dev_dataset = dev_dataset.map(tokenize_and_align_labels, batched=True)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(unique_tags)
)

batch_size = 16  # or something that fits your GPU/CPU memory
args = TrainingArguments(
    output_dir="./ner_baseline",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=3,  # adjust as you see fit
    weight_decay=0.01,
    logging_dir="./logs",  # for TensorBoard
    logging_steps=50
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("./my_ner_model2")