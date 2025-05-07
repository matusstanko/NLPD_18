# train_bert.py
import pandas as pd, numpy as np, ast, sys, torch, matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          Trainer, TrainingArguments, EarlyStoppingCallback)

# Get SLURM job array index
config_index = int(sys.argv[1])

# Grid of configs
experiment_configs = []
exp_id = 1
for epochs in [10, 15, 20]:
    for batch_size in [4, 8, 16, 32, 64]:
        for lr in [1e-5, 2e-5, 3e-5, 4e-5]:
            experiment_configs.append({
                "name": f"exp_{exp_id}",
                "epochs": epochs,
                "batch_size": batch_size,
                "lr": lr
            })
            exp_id += 1

# Safety check for SLURM array overrun
if config_index >= len(experiment_configs):
    print(f" Config index {config_index} is out of range. Only {len(experiment_configs)} configs available.")
    sys.exit(0)

# Load data
df_train = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_train.csv')
df_valid = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_valid.csv')

# Functions for inserting XML tags
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
    merged_entities = merge_adjacent_entities(entities)
    merged_entities.sort(key=lambda x: x['start'])
    offset = 0
    for ent in merged_entities:
        ent_type = ent['entity'].lower()
        start = ent['start'] + offset
        end = ent['end'] + offset
        open_tag = f"<{ent_type}>"
        close_tag = f"</{ent_type}>"
        text = text[:start] + open_tag + text[start:end] + close_tag + text[end:]
        offset += len(open_tag) + len(close_tag)
    return text

# Add XML tags
tqdm.pandas()
df_train['A_XML_statement'] = df_train.progress_apply(lambda r: insert_xml_tags(r['statement'], r['A_raw_entities']), axis=1)
df_valid['A_XML_statement'] = df_valid.progress_apply(lambda r: insert_xml_tags(r['statement'], r['A_raw_entities']), axis=1)

# Tokenizer with special tags
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
special_tokens = ['<per>', '</per>', '<org>', '</org>', '<loc>', '</loc>', '<misc>', '</misc>']
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

# Encode
train_encodings = tokenizer(df_train['A_XML_statement'].tolist(), truncation=True, padding=True)
valid_encodings = tokenizer(df_valid['A_XML_statement'].tolist(), truncation=True, padding=True)
train_labels = df_train['label_binary'].tolist()
valid_labels = df_valid['label_binary'].tolist()

# Dataset class
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

train_dataset = NERDataset(train_encodings, train_labels)
valid_dataset = NERDataset(valid_encodings, valid_labels)

# Select config
config = experiment_configs[config_index]
print(f"Running config {config['name']}")

# Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds)
    }

# Train
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.resize_token_embeddings(len(tokenizer))

training_args = TrainingArguments(
    output_dir=f"./results/{config['name']}",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    per_device_train_batch_size=config['batch_size'],
    per_device_eval_batch_size=config['batch_size'],
    num_train_epochs=config['epochs'],
    learning_rate=config['lr'],
    weight_decay=0.01,
    logging_dir=f"./logs/{config['name']}",
    save_strategy="epoch",
    report_to="none",
    metric_for_best_model="f1",
    greater_is_better=True,
    load_best_model_at_end=True,
    fp16=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.train()
eval_result = trainer.evaluate()

# Save training plot
logs = pd.DataFrame(trainer.state.log_history)
logs = logs[logs["epoch"].notnull()]
stats = logs.groupby("epoch").agg({"loss": "last", "eval_loss": "last", "eval_f1": "last"})

plt.figure(figsize=(10, 5))
plt.title(f"{config['name']} | bs={config['batch_size']} lr={config['lr']} ep={config['epochs']}")
plt.plot(stats.index, stats["loss"], label="Train Loss")
plt.plot(stats.index, stats["eval_loss"], label="Eval Loss")
plt.plot(stats.index, stats["eval_f1"], label="Eval F1", linestyle="--")
plt.xlabel("Epoch"); plt.legend()
plt.tight_layout()
plt.savefig(f"training_curve_{config['name']}.png")
plt.close()

# Save metrics
pd.DataFrame([{
    "name": config["name"],
    "epochs": config["epochs"],
    "batch_size": config["batch_size"],
    "lr": config["lr"],
    "accuracy": eval_result.get("eval_accuracy"),
    "f1": eval_result.get("eval_f1"),
    "precision": eval_result.get("eval_precision"),
    "recall": eval_result.get("eval_recall"),
    "best_f1": trainer.state.best_metric,
    "final_epoch": trainer.state.epoch,
    "best_checkpoint": trainer.state.best_model_checkpoint
}]).to_csv(f"result_{config['name']}.csv", index=False)
