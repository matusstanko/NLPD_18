# ====================== BERT + XML TAGS ======================
# Trains with 15 max epochs, no early stopping
# Saves training curve PNG (loss + F1 over epochs)
# ============================================================

import pandas as pd, numpy as np, ast, torch, random, matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          Trainer, TrainingArguments)

print("1/6 libraries loaded")

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Experiment config
experiment_configs = [{
    "name": "bert_xml_tags",
    "epochs": 3,
    "batch_size": 16,
    "lr": 3e-5
}]

# Load data
df_train = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_train.csv')
df_valid = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_valid.csv')
df_test  = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_test.csv')
print("2/6 data loaded")

# Merge train+valid (can separate if needed)
df_train_merged = pd.concat([df_train, df_valid], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# Add XML tags to statements
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

tqdm.pandas()
df_train_merged['A_XML_statement'] = df_train_merged.progress_apply(
    lambda row: insert_xml_tags(row['statement'], row['A_raw_entities']), axis=1
)
df_test['A_XML_statement'] = df_test.progress_apply(
    lambda row: insert_xml_tags(row['statement'], row['A_raw_entities']), axis=1
)
print("3/6 XML tags inserted")

# Tokenizer with added XML tokens
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
special_tokens = ['<per>', '</per>', '<org>', '</org>', '<loc>', '</loc>', '<misc>', '</misc>']
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

# Encode
train_encodings = tokenizer(df_train_merged['A_XML_statement'].tolist(), truncation=True, padding=True)
test_encodings  = tokenizer(df_test['A_XML_statement'].tolist(), truncation=True, padding=True)

train_labels = df_train_merged['label_binary'].tolist()
test_labels  = df_test['label_binary'].tolist()

# Dataset
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings, self.labels = encodings, labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

train_dataset = NERDataset(train_encodings, train_labels)
test_dataset  = NERDataset(test_encodings,  test_labels)
print("4/6 Converted to datasets")

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

# Training
print("5/6 Training in progress...")
all_results = []

for config in experiment_configs:
    print(f"Training {config['name']}...")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=f"./results/{config['name']}",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        num_train_epochs=config['epochs'],
        learning_rate=config['lr'],
        weight_decay=0.01,
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=False,
        fp16=True,
        report_to="none",
        logging_dir=f"./logs/{config['name']}"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()
    eval_result = trainer.evaluate()

    # Plot training curve
    logs = pd.DataFrame(trainer.state.log_history)
    logs = logs[logs["epoch"].notnull()]
    stats = logs.groupby("epoch").agg({"loss":"last", "eval_loss":"last", "eval_f1":"last"})

    fig, ax1 = plt.subplots()
    line1 = ax1.plot(stats.index, stats["loss"], marker="o", label="train loss")
    line2 = ax1.plot(stats.index, stats["eval_loss"], marker="o", label="eval loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")

    ax2 = ax1.twinx()
    if "eval_f1" in stats.columns:
        line3 = ax2.plot(stats.index, stats["eval_f1"], marker="s", linestyle="--", color="green", label="eval F1")
        ax2.set_ylabel("F1 score")

        # Combine legends from both axes
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    fig.tight_layout()
    plt.title("BERT+XML Training Curve")
    png_name = f"{config['name']}_curve.png"
    plt.savefig(png_name)
    plt.close()
    print(f"Saved plot to {png_name}")

    all_results.append({
        "name": config["name"],
        "epochs": config["epochs"],
        "batch_size": config["batch_size"],
        "lr": config["lr"],
        "accuracy": eval_result.get("eval_accuracy"),
        "f1": eval_result.get("eval_f1"),
        "precision": eval_result.get("eval_precision"),
        "recall": eval_result.get("eval_recall")
    })

results_df = pd.DataFrame(all_results)
results_df.to_csv("bert_with_xml_results.csv", index=False)
print("6/6 Done — results saved to bert_with_xml_results.csv ✅")