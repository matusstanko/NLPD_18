# =========================  BASE MODEL (NO XML TAGS)  =========================
# Train+Valid → TRAIN, Test → EVAL
# 7 epochs max, EarlyStopping patience = 3
# Saves a PNG with train/eval loss and eval‑F1 per epoch
# ============================================================================

# 1. Libraries
import pandas as pd, numpy as np, ast, torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          Trainer, TrainingArguments, EarlyStoppingCallback)

print("1/6 libraries loaded")

# 2. Hyper‑parameters
experiment_configs = [{
    "name": "bert_base_tags",
    "epochs": 7,
    "batch_size": 16,
    "lr": 3e-5
}]

# 3. Data --------------------------------------------------------------------
df_train = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_train.csv')
df_valid = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_valid.csv')
df_test  = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_test.csv')
print("2/6 data loaded")

df_train_merged = (pd.concat([df_train, df_valid], ignore_index=True)
                     .sample(frac=1, random_state=42)
                     .reset_index(drop=True))

# 4. Tokenizer & model -------------------------------------------------------
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(df_train_merged['statement'].tolist(),
                            truncation=True, padding=True)
test_encodings  = tokenizer(df_test['statement'].tolist(),
                            truncation=True, padding=True)

train_labels = df_train_merged['label_binary'].tolist()
test_labels  = df_test['label_binary'].tolist()

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings, self.labels = encodings, labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_ds = SimpleDataset(train_encodings, train_labels)
test_ds  = SimpleDataset(test_encodings,  test_labels)
print("3/6 datasets built")

# 5. Metrics ------------------------------------------------------------------
def compute_metrics(pred):
    lab = pred.label_ids
    p   = pred.predictions.argmax(-1)
    return {"accuracy":  accuracy_score(lab, p),
            "f1":        f1_score(lab, p),
            "precision": precision_score(lab, p),
            "recall":    recall_score(lab, p)}

# 6. Training loop ------------------------------------------------------------
all_results = []

for cfg in experiment_configs:
    print(f"Training run: {cfg['name']}")
    model = BertForSequenceClassification.from_pretrained(
                "bert-base-uncased", num_labels=2)

    args = TrainingArguments(
        output_dir=f"./results/{cfg['name']}",
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=cfg['batch_size'],
        per_device_eval_batch_size=cfg['batch_size'],
        num_train_epochs=cfg['epochs'],
        learning_rate=cfg['lr'],
        weight_decay=0.01,
        metric_for_best_model="f1",
        greater_is_better=True,
        load_best_model_at_end=True,
        fp16=True,
        report_to="none",
        logging_dir=f"./logs/{cfg['name']}"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # << FIXED: patience 3
    )

    trainer.train()
    eval_res = trainer.evaluate()

    # -------- plot train/eval loss & F1 per epoch -----------------------------
    logs = pd.DataFrame(trainer.state.log_history)
    logs = logs[logs["epoch"].notnull()]
    stats = logs.groupby("epoch").agg({"loss":"last",
                                       "eval_loss":"last",
                                       "eval_f1":"last"})

    fig, ax1 = plt.subplots()
    ax1.plot(stats.index, stats["loss"], marker="o", label="train loss")
    ax1.plot(stats.index, stats["eval_loss"], marker="o", label="eval loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")

    ax2 = ax1.twinx()
    ax2.plot(stats.index, stats["eval_f1"], marker="s", linestyle="--",
             label="eval F1", color="green")
    ax2.set_ylabel("F1 score")
    fig.tight_layout()
    fig.legend(loc="upper right")
    plt.title("Training vs Eval curves")
    png_name = f"{cfg['name']}_curve.png"
    plt.savefig(png_name)
    plt.close()
    print(f"Saved plot to {png_name}")

    # -------- store summary ---------------------------------------------------
    all_results.append({
        "name": cfg["name"],
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "accuracy":  eval_res.get("eval_accuracy"),
        "f1":        eval_res.get("eval_f1"),
        "precision": eval_res.get("eval_precision"),
        "recall":    eval_res.get("eval_recall"),
        "best_f1":   trainer.state.best_metric,
        "final_epoch": trainer.state.epoch,
        "best_checkpoint": trainer.state.best_model_checkpoint
    })

pd.DataFrame(all_results).to_csv("bert_base_results.csv", index=False)
print("Training complete — results saved to bert_base_results.csv ✅")