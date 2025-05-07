#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BERT-base klasifikátor **bez** vložených XML tagov.
Vygeneruje: training_curve.png, confusion_matrix.png, roc_curve.png, results_summary.csv
"""

# ========== 0. DEPENDENCIES ==========
import os, torch, pandas as pd, matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          Trainer, TrainingArguments)

# ========== 1. CONFIG ==========
tqdm.pandas()
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # zmaž, ak chceš GPU
device = torch.device("cpu")

CFG = dict(
    name        = "bert_plain",
    epochs      = 7,
    batch_size  = 16,
    lr          = 3e-5,
    data_dir    = "/home/matus/NLPD_18/part1/outputs"   # ← uprav podľa seba
)

# ========== 2. LOAD DATA ==========
df_train = pd.read_csv(f"{CFG['data_dir']}/output_train.csv")
df_valid = pd.read_csv(f"{CFG['data_dir']}/output_valid.csv")
df_test  = pd.read_csv(f"{CFG['data_dir']}/output_test.csv")

df_train_full = pd.concat([df_train, df_valid], ignore_index=True)\
                  .sample(frac=1, random_state=42).reset_index(drop=True)

# ========== 3. TOKENIZE ==========
tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

def encode(texts): 
    return tok(texts, truncation=True, padding=True)

enc_train = encode(df_train_full["statement"].tolist())
enc_valid = encode(df_valid["statement"].tolist())
enc_test  = encode(df_test["statement"].tolist())

# ========== 4. DATASETS ==========
class ClsDataset(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

ds_train = ClsDataset(enc_train, df_train_full["label_binary"].tolist())
ds_valid = ClsDataset(enc_valid, df_valid["label_binary"].tolist())
ds_test  = ClsDataset(enc_test , df_test ["label_binary"].tolist())

# ========== 5. METRICS ==========
def metrics(pred):
    lab, pr = pred.label_ids, pred.predictions.argmax(-1)
    return dict(
        accuracy  = accuracy_score (lab, pr),
        f1        = f1_score       (lab, pr, zero_division=0),
        precision = precision_score(lab, pr, zero_division=0),
        recall    = recall_score   (lab, pr, zero_division=0)
    )

# ========== 6. TRAIN ==========
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

args = TrainingArguments(
    output_dir       = f"./results/{CFG['name']}",
    evaluation_strategy = "epoch",
    logging_strategy    = "epoch",
    save_strategy       = "no",
    per_device_train_batch_size = CFG["batch_size"],
    per_device_eval_batch_size  = CFG["batch_size"],
    num_train_epochs  = CFG["epochs"],
    learning_rate     = CFG["lr"],
    weight_decay      = 0.01,
    fp16              = False,
    report_to         = "none"
)

trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = ds_train,
    eval_dataset    = ds_valid,
    compute_metrics = metrics
)

trainer.train()

# ========== 7. FINAL EVAL ==========
eval_test = trainer.evaluate(ds_test)

# ========== 8. PLOTS ==========
plt.style.use("ggplot")
logs = pd.DataFrame(trainer.state.log_history)
eval_logs = logs[logs["eval_f1"].notnull()]

# 8a. Loss & F1
fig, ax1 = plt.subplots(figsize=(7,4))
ax1.plot(eval_logs["epoch"], eval_logs["eval_loss"], marker="o", label="valid loss")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax2 = ax1.twinx()
ax2.plot(eval_logs["epoch"], eval_logs["eval_f1"], marker="s", linestyle="--", label="valid F1")
ax2.set_ylabel("F1 score")
fig.suptitle("Validation Loss & F1")
fig.legend(loc="upper right"); fig.tight_layout()
plt.savefig("training_curve.png"); plt.close()

# 8b. Confusion matrix
pred_test = trainer.predict(ds_test).predictions.argmax(-1)
cm = confusion_matrix(df_test["label_binary"], pred_test)
ConfusionMatrixDisplay(cm, display_labels=["False","True"]).plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix (test set)")
plt.tight_layout(); plt.savefig("confusion_matrix.png"); plt.close()

# 8c. ROC
probs = trainer.predict(ds_test).predictions[:,1]
fpr, tpr, _ = roc_curve(df_test["label_binary"], probs)
plt.figure(figsize=(5,4))
plt.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve (test set)"); plt.legend(); plt.tight_layout()
plt.savefig("roc_curve.png"); plt.close()

# ========== 9. SAVE SUMMARY ==========
pd.DataFrame([{
    **CFG,
    "accuracy" : eval_test["eval_accuracy"],
    "f1"       : eval_test["eval_f1"],
    "precision": eval_test["eval_precision"],
    "recall"   : eval_test["eval_recall"]
}]).to_csv("results_summary.csv", index=False)

print("✅ Hotovo — všetky výsledky a grafy uložené.")