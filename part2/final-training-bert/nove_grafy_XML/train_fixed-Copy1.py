#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Finálny (self-contained) skript:
 ▸ vloží XML tagy,
 ▸ natrénuje BERT-base-uncased,
 ▸ vyhodnotí na validačnej i testovacej množine,
 ▸ uloží metriku + tri moderné grafy.
"""

# ========== 0. DEPENDENCIES ==========
import os, ast, random, numpy as np, torch, pandas as pd, matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          Trainer, TrainingArguments, set_seed)

# ---------- GLOBAL SEED ----------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)                # neškodí ani na CPU
set_seed(SEED)                                  # HuggingFace helper
torch.backends.cudnn.deterministic = True       # ak pôjdeš na GPU
torch.backends.cudnn.benchmark = False
# -----------------------------------

# ========== 1. BASIC CONFIG ==========
tqdm.pandas()
os.environ["CUDA_VISIBLE_DEVICES"] = ""         # Zmaž, ak chceš GPU
device = torch.device("cpu")

CFG = dict(
    name        = "bert_xml_tags",
    epochs      = 7,
    batch_size  = 16,
    lr          = 3e-5,
    data_dir    = "/home/matus/NLPD_18/part1/outputs"   # ← uprav len túto cestu
)

# ========== 2. LOAD DATA ==========
df_train = pd.read_csv(f"{CFG['data_dir']}/output_train.csv")
df_valid = pd.read_csv(f"{CFG['data_dir']}/output_valid.csv")
df_test  = pd.read_csv(f"{CFG['data_dir']}/output_test.csv")

df_train_full = (
    pd.concat([df_train, df_valid], ignore_index=True)
      .sample(frac=1, random_state=SEED)        # reprodukovateľné premiešanie
      .reset_index(drop=True)
)

# ========== 3. INSERT XML TAGS ==========
def merge_adjacent_entities(entities):
    if not entities: return []
    entities = sorted(entities, key=lambda x: x["start"])
    merged = [entities[0]]
    for cur in entities[1:]:
        last = merged[-1]
        if cur["entity"] == last["entity"] and cur["start"] <= last["end"] + 1:
            last["end"] = cur["end"]
        else:
            merged.append(cur)
    return merged

def insert_xml(text, entities):
    if not entities: return text
    if isinstance(entities, str):
        try: entities = ast.literal_eval(entities)
        except Exception: return text
    merged = merge_adjacent_entities(entities)
    merged.sort(key=lambda x: x["start"])
    offset = 0
    for ent in merged:
        tag = ent["entity"].lower()
        open_tag, close_tag = f"<{tag}>", f"</{tag}>"
        s, e = ent["start"] + offset, ent["end"] + offset
        text = text[:s] + open_tag + text[s:e] + close_tag + text[e:]
        offset += len(open_tag) + len(close_tag)
    return text

for df in (df_train_full, df_valid, df_test):
    df["xml_stmt"] = df.progress_apply(
        lambda r: insert_xml(r["statement"], r["A_raw_entities"]), axis=1
    )

# ========== 4. TOKENIZE ==========
tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
special = ['<per>', '</per>', '<org>', '</org>', '<loc>', '</loc>', '<misc>', '</misc>']
tok.add_special_tokens({'additional_special_tokens': special})

def encode(texts): return tok(texts, truncation=True, padding=True)

enc_train = encode(df_train_full["xml_stmt"].tolist())
enc_valid = encode(df_valid["xml_stmt"].tolist())
enc_test  = encode(df_test ["xml_stmt"].tolist())

# ========== 5. DATASETS ==========
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

# ========== 6. METRICS ==========
def metrics(pred):
    lab, pr = pred.label_ids, pred.predictions.argmax(-1)
    return dict(
        accuracy  = accuracy_score (lab, pr),
        f1        = f1_score       (lab, pr, zero_division=0),
        precision = precision_score(lab, pr, zero_division=0),
        recall    = recall_score   (lab, pr, zero_division=0)
    )

# ========== 7. TRAIN ==========
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.resize_token_embeddings(len(tok))
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
    fp16              = False,      # CPU-safe
    report_to         = "none",
    seed              = SEED        # ← seed pre Trainer & DataLoadery
)

trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = ds_train,
    eval_dataset    = ds_valid,
    compute_metrics = metrics
)

trainer.train()

# ========== 8. FINAL EVAL ==========
eval_test = trainer.evaluate(ds_test)

# ========== 9. LOGS → PLOTS ==========
logs = pd.DataFrame(trainer.state.log_history)

plt.style.use("ggplot")

# -------- 9a. TRAINING CURVE (štýl SpaCy snippetu) --------
train_losses = (
    logs[logs["loss"].notnull()]
        .groupby("epoch")["loss"].last().tolist()
)
val_f1s = (
    logs[logs["eval_f1"].notnull()]
        .groupby("epoch")["eval_f1"].last().tolist()
)
n_iter = len(train_losses)

plt.figure()
plt.plot(range(1, n_iter + 1), train_losses, label="Train loss")
plt.plot(range(1, n_iter + 1), val_f1s, label="Validation F1")
plt.xlabel("Epoch")
plt.title("BERT + XML Training Curve")
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png")
plt.close()

# -------- 9b. CONFUSION MATRIX --------
pred_test = trainer.predict(ds_test).predictions.argmax(-1)
cm = confusion_matrix(df_test["label_binary"], pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["False", "True"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (BERT + XML)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# -------- 9c. ROC CURVE --------
probs = trainer.predict(ds_test).predictions[:, 1]
fpr, tpr, _ = roc_curve(df_test["label_binary"], probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (BERT + XML)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# ========== 10. SAVE TABLE ==========
pd.DataFrame([{
    **CFG,
    "accuracy" : eval_test["eval_accuracy"],
    "f1"       : eval_test["eval_f1"],
    "precision": eval_test["eval_precision"],
    "recall"   : eval_test["eval_recall"]
}]).to_csv("results_summary.csv", index=False)

print("✅ Hotovo – model, metriky aj grafy (seed 42) sú uložené.")