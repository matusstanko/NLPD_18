# === Imports ===
from tqdm.auto import tqdm
tqdm.pandas()
import os, ast, random, numpy as np, torch, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          Trainer, TrainingArguments, set_seed)

# === Seed Setting ===
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# === Config ===
CFG = dict(
    name        = "bert_XML",
    epochs      = 7,
    batch_size  = 16,
    lr          = 3e-5,
    data_dir    = "/home/matus/NLPD_18/part1/outputs"
)

# === Load Data ===
df_train = pd.read_csv(f"{CFG['data_dir']}/output_train.csv")
df_valid = pd.read_csv(f"{CFG['data_dir']}/output_valid.csv")
df_test  = pd.read_csv(f"{CFG['data_dir']}/output_test.csv")

df_train_full = (
    pd.concat([df_train, df_valid], ignore_index=True)
      .sample(frac=1, random_state=SEED)
      .reset_index(drop=True)
)

# === Insert XML Tags ===
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

# === Tokenizer ===
tok = BertTokenizerFast.from_pretrained("bert-base-uncased")
special = ['<per>', '</per>', '<org>', '</org>', '<loc>', '</loc>', '<misc>', '</misc>']
tok.add_special_tokens({'additional_special_tokens': special})

def encode(texts): return tok(texts, truncation=True, padding=True)

enc_train = encode(df_train_full["xml_stmt"].tolist())
enc_valid = encode(df_valid["xml_stmt"].tolist())
enc_test  = encode(df_test["xml_stmt"].tolist())

# === Dataset Wrapper ===
class ClsDataset(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc = enc
        self.labels = labels
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

ds_train = ClsDataset(enc_train, df_train_full["label_binary"].tolist())
ds_valid = ClsDataset(enc_valid, df_valid["label_binary"].tolist())
ds_test  = ClsDataset(enc_test , df_test ["label_binary"].tolist())

# === Metrics ===
def metrics(pred):
    lab, pr = pred.label_ids, pred.predictions.argmax(-1)
    return dict(
        accuracy  = accuracy_score (lab, pr),
        f1        = f1_score       (lab, pr, zero_division=0),
        precision = precision_score(lab, pr, zero_division=0),
        recall    = recall_score   (lab, pr, zero_division=0)
    )

# === Training Setup ===
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.resize_token_embeddings(len(tok))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    report_to         = "none",
    seed              = SEED
)

trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = ds_train,
    eval_dataset    = ds_valid,
    compute_metrics = metrics
)

trainer.train()

# === Evaluation ===
eval_test = trainer.evaluate(ds_test)

# === Logging ===
logs = pd.DataFrame(trainer.state.log_history)
plt.style.use("default")  # Clean white background, consistent colors

# === Training + Validation Curve ===
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
plt.plot(range(1, n_iter+1), train_losses, marker="o", color="tab:blue", label="Train Loss")
plt.plot(range(1, n_iter+1), val_f1s,      marker="s", linestyle="--", color="tab:orange", label="Val F1")
plt.xlabel("Epoch")
plt.title("Training Loss & Validation F1")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("bert_training_validation_curve.png")
plt.close()

# === Confusion Matrix ===
pred_test = trainer.predict(ds_test).predictions.argmax(-1)
cm = confusion_matrix(df_test["label_binary"], pred_test)
disp = ConfusionMatrixDisplay(cm, display_labels=["0", "1"])
fig, ax = plt.subplots()
disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.savefig("confusion_matrix_bert.png")
plt.close()

# === ROC Curve ===
probs = trainer.predict(ds_test).predictions[:, 1]
fpr, tpr, _ = roc_curve(df_test["label_binary"], probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", color="tab:blue")
plt.plot([0, 1], [0, 1], linestyle="--", color="tab:orange")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(False)
plt.tight_layout()
plt.savefig("roc_curve_bert.png")
plt.close()

# === Save Summary ===
pd.DataFrame([{
    **CFG,
    "accuracy" : eval_test["eval_accuracy"],
    "f1"       : eval_test["eval_f1"],
    "precision": eval_test["eval_precision"],
    "recall"   : eval_test["eval_recall"]
}]).to_csv("results_summary.csv", index=False)

print("✅ BERT + XML results and plots saved.")