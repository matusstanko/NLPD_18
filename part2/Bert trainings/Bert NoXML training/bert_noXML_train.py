# Imports
import os, random, numpy as np, torch, pandas as pd, matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             roc_curve, auc)
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          Trainer, TrainingArguments, set_seed)

# Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)              
set_seed(SEED)                                
torch.backends.cudnn.deterministic = True     
torch.backends.cudnn.benchmark = False
tqdm.pandas()

# Parameters
CFG = dict(
    name        = "bert_noXML",
    epochs      = 7,
    batch_size  = 16,
    lr          = 3e-5,
    data_dir    = "/home/matus/NLPD_18/part1/outputs"   
)

# Loading data
df_train = pd.read_csv(f"{CFG['data_dir']}/output_train.csv")
df_valid = pd.read_csv(f"{CFG['data_dir']}/output_valid.csv")
df_test  = pd.read_csv(f"{CFG['data_dir']}/output_test.csv")

# Combine train+validation data
df_train_full = (
    pd.concat([df_train, df_valid], ignore_index=True)
      .sample(frac=1, random_state=SEED)           
      .reset_index(drop=True)
)

# Tokenize using bert-base-uncased
tok = BertTokenizerFast.from_pretrained("bert-base-uncased")

def encode(texts):
    return tok(texts, truncation=True, padding=True)

enc_train = encode(df_train_full["statement"].tolist())
enc_valid = encode(df_valid["statement"].tolist())
enc_test  = encode(df_test["statement"].tolist())

# Define datasets 
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

# Output metrics
def metrics(pred):
    lab, pr = pred.label_ids, pred.predictions.argmax(-1)
    return dict(
        accuracy  = accuracy_score (lab, pr),
        f1        = f1_score       (lab, pr, zero_division=0),
        precision = precision_score(lab, pr, zero_division=0),
        recall    = recall_score   (lab, pr, zero_division=0)
    )

# Training
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
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


# === Evaluation on test set ===
eval_test = trainer.evaluate(ds_test)

# === Plots ===
plt.style.use("ggplot")
logs = pd.DataFrame(trainer.state.log_history)

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

# === Training Loss & Validation F1 (clean style) ===
plt.style.use("default")  # White background and default color palette

plt.figure()
plt.plot(range(1, n_iter + 1), train_losses, marker="o", color="tab:blue", label="Train Loss")
plt.plot(range(1, n_iter + 1), val_f1s, marker="s", linestyle="--", color="tab:orange", label="Val F1")
plt.xlabel("Epoch")
plt.title("Training Loss & Validation F1")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig("bert_noXML_training_validation_curve.png")
plt.close()

# === Confusion Matrix (clean style) ===
plt.style.use("default")  # Ensures white background and default fonts

pred_test = trainer.predict(ds_test).predictions.argmax(-1)
cm = confusion_matrix(df_test["label_binary"], pred_test)
disp = ConfusionMatrixDisplay(cm, display_labels=["0", "1"])

fig, ax = plt.subplots()
disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)

plt.title("Confusion Matrix")
plt.grid(False)          # No grid
plt.tight_layout()       # Tight layout
plt.savefig("confusion_matrix_bert_noXML.png")
plt.close()

# === ROC Curve (custom style to match target image) ===
plt.style.use("default")  # ⬅ Reset style to default (white background)

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
plt.savefig("roc_curve_bert_noXML.png")
plt.close()

# === Save summary ===
pd.DataFrame([{
    **CFG,
    "accuracy" : eval_test["eval_accuracy"],
    "f1"       : eval_test["eval_f1"],
    "precision": eval_test["eval_precision"],
    "recall"   : eval_test["eval_recall"]
}]).to_csv("results_summary_noXML.csv", index=False)

print("✅ BERT noXML results and plots saved.")