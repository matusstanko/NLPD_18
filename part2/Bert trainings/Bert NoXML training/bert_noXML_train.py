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

# Evaluation on test set
eval_test = trainer.evaluate(ds_test)

# Plots
plt.style.use("ggplot")
logs = pd.DataFrame(trainer.state.log_history)

# 1
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
plt.title("BERT-base Training Curve")
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png")
plt.close()

# Confusion matrix
pred_test = trainer.predict(ds_test).predictions.argmax(-1)
cm = confusion_matrix(df_test["label_binary"], pred_test)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["False", "True"])
disp.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix (BERT-base)")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# Roc Curve
probs = trainer.predict(ds_test).predictions[:, 1]
fpr, tpr, _ = roc_curve(df_test["label_binary"], probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (BERT-base)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.close()

# Saving file
pd.DataFrame([{
    **CFG,
    "accuracy" : eval_test["eval_accuracy"],
    "f1"       : eval_test["eval_f1"],
    "precision": eval_test["eval_precision"],
    "recall"   : eval_test["eval_recall"]
}]).to_csv("results_summary.csv", index=False)