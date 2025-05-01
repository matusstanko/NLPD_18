import os
import pandas as pd, torch, matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc)
from transformers import (BertTokenizerFast, BertForSequenceClassification,
                          Trainer, TrainingArguments)

# ========== CONFIG ==========
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
device = torch.device("cpu")

experiment_config = {
    "name": "bert_no_xml_tags",
    "epochs": 7,
    "batch_size": 16,
    "lr": 3e-5
}

# ========== 1. LOAD DATA ==========
df_train = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_train.csv')
df_valid = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_valid.csv')
df_test  = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_test.csv')

df_train_merged = pd.concat([df_train, df_valid], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

# ========== 2. TOKENIZE RAW STATEMENTS (NO XML TAGGING) ==========
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

train_encodings = tokenizer(df_train_merged['statement'].tolist(), truncation=True, padding=True)
test_encodings  = tokenizer(df_test['statement'].tolist(), truncation=True, padding=True)

train_labels = df_train_merged['label_binary'].tolist()
test_labels  = df_test['label_binary'].tolist()

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

# ========== 3. METRICS ==========
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds)
    }

# ========== 4. TRAIN ==========
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)

training_args = TrainingArguments(
    output_dir=f"./results/{experiment_config['name']}",
    evaluation_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    per_device_train_batch_size=experiment_config['batch_size'],
    per_device_eval_batch_size=experiment_config['batch_size'],
    num_train_epochs=experiment_config['epochs'],
    learning_rate=experiment_config['lr'],
    weight_decay=0.01,
    load_best_model_at_end=False,
    report_to="none",
    fp16=False,
    logging_dir=f"./logs/{experiment_config['name']}"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=train_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# ========== 5. EVALUATE ON TEST SET ==========
eval_result = trainer.evaluate(test_dataset)

# ========== 6. PLOTS ==========
# -- Training curve --
logs = pd.DataFrame(trainer.state.log_history)
logs = logs[logs["epoch"].notnull()]
stats = logs.groupby("epoch").agg({"loss":"last", "eval_loss":"last", "eval_f1":"last"})

fig, ax1 = plt.subplots()
ax1.plot(stats.index, stats["loss"], marker="o", label="train loss")
ax1.plot(stats.index, stats["eval_loss"], marker="o", label="eval loss")
ax1.set_xlabel("epoch")
ax1.set_ylabel("loss")

ax2 = ax1.twinx()
ax2.plot(stats.index, stats["eval_f1"], marker="s", linestyle="--", color="green", label="eval F1")
ax2.set_ylabel("F1 score")

fig.tight_layout()
fig.legend(loc="upper right")
plt.title("BERT (No XML) Training Curve")
plt.savefig("bert_no_xml_training_curve.png")
plt.close()

# -- Confusion matrix --
preds = trainer.predict(test_dataset).predictions.argmax(-1)
cm = confusion_matrix(test_labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["False", "True"])
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix (BERT without XML)")
plt.tight_layout()
plt.savefig("confusion_matrix_no_xml.png")
plt.close()

# -- ROC curve --
probs = trainer.predict(test_dataset).predictions[:, 1]
fpr, tpr, _ = roc_curve(test_labels, probs)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (BERT without XML)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve_no_xml.png")
plt.close()

# ========== 7. SAVE RESULTS ==========
results_df = pd.DataFrame([{
    "name": experiment_config["name"],
    "epochs": experiment_config["epochs"],
    "batch_size": experiment_config["batch_size"],
    "lr": experiment_config["lr"],
    "accuracy": eval_result.get("eval_accuracy"),
    "f1": eval_result.get("eval_f1"),
    "precision": eval_result.get("eval_precision"),
    "recall": eval_result.get("eval_recall")
}])
results_df.to_csv("bert_no_xml_results.csv", index=False)

print("✅ DONE — no XML version completed and plots saved.")