# 📁 Part 2 – Final Experiments & Evaluations

This folder contains the final experiments for our project.  
Each script either tunes hyperparameters or runs final training using the **best setup** from previous grid searches.

---

## 📌 Files Overview

### 1. **`Bert_param_eval`**
- Runs grid search for BERT classifier **with XML tags**.
- Trains on **train**, evaluates on **validation**.
- Saves metrics (accuracy, F1, precision, recall) to CSV.
- Includes `.py`, `.ipynb`, and `.sh` for HPC.

---

### 2. **`Bert_ft_final_train`**
- Trains BERT **with XML tags** using best parameters.
- Uses **train + validation** for training, evaluates on **test** set.
- Saves final model and results (F1, accuracy, etc.).

---

### 3. **`Bert_base_train`**
- Uses **base (non-finetuned) BERT**, **no XML tags**.
- Uses same parameters as the best fine-tuned model.
- Evaluates on **test** set.

---

### 4. **(Skipped)** BiLSTM without GloVe
- Tried BiLSTM without GloVe, got very low F1, not used anymore.

---

### 5. **`BiLSTM_param_eval`**
- Grid search for BiLSTM + GloVe **with XML tags**.
- Trains on **train**, evaluates on **validation**.
- Saves best parameter results to CSV.

---

### 6. **`BiLSTM_GloVe_ft_final`**
- Trains BiLSTM + GloVe **with XML tags** using best params.
- Uses **train + validation**, evaluates on **test**.
- Saves final results (F1, accuracy, etc.).

---

### 7. **`BiLSTM_GloVe_base_train`**
- Trains BiLSTM + GloVe **without XML tags**.
- Uses **same parameters** as the best fine-tuned XML model.
- Evaluates on **test** set.

---

## 📁 Other Info

- `glove_data/` → Contains GloVe embeddings (`glove.6B.100d.txt`).
- `results/` → Contains saved model checkpoints (not reused).
- Each script has a matching `.sh` file for HPC and `.ipynb` for testing locally.

---

## ✅ Summary

| Model | XML | Fine-tuned | File |
|-------|-----|------------|------|
| BERT  | ✅  | ✅          | `Bert_ft_final_train` |
| BERT  | ❌  | ❌ (base)   | `Bert_base_train`     |
| BiLSTM + GloVe | ✅ | ✅ | `BiLSTM_GloVe_ft_final` |
| BiLSTM + GloVe | ❌ | ❌ (base) | `BiLSTM_GloVe_base_train` |

---

Let me know if you need help running any file!