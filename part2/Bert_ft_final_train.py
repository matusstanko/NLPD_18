# Import libraries
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
import torch
print('1/6 libraries loaded')


# Choose parameters for BERT 
# These were decided from gridsearch training

experiment_configs = [{
    "name": "best_model_run",
    "epochs": 2,
    "batch_size": 32,
    "lr": 2e-5
}]

# Load train, valid and test data
df_train = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_train.csv')
df_valid = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_valid.csv')
df_test = pd.read_csv('/home/matus/NLPD_18/part1/outputs/output_test.csv')
print('2/6 data loaded')

# Join train+validation data and shuffle 
df_train_merged = pd.concat([df_train, df_valid], ignore_index=True)
df_train_merged = df_train_merged.sample(frac=1, random_state=42).reset_index(drop=True)



# Functions to extract and insert XML tags (entities) into statement text
def merge_adjacent_entities(entities):
    if not entities:
        return []

    # Sort entities by start position
    entities = sorted(entities, key=lambda x: x['start'])
    merged = [entities[0]]

    for current in entities[1:]:
        last = merged[-1]

        # Merge adjacent entities of the same type
        if current['entity'] == last['entity'] and current['start'] <= last['end'] + 1:
            last['end'] = current['end']
        else:
            merged.append(current)

    return merged
def insert_xml_tags(text, entities):
    if not entities:
        return text

    # If entities are a string, try to evaluate it
    if isinstance(entities, str):
        try:
            entities = ast.literal_eval(entities)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing entities: {entities} - {e}")
            return text  # Skip this row or handle it differently

    # Merge adjacent entities
    merged_entities = merge_adjacent_entities(entities)
    
    # Sort entities by their start position
    merged_entities.sort(key=lambda x: x['start'])

    offset = 0
    for ent in merged_entities:
        ent_type = ent['entity']
        start = ent['start'] + offset
        end = ent['end'] + offset

        open_tag = f"<{ent_type}>"
        close_tag = f"</{ent_type}>"

        text = text[:start] + open_tag + text[start:end] + close_tag + text[end:]
        offset += len(open_tag) + len(close_tag)

    return text




tqdm.pandas() # Progress bar

# Use defined functions to add new col with XML tags in statements
df_train_merged['A_XML_statement'] = df_train_merged.progress_apply(
    lambda row: insert_xml_tags(row['statement'], row['A_raw_entities']),
    axis=1
)

df_test['A_XML_statement'] = df_test.progress_apply(
    lambda row: insert_xml_tags(row['statement'], row['A_raw_entities']),
    axis=1
)
print('3/6 train,valid XML extracted and added')




# Add XML tags to tokenizer vocabulary
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# Pridaj špeciálne XML tagy
special_tokens = ['<PER>', '</PER>', '<ORG>', '</ORG>', '<LOC>', '</LOC>', '<MISC>', '</MISC>']
tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

# Load bert for classification
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.resize_token_embeddings(len(tokenizer))

train_encodings = tokenizer(
    df_train_merged['A_XML_statement'].tolist(),
    truncation=True,
    padding=True
)

test_encodings = tokenizer(
    df_test['A_XML_statement'].tolist(),
    truncation=True,
    padding=True
)

train_labels = df_train_merged['label_binary'].tolist()
test_labels = df_test['label_binary'].tolist()





# Save as dataset
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NERDataset(train_encodings, train_labels)
test_dataset = NERDataset(test_encodings, test_labels)
print('4/6 Converted to datasets')




def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds),
        'precision': precision_score(labels, preds),
        'recall': recall_score(labels, preds)
    }



# TRAIN
print('Training in progress')

all_results = []

for config in experiment_configs:
    print(f"Training {config['name']}...")

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
        report_to="none",  # don't use wandb/huggingface
        metric_for_best_model="f1",        # 👈 Use F1 for early stopping
        greater_is_better=True,            # 👈 Higher F1 = better
        load_best_model_at_end=True,
        fp16=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    trainer.train()
    eval_result = trainer.evaluate()

    all_results.append({
    "name": config["name"],
    "epochs": config["epochs"],
    "batch_size": config["batch_size"],
    "lr": config["lr"],
    "accuracy": eval_result.get("eval_accuracy", None),
    "f1": eval_result.get("eval_f1", None),
    "precision": eval_result.get("eval_precision", None),
    "recall": eval_result.get("eval_recall", None),
    "best_f1": trainer.state.best_metric,
    "final_epoch": trainer.state.epoch,
    "best_checkpoint": trainer.state.best_model_checkpoint,
    "best_step": int(trainer.state.best_model_checkpoint.split("-")[-1]) if trainer.state.best_model_checkpoint else None
})



print('5/6 training done. Saving as csv')

results_df = pd.DataFrame(all_results)

results_df.to_csv("bert_ft_final_train.csv", index=False)
print('6/6 saved as .csv')


trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")

