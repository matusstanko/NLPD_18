print('1: Importing libraries')
import torch
import pandas as pd
from transformers import pipeline

# -----------------------------------------------------
# 1. Check for GPU availability
# -----------------------------------------------------
device = 0 if torch.cuda.is_available() else -1
print(f'GPU available: {torch.cuda.is_available()} (using device={device})')

# -----------------------------------------------------
# 2. Load your training data
# -----------------------------------------------------
print('2: Loading training data')
df_all_features = pd.read_csv("./liar2/train.csv")
df = df_all_features[["statement", "label"]]

# -----------------------------------------------------
# 3. Convert labels to binary (true/false)
# -----------------------------------------------------
print('3: Converting labels to binary')
true_labels = [5, 4, 3]  # e.g. mostly-true, half-true, true
false_labels = [0, 1, 2] # e.g. false, pants-fire, mostly-false

def convert_label(label):
    if label in true_labels:  
        return 1
    elif label in false_labels:
        return 0
    else:
        return 'Out-Of-Range'

df = df.copy()
df.loc[:, "label_binary"] = df["label"].apply(convert_label)

# -----------------------------------------------------
# 4. Define the NER models
#    Note: "vinai/bertweet-base" is not a typical NER model
#          but we'll leave it here as in your original code.
# -----------------------------------------------------
print('4: Defining models list')
models = [
    ("A_raw_entities", "Jean-Baptiste/roberta-large-ner-english"),
    ("B_raw_entities", "dbmdz/bert-large-cased-finetuned-conll03-english"),
    ("C_raw_entities", "vinai/bertweet-base")
]

# -----------------------------------------------------
# 5. Function to apply NER models
# -----------------------------------------------------
def apply_ner_models(df, models, device):
    for column_name, model_name in models:
        print(f'Loading pipeline for: {model_name}')
        ner_pipeline = pipeline("ner", model=model_name, device=device)
        df[column_name] = df["statement"].apply(lambda x: ner_pipeline(x))
    return df

# -----------------------------------------------------
# 6. Apply all models to the dataframe
# -----------------------------------------------------
print('5: Applying models...')
df = apply_ner_models(df, models, device=device)

# -----------------------------------------------------
# 7. Save output to CSV
# -----------------------------------------------------
print('6: Saving output...')
df.to_csv('output.csv', index=False)
print('Done!')