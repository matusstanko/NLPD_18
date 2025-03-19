import torch
import pandas as pd
from transformers import pipeline

# 1. GPU or CPU
device = 0 if torch.cuda.is_available() else -1
print("Using device:", device)

# 2. Load your data
df_all_features = pd.read_csv("./liar2/train.csv")
df = df_all_features[["statement", "label"]].copy()

# 3. Convert labels to binary
true_labels = [5, 4, 3]
false_labels = [0, 1, 2]
def convert_label(label):
    if label in true_labels:
        return 1
    elif label in false_labels:
        return 0
    else:
        return 'Out-Of-Range'

df["label_binary"] = df["label"].apply(convert_label)

# 4. Models to run
models = [
    ("A_raw_entities", "Jean-Baptiste/roberta-large-ner-english"),
    ("B_raw_entities", "dbmdz/bert-large-cased-finetuned-conll03-english"),
    ("C_raw_entities", "vinai/bertweet-base")
]

# 5. Function to apply each NER model on all statements (in batches)
def apply_ner_models_in_batches(df, models, device, batch_size=8):
    # Convert the 'statement' column to a list of strings
    statements = df["statement"].tolist()
    
    for new_column, model_name in models:
        print(f"Loading pipeline for: {model_name}")
        ner_pipeline = pipeline("ner", model=model_name, device=device)

        # We'll store the results for each row in this list
        all_results = []

        # Process the data in smaller batches
        for i in range(0, len(statements), batch_size):
            batch_texts = statements[i : i + batch_size]
            # Pass multiple texts to the pipeline at once
            batch_output = ner_pipeline(batch_texts, truncation=True)
            all_results.extend(batch_output)

        # After inference, write back to our dataframe
        df[new_column] = all_results
        
    return df

print("Running NER models in batches...")
df = apply_ner_models_in_batches(df, models, device=device, batch_size=8)

df.to_csv("output.csv", index=False)
print("Done! Saved to output.csv")