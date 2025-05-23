print("1) Importing libraries...")
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    pipeline
)



print("2) Checking GPU availability...")
device = 0 if torch.cuda.is_available() else -1
print(f"   GPU available = {torch.cuda.is_available()}")
print(f"   Using device = {device}")

print("3) Loading dataset from CSV...")
df_all_features = pd.read_csv("/home/matus/NLPD_18/liar2/valid.csv")
df = df_all_features[["statement", "label"]].copy()
print(f"   DataFrame shape = {df.shape}")

print("4) Converting labels to binary...")
true_labels = [5, 4, 3]  # e.g. mostly-true, half-true, true
false_labels = [0, 1, 2] # e.g. false, pants-fire, mostly-false

def convert_label(label):
    if label in true_labels:
        return 1
    elif label in false_labels:
        return 0
    else:
        return 'Out-Of-Range'

df["label_binary"] = df["label"].apply(convert_label)
print(f"   Example rows:\n{df.head()}")

print("5) Defining NER models to use...")
models = [
    ("A_raw_entities", "Jean-Baptiste/roberta-large-ner-english")
]

def apply_ner_models_in_batches(df, models, device, batch_size=8):
    """Apply multiple NER models in mini-batches to the 'statement' column."""
    statements = df["statement"].tolist()
    print(f"   Number of statements = {len(statements)}")
    print(f"   Batch size = {batch_size}")

    for new_column, model_name in models:
        print(f"\n--> Loading tokenizer/model for: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)

        print("   Creating NER pipeline...")
        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            device=device
        )

        all_results = []
        print(f"   Running inference in batches of size {batch_size}...")

        for i in range(0, len(statements), batch_size):
            batch_texts = statements[i : i + batch_size]
            # Print current progress
            print(f"      Processing batch {i//batch_size + 1} (index range: {i} to {i+len(batch_texts)-1})")

            # Call pipeline on multiple texts
            outputs = ner_pipeline(batch_texts)
            # outputs is a list of lists, one sub-list per input text
            all_results.extend(outputs)

        # Make sure length matches
        assert len(all_results) == len(statements), (
            f"Mismatch: got {len(all_results)} results for {len(statements)} statements"
        )

        # Assign to the DataFrame
        #####
        # Clean each entity: convert np.float32 scores to float
        def convert_scores(entities):
            for ent in entities:
                if isinstance(ent.get("score", None), torch.Tensor):
                    ent["score"] = float(ent["score"].item())
                elif isinstance(ent.get("score", None), (np.float32, float)):
                    ent["score"] = float(ent["score"])
            return entities

        # Convert every row of entities
        cleaned_results = [convert_scores(row) for row in all_results]

        # Assign to the DataFrame
        df[new_column] = cleaned_results

    return df

print("6) Applying all NER models in batches...")
df = apply_ner_models_in_batches(df, models, device=device, batch_size=8)

print("7) Saving results to 'output.csv'...")
df.to_csv('output_valid.csv', index=False)

print("All done!")