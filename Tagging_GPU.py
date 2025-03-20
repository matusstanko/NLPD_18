print("1) Importing libraries...")
import torch
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    pipeline
)
import spacy

print("2) Checking GPU availability...")
device = 0 if torch.cuda.is_available() else -1
print(f"   GPU available = {torch.cuda.is_available()}")
print(f"   Using device = {device}")

print("3) Loading dataset from CSV...")
df_all_features = pd.read_csv("./liar2/test.csv")
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
    ("A_raw_entities", "Jean-Baptiste/roberta-large-ner-english"),
#    ("B_raw_entities", "dbmdz/bert-large-cased-finetuned-conll03-english"),
#    ("C_raw_entities", "vinai/bertweet-base")
]

# load spaCy model
spacy_model = spacy.load("en_core_web_sm")

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
            print(f"      Processing batch {i//batch_size + 1} (index range: {i} to {i+len(batch_texts)-1})")

            # Call pipeline on multiple texts
            outputs = ner_pipeline(batch_texts)
            all_results.extend(outputs)

        assert len(all_results) == len(statements), (
            f"Mismatch: got {len(all_results)} results for {len(statements)} statements"
        )

        df[new_column] = all_results
        print(f"   Done with model: {model_name}. New column: {new_column}")

    return df

print("6) Applying all transformer-based NER models in batches...")
df = apply_ner_models_in_batches(df, models, device=device, batch_size=8)

print("7) Applying spaCy NER...")

def extract_entities(text):
    return [{"word": ent.text, "entity": ent.label_} for ent in spacy_model(text).ents]

df["B_raw_entities"] = df["statement"].apply(extract_entities)


print("8) Saving results to 'output.csv'...")
df.to_csv('output_train.csv', index=False)

print("All done!")
