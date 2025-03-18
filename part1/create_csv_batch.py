print('2: Importing libraries')
import pandas as pd
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor

print('3: Loading training data')
df_all_features = pd.read_csv("../liar2/train_sample.csv")
df = df_all_features[["statement", "label"]]

print('5: Adding binary column (true/false)')
true_labels = [5, 4, 3]
false_labels = [0, 1, 2]

def convert_label(label):
    return 1 if label in true_labels else (0 if label in false_labels else 'Out-Of-Range')

df = df.copy()
df.loc[:, "label_binary"] = df["label"].apply(convert_label)

# List of models to use (optimized for GPU)
models = [
    ("A_raw_entities", "Jean-Baptiste/roberta-large-ner-english"),
    ("B_raw_entities", "dbmdz/bert-large-cased-finetuned-conll03-english"),
    ("C_raw_entities", "vinai/bertweet-base")
]

# Function to apply multiple NER models dynamically
def process_statements(model_name, statements):
    ner_pipeline = pipeline("ner", model=model_name, device=0)  # Use GPU
    return list(ner_pipeline(statements, batch_size=8))  # Batch processing

def apply_ner_models(df, models):
    for column_name, model_name in models:
        print(f"Processing {column_name} with {model_name} on GPU...")
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(lambda x: process_statements(model_name, x), df["statement"]))
        df[column_name] = results
    return df

# Apply all models to the dataframe
df = apply_ner_models(df, models)
df.to_csv('output.csv', index=False)