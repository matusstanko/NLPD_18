from transformers import pipeline
from concurrent.futures import ProcessPoolExecutor
import pandas as pd

# Load model once outside the function
model_name = "Jean-Baptiste/roberta-large-ner-english"
ner_pipeline = pipeline("ner", model=model_name, device=-1)  # CPU-only

# Function to process a batch of statements
def process_statements(statements):
    return ner_pipeline(statements, batch_size=8)  # Efficient batch processing

df_all_features = pd.read_csv("../liar2/train_sample.csv")
df = df_all_features[["statement", "label"]]

# Convert labels
true_labels = [5, 4, 3]
false_labels = [0, 1, 2]
df["label_binary"] = df["label"].apply(lambda x: 1 if x in true_labels else 0)

# Process in parallel using ProcessPoolExecutor
with ProcessPoolExecutor() as executor:
    df["A_raw_entities"] = list(executor.map(process_statements, df["statement"]))

df.to_csv('output_A_opt.csv', index=False)