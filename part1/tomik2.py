import torch
import pandas as pd
# Explicitly import model + tokenizer for clarity
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# 1. Check for GPU
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {device}")

# 2. Load your CSV
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

# 5. Function to apply each NER model in *batches* to statements
def apply_ner_models_in_batches(df, models, device, batch_size=8):
    # Convert statements to list
    statements = df["statement"].tolist()
    
    for new_column, model_name in models:
        print(f"Loading tokenizer/model for: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForTokenClassification.from_pretrained(model_name)
        
        # Create the pipeline
        ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            device=device
        )
        
        # We'll collect the NER results for each row in a list
        all_results = []
        
        # Go in batches
        for i in range(0, len(statements), batch_size):
            batch_texts = statements[i : i + batch_size]
            
            # Call pipeline on multiple texts at once
            outputs = ner_pipeline(batch_texts)  
            # For token classification, 'outputs' is a list of lists (one sub-list per input)
            all_results.extend(outputs)
        
        # Now `all_results` should have the same length as `statements`
        df[new_column] = all_results
    
    return df

# 6. Apply the models
print("Running NER models in batches...")
df = apply_ner_models_in_batches(df, models, device=device, batch_size=8)

# 7. Save to CSV
df.to_csv("output.csv", index=False)
print("Done! Saved to output.csv")