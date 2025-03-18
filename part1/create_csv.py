# 2: Importing libraries
import pandas as pd
from transformers import pipeline

# 3: Loading training data
df_all_features = pd.read_csv("../liar2/train_sample.csv")
df = df_all_features[["statement", "label"]]

# 5: Adding binary column (true/false)
true_labels = [5, 4, 3]
false_labels = [0, 1, 2]
def convert_label(label):
    if label in true_labels:  
        return 1
    elif label in false_labels:
        return 0
    else:
        return 'Out-Of-Range'
df = df.copy()
df.loc[:, "label_binary"] = df["label"].apply(convert_label)



# List of models to use
models = [
    ("A_raw_entities", "Jean-Baptiste/roberta-large-ner-english"),
    ("B_raw_entities", "dbmdz/bert-large-cased-finetuned-conll03-english")
    #("C_raw_entities", "vinai/bertweet-base")
]

# Function to apply multiple NER models dynamically
def apply_ner_models(df, models):
    for column_name, model_name in models:
        ner_pipeline = pipeline("ner", model=model_name)
        df[column_name] = df["statement"].apply(lambda x: ner_pipeline(x))
    return df

# Apply all models to the dataframe
df = apply_ner_models(df, models) 
df.to_csv('output.csv', index=False)