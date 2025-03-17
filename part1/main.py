# 2: Importing libraries
import pandas as pd
from transformers import pipeline
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


# 3: Loading training data
df_all_features = pd.read_csv("../liar2/train.csv")

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


ner_pipeline_A = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english")
# NER tagging, cleaning, extracting
def clean_entities_A(entities):
    """Extracts entity type and cleaned word, removing 'Ġ' artifacts."""
    return [(ent["word"].replace("Ġ", ""), ent["entity"]) for ent in entities]


df = df.copy()  

df["A_raw_entities"] = df["statement"].apply(ner_pipeline_A)
df["A_entities"] = df["A_raw_entities"].apply(clean_entities_A)


# Load dbmdz BERT
ner_pipeline_B = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
# NER tagging, merging, extracting
def merge_entities_B(entities):
    """Merges WordPiece tokens correctly while keeping distinct words separate."""
    merged = []
    current_word = ""
    current_entity = None

    for ent in entities:
        word = ent["word"]
        entity = ent["entity"]

        if word.startswith("##"):  # It's a subword, merge it
            current_word += word[2:]  # Remove "##" and append
        else:  # New word starts
            if current_word:  # Store previous word
                merged.append((current_word, current_entity))
            current_word = word  # Start new word
            current_entity = entity

    # Add last merged entity
    if current_word:
        merged.append((current_word, current_entity))

    return merged



df = df.copy()  

df["B_raw_entities"] = df["statement"].apply(ner_pipeline_B)
df["B_entities"] = df["B_raw_entities"].apply(merge_entities_B)


# 7: Spliting df into true,false
df_true = df[df["label_binary"] == 1].copy()
df_false = df[df["label_binary"] == 0].copy()

# 8: Count entities from true/false of each NER model

# Get entities
entities_A_true = [entity for entities in df_true["A_entities"] for entity in entities]
entities_A_false = [entity for entities in df_false["A_entities"] for entity in entities]
entities_B_true = [entity for entities in df_true["B_entities"] for entity in entities]
entities_B_false = [entity for entities in df_false["B_entities"] for entity in entities]


# Counts
counts_A_true_entities = Counter(entities_A_true)
counts_A_false_entities = Counter(entities_A_false)
counts_B_true_entities = Counter(entities_B_true)
counts_B_false_entities = Counter(entities_B_false)

# Create DF
df_counts_A_true = pd.DataFrame(counts_A_true_entities.items(), columns=["Entity", "Count"])
df_counts_A_false = pd.DataFrame(counts_A_false_entities.items(), columns=["Entity", "Count"])
df_counts_B_true = pd.DataFrame(counts_B_true_entities.items(), columns=["Entity", "Count"])
df_counts_B_false = pd.DataFrame(counts_B_false_entities.items(), columns=["Entity", "Count"])


# 10: Visualization of model A,B Real vs Fake

# Sorting and selecting the top 10 entities from each category
df_counts_A_true["Entity"] = df_counts_A_true["Entity"].astype(str)
df_counts_A_false["Entity"] = df_counts_A_false["Entity"].astype(str)
df_counts_B_true["Entity"] = df_counts_B_true["Entity"].astype(str)
df_counts_B_false["Entity"] = df_counts_B_false["Entity"].astype(str)

true_counts_A = df_counts_A_true.sort_values(by="Count", ascending=False).head(10)
false_counts_A = df_counts_A_false.sort_values(by="Count", ascending=False).head(10)
true_counts_B = df_counts_B_true.sort_values(by="Count", ascending=False).head(10)
false_counts_B = df_counts_B_false.sort_values(by="Count", ascending=False).head(10)

# Set modern seaborn style
sns.set_style("whitegrid")
true_color = "#1f77b4"  # Blue for Real
false_color = "#d62728"  # Red for Fake

# Creating horizontal bar charts
fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

# Plot Model A True entity counts
axes[0, 0].barh(true_counts_A["Entity"], true_counts_A["Count"], color=true_color)
axes[0, 0].set_title("Most Used Entities (Real) - Model A", fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel("Count", fontsize=12)
axes[0, 0].invert_yaxis()

# Plot Model A False entity counts
axes[0, 1].barh(false_counts_A["Entity"], false_counts_A["Count"], color=false_color)
axes[0, 1].set_title("Most Used Entities (Fake) - Model A", fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel("Count", fontsize=12)
axes[0, 1].invert_yaxis()

# Plot Model B True entity counts
axes[1, 0].barh(true_counts_B["Entity"], true_counts_B["Count"], color=true_color)
axes[1, 0].set_title("Most Used Entities (Real) - Model B", fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel("Count", fontsize=12)
axes[1, 0].invert_yaxis()

# Plot Model B False entity counts
axes[1, 1].barh(false_counts_B["Entity"], false_counts_B["Count"], color=false_color)
axes[1, 1].set_title("Most Used Entities (Fake) - Model B", fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel("Count", fontsize=12)
axes[1, 1].invert_yaxis()

# Improve layout
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
plt.show()