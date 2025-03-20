import torch 
import pandas as pd  
import spacy 
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline  

# check gpu availability
device = 0 if torch.cuda.is_available() else -1  # set device to GPU if available, else use CPU
print(f"gpu available: {torch.cuda.is_available()}, using device: {device}") # tihs is just for faster processing

# load dataset
df = pd.read_csv("./liar2/train_sample.csv")[["statement", "label"]]  # select only the relevant cols
df["label_binary"] = df["label"].isin([3, 4, 5]).astype(int)  # convert labels to binary (true/false)

# load transformer model
tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english")  # load tokenizer
model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english")  # load model
ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, device=device)  # create NER pipeline

# apply ner in batches
batch_size = 8  # define batch size for processing
statements = df["statement"].tolist() 
print(f"processing {len(statements)} statements in batches of {batch_size}")
all_results = []  # initialize list to store results

for i in range(0, len(statements), batch_size):  # iterate through statements in batches
    print(f"batch {i//batch_size + 1}: {i} to {i+batch_size-1}")  # log batch info
    all_results.extend(ner_pipeline(statements[i:i+batch_size]))  # apply NER and store results

df["A_raw_entities"] = all_results  # save results in DF

# apply spacy ner
spacy_model = spacy.load("en_core_web_sm")  # load spaCy model
def extract_entities(text):
    entities = []
    for ent in spacy_model(text).ents:
        entities.append({"word": ent.text, "entity": ent.label_})
    return entities

df["B_raw_entities"] = df["statement"].apply(extract_entities)  # extract entities

# save results
df.to_csv("output_ABCD.csv", index=False)  # save processed DataFrame to CSV
print("all done!")  # print completion message
