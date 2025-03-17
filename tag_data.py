### 1. SERIAL (VELMI POMALE)

# import pandas as pd
# from transformers import pipeline

# print('Loading file')
# df_all_features = pd.read_csv("../liar2/train.csv")

# print('Separating labels')
# df = df_all_features[["statement", "label"]]

# print('Adding true/false column')
# true_labels = [5, 4, 3]
# false_labels = [0, 1, 2]

# def convert_label(label):
#     if label in true_labels:  
#         return 1
#     elif label in false_labels:
#         return 0
#     else:
#         return 'Out-Of-Range'

# df = df.copy()
# df.loc[:, "label_binary"] = df["label"].apply(convert_label)


# print('NER model A')
# ner_pipeline_A = pipeline("ner", model="Jean-Baptiste/roberta-large-ner-english")
# df = df.copy()  
# df["A_raw_entities"] = df["statement"].apply(ner_pipeline_A)


# print('NER model B')
# ner_pipeline_B = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
# df = df.copy()  
# df["B_raw_entities"] = df["statement"].apply(ner_pipeline_B)


# print('NER model C - News-based')
# ner_pipeline_C = pipeline("ner", model="dslim/bert-large-NER")
# df = df.copy()  
# df["C_raw_entities"] = df["statement"].apply(ner_pipeline_C)

# print('NER model D - Social Media & News')
# ner_pipeline_D = pipeline("ner", model="vinhkhuc/BERTweet-NER")
# df = df.copy()  
# df["D_raw_entities"] = df["statement"].apply(ner_pipeline_D)

# print('NER model E - Flair-based (Non-BERT)')
# ner_pipeline_E = pipeline("ner", model="flair/ner-english-large")
# df = df.copy()  
# df["E_raw_entities"] = df["statement"].apply(ner_pipeline_E)


### 2. 


from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from transformers import pipeline

# Initialize Spark Session
spark = SparkSession.builder.appName("NER-Spark").getOrCreate()

print('Loading file')
df_spark = spark.read.csv("../liar2/train.csv", header=True, inferSchema=True)

print('Separating labels')
df_spark = df_spark.select("statement", "label")

print('Adding true/false column')
true_labels = [5, 4, 3]
false_labels = [0, 1, 2]

def convert_label(label):
    return 1 if label in true_labels else 0 if label in false_labels else 'Out-Of-Range'

df_spark = df_spark.withColumn("label_binary", col("label").cast("integer").alias("label")).rdd.map(lambda row: (row.statement, row.label, convert_label(row.label))).toDF(["statement", "label", "label_binary"])

# Define NER processing function for partitions
def process_partition(iter, model_name):
    ner_pipeline = pipeline("ner", model=model_name)
    return [(row.statement, row.label, row.label_binary, ner_pipeline(row.statement)) for row in iter]

# Apply NER models using mapPartitions
models = {
    "A_raw_entities": "Jean-Baptiste/roberta-large-ner-english",
    "B_raw_entities": "dbmdz/bert-large-cased-finetuned-conll03-english",
    "C_raw_entities": "dslim/bert-large-NER",
    "D_raw_entities": "vinhkhuc/BERTweet-NER",
    "E_raw_entities": "flair/ner-english-large"
}

for col_name, model_name in models.items():
    print(f'Processing {col_name} using {model_name}')
    df_rdd = df_spark.rdd.mapPartitions(lambda iter: process_partition(iter, model_name))
    df_spark = df_rdd.toDF(df_spark.columns + [col_name])

# Save final dataframe as CSV
df_spark.write.csv("ner_results.csv", header=True, mode="overwrite")

print("NER processing completed. Results saved to ner_results.csv")