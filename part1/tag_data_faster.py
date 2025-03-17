from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from transformers import pipeline
import spacy

# --------------------------------------------------------------------------
# 1. SPARK SESSION & DATA LOADING
# --------------------------------------------------------------------------
spark = SparkSession.builder.appName("NER-Spark-Fast").getOrCreate()

print('Loading file...')
df_spark = spark.read.csv("../liar2/train.csv", header=True, inferSchema=True)

print('Selecting statement & label columns...')
df_spark = df_spark.select("statement", "label")

print('Adding true/false column...')
true_labels = [5, 4, 3]
false_labels = [0, 1, 2]

def convert_label(label):
    return 1 if label in true_labels else 0 if label in false_labels else 'Out-Of-Range'

df_spark = (
    df_spark
    .withColumn("label_binary", col("label").cast("integer"))
    .rdd.map(lambda row: (row.statement, row.label, convert_label(row.label)))
    .toDF(["statement", "label", "label_binary"])
)

# Optional: Control number of partitions to avoid re-loading each model too many times.
# Example: df_spark = df_spark.repartition(4)

# --------------------------------------------------------------------------
# 2. PARTITION-LEVEL PROCESSING FUNCTION
# --------------------------------------------------------------------------
def process_partition(rows, model_name):
    """
    For a given partition of rows, load the model/pipeline ONCE,
    then apply it to each row's statement.
    """
    # Load the pipeline (or spaCy model) only once per partition:
    if model_name == "en_core_web_trf":
        # SpaCy pipeline
        nlp_spacy = spacy.load("en_core_web_trf")

        def ner_function(txt):
            doc = nlp_spacy(txt)
            return [
                {
                    "entity": ent.label_,
                    "word": ent.text,
                    "start": ent.start_char,
                    "end": ent.end_char
                }
                for ent in doc.ents
            ]

    elif model_name == "stanford-ner-crf-based":
        # Placeholder for a classical Stanford approach
        def ner_function(txt):
            return [{"entity": "PLACEHOLDER_STANFORD", "word": txt}]

    else:
        # For regular Hugging Face pipelines
        hf_pipeline = pipeline("ner", model=model_name)

        def ner_function(txt):
            return hf_pipeline(txt)

    # Now iterate the rows in this partition and run NER
    results = []
    for row in rows:
        statement, label, label_binary = row
        ner_result = ner_function(statement)
        results.append((statement, label, label_binary, ner_result))

    return results

# --------------------------------------------------------------------------
# 3. MODEL DICTIONARY
# --------------------------------------------------------------------------
models = {
    "A_raw_entities": "Jean-Baptiste/roberta-large-ner-english",
    "B_raw_entities": "dbmdz/bert-large-cased-finetuned-conll03-english",
    "C_raw_entities": "dslim/bert-large-NER",
    "D_raw_entities": "vinhkhuc/BERTweet-NER",
    "E_raw_entities": "flair/ner-english-large",
    # Newly added models:
    "F_raw_entities": "en_core_web_trf",  # SpaCy
    "G_raw_entities": "stanford-ner-crf-based",  # Stanford placeholder
    "H_raw_entities": "xlm-roberta-large-finetuned-conll03-english",  # XLM-R
}

# --------------------------------------------------------------------------
# 4. RUN NER FOR EACH MODEL
# --------------------------------------------------------------------------
for col_name, model_name in models.items():
    print(f'Processing {col_name} with model={model_name} ...')
    # We use .mapPartitions() to ensure we only load the model once per partition.
    df_rdd = df_spark.rdd.mapPartitions(lambda rows: process_partition(rows, model_name))
    df_spark = df_rdd.toDF(df_spark.columns + [col_name])

# --------------------------------------------------------------------------
# 5. SAVE THE RESULT
# --------------------------------------------------------------------------
df_spark.write.csv("ner_results.csv", header=True, mode="overwrite")
print("NER processing completed. Results saved to ner_results.csv.")