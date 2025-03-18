print('2: Importing libraries')
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, IntegerType, ArrayType, StructType, StructField
from transformers import pipeline

# Initialize Spark session
spark = SparkSession.builder.appName("NER_Pipeline").getOrCreate()

print('3: Loading training data')
df_all_features = spark.read.csv("../liar2/train.csv", header=True, inferSchema=True)
df = df_all_features.select("statement", "label")

print('5: Adding binary column (true/false)')
true_labels = {5, 4, 3}
false_labels = {0, 1, 2}

# Define a UDF (User Defined Function) to convert labels
@udf(IntegerType())
def convert_label(label):
    if label in true_labels:  
        return 1
    elif label in false_labels:
        return 0
    return None  # Handle unexpected cases

df = df.withColumn("label_binary", convert_label(col("label")))

# List of models to use
models = [
    ("A_raw_entities", "Jean-Baptiste/roberta-large-ner-english"),
    ("B_raw_entities", "dbmdz/bert-large-cased-finetuned-conll03-english"),
    ("C_raw_entities", "vinai/bertweet-base")
]

# Load models (avoiding reloading them multiple times)
ner_pipelines = {name: pipeline("ner", model=model_name) for name, model_name in models}

# Define a Spark UDF for Named Entity Recognition
@udf(ArrayType(StructType([
    StructField("entity", StringType(), True),
    StructField("word", StringType(), True),
    StructField("score", StringType(), True),
    StructField("start", IntegerType(), True),
    StructField("end", IntegerType(), True)
])))
def apply_ner(text, model_name):
    if text is None:
        return []
    pipeline_model = ner_pipelines.get(model_name)
    return pipeline_model(text) if pipeline_model else []

# Apply all NER models dynamically
for column_name, model_name in models:
    df = df.withColumn(column_name, apply_ner(col("statement"), model_name))

# Save results as a CSV file in SDU Cloud
df.write.csv('output.csv', header=True, mode="overwrite")

print("Processing complete. Results saved to output.csv")