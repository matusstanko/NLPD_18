from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType
from transformers import pipeline
from functools import lru_cache

# Initialize Spark session with increased memory and optimized core usage
spark = SparkSession.builder \
    .appName("NER_Pipeline") \
    .config("spark.driver.memory", "16g") \
    .config("spark.executor.memory", "16g") \
    .config("spark.executor.cores", "16") \
    .config("spark.sql.shuffle.partitions", "32") \
    .getOrCreate()

# Load data
print('Loading training data')
df_all_features = spark.read.csv("./liar2/train.csv", header=True, inferSchema=True)
df = df_all_features.select("statement", "label")

# Define binary label conversion
true_labels = {5, 4, 3}
false_labels = {0, 1, 2}

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

# Cache models per executor to optimize loading
@lru_cache(maxsize=3)
def load_model(model_name):
    return pipeline("ner", model=model_name)

def apply_ner_model(text, model_name):
    if text is None:
        return []
    ner_pipeline = load_model(model_name)  # Load once per executor
    return ner_pipeline(text)

apply_ner_udf = udf(apply_ner_model, ArrayType(StructType([
    StructField("entity", StringType(), True),
    StructField("word", StringType(), True),
    StructField("score", StringType(), True),
    StructField("start", IntegerType(), True),
    StructField("end", IntegerType(), True)
])))

# Apply models dynamically
for column_name, model_name in models:
    df = df.withColumn(column_name, apply_ner_udf(col("statement"), model_name))

# Save results as CSV
df.write.csv('output.csv', header=True, mode="overwrite")

print("Processing complete. Results saved to output.csv")