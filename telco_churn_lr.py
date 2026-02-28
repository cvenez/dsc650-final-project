from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

import happybase
from datetime import datetime

# 1) Spark session with Hive support
spark = (
    SparkSession.builder
    .appName("TelcoChurn_LogReg_to_HBase")
    .enableHiveSupport()
    .getOrCreate()
)

# 2) Load data from Hive
df = spark.sql("""
SELECT
  customerID,
  gender,
  SeniorCitizen,
  Partner,
  Dependents,
  tenure,
  PhoneService,
  MultipleLines,
  InternetService,
  OnlineSecurity,
  OnlineBackup,
  DeviceProtection,
  TechSupport,
  StreamingTV,
  StreamingMovies,
  Contract,
  PaperlessBilling,
  PaymentMethod,
  MonthlyCharges,
  TotalCharges,
  Churn
FROM final_project.telco_churn
""")
#Cast numeric columns explicitly
df = df.withColumn("SeniorCitizen", F.col("SeniorCitizen").cast("int")) \
       .withColumn("tenure", F.col("tenure").cast("int")) \
       .withColumn("MonthlyCharges", F.col("MonthlyCharges").cast("double")) \
       .withColumn("TotalCharges", F.col("TotalCharges").cast("double"))
# 3) Basic cleanup: drop rows with nulls in key fields
df = df.na.drop(subset=["Churn", "tenure", "MonthlyCharges"])

# Some versions of this dataset have TotalCharges null/blank; fill with 0 for modeling
df = df.fillna({"TotalCharges": 0.0})

# 4) Label column (Churn Yes/No -> numeric)
label_indexer = StringIndexer(inputCol="Churn", outputCol="label", handleInvalid="skip")

# 5) Categorical columns to encode
categorical_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]

indexers = [
    StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="skip")
    for c in categorical_cols
]

encoder = OneHotEncoder(
    inputCols=[f"{c}_idx" for c in categorical_cols],
    outputCols=[f"{c}_ohe" for c in categorical_cols],
    handleInvalid="keep"
)

# 6) Numeric columns
numeric_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# 7) Assemble features
assembler = VectorAssembler(
    inputCols=[f"{c}_ohe" for c in categorical_cols] + numeric_cols,
    outputCol="features",
    handleInvalid="skip"
)

# 8) Model
lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=50)

pipeline = Pipeline(stages=[label_indexer] + indexers + [encoder, assembler, lr])

# 9) Train/test split
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

model = pipeline.fit(train_df)
pred = model.transform(test_df).cache()

# 10) Metrics
auc_eval = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
auc = auc_eval.evaluate(pred)

acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
f1_eval  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
prec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
rec_eval  = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")

accuracy = acc_eval.evaluate(pred)
f1 = f1_eval.evaluate(pred)
precision = prec_eval.evaluate(pred)
recall = rec_eval.evaluate(pred)

print("=== Telco Churn Logistic Regression Metrics ===")
print(f"AUC (areaUnderROC): {auc}")
print(f"Accuracy: {accuracy}")
print(f"F1: {f1}")
print(f"Weighted Precision: {precision}")
print(f"Weighted Recall: {recall}")

# 11) Write metrics to HBase
run_id = "run_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")
metrics = [
    (run_id, "cf:auc", str(auc)),
    (run_id, "cf:accuracy", str(accuracy)),
    (run_id, "cf:f1", str(f1)),
    (run_id, "cf:precision", str(precision)),
    (run_id, "cf:recall", str(recall)),
]

def write_partition(partition):
    connection = happybase.Connection(host="master")
    connection.open()
    table = connection.table("telco_metrics")
    for row_key, col, val in partition:
        table.put(row_key, {col: val})
    connection.close()

spark.sparkContext.parallelize(metrics, 1).foreachPartition(write_partition)

print(f"Metrics written to HBase table telco_metrics under row key: {run_id}")

spark.stop()
