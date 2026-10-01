import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = (
    SparkSession.builder
    .appName("MIMIC-IV")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "200")
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
    .getOrCreate()
)
DATA_PATH = "physionet.org/files/mimiciv/3.1/"

df_hosp_lab = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .option("multiLine", "false")
    .csv(DATA_PATH + "hosp/labevents.csv.gz")
)

df_admissions = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("multiLine", "false")
    .csv(DATA_PATH + "hosp/admissions.csv.gz")
)

df_patients = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .option("multiLine", "false")
    .csv(DATA_PATH + "hosp/patients.csv.gz")
)

df_icu_char = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "false")
    .option("multiLine", "false")
    .csv(DATA_PATH + "icu/chartevents.csv.gz")
)

SaO2_id = 50817
SpO2_id = 220277

df_admissions = df_admissions.withColumn("admittime", F.to_timestamp("admittime")) \
                             .withColumn("dischtime", F.to_timestamp("dischtime"))
df_hosp_lab = df_hosp_lab.withColumn("charttime", F.to_timestamp("charttime"))
df_icu_char = df_icu_char.withColumn("charttime", F.to_timestamp("charttime"))

df_hosp_lab = df_hosp_lab.withColumn("itemid", df_hosp_lab["itemid"].cast("int"))
df_icu_char = df_icu_char.withColumn("itemid", df_icu_char["itemid"].cast("int"))

df_SaO2 = df_hosp_lab.filter((df_hosp_lab['itemid'] == SaO2_id)&(df_hosp_lab['valuenum'] >= 70)&(df_hosp_lab['valuenum'] <= 100))
df_SpO2 = df_icu_char.filter((df_icu_char['itemid'] == SpO2_id)&(df_icu_char['valuenum'] >= 70)&(df_icu_char['valuenum'] <= 100))

df_SaO2.cache()
df_SpO2.cache()

df_SaO2.count(),df_SpO2.count()

df_SaO2 = df_SaO2.withColumn("join_date", F.to_date("charttime"))
df_SpO2 = df_SpO2.withColumn("join_date", F.to_date("charttime"))

sao2_df = df_SaO2.select(
    "subject_id", "join_date",
    F.col("charttime").alias("sao2_time"),
    F.col("valuenum").alias("SaO2")
)

spo2_df = df_SpO2.select(
    "subject_id", "join_date", "stay_id",
    F.col("charttime").alias("spo2_time"),
    F.col("valuenum").alias("SpO2")
)

time_window_seconds = 5 * 60

oxygen_pairs = sao2_df.join(
    spo2_df,
    on=[
        sao2_df.subject_id == spo2_df.subject_id,
        sao2_df.join_date == spo2_df.join_date,

        spo2_df.spo2_time >= sao2_df.sao2_time - F.expr(f"INTERVAL {time_window_seconds} SECONDS"),
        spo2_df.spo2_time <= sao2_df.sao2_time + F.expr(f"INTERVAL {time_window_seconds} SECONDS")
    ],
    how="inner"
).drop(spo2_df.subject_id).drop(spo2_df.join_date)

df = oxygen_pairs.join(
    df_patients,
    on="subject_id"
).join(
    df_admissions,
    on="subject_id"
)

df = df.filter(
    (F.col("sao2_time") >= F.col("admittime")) &
    (F.col("sao2_time") <= F.col("dischtime"))
)

df_pd = df.toPandas()
df_pd.to_csv("input/merged_data.csv", index=False)
df.count()

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("input/merged_data.csv")
    .withColumn("subject_id", F.col("subject_id").cast("int"))
    .withColumn("sao2_time", F.to_timestamp("sao2_time"))
)

lab_ids = {
    "hemoglobin": [50811, 51222], "hematocrit": [51221],
    "creatinine": [50912], "lactate": [50813],
    "pH": [50820], "pCO2": [50818], "pO2": [50821],
    "carboxyhemoglobin": [50804], "methemoglobin": [50814]
}

icu_ids = {
    "heart_rate": [220045],
    "respiratory_rate": [220210],
    "sbp_non_invasive": [220179],
    "dbp_non_invasive": [220180],
    "mbp_non_invasive": [220181],
    "temperature": [223762],
    "weight": [224639, 226512],
    "height": [226730]
}

all_itemids = [item for sublist in lab_ids.values() for item in sublist]
all_icu_itemids = [item for sublist in icu_ids.values() for item in sublist]

df_labs_filtered = df_hosp_lab.filter(F.col("itemid").cast("int").isin(all_itemids)) \
    .select(
        F.col("subject_id").cast("int"),
        F.col("charttime").cast("timestamp").alias("lab_time"),
        F.col("itemid").cast("int"),
        F.col("valuenum").cast("float")
    )

df_icu_filtered = df_icu_char.filter(F.col("itemid").cast("int").isin(all_icu_itemids)) \
    .select(
        F.col("subject_id").cast("int"),
        F.col("charttime").cast("timestamp").alias("icu_time"),
        F.col("itemid").cast("int"),
        F.col("valuenum").cast("float")
    )

def map_itemid_lab(id_col):
    expr = F.when(id_col.isin(lab_ids["hemoglobin"]), "cbc_hemoglobin")
    for label, ids in lab_ids.items():
        if label != "hemoglobin":
            expr = expr.when(id_col.isin(ids), f"lab_{label}")
    return expr.otherwise("unknown")

df_labs_pivoted = df_labs_filtered.withColumn("feature", map_itemid_lab(F.col("itemid"))) \
    .groupBy("subject_id", "lab_time") \
    .pivot("feature") \
    .avg("valuenum")

def map_itemid_icu(id_col):
    expr = F.when(F.lit(False), "dummy")
    for label, ids in icu_ids.items():
        expr = expr.when(id_col.isin(ids), f"icu_{label}")
    return expr.otherwise("unknown")

df_icu_pivoted = df_icu_filtered.withColumn("feature", map_itemid_icu(F.col("itemid"))) \
    .groupBy("subject_id", "icu_time") \
    .pivot("feature") \
    .avg("valuenum")

from datetime import timedelta

spark.conf.set("spark.sql.adaptive.enabled", "true")

events_df = df.alias("e")

subject_ids = events_df.select("subject_id").distinct()
bounds = events_df.select(
    F.min("sao2_time").alias("min_sao2"),
    F.max("sao2_time").alias("max_sao2")
).first()

min_lab_time = bounds["min_sao2"] - timedelta(hours=24)
max_lab_time = bounds["max_sao2"]

labs_base = (
    df_labs_pivoted
    .withColumn("subject_id", F.col("subject_id").cast("int"))
    .join(F.broadcast(subject_ids), on="subject_id", how="inner")
    .filter((F.col("lab_time") >= F.lit(min_lab_time)) & (F.col("lab_time") <= F.lit(max_lab_time)))
)
labs_df = labs_base.alias("l")

icu_base = (
    df_icu_pivoted
    .withColumn("subject_id", F.col("subject_id").cast("int"))
    .join(F.broadcast(subject_ids), on="subject_id", how="inner")
    .filter((F.col("icu_time") >= F.lit(min_lab_time)) & (F.col("icu_time") <= F.lit(max_lab_time)))
)
icu_df = icu_base.alias("i")

lab_feature_cols = [c for c in labs_base.columns if c not in ["subject_id", "lab_time"]]

icu_feature_cols = [c for c in icu_base.columns if c not in ["subject_id", "icu_time"]]

joined = events_df.join(
    labs_df,
    on=[
        F.col("e.subject_id") == F.col("l.subject_id"),
        F.col("l.lab_time") <= F.col("e.sao2_time"),
        F.col("l.lab_time") >= F.col("e.sao2_time") - F.expr("INTERVAL 24 HOURS")
    ],
    how="left"
).select(
    F.col("e.*"),
    F.col("l.lab_time"),
    *[F.col(f"l.{c}") for c in lab_feature_cols]
)

window_spec_lab = Window.partitionBy(F.col("subject_id"), F.col("sao2_time")).orderBy(F.col("lab_time").desc())

joined_with_labs = (
    joined
    .withColumn("rn", F.row_number().over(window_spec_lab))
    .filter(F.col("rn") == 1)
    .drop("rn", "lab_time")
)

joined_with_labs_aliased = joined_with_labs.alias("jl")

joined_with_icu = joined_with_labs_aliased.join(
    icu_df,
    on=[
        F.col("jl.subject_id") == F.col("i.subject_id"),
        F.col("i.icu_time") <= F.col("jl.sao2_time"),
        F.col("i.icu_time") >= F.col("jl.sao2_time") - F.expr("INTERVAL 1 HOURS")
    ],
    how="left"
).select(
    F.col("jl.*"),
    F.col("i.icu_time"),
    *[F.col(f"i.{c}") for c in icu_feature_cols]
)

window_spec_icu = Window.partitionBy(F.col("subject_id"), F.col("sao2_time")).orderBy(F.col("icu_time").desc())

df_final_spark = (
    joined_with_icu
    .withColumn("rn", F.row_number().over(window_spec_icu))
    .filter(F.col("rn") == 1)
    .drop("rn", "icu_time")
)

df_final_spark.write.mode("overwrite").parquet("input/merged_data_with_labs_and_icu.parquet")

print("Final Dataset Columns:")
print(f"Total columns: {len(df_final_spark.columns)}")
print("\nLab Feature Columns:")
lab_cols = [c for c in df_final_spark.columns if c.startswith("lab_") or c.startswith("cbc_")]
for col in lab_cols:
    print(f"  - {col}")

print("\nICU Feature Columns:")
icu_cols = [c for c in df_final_spark.columns if c.startswith("icu_")]
for col in icu_cols:
    print(f"  - {col}")

print("\nOxygen Measurement Columns:")
oxygen_cols = ["SaO2", "SpO2", "sao2_time", "spo2_time"]
for col in oxygen_cols:
    if col in df_final_spark.columns:
        print(f"  - {col}")

df_final = spark.read.parquet("input/merged_data_with_labs_and_icu.parquet").limit(5000).toPandas()
print(f"\nFinal Dataset Shape: {df_final.shape}")