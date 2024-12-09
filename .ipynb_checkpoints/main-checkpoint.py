from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Projekt test").getOrCreate()

df = spark.read.csv("data/mountains_vs_beaches_preferences.csv",
                    sep=';',
                    inferSchema=True,
                    header=True)

df.show(10)

df.describe().show()

print(df.schema)