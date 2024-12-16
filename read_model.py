import os
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql.types import StructType, StructField, \
    DoubleType, IntegerType, StringType

def prepare_data(spark, input: dict):
  schema = StructType([
      StructField("Age", IntegerType(), True),
      StructField("Gender", StringType(), True),
      StructField("Income", IntegerType(), True),
      StructField("Education_Level", StringType(), True),
      StructField("Travel_Frequency", IntegerType(), True),
      StructField("Preferred_Activities", StringType(), True),
      StructField("Vacation_Budget", IntegerType(), True),
      StructField("Location", StringType(), True),
      StructField("Proximity_to_Mountains", IntegerType(), True),
      StructField("Proximity_to_Beaches", IntegerType(), True),
      StructField("Favorite_Season", StringType(), True),
      StructField("Pets", IntegerType(), True),
      StructField("Environmental_Concerns", IntegerType(), True),
      StructField("Gender_indexed", DoubleType(), True),
      StructField("Education_Level_indexed", DoubleType(), True),
      StructField("Preferred_Activities_indexed", DoubleType(), True),
      StructField("Location_indexed", DoubleType(), True),
      StructField("Favorite_Season_indexed", DoubleType(), True)
  ])
  
  data = list(input.values())

  gender_index = {"male": 0.0, "female": 1.0, "non-binary": 2.0}
  education_index = {"doctorate": 0.0, "bachelor": 1.0, "high school": 2.0, "master": 3.0}
  activities_index = {"sunbathing": 0.0, "skiing": 1.0, "hiking": 2.0, "swimming": 3.0}
  location_index = {"suburban": 0.0, "urban": 1.0, "rural": 2.0}
  season_index = {"summer": 0.0, "fall": 1.0, "winter": 2.0, "spring": 3.0}

  data.append(gender_index.get(input['Gender'].lower(), -1.0))
  data.append(education_index.get(input['Education_Level'].lower(), -1.0))
  data.append(activities_index.get(input['Preferred_Activities'].lower(), -1.0))
  data.append(location_index.get(input['Location'].lower(), -1.0))
  data.append(season_index.get(input['Favorite_Season'].lower(), -1.0))

  input_row = Row(*[field.name for field in schema.fields])
  input_df = spark.createDataFrame([input_row(*data)], schema)

  return input_df

spark = SparkSession.builder \
  .appName("BD wczytanie modelu") \
  .config("spark.hadoop.io.nativeio.disable", "true") \
  .config("spark.master", "local[*]") \
  .getOrCreate()

example_input = {
    "Age": 35,
    "Gender": "Male",                 # Gender: male, female, non-binary
    "Income": 50000,
    "Education_Level": "Bachelor",    # Income: doctorate, bachelor, high school, master
    "Travel_Frequency": 3,
    "Preferred_Activities": "Hiking", # Preferred Activities: sunbathing, skiing, hiking, swimming
    "Vacation_Budget": 2000,
    "Location": "Urban",              # Location: suburban, urban, rural
    "Proximity_to_Mountains": 5,
    "Proximity_to_Beaches": 10,
    "Favorite_Season": "Summer",      # Favorite Season: summer, fall, winter, spring
    "Pets": 1,
    "Environmental_Concerns": 3
}

input_df = prepare_data(spark, example_input)

# Ścieżka do modelu
model_path = "file:///" + os.path.abspath("./model/DecisionTreeClassifierAdjusted")
print(f"Loading model from: {model_path}")

# Wczytywanie modelu
try:
    model = PipelineModel.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
    
predictions = model.transform(input_df)
predictions.select("Age", "Gender", "Income", "Prediction").show()