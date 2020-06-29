from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


# Create Spark Context
sc = SparkContext.getOrCreate()

# Create Spark Session
spark = SparkSession(sc)

# Save path in variable
path = 'hdfs:///user/pknees/RSC20/training.tsv'
val = 'hdfs:///user/pknees/RSC20/test.tsv'

#Reading in the file
df = spark.read \
          .load(path,
           format="csv",delimiter="\x01")

df_val = spark.read \
          .load(val,
           format="csv",delimiter="\x01")

# Changing Column names
df = df.toDF("text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified",\
               "engaging_user_account_creation", "engaged_follows_engaging", "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp")

df_val = df_val.toDF("text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified",\
               "engaging_user_account_creation", "engaged_follows_engaging")

id_features = ["tweet_id","engaging_user_id","engaged_with_user_id"]

numeric_features = ["tweet_timestamp",
                    "engaged_with_user_follower_count", "engaged_with_user_following_count", "engaged_with_user_account_creation",
                    "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_account_creation"
                   ]

categorical_features = ["tweet_type", "language", 
                        "engaged_with_user_is_verified", "engaging_user_is_verified", "engaged_follows_engaging"
                       ]

text_features = ["text_tokens", "hashtags", "present_media", "present_links", "present_domains"]

label_columns = ["reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]

from pyspark.sql import functions as f
#for feature in text_features:
#    text_feature_split = f.split(df[feature], '\t')
#    df = df.withColumn(feature, f.when(f.col(feature).isNotNull(), text_feature_split).otherwise(f.array().cast("array<string>")))
    

from pyspark.sql.types import IntegerType
for feature in numeric_features:
    df = df.withColumn(feature,f.col(feature).cast(IntegerType()))
    
for feature in id_features:
    output_col = feature + "_hashed"
    df = df.withColumn(output_col, (f.hash(f.col(feature))))
    df = df.withColumn(output_col, f.when(f.col(output_col) < 0, f.col(output_col)*-1%50).otherwise(f.col(output_col)%50))
    
for col in label_columns:
    df = df.withColumn(col, f.when(f.col(col).isNotNull(), 1).otherwise(0))
    

##### Same preprocessing for validation (without label_columns transformation)
#for feature in text_features:
#    text_feature_split = f.split(df_val[feature], '\t')
#    df_val = df_val.withColumn(feature, f.when(f.col(feature).isNotNull(), text_feature_split).otherwise(f.array().cast("array<string>")))
    
for feature in numeric_features:
    df_val = df_val.withColumn(feature,f.col(feature).cast(IntegerType()))
    
for feature in id_features:
    output_col = feature + "_hashed"
    df_val = df_val.withColumn(output_col, (f.hash(f.col(feature))))
    df_val = df_val.withColumn(output_col, f.when(f.col(output_col) < 0, f.col(output_col)*-1%50).otherwise(f.col(output_col)%50))
    
    
# Set the numbers of quantiles/buckets for the baseline approach
nq = 50

from pyspark.ml.feature import QuantileDiscretizer, StringIndexer, FeatureHasher, HashingTF, OneHotEncoderEstimator, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

AllQuantileDiscretizers = [QuantileDiscretizer(numBuckets=nq,
                                      inputCol=col,
                                      outputCol=(col + "_bucketized"),
                                      handleInvalid="keep") for col in numeric_features]

AllStringIndexers = [StringIndexer(inputCol=col, 
                            outputCol=(col + "_indexed")) for col in categorical_features]

### FeatureHasher has been adapted to a hardcoded feature hashing + bucketing in the preprocessing step
#AllFeatureHashers = [FeatureHasher(numFeatures=nq,
#                           inputCols=[col],
#                           outputCol=(col + "_hashed")) for col in id_features]


#AllHashingTF = [HashingTF(inputCol=col, 
#                          outputCol=(col + "_vectorized")) for col in text_features]

to_onehot_features = [col + "_bucketized" for col in numeric_features]
to_onehot_features.extend(col + "_indexed" for col in categorical_features)
to_onehot_features.extend(col + "_hashed" for col in id_features)

onehot_features = [col + "_oneHot" for col in numeric_features]
onehot_features.extend(col + "_oneHot" for col in categorical_features)
onehot_features.extend(col + "_oneHot" for col in id_features)

encoder = OneHotEncoderEstimator(inputCols=to_onehot_features,
                                 outputCols=onehot_features)

assembler_features = VectorAssembler(
    inputCols=onehot_features,
    outputCol="features_oneHot")

#assembler_labels = VectorAssembler(
#    inputCols=label_columns,
#    outputCol="label")

AllRFModels = [RandomForestClassifier(labelCol=col, featuresCol="features_oneHot",predictionCol=(col+"_prediction"),probabilityCol=(col+"_probability"),rawPredictionCol=(col+"_raw_prediction"), numTrees=10) for col in label_columns]

from pyspark.ml import Pipeline

AllStages = list()
AllStages.extend(AllQuantileDiscretizers)
AllStages.extend(AllStringIndexers)
#AllStages.extend(AllFeatureHashers) #depreciated
#AllStages.extend(AllHashingTF)
AllStages.append(encoder)
AllStages.append(assembler_features)
AllStages.extend(AllRFModels)

pipeline = Pipeline(stages=AllStages)
pipeline_model = pipeline.fit(df)
pipeline_model.write().overwrite().save("pipeline_model_twitter_group13")
new_train = pipeline_model.transform(df_val).select(["tweet_id", "engaging_user_id", "reply_timestamp_prediction", "retweet_timestamp_prediction", "retweet_with_comment_timestamp_prediction", "like_timestamp_prediction"])
new_train.withColumnRenamed("engaging_user_id","user_id")
new_train.write.csv('prediction_like_timestamp_twitter_Group13.csv')