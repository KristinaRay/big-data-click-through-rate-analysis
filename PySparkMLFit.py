import io
import sys

from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import GBTRegressor

# Используйте как путь куда сохранить модель
MODEL_PATH = 'spark_ml_model'


def process(spark, train_data, test_data):
    #train_data - путь к файлу с данными для обучения модели
    #test_data - путь к файлу с данными для оценки качества модели
    #Ваш код
    train = spark.read.parquet('train_data')
    test = spark.read.parquet('test_data')
    feature = VectorAssembler(inputCols=['ad_id', 'target_audience_count', \
                                         'has_video', 'is_cpm', 'is_cpc',\
                                         'ad_cost', 'day_count'], outputCol="features")
    # Train a GBT model.
    gbt = GBTRegressor(labelCol="ctr", featuresCol="features", maxIter=10)
    
    # Chain VectorAssembler and estimator in a Pipeline
    pipeline = Pipeline(stages=[feature, gbt])

    paramGrid = ParamGridBuilder()\
        .addGrid(rf.maxDepth, [2, 3, 4, 5, 6])\
        .addGrid(rf.numTrees, [3, 6, 8, 9, 10])\
        .build()

    # A TrainValidationSplit requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    tvs = TrainValidationSplit(estimator=pipeline,
                           estimatorParamMaps=paramGrid,
                           evaluator = RegressionEvaluator(labelCol="ctr", predictionCol="prediction"), seed=42)

    # Run TrainValidationSplit, and choose the best set of parameters.
    model_gbt_tvs = tvs.fit(train)

    # Make predictions.
    predictions = model_gbt_tvs.transform(test)
    rmse_tvs_gbt = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
    print("RMSE: ", rmse_tvs_gbt)
    
    
    # Save the model
    model_gbt_tvs.write().save(MODEL_PATH)

    
    
    
    
def main(argv):
    train_data = argv[0]
    print("Input path to train data: " + train_data)
    test_data = argv[1]
    print("Input path to test data: " + test_data)
    spark = _spark_session()
    process(spark, train_data, test_data)


def _spark_session():
    return SparkSession.builder.appName('PySparkMLFitJob').getOrCreate()


if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Train and test data are require.")
    else:
        main(arg)
