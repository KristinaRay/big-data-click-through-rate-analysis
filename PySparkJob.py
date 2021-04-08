#!/usr/bin/env python
# coding: utf-8

# In[2]:


import io
import sys, os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, datediff, when, format_number
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


# In[3]:


def process(spark, input_file, target_path):
    # TODO Ваш код
    df = spark.read.parquet('input_file')
    df = df.withColumn('is_cpm',  when(df['ad_cost_type'] == 'CPM', 1).otherwise(0))
    df = df.withColumn('is_cpc',  when(df['ad_cost_type'] == 'CPC', 1).otherwise(0))
    df=df.drop('ad_cost_type', 'compaign_union_id', 'platform', 'client_union_id','time')
    new_df=df.groupBy('ad_id').agg(F.max(F.col('date')).alias('max_date'),    F.min(F.col('date')).alias('min_date'))
    new_df = new_df.withColumn('day_count',  datediff(new_df.max_date, new_df.min_date))
    df = df.join(new_df.select('ad_id', 'day_count'), ['ad_id'])
    d1 = df.filter(df.event == 'click').groupBy('ad_id').count()
    d1 = d1.withColumnRenamed('count', 'clicks')
    d2 = df.filter(df.event == 'view').groupBy('ad_id').count()
    d2 = d2.withColumnRenamed('count', 'views')
    d = d1.join(d2.select('ad_id', 'views'), ['ad_id'])
    d = d.withColumn('CTR', format_number(d.clicks/d.views, 1).cast(DoubleType())) # находим CTR и переводим в double
    df=df.join(d, 'ad_id',  'outer').select('ad_id', 'target_audience_count', 'has_video', 'is_cpm', 'is_cpc', 'ad_cost', 'day_count', 'CTR')
    train, test, validate = df.randomSplit([0.5, 0.25, 0.25]) # Разбиваем на train, test, validate
    train = df.write.parquet('target_path/train') # Сохраняем
    test = df.write.parquet('target_path/test')
    validate = df.write.parquet('target_path/validate')


# In[4]:


def main(argv):
    input_path = argv[0]
    print("Input path to file: " + input_path)
    target_path = argv[1]
    print("Target path: " + target_path)
    spark = _spark_session()
    process(spark, input_path, target_path)


def _spark_session():
    return SparkSession.builder.appName('PySparkJob').getOrCreate()

if __name__ == "__main__":
    arg = sys.argv[1:]
    if len(arg) != 2:
        sys.exit("Input and Target path are require.")
    else:
        main(arg)



