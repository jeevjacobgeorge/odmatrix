from pyspark.sql import SparkSession

def get_spark_session():
    # Replace with the correct hostname or IP where Hadoop is running
    spark = SparkSession.builder \
        .appName("DjangoHadoopIntegration") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://localhost:9000") \
        .config("spark.hadoop.yarn.resourcemanager.address", "localhost:8032") \
        .getOrCreate()
    return spark
