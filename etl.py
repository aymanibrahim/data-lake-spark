import configparser
from datetime import datetime
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, from_unixtime, monotonically_increasing_id
from pyspark.sql.functions import year, month, dayofmonth, hour, weekofyear, date_format, dayofweek


config = configparser.ConfigParser()
config.read('dl.cfg')

os.environ['AWS_ACCESS_KEY_ID']=config['AWS']['AWS_ACCESS_KEY_ID']
os.environ['AWS_SECRET_ACCESS_KEY']=config['AWS']['AWS_SECRET_ACCESS_KEY']


def create_spark_session():
    """Create spark session instance
    """
    spark = SparkSession \
        .builder \
        .appName("Sparkify App") \
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return spark


def process_song_data(spark, input_data, output_data):
    """Process the songs data file, extract and create songs and artists tables
    :param spark: spark session instance
    :param input_data: input file path
    :param output_data: output file path
    """
    # get filepath to song data file
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    
    # read song data file
    df = spark.read.json(song_data)

    # extract columns to create songs table
    songs_table = df.select("song_id",
                            "title",
                            "artist_id",
                            "year",
                            "duration") \
                     .drop_duplicates()
    
    # write songs table to parquet files partitioned by year and artist
    songs_table.write \
               .parquet(os.path.join(output_data, "songs"), \
                        mode='overwrite', \
                        partitionBy=["year","artist_id"])

    # extract columns to create artists table
    artists_table = df.select("artist_id",
                              "artist_name",
                              "artist_location",
                              "artist_latitude",
                              "artist_longitude") \
                      .drop_duplicates()
        
    # write artists table to parquet files
    artists_table.write \
                 .parquet(os.path.join(output_data, 'artists'), \
                          mode='overwrite')


def process_log_data(spark, input_data, output_data):
    """Process the log data file, extract and create users, time and songplays tables
    :param spark: spark session instance
    :param input_data: input file path
    :param output_data: output file path
    """
    # get filepath to log data file
    log_data = os.path.join(input_data, 'log-data/*/*/*.json')

    # read log data file
    df = spark.read.json(log_data)
    
    # filter by actions for song plays
    df = df.filter(df.page == "NextSong")

    # extract columns for users table    
    users_table = df.select(
                        col("userId")   .alias("user_id"),
                        col("firstName").alias("first_name"),
                        col("lastName") .alias("last_name"),
                        "gender",
                        "level") \
                    .drop_duplicates(subset=['user_id'])
        
    # write users table to parquet files
    users_table.write \
               .parquet(os.path.join(output_data, "users"), \
                        mode='overwrite')   
        
    # create timestamp column from original timestamp column
    get_timestamp = udf(lambda x: str(int(int(x)/1000)))
    df = df.withColumn('timestamp', get_timestamp("ts"))
    
    # create datetime column from original timestamp column
    get_datetime = udf(lambda x: str(datetime.fromtimestamp(int(x) / 1000.0)))
    df = df.withColumn("datetime", get_datetime("ts"))     
    
    
    # extract columns to create time table
    time_table = df.withColumn("hour",    hour("datetime")) \
                   .withColumn("day",     dayofmonth("datetime")) \
                   .withColumn("week",    weekofyear("datetime")) \
                   .withColumn("month",   month("datetime")) \
                   .withColumn("year",    year("datetime")) \
                   .withColumn("weekday", dayofweek("datetime")) \
                   .select(date_format(from_unixtime("timestamp"), 'h:m:s') \
                           .alias('start_time'), \
                           "hour", 
                           "day", 
                           "week", 
                           "month", 
                           "year", 
                           "weekday") \
                   .drop_duplicates()

    # write time table to parquet files partitioned by year and month
    time_table.write \
              .parquet(os.path.join(output_data, "time"), \
                       mode='overwrite', \
                       partitionBy=["year","month"])
    
    
    # read in song data to use for songplays table
    song_data = os.path.join(input_data, 'song_data/*/*/*/*.json')
    song_df = spark.read.json(song_data)
    
    # extract columns from joined song and log datasets to create songplays table 
    
    # match the songs based on song title, artist and duration
    df = df.join(song_df, [song_df.title == df.song, \
                           song_df.artist_name == df.artist, \
                           song_df.duration == df.length])
        
    songplays_table = df.select(
        monotonically_increasing_id().alias('songplay_id'),
        date_format(from_unixtime("timestamp"), 'h:m:s').alias('start_time'), 
        col('userId')                .alias('user_id'),
        col('level'),                 
        col('song_id'),               
        col('artist_id'),             
        col('sessionId')             .alias('session_id'),
        col('location'),              
        col('userAgent')             .alias('user_agent'),
        col('year'),
        month('datetime')            .alias('month')
    )

    # write songplays table to parquet files partitioned by year and month
    songplays_table.write \
                   .partitionBy('year', 'month') \
                   .parquet(os.path.join(output_data, 'songplays'), \
                            mode='overwrite')   


def main():
    """Create Spark session, extract songs and log data from S3, transform them into dimensional tables, and load them back to S3 in Parquet format
    """
    spark = create_spark_session()
    input_data  = "s3://udacity-dend/"
    output_data = "s3://spark-space/sparkify/output/"
    
    process_song_data(spark, input_data, output_data)    
    process_log_data(spark, input_data, output_data)


if __name__ == "__main__":
    main()
