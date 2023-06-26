from googleapiclient.discovery import build
import matplotlib.pyplot as plt
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, StructType, StructField, ArrayType, FloatType
from utils import get_sentiment, plot_to_base64


def youtube_authentication(api_key):
    youtube_api = build('youtube', 'v3', developerKey=api_key)
    try:
        youtube_api.search().list(q="man", part='snippet', type='video', maxResults=1).execute()
    except:
        return False
    return youtube_api


def search_raw_videos(keyword, youtube_api):
    videos = youtube_api.search().list(q=keyword, part='snippet', type='video', maxResults=50).execute()
    return videos


def create_youtube_databases(mongo_client):
    raw_videos_db = mongo_client["raw_videos"]
    processed_videos_db = mongo_client['processed_videos']
    return raw_videos_db, processed_videos_db


def store_raw_videos(keyword, videos, raw_videos_db):
    for video in videos["items"]:
        video_data = video
        existing_video = raw_videos_db[keyword].find_one({"etag": video_data["etag"]})

        if existing_video:
            raw_videos_db[keyword].replace_one({"etag": video_data["etag"]}, video_data)
        else:
            raw_videos_db[keyword].insert_one(video_data)


def process_videos(videos, spark):
    schema = StructType([
        StructField("video_id", StringType(), True),
        StructField("video_title", StringType(), True),
        StructField("description", StringType(), True),
        StructField("channel_title", StringType(), True)
    ])

    sentiment_udf = udf(get_sentiment, ArrayType(FloatType()))

    video_data = [(video['id']['videoId'], video['snippet']['title'], video['snippet']['description'],
                   video['snippet']['channelTitle'])
                  for video in videos["items"]]

    video_df = spark.createDataFrame(video_data, schema)

    processed_videos_df = video_df.withColumn("sentiment_scores", sentiment_udf(col("video_title")))

    sentiment_columns = ["compound_score", "positive_score", "negative_score"]
    for i, column_name in enumerate(sentiment_columns):
        processed_videos_df = processed_videos_df.withColumn(column_name, col("sentiment_scores")[i])

    processed_videos_df = processed_videos_df.drop("sentiment_scores")

    return processed_videos_df


def store_processed_videos(keyword, processed_videos_df, mongo_url):
    processed_videos_df.write.format("com.mongodb.spark.sql.DefaultSource") \
        .mode("overwrite") \
        .option("uri", mongo_url) \
        .option("database", "processed_videos") \
        .option("collection", keyword) \
        .save()


def youtube_analysis(keyword, mongo_url, spark):
    df2 = spark.read.format("mongo") \
        .option("uri", mongo_url) \
        .option("database", "processed_videos") \
        .option("collection", keyword) \
        .load()

    sentiment_counts = df2.groupBy("compound_score").count()
    pos_row = sentiment_counts.filter(sentiment_counts["compound_score"] > 0).select("count").first()
    pos_count = pos_row[0] if pos_row is not None else 0
    neg_row = sentiment_counts.filter(sentiment_counts["compound_score"] < 0).select("count").first()
    neg_count = neg_row[0] if neg_row is not None else 0
    neut_row = sentiment_counts.filter(sentiment_counts["compound_score"] == 0).select("count").first()
    neut_count = neut_row[0] if neut_row is not None else 0
    youtube_pie_chart = generate_youtube_pie_chart(pos_count, neg_count, neut_count)
    youtube_pie_chart_image = plot_to_base64(youtube_pie_chart)

    positive_values = df2.select("positive_score").rdd.flatMap(lambda x: x).collect()
    negative_values = df2.select("negative_score").rdd.flatMap(lambda x: x).collect()
    youtube_scatter_plot = generate_youtube_scatter_plot(positive_values, negative_values)
    youtube_scatter_plot_image = plot_to_base64(youtube_scatter_plot)

    top_positive_videos = df2.filter(df2["compound_score"] > 0).orderBy(df2["compound_score"].desc()).limit(
        5).toPandas()
    top_negative_videos = df2.filter(df2["compound_score"] < 0).orderBy(df2["compound_score"]).limit(5).toPandas()
    return youtube_pie_chart_image, youtube_scatter_plot_image, top_positive_videos, top_negative_videos


def generate_youtube_pie_chart(pos_count, neg_count, neut_count):
    fig, ax = plt.subplots()

    pie_count = []
    pie_labels = []
    pie_colors = []

    if pos_count > 0:
        pie_count.append(pos_count)
        pie_labels.append("Positive Videos")
        pie_colors.append("Green")

    if neg_count > 0:
        pie_count.append(neg_count)
        pie_labels.append("Negative Videos")
        pie_colors.append("Red")

    if neut_count > 0:
        pie_count.append(neut_count)
        pie_labels.append("Neutral Videos")
        pie_colors.append("Grey")

    ax.set_title('Sentiment Pie Chart')
    ax.pie(pie_count, labels=pie_labels, colors=pie_colors, autopct='%1.0f%%')
    return fig


def generate_youtube_scatter_plot(positive_values, negative_values):
    fig, ax = plt.subplots()

    ax.scatter(positive_values, negative_values)
    ax.set_xlabel('Positive')
    ax.set_ylabel('Negative')
    ax.set_title('Sentiment Scatter Plot')
    return fig