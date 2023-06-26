import io
import base64
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pyspark.sql import SparkSession


def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    comp_score = analyzer.polarity_scores(text)["compound"]
    pos_score = analyzer.polarity_scores(text)["pos"]
    neg_score = analyzer.polarity_scores(text)["neg"]
    return comp_score, pos_score, neg_score


def plot_to_base64(plot):
    # Convert the plot object to a base64-encoded image
    buffer = io.BytesIO()
    plot.savefig(buffer, format='png')
    buffer.seek(0)
    plot_image = base64.b64encode(buffer.getvalue()).decode()
    return plot_image


def get_spark_session():
    return SparkSession.builder \
        .appName("webapp") \
        .config("spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:3.0.1") \
        .getOrCreate()
