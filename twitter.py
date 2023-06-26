import tweepy
import matplotlib.pyplot as plt
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, StructType, StructField, ArrayType, FloatType, IntegerType
from utils import get_sentiment, plot_to_base64


def twitter_authentication(con_key, con_sec, acc_tok, acc_sec):
    auth = tweepy.OAuthHandler(con_key, con_sec)
    auth.set_access_token(acc_tok, acc_sec)
    twitter_api = tweepy.API(auth)
    try:
        twitter_api.verify_credentials()
    except:
        return False
    return twitter_api


def search_raw_tweets(keyword, twitter_api):
    tweets = twitter_api.search_tweets(keyword, tweet_mode="extended", lang="en", result_type="popular", count=10)
    return tweets


def create_twitter_databases(mongo_client):
    raw_tweets_db = mongo_client["raw_tweets"]
    processed_tweets_db = mongo_client['processed_tweets']
    return raw_tweets_db, processed_tweets_db


def store_raw_tweets(keyword, tweets, raw_tweets_db):
    for tweet in tweets:
        tweet_data = tweet._json
        existing_tweet = raw_tweets_db[keyword].find_one({"id": tweet_data["id"]})

        if existing_tweet:
            raw_tweets_db[keyword].replace_one({"id": tweet_data["id"]}, tweet_data)
        else:
            raw_tweets_db[keyword].insert_one(tweet_data)


def process_tweets(tweets, spark):
    schema = StructType([
        StructField("tweet_id", StringType(), True),
        StructField("full_text", StringType(), True),
        StructField("user", StringType(), True),
        StructField("retweet_count", IntegerType(), True),
        StructField("favorite_count", IntegerType(), True),
        StructField("lang", StringType(), True),
        StructField("retweeted", StringType(), True)
    ])

    sentiment_udf = udf(get_sentiment, ArrayType(FloatType()))

    tweet_data = [(tweet.id, tweet.full_text, tweet.user.name, tweet.retweet_count,
                   tweet.favorite_count, tweet.lang, tweet.retweeted) for tweet in tweets]

    tweets_df = spark.createDataFrame(tweet_data, schema)

    processed_tweets_df = tweets_df.withColumn("sentiment_scores", sentiment_udf(col("full_text")))

    sentiment_columns = ["compound_score", "positive_score", "negative_score"]
    for i, column_name in enumerate(sentiment_columns):
        processed_tweets_df = processed_tweets_df.withColumn(column_name, col("sentiment_scores")[i])

    processed_tweets_df = processed_tweets_df.drop("sentiment_scores")
    return processed_tweets_df


def store_processed_tweets(keyword, processed_tweets_df, mongo_url):
    processed_tweets_df.write.format("com.mongodb.spark.sql.DefaultSource") \
        .mode("overwrite") \
        .option("uri", mongo_url) \
        .option("database", "processed_tweets") \
        .option("collection", keyword) \
        .save()


def twitter_analysis(keyword, mongo_url, spark):
    # Load DataFrame from MongoDB
    df2 = spark.read.format("mongo") \
        .option("uri", mongo_url) \
        .option("database", "processed_tweets") \
        .option("collection", keyword) \
        .load()

    sentiment_counts = df2.groupBy("compound_score").count()
    pos_row = sentiment_counts.filter(sentiment_counts["compound_score"] > 0).select("count").first()
    pos_count = pos_row[0] if pos_row is not None else 0
    neg_row = sentiment_counts.filter(sentiment_counts["compound_score"] < 0).select("count").first()
    neg_count = neg_row[0] if neg_row is not None else 0
    neut_row = sentiment_counts.filter(sentiment_counts["compound_score"] == 0).select("count").first()
    neut_count = neut_row[0] if neut_row is not None else 0
    twitter_pie_chart = generate_twitter_pie_chart(pos_count, neg_count, neut_count)
    twitter_pie_chart_image = plot_to_base64(twitter_pie_chart)

    positive_values = df2.select("positive_score").rdd.flatMap(lambda x: x).collect()
    negative_values = df2.select("negative_score").rdd.flatMap(lambda x: x).collect()
    twitter_scatter_plot = generate_twitter_scatter_plot(positive_values, negative_values)
    twitter_scatter_plot_image = plot_to_base64(twitter_scatter_plot)

    top_positive_tweets = df2.filter(df2["compound_score"] > 0).orderBy(df2["compound_score"].desc()).limit(
        5).toPandas()
    top_negative_tweets = df2.filter(df2["compound_score"] < 0).orderBy(df2["compound_score"]).limit(5).toPandas()
    return twitter_pie_chart_image, twitter_scatter_plot_image, top_positive_tweets, top_negative_tweets


def generate_twitter_pie_chart(pos_count, neg_count, neut_count):
    fig, ax = plt.subplots()

    pie_count = []
    pie_labels = []
    pie_colors = []

    if pos_count > 0:
        pie_count.append(pos_count)
        pie_labels.append("Positive Tweets")
        pie_colors.append("Green")

    if neg_count > 0:
        pie_count.append(neg_count)
        pie_labels.append("Negative Tweets")
        pie_colors.append("Red")

    if neut_count > 0:
        pie_count.append(neut_count)
        pie_labels.append("Neutral Tweets")
        pie_colors.append("Grey")

    ax.set_title('Sentiment Pie Chart')
    ax.pie(pie_count, labels=pie_labels, colors=pie_colors, autopct='%1.0f%%')
    return fig


def generate_twitter_scatter_plot(positive_values, negative_values):
    fig, ax = plt.subplots()

    ax.scatter(positive_values, negative_values)
    ax.set_xlabel('Positive')
    ax.set_ylabel('Negative')
    ax.set_title('Sentiment Scatter Plot')
    return fig