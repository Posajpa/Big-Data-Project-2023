import mimetypes
import matplotlib.pyplot as plt
from pyspark.sql.functions import udf, length, col, explode
from pyspark.sql.types import ArrayType, StringType, FloatType
from utils import get_sentiment, plot_to_base64


def detect_file_format(dataset_file):
    filename = dataset_file.filename
    file_format, _ = mimetypes.guess_type(filename)
    if file_format == 'text/plain':
        return "txt"
    elif file_format == 'application/json':
        return "json"
    elif file_format == 'text/csv':
        return "csv"
    else:
        return False


def store_dataset(dataset_file, file_format, mongo_url, temp_file, spark):
    if file_format == 'txt':
        data = spark.read.text(temp_file.name)
        data.write.format("com.mongodb.spark.sql.DefaultSource") \
            .mode("append") \
            .option("replaceDocument", "false") \
            .option("uri", mongo_url) \
            .option("database", 'raw_text_files') \
            .option("collection", dataset_file.filename) \
            .save()

    elif file_format == 'json':
        data = spark.read.option("multiline", "true").json(temp_file.name)
        data.write.format("com.mongodb.spark.sql.DefaultSource") \
            .mode("append") \
            .option("replaceDocument", "false") \
            .option("uri", mongo_url) \
            .option("database", 'raw_json_files') \
            .option("collection", dataset_file.filename) \
            .save()

    elif file_format == 'csv':
        data = spark.read.csv(temp_file.name, header=True)
        data.write.format("com.mongodb.spark.sql.DefaultSource") \
            .mode("append") \
            .option("replaceDocument", "false") \
            .option("uri", mongo_url) \
            .option("database", 'raw_csv_files') \
            .option("collection", dataset_file.filename) \
            .save()


def preprocess_dataset(dataset_file, file_format, mongo_url, temp_file, spark):
    def tokenize_sentences(text):
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)

    tokenize_sentences_udf = udf(tokenize_sentences, ArrayType(StringType()))

    if file_format == 'txt':
        df = spark.read.text(temp_file.name)
        df = df.withColumn('text', tokenize_sentences_udf(df['value']))
        df = df.select(explode(col('text')).alias('text'))
        df.write.format("com.mongodb.spark.sql.DefaultSource") \
            .mode("append") \
            .option("replaceDocument", "false") \
            .option("uri", mongo_url) \
            .option("database", 'preprocessed_txt_files') \
            .option("collection", dataset_file.filename) \
            .save()
        return df

    if file_format == 'csv':
        df = spark.read.csv(temp_file.name, header=True, inferSchema=True)
        text_column = None
        max_length = 0
        for column in df.columns:
            if df.schema[column].dataType == StringType():
                column_length = df.select(length(col(column))).agg({"length(" + column + ")": "max"}).collect()[0][0]
                if column_length > max_length:
                    max_length = column_length
                    text_column = column
        df = df.select(col(text_column).alias('text'))
        df.write.format("com.mongodb.spark.sql.DefaultSource") \
            .mode("append") \
            .option("replaceDocument", "false") \
            .option("uri", mongo_url) \
            .option("database", 'preprocessed_csv_files') \
            .option("collection", dataset_file.filename) \
            .save()
        return df

    if file_format == 'json':
        df = spark.read.option("multiline", "true").json(temp_file.name)
        text_column = None
        max_length = 0
        for column in df.columns:
            if df.schema[column].dataType == StringType():
                column_length = df.select(length(col(column))).agg({"length(" + column + ")": "max"}).collect()[0][0]
                if column_length > max_length:
                    max_length = column_length
                    text_column = column
        df = df.select(col(text_column).alias('text'))
        df.write.format("com.mongodb.spark.sql.DefaultSource") \
            .mode("append") \
            .option("replaceDocument", "false") \
            .option("uri", mongo_url) \
            .option("database", 'preprocessed_json_files') \
            .option("collection", dataset_file.filename) \
            .save()
        return df


def process_dataset(df, keyword, mongo_url):
    df_filtered = df.filter(df['text'].contains(keyword))
    if df_filtered.isEmpty():
        return False
    else:
        sentiment_udf = udf(get_sentiment, ArrayType(FloatType()))
        processed_df = df_filtered.withColumn("sentiment_scores", sentiment_udf(col("text")))

        sentiment_columns = ["compound_score", "positive_score", "negative_score"]
        for i, column_name in enumerate(sentiment_columns):
            processed_df = processed_df.withColumn(column_name, col("sentiment_scores")[i])

        processed_df = processed_df.drop("sentiment_scores")

        processed_df.write.format("com.mongodb.spark.sql.DefaultSource") \
            .mode("overwrite") \
            .option("uri", mongo_url) \
            .option("database", "processed_tweets") \
            .option("collection", keyword) \
            .save()
    return processed_df


def dataset_analysis(processed_df):
    sentiment_counts = processed_df.groupBy("compound_score").count()
    pos_row = sentiment_counts.filter(sentiment_counts["compound_score"] > 0).select("count").first()
    pos_count = pos_row[0] if pos_row is not None else 0
    neg_row = sentiment_counts.filter(sentiment_counts["compound_score"] < 0).select("count").first()
    neg_count = neg_row[0] if neg_row is not None else 0
    neut_row = sentiment_counts.filter(sentiment_counts["compound_score"] == 0).select("count").first()
    neut_count = neut_row[0] if neut_row is not None else 0
    dataset_pie_chart = generate_dataset_pie_chart(pos_count, neg_count, neut_count)
    dataset_pie_chart_image = plot_to_base64(dataset_pie_chart)

    positive_values = processed_df.select("positive_score").rdd.flatMap(lambda x: x).collect()
    negative_values = processed_df.select("negative_score").rdd.flatMap(lambda x: x).collect()
    dataset_scatter_plot = generate_dataset_scatter_plot(positive_values, negative_values)
    dataset_scatter_plot_image = plot_to_base64(dataset_scatter_plot)

    top_positive_sentences = processed_df.filter(processed_df["compound_score"] > 0).orderBy(
        processed_df["compound_score"].desc()).limit(5).toPandas()
    top_negative_sentences = processed_df.filter(processed_df["compound_score"] < 0).orderBy(
        processed_df["compound_score"]).limit(5).toPandas()
    return dataset_pie_chart_image, dataset_scatter_plot_image, top_positive_sentences, top_negative_sentences


def generate_dataset_pie_chart(pos_count, neg_count, neut_count):
    fig, ax = plt.subplots()

    pie_count = []
    pie_labels = []
    pie_colors = []

    if pos_count > 0:
        pie_count.append(pos_count)
        pie_labels.append("Positive Sentences")
        pie_colors.append("Green")

    if neg_count > 0:
        pie_count.append(neg_count)
        pie_labels.append("Negative Sentences")
        pie_colors.append("Red")

    if neut_count > 0:
        pie_count.append(neut_count)
        pie_labels.append("Neutral Sentences")
        pie_colors.append("Grey")

    ax.set_title('Sentiment Pie Chart')
    ax.pie(pie_count, labels=pie_labels, colors=pie_colors, autopct='%1.0f%%')
    return fig


def generate_dataset_scatter_plot(positive_values, negative_values):
    fig, ax = plt.subplots()

    ax.scatter(positive_values, negative_values)
    ax.set_xlabel('Positive')
    ax.set_ylabel('Negative')
    ax.set_title('Sentiment Scatter Plot')
    return fig