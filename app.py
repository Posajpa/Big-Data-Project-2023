import tempfile
from pymongo import MongoClient
from flask import Flask, render_template, request, make_response
from twitter import (
    twitter_authentication, search_raw_tweets, create_twitter_databases,
    store_raw_tweets, process_tweets, store_processed_tweets, twitter_analysis
)
from youtube import (
    youtube_authentication, search_raw_videos, create_youtube_databases,
    store_raw_videos, process_videos, store_processed_videos, youtube_analysis
)
from custom_dataset import (
    detect_file_format, store_dataset, preprocess_dataset, process_dataset, dataset_analysis
)
from utils import get_spark_session

mongo_url = "mongodb://myapp-mongo-1:27017"
mongo_client = MongoClient(mongo_url)

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


# Route for index page
@app.route('/', methods=['GET', 'POST'])
def index():
    response = make_response(render_template('index.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# Route for Twitter analysis
@app.route('/twitter', methods=['GET', 'POST'])
def twitter():
    if request.method == 'POST':
        spark = get_spark_session()
        consumer_key = request.form['consumer_key']
        consumer_secret = request.form['consumer_secret']
        access_token = request.form['access_token']
        access_token_secret = request.form['access_token_secret']
        keyword = request.form['keyword']
        twitter_api = twitter_authentication(consumer_key, consumer_secret, access_token, access_token_secret)

        if not twitter_api:
            error_message = 'Invalid credentials. Please try again.'
            return render_template('entry_twitter.html', error_message=error_message)

        tweets = search_raw_tweets(keyword, twitter_api)
        if not tweets:
            error_message = 'No Tweets found for the Keyword. Please try using a different Keyword.'
            return render_template('entry_twitter.html', error_message=error_message)

        raw_tweets_db, processed_tweets_db = create_twitter_databases(mongo_client)
        store_raw_tweets(keyword, tweets, raw_tweets_db)
        processed_tweets_df = process_tweets(tweets, spark)
        store_processed_tweets(keyword, processed_tweets_df, mongo_url)

        twitter_pie_chart_image, twitter_scatter_plot_image, top_positive_tweets, \
            top_negative_tweets = twitter_analysis(keyword, mongo_url, spark)

        spark.stop()
        return render_template('results_twitter.html', twitter_pie_chart_image=twitter_pie_chart_image,
                               keyword=keyword, top_positive_tweets=top_positive_tweets,
                               top_negative_tweets=top_negative_tweets,
                               twitter_scatter_plot_image=twitter_scatter_plot_image)

    return render_template('entry_twitter.html')


# Route for YouTube analysis
@app.route('/youtube', methods=['GET', 'POST'])
def youtube():
    if request.method == 'POST':
        spark = get_spark_session()
        api_key = request.form['api_key']
        keyword = request.form['keyword']
        youtube_api = youtube_authentication(api_key)

        if not youtube_api:
            error_message = 'Invalid credentials. Please try again.'
            return render_template('entry_youtube.html', error_message=error_message)

        videos = search_raw_videos(keyword, youtube_api)
        if not videos:
            error_message = 'No Videos found for the Keyword. Please try using a different Keyword.'
            return render_template('entry_youtube.html', error_message=error_message)

        raw_videos_db, processed_videos_db = create_youtube_databases(mongo_client)
        store_raw_videos(keyword, videos, raw_videos_db)
        processed_videos_df = process_videos(videos, spark)
        store_processed_videos(keyword, processed_videos_df, mongo_url)

        youtube_pie_chart_image, youtube_scatter_plot_image, top_positive_videos, \
            top_negative_videos = youtube_analysis(keyword, mongo_url, spark)

        spark.stop()
        return render_template('results_youtube.html', youtube_pie_chart_image=youtube_pie_chart_image,
                               keyword=keyword, top_positive_videos=top_positive_videos,
                               top_negative_videos=top_negative_videos,
                               youtube_scatter_plot_image=youtube_scatter_plot_image)

    return render_template('entry_youtube.html')


# Route for custom dataset analysis
@app.route('/upload_dataset', methods=['GET', 'POST'])
def upload_dataset():
    if request.method == 'POST':
        spark = get_spark_session()
        dataset_file = request.files['dataset_file']
        file_format = detect_file_format(dataset_file)
        keyword = request.form['keyword']

        if not file_format:
            error_message = 'Unsupported file type. Please try using a different Dataset.'
            return render_template('entry_custom_dataset.html', error_message=error_message)

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        dataset_file.save(temp_file.name)
        store_dataset(dataset_file, file_format, mongo_url, temp_file, spark)
        df = preprocess_dataset(dataset_file, file_format, mongo_url, temp_file, spark)
        processed_df = process_dataset(df, keyword, mongo_url)

        if not processed_df:
            error_message = 'Keyword not found in Dataset. Please try using a different Keyword or Dataset.'
            return render_template('entry_custom_dataset.html', error_message=error_message)

        dataset_pie_chart_image, dataset_scatter_plot_image, top_positive_sentences, \
            top_negative_sentences = dataset_analysis(processed_df)

        spark.stop()
        return render_template('results_custom_dataset.html', dataset_pie_chart_image=dataset_pie_chart_image,
                               keyword=keyword, top_positive_sentences=top_positive_sentences,
                               top_negative_sentences=top_negative_sentences,
                               dataset_scatter_plot_image=dataset_scatter_plot_image)

    return render_template('entry_custom_dataset.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
