import numpy as np
import pandas as pd
import re
import string
import json
import datetime
import tensorflow as tf
import contractions
import matplotlib.pyplot as plt
from textblob import TextBlob
from sklearn.model_selection import train_test_split
# NLTK
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
# Keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop, Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model


# The keras model is mostly based on
# https://towardsdatascience.com/an-easy-tutorial-about-sentiment-analysis-with-deep-learning-and-keras-2bf52b9cba91
# https://github.com/sergiovirahonda/TweetsSentimentAnalysis/blob/main/TweetsSentimentPredictions.ipynb

# Use TextBlob for sentiment analysis
def analyze_sentiment(text_string):
    """
    Use TextBlob for sentiment analysis.
    :param text_string: a string of text to be analyzed.
    :return: the corresponding string label of the sentiment analysis of the text.
    """
    analysis = TextBlob(text_string)
    if analysis.sentiment.polarity > 0:
        return 'pos'
    elif analysis.sentiment.polarity == 0:
        return 'neu'
    else:
        return 'neg'


# from https://github.com/sergiovirahonda/TweetsSentimentAnalysis/blob/main/TweetsSentimentPredictions.ipynb


# regular expressions from https://blog.chapagain.com.np/python-nltk-twitter-sentiment-analysis-natural-language-processing-nlp/
def preprocessing(tweet_text):
    """
    Cleaning a single tweet text: removing links and twitter specific text and symbols.
    Stemming and tokenizing the tweets and removing stopwords.
    :param tweet_text: tweet to be cleaned
    :return: Cleaned and tokenized tweet as a list of tokens
    """
    # Objects used for text cleaning
    tweet_tokenizer = TweetTokenizer(strip_handles=True, preserve_case=False)
    stopwords_english = stopwords.words('english')
    stemmer = PorterStemmer()

    # remove stock market tickers like $GE
    tweet_text = re.sub(r'\$\w*', '', tweet_text)

    # remove old style retweet text "RT"
    tweet_text = re.sub(r'^RT[\s]+', '', tweet_text)

    # remove hyperlinks
    tweet_text = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet_text)

    # remove hashtags
    # only removing the hash # sign from the word
    tweet_text = re.sub(r'#', '', tweet_text)

    # tokenize tweets
    tweet_tokens = tweet_tokenizer.tokenize(tweet_text)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def label_encoding(text_labels):
    """
    Encode the labels to integers to be used by the model.
    :param text_labels: string labels used in sentiment analysis
    :return: n*n matrix where n is the number of categories/labels consisting of the encoded labels
    """
    encoded_labels = []
    for label in text_labels:
        if label == 'neu':
            encoded_labels.append(1)
        elif label == 'neg':
            encoded_labels.append(0)
        elif label == 'pos':
            encoded_labels.append(2)
    encoded_labels = np.array(encoded_labels)
    return to_categorical(encoded_labels)


def get_2d_array(data_strings, keras_tokenizer):
    """
    Tokenizes the texts/strings, turns them into a sequence of integers, and pads them, ready for fitting a model.
    :param keras_tokenizer: Keras tokenizer used to tokenize the tweets
    :param data_strings: All tweets to be used in in the model as a list of strings
    :return: 2d array consisting of list of lists of integers - one list for each tokenized padded twitter text
    """
    # Tokenize the sentence, keeping only MAX_WORDS most common words

    sequences = keras_tokenizer.texts_to_sequences(data_strings)
    array = pad_sequences(sequences, maxlen=200)  # Maximum length of words for each text
    return array


def predict(trained_model, texts, keras_tokenizer, dates=None):
    """
    Predict sentiments from a set of texts/tweets and with trained model and tokenizer.
    :param trained_model: The trained model used for prediction
    :param texts: List of strings to be analyzed
    :param keras_tokenizer: Tokenizer used to prepare the data
    :param dates: a list of dates for the tweets
    :return:
    """
    preprocessed_tweets = []
    for text in texts:
        p_tweet = preprocessing(text)
        preprocessed_tweets.append(p_tweet)
    preprocessed_tweets = [' '.join(preprocessed_tweet) for preprocessed_tweet in preprocessed_tweets]
    texts_2d = get_2d_array(preprocessed_tweets, keras_tokenizer)
    sentiment = ['Negative', 'Neutral', 'Positive']
    results = trained_model.predict(texts_2d)
    text_results = []
    int_results = []
    for result in results:
        sentiment_value = np.around(result, decimals=0).argmax()
        int_results.append(sentiment_value)
        text_results.append(sentiment[sentiment_value])

    df = pd.DataFrame(columns=['Text', 'Prediction Sentiment', 'sentiment_value', 'dates'])
    df['Text'] = preprocessed_tweets
    df['Prediction Sentiment'] = text_results
    df['sentiment_value'] = int_results
    df['dates'] = dates

    return df


if __name__ == '__main__':

    """
    Create and train the model using Keras/tensorflow
    """

    # Some code to make tensorflow work with my GPU by enabling memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    # Use tweets from NLTK twitter samples for training
    tweets = twitter_samples.strings('tweets.20150430-223406.json')
    # pos_tweets = twitter_samples.strings('negative_tweets.json')

    # Classify the tweets using sentiment analysis
    sentiments = []
    for tweet in tweets:
        sentiments.append(analyze_sentiment(tweet))

    # Organize the tweets and their sentiments in a dataframe
    data = pd.DataFrame(columns=["tweet_text", "sentiment"])
    data['tweet_text'] = tweets
    data['sentiment'] = sentiments

    # Clean the tweets
    preprocessed_tweets = []
    for tweet in tweets:
        preprocessed_tweets.append(preprocessing(tweet))

    # add the processed tweets to the dataframe
    data['processed_tweets'] = [' '.join(preprocessed_tweet) for preprocessed_tweet in preprocessed_tweets]
    # print(data.groupby('sentiment').nunique())

    # Encode the preprocessed tweets and their labels for machine learning
    MAX_WORDS = 5000  # only using 5000 most frequent words
    processed_texts = np.array(data['processed_tweets'])
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(processed_texts)
    tweet_array = get_2d_array(processed_texts, tokenizer)
    labels = np.array(data['sentiment'])
    encoded_labels = label_encoding(labels)

    # Split the data into train and test using scikit-learn: 90/10 split
    X_train, X_test, y_train, y_test = train_test_split(tweet_array, encoded_labels, train_size=0.9)

    # Create model with 3 layers using Keras Sequential model
    model = Sequential()
    # input_dim: vocabulary size; output_dim: dimension of dense embedding (dimension of each word vector)
    # Using word embeddings as text representation: map words to vectors
    model.add(layers.Embedding(input_dim=MAX_WORDS, output_dim=64))  # returns 3D floating-point tensor
    model.add(layers.LSTM(15, dropout=0.5))  # Add LSTM layer and use dropout to prevent some overfitting
    model.add(layers.Dense(3, activation='softmax'))  # Add the output layer with dimension 3 (pos, neg, neu)

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    es = EarlyStopping(monitor='val_accuracy', mode='max', patience=4)  # Use early stopping to prevent some overfitting
    checkpoint = ModelCheckpoint("best_model.hdf5", monitor='val_accuracy', verbose=1, save_best_only=True,
                                 mode='auto', period=1, save_weights_only=False)  # Save best model
    # Fit the model
    # Comment out this to use old model
    # history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[es, checkpoint])
    best_model = load_model("best_model.hdf5")  # Use the model with the highest accuracy

    print(f"Final accuracy: {best_model.evaluate(X_test, y_test)[1]}")

    """
    Testing the model on Trump tweets from one month before to one month after the 2020 election.
    """
    # Load the Trump tweets to a pandas dataframe
    trump_tweets = pd.read_json(r'realDonaldTrump.json')
    trump_tweets.set_index(['timeStamp'])
    # Find tweets from one month before to one month after the 2020 election (03. nov. 2020)
    election_tweets_df = trump_tweets[trump_tweets.timeStamp.between(np.datetime64('2020-10-03'),
                                                                     np.datetime64('2020-12-03'),
                                                                     inclusive=True)]
    dates = np.array(list(election_tweets_df['timeStamp']))
    election_tweets = election_tweets_df['text']

    sentiments1 = []
    labeled_sentiments = []
    sentiment_strings = ['neg', 'neu', 'pos']
    for tweet in election_tweets:
        analyzed_tweet = analyze_sentiment(tweet)
        sentiments1.append(analyzed_tweet)
        labeled_sentiments.append(sentiment_strings.index(analyzed_tweet))

    sentiments = label_encoding(sentiments1)

    # Predict sentiment of Trump tweets using the best model
    predictions = predict(best_model, election_tweets, tokenizer, dates)
    predicted_sentiments = predictions['sentiment_value']
    # Calculate accuracy
    wrong_count = 0
    for i in range(len(predicted_sentiments)):
        if predicted_sentiments[i] != labeled_sentiments[i]:
            wrong_count += 1
    print(f"Accuracy on Trump tweets: {(len(predicted_sentiments) - wrong_count)/len(predicted_sentiments)}")

    # Group the tweet sentiments by date
    grouped_by_date = predictions[['dates', 'sentiment_value']].groupby(pd.Grouper(key='dates', freq='D')).agg(list)
    # Find the sentiments for the tweets belonging to a date/chunk
    sentiments_per_chunk = []
    for date in grouped_by_date['sentiment_value']:
        group_sent = np.zeros(3)
        for value in date:
            group_sent[value] += 1
        sentiments_per_chunk.append(group_sent)

    # Find all positive and negative labeled ones
    positive = [x[2] for x in sentiments_per_chunk]
    negative = [x[0] for x in sentiments_per_chunk]

    # Calculate average number of positive and negative tweets for the entire period
    avg_pos = sum(positive)/len(positive)
    avg_neg = sum(negative)/len(negative)
    # Calculate average number of positive and negative tweets for the first and second half of the period
    avg_pos_first = sum(positive[:30])/len(positive[:30])
    avg_neg_first = sum(negative[:30])/len(negative[:30])
    avg_pos_second = sum(positive[30:])/len(positive[30:])
    avg_neg_second = sum(negative[30:]) / len(negative[30:])

    print(f"Change in avg sentiment from first to second month: \n\tPositive:"
          f" {avg_pos_first - avg_pos_second}\n\tNegative: {avg_neg_first - avg_neg_second}")
    sentiment_strings = ['neg', 'neu', 'pos']

    # Plot the number of positive and the number of negative tweets for each day,
    # as well as the average over the period.
    plt.plot(positive, color='g', label="positive")
    plt.plot(negative, color='b', label="neagtive")
    plt.xlabel('Day')
    plt.ylabel('# positive or negative tweets')
    plt.axhline(y=avg_pos, color='g', linestyle='-')
    plt.axhline(y=avg_neg, color='b', linestyle='-')
    plt.legend()
    plt.show()


