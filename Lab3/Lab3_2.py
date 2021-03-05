import pandas as pd
import numpy as np
import ast
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score


tweet_folder = Path("./tweets/")

# Load Elon Musk tweets
elon_tweets = pd.read_csv(tweet_folder / "elonmusk_tweets.csv")
# Get the text from tweets and remove the 'b' prefix
elon_tweet_text = elon_tweets["text"].apply(lambda x: ast.literal_eval(x).decode("utf-8"))
elon_tweet_tests = elon_tweet_text[500:600]
elon_tweet_text = elon_tweet_text[:500]

# Vectorize tweets
tfidf = TfidfVectorizer(stop_words=stopwords.words('english'), analyzer='word')
elon_labels = np.zeros(len(elon_tweet_text))
elon_vectors = tfidf.fit_transform(elon_tweet_text)

# Load Trump tweets
with open(tweet_folder / "realDonaldTrump.json", encoding='utf-8') as file:
    data = json.load(file)
    donald_tweets = []
    for tweet in data:
        donald_tweets.append(tweet['text'])

donald_test = donald_tweets[500:600]
donald_tweets = donald_tweets[:500]

donald_labels = np.ones(len(donald_tweets))
labels = np.append(elon_labels, donald_labels)  # Elon is 0 and Trump is 1
tweets = np.append(elon_tweet_text, donald_tweets)

features = tfidf.fit_transform(tweets)
features_nd = features.toarray()
X_train, X_test, y_train, y_test = train_test_split(features_nd, labels, train_size=0.80, random_state=1234)

# Extra test set
test_tweets = np.append(np.asarray(elon_tweet_tests), np.asarray(donald_test))
test_labels = np.append(np.zeros(len(elon_tweet_tests)), np.ones(len(donald_test)))
zipped = list(zip(test_tweets, test_labels))
random.shuffle(zipped)
test_tweets, test_labels = zip(*zipped)

# Create Bayes classifier
model = MultinomialNB()
model = model.fit(X=X_train, y=y_train)

# Simple error rate
wrong = 0.0
for i, t_tweet in enumerate(test_tweets):
    t = [t_tweet]
    test_tweet_vector = tfidf.transform(t)
    predicted = model.predict(test_tweet_vector)
    if predicted[0] != test_labels[i]:
        wrong += 1

print(f'error rate: {wrong/len(test_tweets)}')

predicted = model.predict(X_test)
print(accuracy_score(y_test, predicted))

# Check probability of a specific tweet.
test_tweet = ["Just agree to do Clubhouse with @kanyewest"]
test_tweet_vector = tfidf.transform(test_tweet)
predicted_probability = model.predict_proba(test_tweet_vector)
print(predicted_probability)
