from pathlib import Path
import json
import nltk
import re
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import MLE

tokenizer = nltk.RegexpTokenizer(r"\w+")


# Load Donald Trump Tweets
with open("./realDonaldTrump.json", encoding='utf-8') as file:
    data = json.load(file)
    donald_tweets = []
    for tweet in data:
        # Tokenize, remove punctuation marks and simple URLS
        tweet_text = tokenizer.tokenize(re.sub(r'http\S+', '', tweet['text'].lower()))
        if len(tweet_text) != 0:
            donald_tweets.append(tweet_text)


train, vocab = padded_everygram_pipeline(3, donald_tweets)

lm = MLE(3)

lm.fit(train, vocab)


# Find the most likely completion of the provided sentence, limited by a # of tries.
def predict_sent(preceding, tries):
    input_sent = preceding
    preceding = tokenizer.tokenize(preceding.lower())
    i = 1
    while i < tries:
        print(i)
        sent = lm.generate(i, text_seed=preceding, random_seed=3)
        i += 1
        if sent[-1] == '</s>':
            sent = sent[:-1]
            return input_sent + " " + " ".join(sent)

    return f"No match in {tries}"


print(predict_sent("make America", 10))

# print(lm.generate(2, text_seed=['make', 'america'], random_seed=3))
