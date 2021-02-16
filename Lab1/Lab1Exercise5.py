import json, re
import pandas as pd
from nltk.corpus import stopwords
from pathlib import Path
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.probability import FreqDist
from nltk.corpus.reader.twitter import TwitterCorpusReader

stopwords = stopwords.words('english')
stopwords.append('rt')
stopwords.append('https')
tweet_folder = Path("./tweets/terms/")

# twitter_corpus = TwitterCorpusReader("./tweets/terms/", '.*\.json')
# twitter_tokens = twitter_corpus.strings()
# docs = twitter_corpus.docs()

def print_frequency(data):
    print(pd.DataFrame(data).to_string(index=False, header=False))

def create_corpus(json_folder):
    '''
    Writes the text content of tweets to txt files in the corpus folder
    :param json_folder:
    :return:
    '''

    for file in json_folder.iterdir():
        with open(file) as f:
            name = file.name
            tweets = json.load(f)

        textfile_name = str(name)[:-5] + '.txt'

        with open(Path('./corpus') / textfile_name, 'w', encoding='utf-8') as f:

            for tweet in tweets:
                text = tweet.get("full_text")
                text = re.sub(r"http\S+", "", text)  # remove links from corpora
                f.write(text)


create_corpus(tweet_folder)

# Create NLTK corpus from txt files
corpus_folder = Path("./corpus/")
corpus = PlaintextCorpusReader('./corpus/', '.*')

print(corpus.words('anime.txt'))



# Task a)
def filter_corpus(corpus, file=None):
    '''
    Removes english stopwords
    :param tokens:
    :return:
    '''

    if file is not None:
        tokens = corpus.raw(file).split(' ')  # Using split to keep hashtags
    else:
        tokens = corpus.raw(corpus.fileids()).split(' ')
    filtered_words = []
    i = 0
    while i < len(tokens):
        word = tokens[i].lower()

        if word not in stopwords and word not in ['', '-']:
            filtered_words.append(word)
        i += 1
    return filtered_words


words = filter_corpus(corpus)


# Task b)
def top_ten_words_text(tokenized_words):
    freqDist = FreqDist(tokenized_words)
    return freqDist.most_common(10)

print("TOP 10 WORDS FOR THE ENTIRE CORPUS: ")
print_frequency(top_ten_words_text(words))

# Task c)
def top_ten_words_corpus(corpus):
    fileids = corpus.fileids()

    for file in fileids:
        filtered = filter_corpus(corpus, file)
        print(f'TOP 10 WORDS FOR {file}')
        print_frequency(top_ten_words_text(filtered))
        print('')

top_ten_words_corpus(corpus)

# Task d)
def top_10_hash_tags(corpus):
    fileids = corpus.fileids()
    most_common = {}
    for file in fileids:
        filtered = filter_corpus(corpus, file)
        tags = [token for token in filtered if token.startswith('#')]
        most_common_tags = FreqDist(tags).most_common()
        for tag in most_common_tags:
            if tag[0].startswith('#'):
                if tag[0] not in most_common.keys():
                    most_common[tag[0]] = 0
                most_common[tag[0]] += tag[1]

    return [(k, v) for k, v in sorted(most_common.items(), key=lambda item: item[1], reverse=True)][:10]

print_frequency(top_10_hash_tags(corpus))

