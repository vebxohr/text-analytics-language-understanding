import nltk
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from collections import defaultdict
from itertools import islice
from nltk.probability import ConditionalFreqDist
import matplotlib.pyplot as plt

def preprocess():
    """
    Read html file using BeautifulSoup, tokenize, and pos tag
    :return: NLTK name entity chunks
    """
    #
    with open("dump.html") as file:
        html = file.read()
        raw = BeautifulSoup(html, 'html.parser').get_text()
        tokens = nltk.word_tokenize(raw)
        sent_tokens = [nltk.word_tokenize(sent) for sent in nltk.sent_tokenize(raw)]
        sent_pos = nltk.pos_tag_sents(sent_tokens)
    return nltk.ne_chunk_sents(sent_pos)


def frequencies(chunks):
    """

    :param chunks: NLTK name entity chunks
    :return ConditonalFreqDist: frequency distribution of each entity
    """
    cfdist= ConditionalFreqDist()

    for tree in chunks:
        for subtree in islice(tree.subtrees(), 1, None):  # Skip first iteration as this is the complete tree
            # substring = []
            condition = subtree.label().lower()
            for word_tuple in subtree:
                cfdist[condition][word_tuple[0].lower()] += 1
                # substring.append(word_tuple[0])
            #substring = " ".join(substring).lower()
            #cfdist[condition][substring] += 1

    return cfdist



if __name__ == '__main__':
    # Task a)
    chunks = preprocess()

    # task 1 b)
    dist_frequencies = frequencies(chunks)
    index = []
    freqs = []
    for freq in dist_frequencies:
        freqs.append(dist_frequencies.get(freq).N())
        index.append(freq)

    df = pd.DataFrame(data=freqs, columns=["frequencies"], index=index)
    print(df)

    # Task c)
    # Only top 10 persons included for readability
    top_persons = dist_frequencies['person'].most_common(10)

    x_values = [person[0] for person in top_persons]
    y_values = [person[1] for person in top_persons]

    # Plot frequencies
    plt.bar(x_values, y_values)
    plt.xlabel('Person')
    plt.ylabel('Frequency')
    plt.savefig('person_freq.png')
    plt.show()


