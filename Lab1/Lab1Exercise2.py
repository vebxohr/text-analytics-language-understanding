from nltk import corpus
import pandas as pd
import matplotlib.pyplot as plt

# task a)
search_words = ['men', 'women', 'people']
frequencies = {}
for text in corpus.state_union.fileids():
    year = text[:4]
    words = corpus.state_union.words(text)
    frequencies[year] = [0, 0, 0]
    for i in range(len(search_words)):
        frequencies.get(year)[i] += words.count(search_words[i])
data = pd.DataFrame(frequencies, index=search_words)
print(data.T)

# task b)
data = data.T
plt.figure()
data.plot()
plt.show()
data.plot.bar()
plt.show()