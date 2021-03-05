from _collections import defaultdict
from operator import itemgetter
from nltk.corpus import brown
import nltk
import pandas as pd


# Task a)

tags = brown.tagged_words(tagset='universal')
tags_fd = nltk.FreqDist(tag for (word, tag) in tags)
df = pd.DataFrame(tags_fd.most_common()).to_string(index=False, header=False)
print("Task a)")
print(df, end='\n\n')


# Task b)
print("Task b)")
ambiguous_words = set()
tags_cfd = nltk.ConditionalFreqDist((word.lower(), tag) for (word, tag) in tags)
for word in tags_cfd.conditions():
    # if two or more tags
    if len(tags_cfd[word].keys()) >= 2:
        ambiguous_words.add(word)

print("Number of ambigious words:", len(ambiguous_words), end='\n\n')


# Task c)
print("Task c)")
words = brown.words()
print("Percentage of ambigious words:",(len(ambiguous_words)/len(set(words)) * 100), end="\n\n")


# Task d)
print("Task d)")
most_ambiguous_words = defaultdict(int)
for word in tags_cfd.conditions():
    most_ambiguous_words[word] += len(tags_cfd[word].keys())

sorted_most_amibious_words = sorted(most_ambiguous_words.items(), key=itemgetter(1), reverse=True)
print(list(sorted_most_amibious_words)[:10], end='\n\n')

sentences = brown.sents()

for word, count in list(sorted_most_amibious_words)[:10]:
    counter = 0
    print(f'Sentences with word "{word}"')
    for sentence in sentences:
        if word in sentence:
            print(nltk.pos_tag(sentence, tagset='universal'))
            counter += 1
            if counter >= 4:  # Print 4 sentences for each word
                print('\n')
                break


