from nltk.corpus import brown, nps_chat
import nltk


# Initialize all training and test data
tokens_brown = brown.sents()
tokens_nps_chat = nps_chat.posts()
tagged_sents_brown = brown.tagged_sents()
tagged_posts_nps_chat = nps_chat.tagged_posts()

size_brown_09 = int(len(tagged_sents_brown) * 0.9)
size_brown_05 = int(len(tagged_sents_brown) * 0.5)
size_nps_chat_09 = int(len(tagged_posts_nps_chat) * 0.9)
size_nps_chat_05 = int(len(tagged_posts_nps_chat) * 0.5)
train_sents_brown_09 = tagged_sents_brown[:size_brown_09]
test_sents_brown_09 = tagged_sents_brown[size_brown_09:]
train_sents_brown_05 = tagged_sents_brown[:size_brown_05]
test_sents_brown_05 = tagged_sents_brown[size_brown_05:]

train_posts_nps_chat_09 = tagged_posts_nps_chat[:size_nps_chat_09]
test_posts_nps_chat_09 = tagged_posts_nps_chat[size_nps_chat_09:]
train_posts_nps_chat_05 = tagged_posts_nps_chat[:size_nps_chat_05]
test_posts_nps_chat_05 = tagged_posts_nps_chat[size_nps_chat_05:]

patterns = [
    (r'.*ing$', 'VBG'),  # gerunds
    (r'.*ed$', 'VBD'),  # simple past
    (r'.*es$', 'VBZ'),  # 3rd singular present
    (r'.*ould$', 'MD'),  # modals
    (r'.*\'s$', 'NN$'),  # possessive nouns
    (r'.*s$', 'NNS'),  # plural nouns
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD')
]

# Task a)
print("Task a)")

# lookup tagger from NLTK book Ch 5, 4.3
def lookup_tagger(words, tagged_words):
    fd = nltk.FreqDist(words)
    cfd = nltk.ConditionalFreqDist(tagged_words)
    most_freq_words = fd.most_common(200)
    likely_tags = dict((word, cfd[word].max()) for (word, _) in most_freq_words)
    return nltk.UnigramTagger(model=likely_tags, backoff=nltk.DefaultTagger('NN'))


brown_lookup_tagger = lookup_tagger(brown.words(), brown.tagged_words())
print("Lookup Tagger brown 90/10: ", brown_lookup_tagger.evaluate(test_sents_brown_09))
print("Lookup Tagger brown 50/50: ", brown_lookup_tagger.evaluate(test_sents_brown_05))

regex_tagger_brown_09 = nltk.RegexpTagger(patterns, backoff=brown_lookup_tagger)
print("Regexp tagger accuracy brown 90/10 lookup backoff: ", regex_tagger_brown_09.evaluate(test_sents_brown_09))
regex_tagger_brown_05 = nltk.RegexpTagger(patterns, backoff=brown_lookup_tagger)
print("Regexp tagger accuracy brown 50/50 lookup backoff: ", regex_tagger_brown_05.evaluate(test_sents_brown_05))
print()

# Test with fewer words
print("Test regexp with fewer words for lookup tagger")
brown_lookup_tagger = lookup_tagger(brown.words()[:1000], brown.tagged_words()[:1000])
regex_tagger_brown_09 = nltk.RegexpTagger(patterns, backoff=brown_lookup_tagger)
print("Regexp tagger accuracy brown 90/10 lookup backoff: ", regex_tagger_brown_09.evaluate(test_sents_brown_09))
regex_tagger_brown_05 = nltk.RegexpTagger(patterns, backoff=brown_lookup_tagger)
print("Regexp tagger accuracy brown 50/50 lookup backoff: ", regex_tagger_brown_05.evaluate(test_sents_brown_05))
print()

# Unigram with fewer sentences
bigram_tagger_brown_09 = nltk.BigramTagger(train_sents_brown_09, backoff=brown_lookup_tagger)
print("Bigram tagger accuracy brown 90/10 with lookup backoff: ", bigram_tagger_brown_09.evaluate(test_sents_brown_09))
train_sents_brown_09 = tagged_sents_brown[:size_brown_09][:1000]
test_sents_brown_09 = tagged_sents_brown[size_brown_09:][:1000]
brown_lookup_tagger = lookup_tagger(brown.words(), brown.tagged_words())
bigram_tagger_brown_09 = nltk.BigramTagger(train_sents_brown_09, backoff=brown_lookup_tagger)
print("Bigram with fewer sentences")
print("Bigram tagger accuracy brown 90/10 with lookup backoff: ", bigram_tagger_brown_09.evaluate(test_sents_brown_09))