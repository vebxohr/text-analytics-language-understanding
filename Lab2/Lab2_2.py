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


# Task a)
print("Task a)")
tags_brown = [tag for word, tag in brown.tagged_words()]
tags_nps_chat = [tag for word, tag in nps_chat.tagged_words()]

# Find most common tags
max_brown = nltk.FreqDist(tags_brown).max()  # NN
max_nps_chat = nltk.FreqDist(tags_nps_chat).max()  # UH

# Create default taggers
default_tagger_brown = nltk.DefaultTagger(max_brown)
default_tagger_nps_chat = nltk.DefaultTagger(max_nps_chat)

# Evaluate the Default taggers
print("Accuracy Brown default tagger 90/10: ", default_tagger_brown.evaluate(test_sents_brown_09))
default_tagger_brown.tag(train_sents_brown_05)
print("Accuracy Brown default tagger 50/50: ", default_tagger_brown.evaluate(test_sents_brown_05))
default_tagger_nps_chat.tag(train_posts_nps_chat_09)
print("Accuracy NPS chat default tagger 90/10: ", default_tagger_nps_chat.evaluate(test_posts_nps_chat_09))
default_tagger_nps_chat.tag(train_posts_nps_chat_05)
print("Accuracy NPS chat default tagger 50/50: ", default_tagger_nps_chat.evaluate(test_posts_nps_chat_05))
print()




# Task b)
print("Task b)")
# Patterns from NLTK book
patterns= [
    (r'.*ing$', 'VBG'),  # gerunds
    (r'.*ed$', 'VBD'),  # simple past
    (r'.*es$', 'VBZ'),  # 3rd singular present
    (r'.*ould$', 'MD'),  # modals
    (r'.*\'s$', 'NN$'),  # possessive nouns
    (r'.*s$', 'NNS'),  # plural nouns
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD')
]


# Unigram tagger
print("Unigram tagger default tagger backoff")
unigram_tagger_brown_09 = nltk.UnigramTagger(train_sents_brown_09, backoff=nltk.DefaultTagger('NN'))
print("Unigram tagger accuracy brown 90/10: ", unigram_tagger_brown_09.evaluate(test_sents_brown_09))
unigram_tagger_brown_05 = nltk.UnigramTagger(train_sents_brown_05, backoff=nltk.DefaultTagger('NN'))
print("Unigram tagger accuracy brown 50/50: ", unigram_tagger_brown_05.evaluate(test_sents_brown_05))

unigram_tagger_nps_chat_09 = nltk.UnigramTagger(train_posts_nps_chat_09, backoff=nltk.DefaultTagger('UH'))
print("Unigram tagger accuracy NPS Chat 90/10: ", unigram_tagger_nps_chat_09.evaluate(test_posts_nps_chat_09))
unigram_tagger_nps_chat_05 = nltk.UnigramTagger(train_posts_nps_chat_05, backoff=nltk.DefaultTagger('UH'))
print("Unigram tagger accuracy NPS Chat 50/50: ", unigram_tagger_nps_chat_05.evaluate(test_posts_nps_chat_05))
print()

# Bigram tagger
print("Bigram tagger unigram tagger backoff")
bigram_tagger_brown_09 = nltk.BigramTagger(train_sents_brown_09, backoff=unigram_tagger_brown_09)
print("Bigram tagger accuracy brown 90/10: ", bigram_tagger_brown_09.evaluate(test_sents_brown_09))
bigram_tagger_brown_05 = nltk.BigramTagger(train_sents_brown_05, backoff=unigram_tagger_brown_05)
print("Bigram tagger accuracy brown 50/50: ", bigram_tagger_brown_05.evaluate(test_sents_brown_05))

bigram_tagger_nps_chat_09 = nltk.BigramTagger(train_posts_nps_chat_09, backoff=unigram_tagger_nps_chat_09)
print("Bigram tagger accuracy NPS Chat 90/10: ", bigram_tagger_nps_chat_09.evaluate(test_posts_nps_chat_09))
bigram_tagger_nps_chat_05 = nltk.BigramTagger(train_posts_nps_chat_05, backoff=unigram_tagger_nps_chat_05)
print("Bigram tagger accuracy NPS Chat 50/50: ", bigram_tagger_nps_chat_05.evaluate(test_posts_nps_chat_05))
print()

# Regexp tagger
print("Regexp taggers with different backoff")
regex_tagger_brown_09 = nltk.RegexpTagger(patterns, backoff=bigram_tagger_brown_09)
print("Regexp tagger accuracy brown 90/10 bigram backoff: ", regex_tagger_brown_09.evaluate(test_sents_brown_09))
regex_tagger_brown_05 = nltk.RegexpTagger(patterns, backoff=bigram_tagger_brown_05)
print("Regexp tagger accuracy brown 50/50 bigram backoff: ", regex_tagger_brown_05.evaluate(test_sents_brown_05))
regex_tagger_brown_09 = nltk.RegexpTagger(patterns, backoff=unigram_tagger_brown_09)
print("Regexp tagger accuracy brown 90/10 unigram backoff: ", regex_tagger_brown_09.evaluate(test_sents_brown_09))
regex_tagger_brown_05 = nltk.RegexpTagger(patterns, backoff=unigram_tagger_brown_05)
print("Regexp tagger accuracy brown 50/50 unigram backoff: ", regex_tagger_brown_05.evaluate(test_sents_brown_05))

regex_tagger_nps_chat_09 = nltk.RegexpTagger(patterns, backoff=unigram_tagger_nps_chat_09)
print("Regexp tagger accuracy NPS Chat 90/10 unigram backoff: ", regex_tagger_nps_chat_09.evaluate(test_posts_nps_chat_09))
regex_tagger_nps_chat_05 = nltk.RegexpTagger(patterns, backoff=unigram_tagger_nps_chat_05)
print("Regexp tagger accuracy NPS Chat 50/50 unigram backoff: ", regex_tagger_nps_chat_05.evaluate(test_posts_nps_chat_05))
regex_tagger_nps_chat_09 = nltk.RegexpTagger(patterns, backoff=bigram_tagger_nps_chat_09)
print("Regexp tagger accuracy NPS Chat 90/10 bigram backoff: ", regex_tagger_nps_chat_09.evaluate(test_posts_nps_chat_09))
regex_tagger_nps_chat_05 = nltk.RegexpTagger(patterns, backoff=bigram_tagger_nps_chat_05)
print("Regexp tagger accuracy NPS Chat 50/50 bigram backoff: ", regex_tagger_nps_chat_05.evaluate(test_posts_nps_chat_05))
print()



#print("Accuracy Brown regex tagger: ", regex_tagger_brown.evaluate(list(tokens_brown[int(len(tokens_brown) * 0.5):])))
#print("Accuracy NPS chat regex tagger: ", regex_tagger_nps_chat.evaluate(test_posts_nps_chat_09))


