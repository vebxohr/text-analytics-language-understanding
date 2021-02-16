from nltk.corpus import brown, nps_chat
import nltk


# Initialize all training and test data
tokens_brown = brown.sents()
tokens_nps_chat = nps_chat.words()
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
tags_brown = [tag for word, tag in brown.tagged_words()]
tags_nps_chat = [tag for word, tag in nps_chat.tagged_words()]

# Find most common tags
max_brown = nltk.FreqDist(tags_brown).max()  # NN
max_nps_chat = nltk.FreqDist(tags_nps_chat).max()  # UH

# Create default taggers
default_tagger_brown = nltk.DefaultTagger(max_brown)
default_tagger_nps_chat = nltk.DefaultTagger(max_nps_chat)

# Tag the corpora
default_tagger_brown.tag(train_sents_brown_09)
print("Accuracy Brown default tagger 90/10: ", default_tagger_brown.evaluate(test_sents_brown_09))
default_tagger_brown.tag(train_sents_brown_05)
print("Accuracy Brown default tagger 50/50: ", default_tagger_brown.evaluate(test_sents_brown_05))
default_tagger_nps_chat.tag(train_posts_nps_chat_09)
print("Accuracy NPS chat default tagger 90/10: ", default_tagger_nps_chat.evaluate(test_posts_nps_chat_09))
default_tagger_nps_chat.tag(train_posts_nps_chat_05)
print("Accuracy NPS chat default tagger 50/50: ", default_tagger_nps_chat.evaluate(test_posts_nps_chat_05))

# Accuracy of the default taggers



# Task b)

# Patterns from NLTK book
patterns_brown = [
    (r'.*ing$', 'VBG'),  # gerunds
    (r'.*ed$', 'VBD'),  # simple past
    (r'.*es$', 'VBZ'),  # 3rd singular present
    (r'.*ould$', 'MD'),  # modals
    (r'.*\'s$', 'NN$'),  # possessive nouns
    (r'.*s$', 'NNS'),  # plural nouns
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'NN')  # nouns (default)
]
patterns_nps_chat = [
    (r'.*ing$', 'VBG'),  # gerunds
    (r'.*ed$', 'VBD'),  # simple past
    (r'.*es$', 'VBZ'),  # 3rd singular present
    (r'.*ould$', 'MD'),  # modals
    (r'.*\'s$', 'NN$'),  # possessive nouns
    (r'.*s$', 'NNS'),  # plural nouns
    (r'^-?[0-9]+(\.[0-9]+)?$', 'CD'),  # cardinal numbers
    (r'.*', 'UH')  # nouns (default)
]

regex_tagger_brown = nltk.RegexpTagger(patterns_brown)
regex_tagger_nps_chat = nltk.RegexpTagger(patterns_nps_chat)

tokens_brown = brown.tagged_words()
regex_tagger_brown.tag(list(tokens_brown[:int(len(tokens_brown) * 0.9)]))
regex_tagger_nps_chat.tag(train_posts_nps_chat_09)

print("Accuracy Brown regex tagger: ", regex_tagger_brown.evaluate(list(tokens_brown[int(len(tokens_brown) * 0.9):])))
print("Accuracy NPS chat regex tagger: ", regex_tagger_nps_chat.evaluate(tagged_posts_nps_chat))