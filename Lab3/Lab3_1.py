from collections import defaultdict

from nltk.corpus import names
from nltk.classify import apply_features
import nltk
import random

random.seed(12345)


def gender_features(word):
    # Code from NLTK book chapter 6
    # features = defaultdict(lambda: 0)
    features = {'suffix1': word[-1:].lower(), 'suffix2': word[-2:].lower()}
    # features = {'name': word}
    # for char in word:
    #     features[char] += 1
    return features


# Code from NLTK book chapter 6
labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                 [(name, 'female') for name in names.words('female.txt')])
random.shuffle(labeled_names)

size_names = int(len(labeled_names) * 0.8)

# Split into training and test data, 80/20 split
train_set = apply_features(gender_features, labeled_names[:size_names])
test_set = apply_features(gender_features, labeled_names[size_names:])

# Create classifiers
bayes_classifier = nltk.NaiveBayesClassifier.train(train_set)
tree_classifier = nltk.DecisionTreeClassifier.train(train_set)
# 10 iterations as the accuracy quickly converges
max_entropy_classifier = nltk.MaxentClassifier.train(train_set, trace=1, max_iter=10)

print(f'Naive Bayes Classifier Accuracy {nltk.classify.accuracy(bayes_classifier, test_set)}')
print(f'Decision Tree Classifier Accuracy {nltk.classify.accuracy(tree_classifier, test_set)}')
print(f'Maximum Entropy Classifier Accuracy {nltk.classify.accuracy(max_entropy_classifier, test_set)}')
