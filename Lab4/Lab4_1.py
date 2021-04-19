import nltk
from nltk.corpus import brown
import re


sentences = brown.tagged_sents() # get PoS tagged sentences
sentence = "the little cat sat on the mat in the house on a hill"
sentence = nltk.pos_tag(nltk.word_tokenize(sentence))
#sentences = [sentence]

grammar = r"""
NP: {<DT>? <JJ>* <NN>*} # Noun Phrase
P: {<IN>}           # Preposition
V: {<V.*>}          # Verb
CLAUSE: {<V> (<P>* <NP>*)*} # Verb followed by a combination of prepositions and Noun Phrases
"""
# grammar = r"""
#     NP: {<DT>?<JJ>*<NN>*} # NP
#     P: {<IN>} # Preposition
#     V: {V.*} # Verb
#     """

tuples = []
for sent in sentences:
    parser = nltk.RegexpParser(grammar)
    tree = parser.parse(sent)
    for subtree in tree.subtrees():
        if subtree.label() == 'CLAUSE':
            a = []
            for subsubtree in subtree.subtrees():
                if subsubtree.label() == 'CLAUSE':  # First subtree is the same as parent tree so skip
                    continue
                if subsubtree.label() == 'NP':
                    a.append('NP')
                else:
                    a.append(subsubtree[0][0])

            tuples.append(a)


for tuple in tuples[:20]:
    print(tuple)

