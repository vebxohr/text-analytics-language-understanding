import nltk

grammar = r"""
NP: {<NNP>∗}
{<DT>?<JJ>?The<NNS>}
{<NN> <NN>}
"""


def load_txt_to_sentence(filename):
    """
    Load txt file to tokenized sentences with PoS tags
    :param filename: filepath
    :return: List of tokenized sentences with PoS tags
    """
    sentences = []
    with open(filename, 'r') as file:

        for line in file:
            if line != '\n':
                newline = line.replace('\n', "").replace("\'", "\'")
                sentences.append(nltk.word_tokenize(newline.lower()))

    return [nltk.pos_tag(sent) for sent in sentences]


space_sentences = load_txt_to_sentence("SpaceX.txt")


matches = []
trees = []
# Chunk each sentence
for sent in space_sentences:

    parser = nltk.RegexpParser(grammar)
    tree = parser.parse(sent)
    trees.append(tree)

    for subtree in tree.subtrees():
        match = None
        if subtree.label() == 'NP':
            match = []
            for leave in subtree.leaves():
                match.append(leave[0])
        if match:
            matches.append(match)

# Print the NP chunks for 5 first sentences
for tree in trees[:5]:
    print(tree)
    print('\n')

# Print the matching texts
print(matches)
