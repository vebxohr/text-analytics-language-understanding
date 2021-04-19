import nltk

GRAMMAR = nltk.CFG.fromstring("""
S -> NP V
NP -> Det N
V -> "run" | "runs"
Det -> "This" | "These"
N -> "dog" | "dogs"
""")


def simple_parser(sentence):
    parser = nltk.ChartParser(GRAMMAR)
    try:
        trees = list(parser.parse(sentence))
        for tree in trees:
            print(tree)

        return trees[0]
    except ValueError as error:
        print(error)
        return None


simple_parser("This dog runs".split())
simple_parser("These dogs run".split())

