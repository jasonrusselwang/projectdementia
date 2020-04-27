import spacy
from statistics import mean
import numpy as np

nlp = spacy.load("en_core_web_sm")


def test_case():
    doc = nlp(open("2009_04_02.txt").read())
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
              token.shape_, token.is_alpha, token.is_stop)


def get_vocab_dict(file):
    # checks frequency of lemmatized vocabulary
    vocab_freq = {}
    doc = nlp(open(file).read())
    for token in doc:
        # building the vocab dictionary
        if token.lemma_ in vocab_freq:
            vocab_freq[token.lemma_] += 1
        if token.lemma_ not in vocab_freq:
            vocab_freq[token.lemma_] = 1
    sorted_vocab = sorted((value, key) for (key, value) in vocab_freq.items())
    print(sorted_vocab)


def get_pos_dict(file):
    pos_freq = {}
    doc = nlp(open(file).read())
    for token in doc:
        if token.pos_ in pos_freq:
            pos_freq[token.pos_] += 1
        if token.pos_ not in pos_freq:
            pos_freq[token.pos_] = 1
    return pos_freq


def sentence_length(file):
    doc = nlp(open(file).read())
    length_list = []
    word_count = 0
    for token in doc:
        if token.pos_ == 'PUNCT':
            length_list.append(word_count)
            word_count = 0
        else:
            word_count += 1
            print(word_count)
    print(length_list)
    print(mean(length_list))

def markov_tracker(file):
    # open the file and make it readable in spaCy
    doc = nlp(open(file).read())
    # create the list of lists to house the transition matrix
    transition_matrix = []
    # grab the part-of-speech dictionary so we can make a base dictionary to make our life easier later
    pos_dict = get_pos_dict(file)
    for i in pos_dict.keys():
        pos_dict[i] = 0
    print(pos_dict)
    iterate = pos_dict
    # create a tracker to keep track of the pos after the pos we're looking for transitions for
    tracker = False
    for item in iterate.keys():
        before = item
        transition_dict = {}
        for token in doc:
            if tracker:
                next_pos = token.pos_
                if next_pos in transition_dict:
                    transition_dict[next_pos] += 1
                else:
                    transition_dict[next_pos] = 1
                tracker = False
            if token.pos_ == before:
                tracker = True
        print(before)
        total = sum(transition_dict.values())
        for transition in transition_dict:
            transition_dict[transition] = (transition_dict[transition]/total)*100

        row = pos_dict
        for item in row.keys():
            if item in transition_dict.keys():
                row[item] = transition_dict[item]
        print(row)
        print(sum(row.values()))
        print("______________________________________________________________________________________")
        transition_matrix.append(list(row.values()))
        for i in pos_dict.keys():
            pos_dict[i] = 0
    matrix = np.array(transition_matrix)
    print(list(pos_dict.keys()))
    with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
        print(matrix)

if __name__ == "__main__":
    markov_tracker("2009Munich.txt")
