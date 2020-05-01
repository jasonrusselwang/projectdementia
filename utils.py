import spacy
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import syllables

nlp = spacy.load("en_core_web_sm")


def spaCy_demo(file):
    doc = nlp(open(file).read())
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)


def get_vocab_dict(file, n=2, m=20):
    # checks frequency of vocabulary
    vocab_freq = {}
    doc = nlp(open(file).read())
    for token in doc:
        # building the vocab dictionary
        if token.text in vocab_freq:
            vocab_freq[token.text] += 1
        if token.text not in vocab_freq:
            vocab_freq[token.text] = 1
    sorted_vocab = sorted((value, key) for (key, value) in vocab_freq.items())
    filtered_dict = {}
    for i in vocab_freq.keys():
        if vocab_freq[i] >= n:
            filtered_dict[i] = vocab_freq[i]
    # return the top n frequented words
    sorted_filtered_dict = sorted(filtered_dict.items(), key=lambda x: x[1], reverse=True)[0:m]
    m_sum = sum(dict(sorted_filtered_dict).values())
    total_words = sum(vocab_freq.values())
    top_frac = round((m_sum / total_words) * 100, 3)
    return top_frac, sorted_filtered_dict


def sectioned_vocabulary(file, n=4):
    doc = nlp(open(file).read())
    # make the document into a list of words
    word_list = []
    for token in doc:
        word_list.append(token.text)
    # section off the word list into fourths
    section = round(len(word_list) / n)
    # find the lengths of the unique words in each section and calculate the new words added per fourth
    unique_words = {}
    for i in list(range(n)):
        last = len(set(word_list[:section * i]))
        next = len(set(word_list[:section * (i + 1)]))
        if i == n - 1:
            next = len(set(word_list))
        delta = next - last
        unique_words[i] = delta
    return str(file), unique_words


def plot_vocabulary(tup1, tup2=None):
    # input two tuples from sectioned_vocabulary with name and dict
    # plot 1 code shown below
    name1 = tup1[0]
    dict1 = tup1[1]
    # split into two axes
    x = np.array(list(dict1.keys()))
    y = np.array(list(dict1.values()))
    # get linear and polynomial fits
    m, b = np.polyfit(x, y, 1)  # linear fit
    print("{} Linear Coefficients: {}x + {}".format(name1, round(m, 4), round(b, 4)))
    coefficients = np.polyfit(x, y, 3)  # poly3 fit
    poly = np.poly1d(coefficients)
    new_x = np.linspace(x[0], x[-1])
    new_y = poly(new_x)
    # optional second input for comparison
    if tup2:
        name2 = tup2[0]
        dict2 = tup2[1]
        # plot 2 code shown below
        # split the dict into two sets of axes
        a = np.array(list(dict2.keys()))
        e = np.array(list(dict2.values()))
        # linear and polynomial fits
        c, d = np.polyfit(a, e, 1)  # linear fit
        print("{} Linear Coefficients: {}x + {}".format(name2, round(c, 4), round(d, 4)))
        coefficients2 = np.polyfit(a, e, 3)  # poly3 fit
        poly2 = np.poly1d(coefficients2)
        print(coefficients2)
        new_a = np.linspace(a[0], a[-1])
        new_e = poly2(new_a)
        plt.subplot(1, 2, 2)
        plt.bar(a, e, 0.5, color='g')
        plt.xlabel('sections(n={})'.format(len(dict2.keys())))
        plt.ylabel('New words added per section')
        plt.plot(a, c * x + d, 2)  # linear
        plt.plot(new_a, new_e, 2)  # polynomial
        plt.title(name2)
        plt.subplot(1, 2, 1)
    plt.bar(x, y, 0.5, color='g')
    plt.xlabel('sections(n={})'.format(len(dict1.keys())))
    plt.ylabel('New words added per section')
    plt.plot(x, m * x + b, 1)  # linear
    plt.plot(new_x, new_y, 1)  # polynomial
    plt.title(name1)
    plt.tight_layout()
    plt.show()


def syllable_check(file):
    doc = nlp(open(file).read())
    syllable_dict = {}
    for token in doc:
        if token.pos_ in ['SPACE', 'PUNCT', 'X', 'SYM', 'NUM']:
            continue
        s_num = syllables.estimate(token.text)
        if s_num in syllable_dict.keys():
            syllable_dict[s_num] += 1
        else:
            syllable_dict[s_num] = 1
    plt.bar(syllable_dict.keys(), syllable_dict.values(), 0.5, color='g')
    plt.show()
    return syllable_dict


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
    print(length_list)
    print('Mean Sentence Length: {}'.format(round(mean(length_list))))


def markov_tracker(file):
    # open the file and make it readable in spaCy
    doc = nlp(open(file).read())
    # create the list of lists to house the transition matrix
    transition_matrix = []
    # get the POS dictionary of the file and create a template dict to fill in for easy matrix generation
    pos_dict = get_pos_dict(file)
    for i in pos_dict.keys():
        pos_dict[i] = 0
    # print(pos_dict)
    iterate = pos_dict
    # create a tracker to keep track of the POS after the POS we're looking for transitions for
    tracker = False
    # loop through each pos type in the speech to create a transition matrix
    for item in iterate.keys():
        transition_dict = {}
        for token in doc:
            if tracker:
                next_pos = token.pos_
                if next_pos in transition_dict:
                    transition_dict[next_pos] += 1
                else:
                    transition_dict[next_pos] = 1
                tracker = False
            if token.pos_ == item:
                tracker = True
        # print the part of speech and the transitions probabilities for that pos
        # print(item)
        total = sum(transition_dict.values())
        for transition in transition_dict:
            if transition == 0:
                transition_dict[transition] = 0
            else:
                transition_dict[transition] = round((transition_dict[transition] / total), 2)
        row = pos_dict
        for pos in row.keys():
            if pos in transition_dict.keys():
                row[pos] = transition_dict[pos]
        # print(row)
        # print(sum(row.values()))
        # print("______________________________________________________________________________________")
        transition_matrix.append(list(row.values()))
        for i in pos_dict.keys():
            pos_dict[i] = 0
    matrix = np.array(transition_matrix)
    pos_map = list(pos_dict.keys())
    # with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.2f}'.format}, linewidth=100):
    #     print(matrix)
    return pos_map, matrix


def bmatrix(a):
    # input np.array, returns a bmatrix in LaTeX
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    return '\n'.join(rv)


if __name__ == "__main__":
    for n in [.1,.15,.2]:
        # return pos_map and transition matrix for all pos
        old = markov_tracker("2009Munich.txt")
        new = markov_tracker("2020Kickoff.txt")
        m1 = old[1].flatten()
        m2 = new[1].flatten()
        slice = round(len(m1) * n)
        slice2 = round(len(m2) * n)
        reversed_m1 = sorted(m1, reverse=True)
        reversed_m2 = sorted(m2, reverse=True)
        print('Sum of Top {}% of Transition Probabilities'.format(n*100))
        print(sum(reversed_m1[:slice]))
        print(sum(reversed_m2[:slice2]))
    # m1 = np.array(list(filter(lambda a: a != 0.0, m1)))
    # m2 = np.array(list(filter(lambda a: a != 0.0, m2)))
    # m11 = {}
    # for i in m1:
    #     if i in m11.keys():
    #         m11[i] += 1
    #     else:
    #         m11[i] = 1
    # print(m1)
    # print(len(m1))
    # print(m2)
    # print(len(m2))
    # plt.subplot(1, 2, 1)
    # plt.hist(m1, bins=20)
    # plt.ylabel('Probability')
    # plt.xlabel('Data')
    # plt.subplot(1, 2, 2)
    # plt.hist(m2, bins=20)  # `density=False` would make counts
    # plt.ylabel('Probability')
    # plt.xlabel('Data')
    # plt.tight_layout()
    # plt.show()


    # return total section of the top 20 words compared to total words
    # old = get_vocab_dict("2009Munich.txt", 2, 20)
    # new = get_vocab_dict("2020Kickoff.txt", 2, 20)
    # print(old)
    # print(new)

    # plots sectioned graphs for new words added
    # for i in [4, 10, 25, 50, 100]:
    #     old = sectioned_vocabulary("2009Munich.txt", i)
    #     new = sectioned_vocabulary("2020Kickoff.txt", i)
    #     print('{} Sections:'.format(i))
    #     plot_vocabulary(old,new)
