import re
from collections import Counter
import utils


def find_words(text): return re.findall(r'\w+', text.lower())


def count_words(corpus):
    """

    :param corpus: A file that will be used to calculate our stats
    :return: Returns a Counter object that maps each word to the amount of times it appeared
    """
    WORDS = Counter(find_words(open(corpus).read()))
    print(corpus)
    return WORDS


class SpellCorrector:
    """
    The SpellCorrector is copied the functionality of the Peter Norvig's
    spell-corrector in http://norvig.com/spell-correct.html
    """

    def __init__(self, **kwargs):
        """
        :param corpus: the corpus to use in order to get the statistics.
        """
        corpus = kwargs.get("corpus")
        dictionary = kwargs.get("dictionary", None)
        save_to = kwargs.get("save_to", None)
        if dictionary is not None:
            print("Reading from dict", dictionary)
            self.WORDS = utils.read_dictionary(dictionary)
        else:
            print("Calculating stats from corpus", corpus)
            self.WORDS = Counter(count_words(corpus))
            if save_to is not None:
                utils.dict_to_file(self.WORDS, save_to)
        self.N = sum(self.WORDS.values())  # The number of words in our corpus

    def p(self, word):
        """
        Probability of `word`.

        :param word: The word that we want to find its probability
        :return: The probability of 'word'
        """
        return self.WORDS[word] / self.N

    def correction(self, word):
        """
        Most probable spelling correction for word.
        :param word: The word we want to correct
        :return: The correction we guessed
        """
        return max(self.candidates(word), key=self.p)

    def candidates(self, word):
        """
        Generate possible spelling corrections for word.
        :param word: The word we want to correct
        :return: A set of possible corrections
        """
        return self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word]

    def known(self, words):
        """
        The subset of `words` that appear in the dictionary of WORDS.
        :param words:
        :return:
        """
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        """
        All edits that are one edit away from `word`.
        :param word:
        :return:
        """
        letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        """
        All edits that are two edits away from `word`.
        :param word:
        :return:
        """
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))


def unit_tests():
    # sc = SpellCorrector(corpus="../MyData/big.txt", save_to="../MyData/word_freq.txt")
    sc = SpellCorrector(corpus="../MyData/big.txt")
    # sc = SpellCorrector(dictionary="../MyData/word_freq.txt")
    assert sc.correction('speling') == 'spelling'  # insert
    assert sc.correction('korrectud') == 'corrected'  # replace 2
    assert sc.correction('bycycle') == 'bicycle'  # replace
    assert sc.correction('inconvient') == 'inconvenient'  # insert 2
    assert sc.correction('arrainged') == 'arranged'  # delete
    assert sc.correction('peotry') == 'poetry'  # transpose
    assert sc.correction('peotryy') == 'poetry'  # transpose + delete
    assert sc.correction('word') == 'word'  # known
    assert sc.correction('quintessential') == 'quintessential'  # unknown
    assert find_words('This is a TEST.') == ['this', 'is', 'a', 'test']
    assert Counter(find_words('This is a test. 123; A TEST this is.')) == (
        Counter({'123': 1, 'a': 2, 'is': 2, 'test': 2, 'this': 2}))
    assert len(sc.WORDS) == 32198
    assert sum(sc.WORDS.values()) == 1115585
    assert sc.WORDS.most_common(10) == [
        ('the', 79809),
        ('of', 40024),
        ('and', 38312),
        ('to', 28765),
        ('in', 22023),
        ('a', 21124),
        ('that', 12512),
        ('he', 12401),
        ('was', 11410),
        ('it', 10681)]
    assert sc.WORDS['the'] == 79809
    assert sc.p('quintessential') == 0
    assert 0.07 < sc.p('the') < 0.08
    return 'unit_tests pass'


def spelltest(tests, sc, verbose=False):
    """Run correction(wrong) on all (right, wrong) pairs; report results."""
    import time
    start = time.clock()
    good, unknown = 0, 0
    n = len(tests)
    for right, wrong in tests:
        w = sc.correction(wrong)
        good += (w == right)
        if w != right:
            unknown += (right not in sc.WORDS)
            if verbose:
                print('correction({}) => {} ({}); expected {} ({})'
                      .format(wrong, w, sc.WORDS[w], right, sc.WORDS[right]))
    dt = time.clock() - start
    print('{:.0%} of {} correct ({:.0%} unknown) at {:.0f} words per second '
          .format(good / n, n, unknown / n, n / dt))


def test_set(lines):
    """Parse 'right: wrong1 wrong2' lines into [('right', 'wrong1'), ('right', 'wrong2')] pairs."""
    return [(right, wrong)
            for (right, wrongs) in (line.split(':') for line in lines)
            for wrong in wrongs.split()]


if __name__ == '__main__':
    print(unit_tests())
    # spelltest(test_set(open('spell-testset1.txt')))
    # spelltest(test_set(open('spell-testset2.txt')))
