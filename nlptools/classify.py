from sklearn.model_selection import train_test_split

from nlptools.model import lr
from nlptools.preprocess import clean_text

class Classifier:
    def __init__(self):
        self.model = None

    def preprocess(self):
        '''
        '''
        pass

    def read_data(self, data_fn):
        X, Y = zip(*[line.strip.rsplit("\t", 1) for line in open(data_fn, encoding='utf8')])
        return list(X), list(Y)

    def read(self, train_fn, dev_fn="", dev_size=0.2):
        train_x, train_y = self.read_data(train_fn)
        if dev_fn:
            dev_x, dev_y = self.read_data(dev_fn)
        else:
            train_x, train_y, dev_x, dev_y = train_test_split(train_x, train_y, dev_size)
        return (train_x, trainy), (dev_x, dev_y)

    def train(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()

    def predict(self):
        raise NotImplementedError()

    def save(self, model_fn):
        pass

    def load(self, model_fn):
        self.model = None


class WordFreqLR(Classifier):
    def __init__(self):
        super(WordFreqLR, self).__init__()
        self.model = None

    def build_freq(self, X, Y):
        """Build frequencies.
        Input:
            X: a list of tweets
            Y: an m x 1 array with the sentiment label of each tweet
                (either 0 or 1)
        Output:
            freqs: a dictionary mapping each (word, sentiment) pair to its
            frequency
        """
        freqs = {}
        for x, y in zip(X, Y):
            for word in self.tokenize(X):
                pair = (word, y)
                if pair in freqs:
                    freqs[pair] += 1
                else:
                    freqs[pair] = 1
        return freqs

    def train(self, corpus):

class NavieBayes(Classifier):
    def __init__(self):
        super(NavieBayes, self).__init__()

    def classify(self, corpus):

class SBERTClassifier(Classifier):
    def __init__(self):
        super(SBERTClassifier, self).__init__()

    def

class BERTClassifier(Classifier):


