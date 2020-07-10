from sklearn.model_selection import train_test_split

from nlptools.model.lr import *
from nlptools.preprocess import clean_text

class Classifier:
    def __init__(self):
        self.model = None

    def preprocess(self, text):
        text = clean_text(text)
        return text

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
    
    def process_labels(self, labels):
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.unique_label_cnt = len(self.label2id)
        labels = [self.label2id[label] for label in labels]
        return labels

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
    def __init__(self, tokenizer=None):
        super(WordFreqLR, self).__init__()
        self.model = None
        self.tokenizer = tokenizer
        self.freqs = {}
        self.theta = None

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
        corpus_tokens = []
        for x, y in zip(X, Y):
            words = self.tokenizer.tokenize(x)
            corpus_tokens.append(words)
            for word in words:
                pair = (word, y)
                if pair in freqs:
                    self.freqs[pair] += 1
                else:
                    self.freqs[pair] = 1
        return corpus_tokens

    def extract_features(self, tokens):
        feat = np.zeros((1, len(self.unique_label_cnt)))
        feat[0, 0] = 1
        for token in tokens:
            for i in range(len(self.unique_label_cnt)):
                feat[0, i] += self.freqs.get((token, j - 1), 0)
        return feat

    def gradientDescent(self, x, y, theta, alpha, num_class, num_iters):
        m = x.shape[0]
        for _ in range(0, num_iters):
            z = np.dot(x, theta)
            h = softmax(z)
            J = -1 / m * np.sum(np.dot(y.T, np.log(h)) + np.dot((1 - y).T, np.log(1 - h)))
            theta = theta - (alpha / m) * np.dot(x.T, h - y)
        J = float(J)
        return J, theta

    def train(self, corpus, labels, lr=1e-9, num_iters=1000):
        corpus = [self.preprocess(doc) for doc in corpus]
        unique_labels = set(labels)
        labels = self.process_labels(labels)

        corpus_tokens = self.build_freq(corpus, labels)

        feats = np.zeros(len(corpus), self.unique_label_cnt)
        for i, tokens in enumerate(corpus_tokens):
            feats[i, :] = self.extract_features(tokens)

        J, theta = gradientDescent(feats, labels, np.zeros((self.unique_label_cnt + 1, 1)), lr, num_iters)
        self.theta = theta

    def eval(self, corpus, labels, metric=["acc"]):
        corpus = [self.preprocess(doc) for doc in corpus]
        corpus_tokens = [self.tokenizer.tokenize(doc) for doc in corpus]
        labels = [self.label2id[label] for label in labels]
        feats = self.extract_features

    def predict(self, text):
        tokens = self.tokenizer.tokenize(self.preprocess(text))
        feat = self.extract_features(tokens)
        y_pred = softmax(np.dot(feat, self.feat))
        predict_label = self.id2label[np.argmax(y_pred)]
        return predict_label


class NavieBayes(Classifier):
    def __init__(self):
        super(NavieBayes, self).__init__()
        self.freqs = {}
        self.loglikelihood = {}
        self.logprior = 0

    def process_doc(self, doc):
        return self.tokenizer.tokenize(self.preprocess(doc))

    def build_freq(self, corpus, labels):
        for doc, label in zip(corpus, labels):
            for word in self.process_doc(doc):
                pair = (word, label)
                if pair in counter:
                    self.freqs[pair] += 1
                else:
                    self.freqs[pair] = 1

    def train(self, corpus, labels):
        self.build_freq(corpus, labels)
        labels = self.process_labels(labels)
        vocab = set([pair[0] for pair in self.freqs.keys()])
        V = len(vocab)
        label_cnt = defaultdict(int)
        for pair, cnt in self.freqs.items():
            label_cnt[pair[1]] += cnt
        
        D = len(corpus)
        D_pos = sum(train_y)

        # Calculate D_neg, the number of negative documents (*hint: compute using D and D_pos)
        D_neg = D - D_pos

        # Calculate logprior
        self.logprior = np.log(D_pos) - np.log(D_neg)

        # For each word in the vocabulary...
        for word in vocab:
            # get the positive and negative frequency of the word
            freq_pos = freqs.get((word, 1), 0)
            freq_neg = freqs.get((word, 0), 0)

            # calculate the probability that each word is positive, and negative
            # hint: use V instead of V_pos and V_neg in the denominator
            p_w_pos = (freq_pos + 1) / (N_pos + V)
            p_w_neg = (freq_neg + 1) / (N_neg + V)

            # calculate the log likelihood of the word
            self.loglikelihood[word] = np.log(p_w_pos / p_w_neg)

    def predict(self, text):
        tokens = self.tokenizer.tokenize(self.preprocess(text))
        p = 0
        p += self.logprior
        for token in tokens:
            p += self.loglikelihood.get(token, 0)
        return p

class SBERTClassifier(Classifier):
    def __init__(self):
        super(SBERTClassifier, self).__init__()


class BERTClassifier(Classifier):
    def __init__(self):
        super(BERTClassifier, self).__init__()
