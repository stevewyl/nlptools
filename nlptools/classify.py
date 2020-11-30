from sklearn.model_selection import train_test_split

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
        return (train_x, train_y), (dev_x, dev_y)

    def process_labels(self, labels):
        self.label2id = {label: i for i, label in enumerate(labels)}
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


class SBERTClassifier(Classifier):
    def __init__(self):
        super(SBERTClassifier, self).__init__()


class BERTClassifier(Classifier):
    def __init__(self):
        super(BERTClassifier, self).__init__()
