import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy


class SentimentTwitterClassifier:

    def __init__(self, test_ratio=0.7):
        self.test_ratio = test_ratio
        self.classifier = None
        self.pos = []
        self.neg = []

    def format_sentence(self, sent):
        return({word: True for word in nltk.word_tokenize(sent)})

    def preparation_data(self):
        with open("./pos_tweets.txt") as f:
            for i in f:
                self.pos.append([self.format_sentence(i), 'pos'])

        with open("./neg_tweets.txt") as f:
            for i in f:
                self.neg.append([self.format_sentence(i), 'neg'])

    def train_naivebayes_classifier(self):
        self.preparation_data()
        pos_cutoff = int(self.test_ratio * len(self.pos))
        neg_cutoff = int(self.test_ratio * len(self.neg))

        # next, split labeled data into the training and test data
        training = self.pos[:pos_cutoff] + self.neg[:neg_cutoff]
        self.test = self.pos[pos_cutoff:] + self.neg[neg_cutoff:]

        self.classifier = NaiveBayesClassifier.train(training)

    def show_info_features(self):
        self.classifier.show_most_informative_features()

    def show_accuracy(self):
        print(accuracy(self.classifier, self.test))

    def classify_sentence(self, sentence):
        print("Classify sentence: %s" % sentence)
        print(self.classifier.classify(self.format_sentence(sentence)))
