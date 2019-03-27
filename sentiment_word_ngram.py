# Load and prepare the dataset
import random

import nltk
from nltk import ngrams
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier, MaxentClassifier

nltk.download('movie_reviews')


# feature extractor function
def bag_of_words(words):
    return dict([(word, True) for word in words])


def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})


# ngram feature
def bag_of_ngrams(words, n=2):
    ngs = [ng for ng in iter(ngrams(words, n))]
    return bag_of_words(ngs)


# word feature and ngram feature
def bag_of_all(words, n=2):
    all_features = bag_of_words(words)
    ngram_features = bag_of_ngrams(words, n=n)
    all_features.update(ngram_features)
    return all_features


documents = [(
    list(movie_reviews.words(fileid)), category)
    for category in movie_reviews.categories()
    for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

test_text = "I love this movie, very interesting"
test_sentence = "Sound is annoying"

# Train Naive Bayes classifier
data_sets = [(bag_of_all(d), c) for (d, c) in documents]
train_set, test_set = data_sets[100:], data_sets[:100]

# word feature and ngram feature
nb_all_classifier = NaiveBayesClassifier.train(train_set)
print("Apply word feature and ngram feature")
nb_all_classifier.show_most_informative_features(10)

# predict sentiment
print("Bag of all Naive Bayes,Sentence: %s" % test_sentence)
print(nb_all_classifier.classify(bag_of_all(
    format_sentence(test_sentence))))

print("Bag of all Naive Bayes,Sentence: %s" % test_text)
print(nb_all_classifier.classify(bag_of_all(
    format_sentence(test_text))))

# word feature and ngram feature
maxent_all_classifier = MaxentClassifier.train(train_set, "megam")
print("Apply word feature and ngram feature")
maxent_all_classifier.show_most_informative_features(10)

# predict sentiment
print("Maxent classifier, Sentence: %s" % test_sentence)
print(maxent_all_classifier.classify(bag_of_all(
    format_sentence(test_sentence))))

print("Maxent Classifier,Sentence: %s" % test_text)
print(maxent_all_classifier.classify(bag_of_all(
    format_sentence(test_text))))
