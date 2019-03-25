# Load and prepare the dataset
import random
import string

import nltk
from nltk import ngrams
from nltk.corpus import movie_reviews, stopwords
from nltk.classify import NaiveBayesClassifier, MaxentClassifier
from nltk.classify.util import accuracy

nltk.download('movie_reviews')
nltk.download('stopwords')
stopwords_english = stopwords.words('english')


# feature extractor function
def bag_of_words(words):
    words_clean = []

    for word in words:
        word = word.lower()
        if word not in stopwords_english and word not in string.punctuation:
            words_clean.append(word)

    words_dictionary = dict([word, True] for word in words_clean)

    return words_dictionary


def format_sentence(sent):
    return({word: True for word in nltk.word_tokenize(sent)})


def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features


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


# Train Naive Bayes classifier
featuresets = [(document_features(d), c) for (d, c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
classifier = NaiveBayesClassifier.train(train_set)

print("Naive Bayes Accuracy")
print(accuracy(classifier, test_set))

classifier.show_most_informative_features(5)

print("Sentiment")
print("Sentence: This was an amazing movie")
print(classifier.classify(bag_of_words(
    format_sentence("This was an amazing movie"))))

print("Sentence: This was a worse movie")
print(classifier.classify(bag_of_words(
    format_sentence("This was a worse movie"))))

print("Sentence: This was not an good movie")
print(classifier.classify(bag_of_words(
    format_sentence("This was not an good movie"))))

test_text = "I love this movie, very interesting"
test_sentence = "Sound is annoying"

print("Naive Bayes, Sentiment Analysis with a sentence: %s" % test_sentence)
print(classifier.classify(document_features(test_sentence.split())))

test_set = document_features(test_text.split())
print("Sentence: %s" % test_text)
print("Naive Bayes")
print(classifier.classify(test_set))

# Maxent Classifier
maxent_classifier = MaxentClassifier.train(train_set, "megam")
print("Maxent")
print(maxent_classifier.classify(test_set))

print("Maxent Classifier, Sentiment Analysis with a sentence: %s" % test_sentence)
print(maxent_classifier.classify(document_features(test_sentence.split())))

print("Maxent Informative Feature")
maxent_classifier.show_most_informative_features(10)
