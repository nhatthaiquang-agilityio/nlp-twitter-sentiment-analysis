# nlp-twitter-sentiment-analysis


### Run
```
sen = SentimentTwitterClassifier()
sen.train_naivebayes_classifier()
sen.show_info_features()
sen.show_accuracy()

sen.classify_sentence("This was an amazing movie")

sen.classify_sentence("This was an worse movie")

sen.classify_sentence("This was not an good movie")
```

### Reference
[Analyzing Sentiment with Python and nltk](https://www.twilio.com/blog/2017/09/sentiment-analysis-python-messy-data-nltk.html)

[Twitter Sentiment](https://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/)

[NLTK Classify](https://github.com/EdmundMartin/nltk_classify)

