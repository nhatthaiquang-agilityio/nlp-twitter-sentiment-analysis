# nlp-twitter-sentiment-analysis
+ Naive Bayes Classifier
+ Maxent Classifier(Megam, GIS)

### Required
+ [Install Megam](https://textminingonline.com/dive-into-nltk-part-viii-using-external-maximum-entropy-modeling-libraries-for-text-classification)


### Overview
![Overview](./overview.png)

### Processing Steps
+ Preparation data
+ Feature Extraction(word feature and ngram)
+ Training Classifier
+ Classification
+ Accuracy

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
[NLTK Classify](http://www.nltk.org/howto/classify.html)

[Analyzing Sentiment with Python and nltk](https://www.twilio.com/blog/2017/09/sentiment-analysis-python-messy-data-nltk.html)

[Twitter Sentiment](https://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/)

[NLTK Classify Example](https://github.com/EdmundMartin/nltk_classify)

[Text Classification to Sentiment Analysis](https://textminingonline.com/dive-into-nltk-part-ix-from-text-classification-to-sentiment-analysis)

[External Maximum Entropy Modeling for Text Classification](https://textminingonline.com/dive-into-nltk-part-viii-using-external-maximum-entropy-modeling-libraries-for-text-classification)