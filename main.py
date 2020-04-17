from nltk.classify import NaiveBayesClassifier

from textblob import TextBlob


def word_feats(words):
    return dict([(word, True) for word in words])
    
positive_vocab = ['awesome','outstanding','fantastic','terrific','good','nice','great']
negative_vocab = ['bad','terrible','useless','hate']

positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]

train_set = negative_features + positive_features
classifier = NaiveBayesClassifier.train(train_set)

neg = 0
pos = 0
sentence = input("Enter the moview review : ")
words = sentence.lower().split(' ')
classResult = 0
for word in words:
    classResult += TextBlob(word).sentiment.polarity
if classResult <0:
    neg = neg + 1
if classResult >0:
    pos = pos + 1
print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))
print('Overall Sentiment is : ' , (float(pos)/len(words)) - (float(neg)/len(words)))
