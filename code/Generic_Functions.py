import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer
import pandas as pd


def clean_comment(comment):

    comment = re.sub(r'(\w)\1{2,}', r'\1', comment)

    comment = re.sub(r"what's", "what is ", comment)

    comment = re.sub(r"\'ve", " have ", comment)

    comment = re.sub(r"can't", "cannot ", comment)

    comment = re.sub(r"n't", " not ", comment)

    comment = re.sub(r"i'm", "i am ", comment)

    comment = re.sub(r"\'re", " are ", comment)

    comment = re.sub(r"\'d", " would ", comment)

    comment = re.sub(r"\'ll", " will ", comment)

    comment = re.sub('\W', ' ', comment)

    comment = re.sub('\s+', ' ', comment)

    comment = comment.strip(' ')

    return comment


def preprocess_comments(data):

    stop_words = set(stopwords.words('english'))

    stemmer = SnowballStemmer("english")

    documents = [" ".join([stemmer.stem(word) for word in clean_comment(comment).split(" ")
                           if word not in stop_words if word.isalpha()]) for comment in data]

    return documents


def split_data(x, y, test_size=0.30, random_state=2, **kwargs):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state, **kwargs)

    return x_train, x_test, y_train, y_test


def generate_features(x_train, ngram_range=(1, 1), stop_words='english', strip_accents='unicode',
                      sublinear_tf=True, max_features=5000, **kwargs):

    tf_idf_vec = TfidfVectorizer(ngram_range=ngram_range, stop_words=stop_words, strip_accents=strip_accents,
                                 sublinear_tf=sublinear_tf, max_features=max_features, **kwargs)

    tf_idf_vec.fit(x_train)

    train_tf_idf = tf_idf_vec.transform(x_train)

    return train_tf_idf, tf_idf_vec, tf_idf_vec.get_feature_names()


def model_fitting_and_get_training_accuracy(base_model, model, train_tf_idf, y_train, **kwargs):

    classifier = model(base_model(**kwargs))

    classifier.fit(train_tf_idf, y_train)

    predictions_train = classifier.predict(train_tf_idf)

    predictions_train = pd.DataFrame(predictions_train.toarray())

    predictions_train_prob = classifier.predict_proba(train_tf_idf)

    predictions_train_prob = pd.DataFrame(predictions_train_prob.toarray())

    accuracy_train = accuracy_score(predictions_train, y_train)

    accuracy_train_label = [accuracy_score(predictions_train.iloc[:, i], y_train.iloc[:, i]) for i in range(0, 6)]

    roc_curve_train = [roc_curve(y_train.iloc[:, i], predictions_train_prob.iloc[:, i]) for i in range(0, 6)]

    fpr_train = [fpr[0] for fpr in roc_curve_train]

    tpr_train = [tpr[1] for tpr in roc_curve_train]

    auc_thresholds_train = [auc_thresholds[2] for auc_thresholds in roc_curve_train]

    roc_score_train = [roc_auc_score(y_train.iloc[:, i], predictions_train_prob.iloc[:, i]) for i in range(0, 6)]

    return predictions_train, predictions_train_prob, accuracy_train, classifier, \
        fpr_train, tpr_train, auc_thresholds_train, roc_score_train, accuracy_train_label


def get_test_accuracy(tf_idf_vec, x_test, classifier, y_test):

    test_tf_idf = tf_idf_vec.transform(x_test)

    predictions_test = classifier.predict(test_tf_idf)

    predictions_test = pd.DataFrame(predictions_test.toarray())

    predictions_test_prob = classifier.predict_proba(test_tf_idf)

    predictions_test_prob = pd.DataFrame(predictions_test_prob.toarray())

    accuracy_test = accuracy_score(predictions_test, y_test)

    accuracy_test_label = [accuracy_score(predictions_test.iloc[:, i], y_test.iloc[:, i]) for i in range(0, 6)]

    roc_curve_test = [roc_curve(y_test.iloc[:, i], predictions_test_prob.iloc[:, i]) for i in range(0, 6)]

    fpr_test = [fpr[0] for fpr in roc_curve_test]

    tpr_test = [tpr[1] for tpr in roc_curve_test]

    auc_thresholds_test = [auc_thresholds[2] for auc_thresholds in roc_curve_test]

    roc_score_test = [roc_auc_score(y_test.iloc[:, i], predictions_test_prob.iloc[:, i]) for i in range(0, 6)]

    return predictions_test, predictions_test_prob, accuracy_test, fpr_test, tpr_test, auc_thresholds_test, \
        roc_score_test, accuracy_test_label
