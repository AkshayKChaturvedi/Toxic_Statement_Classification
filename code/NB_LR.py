import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import re
import string
from Generic_Functions import preprocess_comments, split_data, generate_features
import numpy as np

data = pd.read_csv('C:/Users/Dell/Desktop/train.csv')

# Pre-processing of comments i.e removing stopwords, numbers and stemming of words
clean_comments = preprocess_comments(data['comment_text'])

data['comment_text'] = clean_comments

# Splitting data for training and testing
x_train, x_test, y_train, y_test = split_data(data['comment_text'], data.iloc[:, 2:])

# Link to Jeremy's code:- https://www.kaggle.com/jhoward/nb-svm-strong-linear-baseline

# -------------------------------------Jeremy's Pre-processing and Functions-------------------------------------------

re_tok = re.compile('([' + string.punctuation + '“”¨«»®´·º½¾¿¡§£₤‘’])')


def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()


train_tf_idf, tf_idf_vec, feature_names = generate_features(x_train=x_train, ngram_range=(1, 2), tokenizer=tokenize,
                                                            min_df=3, strip_accents='unicode', use_idf=1,
                                                            smooth_idf=1, sublinear_tf=1)


def pr(y_i, y):
    p = train_tf_idf[y == y_i].sum(0)
    return (p+1) / ((y == y_i).sum()+1)


def get_mdl(y):
    y = y.values
    r = np.log(pr(1, y) / pr(0, y))
    m = LogisticRegression(C=1, dual=True)
    x_nb = train_tf_idf.multiply(r)
    return m.fit(x_nb, y), r


# ------------------------------------------------------End of Jeremy's Function---------------------------------------

test_tf_idf = tf_idf_vec.transform(x_test)

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

preds = np.zeros((len(x_test), len(label_cols)))

predictions_test = np.zeros((len(x_test), len(label_cols)))

accuracy_label = np.zeros(len(label_cols))

roc_score_test = np.zeros(len(label_cols))

for i, j in enumerate(label_cols):
    print('fit', j)
    m, r = get_mdl(y_train[j])
    preds[:, i] = m.predict_proba(test_tf_idf.multiply(r))[:, 1]
    predictions_test[:, i] = m.predict(test_tf_idf)
    accuracy_label[i] = accuracy_score(predictions_test[:, i], y_test.iloc[:, i])
    roc_score_test[i] = roc_auc_score(y_test.iloc[:, i], preds[:, i])

print('accuracy_label : ', accuracy_label)
accuracy_test = accuracy_score(predictions_test, y_test)
print('Total accuracy : ', accuracy_test)
print('ROC_AUC : ', roc_score_test)
