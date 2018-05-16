import pandas as pd
from sklearn.linear_model import LogisticRegression
from Generic_Functions import preprocess_comments, split_data, generate_features, \
    model_fitting_and_get_training_accuracy, get_test_accuracy
from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain, LabelPowerset

data = pd.read_csv('C:/Users/Dell/Desktop/train.csv')

# Pre-processing of comments i.e removing stopwords, numbers and stemming of words
clean_comments = preprocess_comments(data['comment_text'])

data['comment_text'] = clean_comments

# Splitting data for training and testing
x_train, x_test, y_train, y_test = split_data(data['comment_text'], data.iloc[:, 2:])

# Generating TF-IDF features
train_tf_idf, tf_idf_vec, feature_names = generate_features(x_train)

# --------------------------------------------Binary Relevance-----------------------------------------------------

# Fitting a Logistic Regression model with Binary Relevance problem transformation method and get training accuracy
br_predictions_train, br_predictions_train_prob, br_accuracy_train, br_classifier, \
    br_fpr_train, br_tpr_train, br_auc_thresholds_train, br_roc_score_train, br_accuracy_train_label = \
    model_fitting_and_get_training_accuracy(LogisticRegression, BinaryRelevance, train_tf_idf, y_train)

print('Binary Relevance Starts')

print('Accuracy on Training Data of Logistic Regression with Binary Relevance: '
      '{br_accuracy_train}'.format(br_accuracy_train=br_accuracy_train))

print('Accuracy for each label on Training Data of Logistic Regression with Binary Relevance: '
      '{br_accuracy_train_label}'.format(br_accuracy_train_label=br_accuracy_train_label))

# Applying the fitted model above to the test data and get model accuracy on the test data
br_predictions_test, br_predictions_test_prob, br_accuracy_test, br_fpr_test, br_tpr_test, br_auc_thresholds_test, \
    br_roc_score_test, br_accuracy_test_label = get_test_accuracy(tf_idf_vec, x_test, br_classifier, y_test)

print('Accuracy on Test Data of Logistic Regression with Binary Relevance: '
      '{br_accuracy_test}'.format(br_accuracy_test=br_accuracy_test))

print('Accuracy for each label on Test Data of Logistic Regression with Binary Relevance: '
      '{br_accuracy_test_label}'.format(br_accuracy_test_label=br_accuracy_test_label))

print('Binary Relevance Ends')

# -------------------------------------------------------------------------------------------------------------------
# --------------------------------------------Classifier Chain-------------------------------------------------------

# Fitting a Logistic Regression model with Classifier Chain problem transformation method and get training accuracy
cc_predictions_train, cc_predictions_train_prob, cc_accuracy_train, cc_classifier, \
    cc_fpr_train, cc_tpr_train, cc_auc_thresholds_train, cc_roc_score_train, cc_accuracy_train_label = \
    model_fitting_and_get_training_accuracy(LogisticRegression, ClassifierChain, train_tf_idf, y_train)

print('Classifier Chain Starts')

print('Accuracy on Training Data of Logistic Regression with Classifier Chain: '
      '{cc_accuracy_train}'.format(cc_accuracy_train=cc_accuracy_train))

print('Accuracy for each label on Training Data of Logistic Regression with Classifier Chain: '
      '{cc_accuracy_train_label}'.format(cc_accuracy_train_label=cc_accuracy_train_label))

# Applying the fitted model above to the test data and get model accuracy on the test data
cc_predictions_test, cc_predictions_test_prob, cc_accuracy_test, cc_fpr_test, cc_tpr_test, cc_auc_thresholds_test, \
    cc_roc_score_test, cc_accuracy_test_label = get_test_accuracy(tf_idf_vec, x_test, cc_classifier, y_test)

print('Accuracy on Test Data of Logistic Regression with Classifier Chain: '
      '{cc_accuracy_test}'.format(cc_accuracy_test=cc_accuracy_test))

print('Accuracy for each label on Test Data of Logistic Regression with Classifier Chain: '
      '{cc_accuracy_test_label}'.format(cc_accuracy_test_label=cc_accuracy_test_label))

print('Classifier Chain Ends')

# -------------------------------------------------------------------------------------------------------------------
# --------------------------------------------Label Power Set---------------------------------------------------------

# Fitting a Logistic Regression model with Label Power Set problem transformation method and get training accuracy
lp_predictions_train, lp_predictions_train_prob, lp_accuracy_train, lp_classifier, \
    lp_fpr_train, lp_tpr_train, lp_auc_thresholds_train, lp_roc_score_train, lp_accuracy_train_label = \
    model_fitting_and_get_training_accuracy(LogisticRegression, LabelPowerset, train_tf_idf, y_train)

print('Label Power Set Starts')

print('Accuracy on Training Data of Logistic Regression with Label Power Set: '
      '{lp_accuracy_train}'.format(lp_accuracy_train=lp_accuracy_train))

print('Accuracy for each label on Training Data of Logistic Regression with Label Power Set: '
      '{lp_accuracy_train_label}'.format(lp_accuracy_train_label=lp_accuracy_train_label))

# Applying the fitted model above to the test data and get model accuracy on the test data
lp_predictions_test, lp_predictions_test_prob, lp_accuracy_test, lp_fpr_test, lp_tpr_test, lp_auc_thresholds_test, \
    lp_roc_score_test, lp_accuracy_test_label = get_test_accuracy(tf_idf_vec, x_test, lp_classifier, y_test)

print('Accuracy on Test Data of Logistic Regression with Label Power Set: '
      '{lp_accuracy_test}'.format(lp_accuracy_test=lp_accuracy_test))

print('Accuracy for each label on Test Data of Logistic Regression with Label Power Set: '
      '{lp_accuracy_test_label}'.format(lp_accuracy_test_label=lp_accuracy_test_label))

print('Label Power Set Ends')
