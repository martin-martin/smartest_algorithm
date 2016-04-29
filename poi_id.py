#!/usr/bin/python

import sys
import pickle
import pprint
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'bonus', 'total_payments', 'shared_receipt_with_poi',
                'from_poi_to_this_person', 'from_this_person_to_poi' ]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

###
### 1 - SELECTING FEATURES
###


###
### Task 2: Remove outliers
###

# preparing a pandas DF for easier access
import pandas as pd
enron_df = pd.DataFrame(data_dict).transpose()

# reducing the dataframe for better overview, removing the features
# that I decided not to use
for c in enron_df.columns:
    if c not in features_list:
        enron_df.drop(c, axis=1, inplace=True)

# the feature combination where I detected one erroneous outlier
from matplotlib import pyplot as plt
plt.scatter(enron_df['bonus'], enron_df['salary'])

# transforming the 'NaN' strings to 'None' values
enron_df.replace(['NaN'], [None], inplace=True)
# removing the outlier
outlier = enron_df['salary'].max()
enron_df = enron_df[enron_df.salary != outlier]

# removing the bogus company entry
agency = "THE TRAVEL AGENCY IN THE PARK"
enron_df = enron_df.drop([agency])


###
### Task 3: Create a new feature
###

def email_perc(row):
    """Calculates the ratio of Emails sent / received with poi involved for one row."""
    emails_sent_to_poi = row['from_this_person_to_poi']
    emails_received_from_poi = row['from_poi_to_this_person']

    ratio = 0
    # removing rows that have the value 0 in either
    # to avoid "DivisionByZero" Errors
    if emails_sent_to_poi != 0 and emails_received_from_poi != 0:
        ratio = float(emails_sent_to_poi) / emails_received_from_poi
    return ratio

enron_df["sent_received_ratio"] = enron_df.apply(lambda row: email_perc(row), axis=1)
# feature is already added to features_list


###
### TRANSFORMATION
###

# dropping rows containing NaNs
enron_no_na = enron_df.dropna()
# reformat the pandas df to a dict, for further processing with the lesson code
no_na_dataset = enron_no_na.to_dict(orient='index')

# creating a dataset without NaN values (removing the rows that contain NaN)
no_na_data = featureFormat(no_na_dataset, features_list, sort_keys = True)
no_na_labels, no_na_features = targetFeatureSplit(no_na_data)

from sklearn.cross_validation import train_test_split
# features filtered, NaN values removed
no_na_features_train, no_na_features_test, no_na_labels_train, no_na_labels_test = train_test_split(no_na_features,
                                                                                                    no_na_labels,
                                                                                                    test_size=0.3,
                                                                                                    random_state=42)

# creating a list for testing:
no_na_list = [no_na_features_train, no_na_features_test, no_na_labels_train, no_na_labels_test]

# final dataset-choice extra variable for testing compatibility
my_dataset = no_na_dataset


###
### Task 4: Try a variety of classifiers
###

def get_CM_nums(true_labels, predictions, CM_type):
    """Calculates the number of elements in the different cells of a confusion matrix.

    Takes as input the true labels and the predictions, both as lists.
    Further a specification of the metric wanted as an abbreviated string:
    'TP' for True Positives, 'FP' for False Positives
    'TN' for True Negatives, 'FN' for False Negatives
    Returns the amount of the specified metric.
    """
    import numpy as np

    def error():
        print "Error: please enter 'TP', 'TN', 'FP', or 'FN'."

    def binary(CM_string):
        """Encodes Positives with '1' and Negatives with '0'."""
        if CM_string.endswith("P"):
            return 1
        elif CM_string.endswith("N"):
            return 0
        else:
            return error()

    if len(CM_type) == 2:
        CM_encode = binary(CM_type)
        if CM_type.startswith("T"):
            cpp = [1 for j in zip(true_labels, predictions) if j[0] == j[1] and j[1] == CM_encode]
        elif CM_type.startswith("F"):
            cpp = [1 for j in zip(true_labels, predictions) if j[0] != j[1] and j[1] == CM_encode]
        else:
            return error()
        num_cpp = np.sum(cpp)
        return int(num_cpp)
    else:
        return error()

def construct_CM(true_labels, predictions):
    """Wrapper function to calculate the confusion matrix and returns a formatted string.

    Takes as input the true labels and the predictions, both as lists.
    Calls get_CM_nums() with all possible inputs (TP, FP, TN, FN)
    Returns a formatted string representing the confusion martrix that is easy to read.
    """
    num_TP = get_CM_nums(true_labels, predictions, "TP")
    num_TN = get_CM_nums(true_labels, predictions, "TN")
    num_FP = get_CM_nums(true_labels, predictions, "FP")
    num_FN = get_CM_nums(true_labels, predictions, "FN")

    return """confusion matrix:
              predicted class
              _Yes_|__No_
actual | Yes |  {0}  |  {3}
class  | No  |  {2}  |  {1}""".format(num_TP, num_TN, num_FP, num_FN)

def calculate_f1(true_labels, predictions):
    """Calculates statistical metrics for the classifier's prediction.

    Takes as input the true labels and the predictions, both as lists.
    Returns a string containing the scores for precision recall and the f1-score.
    """
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1_score =  f1_score(true_labels, predictions)
    return """precision: {0}
recall:    {1}
f1_score:  {2}""".format(precision, recall, f1_score)

def test_classifier(classifier_obj, data, scale=False):
    """Measures the time and accuracy of a given classifier.

    Takes as input a classifiert object (tuned or untuned)
    and a list containing the training and testing features and labels
    in this form:
    data = [features_train, features_test, labels_train, labels_test]
    Prints processing time and overall accuracy score.
    """
    from time import time
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import MinMaxScaler

    # scaling the data when the type of algorithm demands this
    if scale == True:
        scaler = MinMaxScaler()
        features_train = scaler.fit_transform(data[0])
        features_test = scaler.transform(data[1])
    else:
        features_train = data[0]
        features_test = data[1]
    labels_train = data[2]
    labels_test = data[3]

    classifier_obj.fit(features_train, labels_train)
    pred = classifier_obj.predict(features_test)

    print construct_CM(labels_test, pred)
    print calculate_f1(labels_test, pred)


def test_a_lot(training_test_list):
    """Wrapper function that calculates performance results for different classifiers and prints the results.

    Takes as input a list of test and training data in the following form:
    'training_test_list = [features_train, features_test, labels_train, labels_test]'
    Calls the functions test_classifier() on Naive Bayes, SVM (with different settings),
    Decision Trees (with different settings), ans K-nearest neighbors (with different settings).
    Prints all results in formatted output.
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    print "PERFORMANCE RESULTS OF DIFFERENT CLASSIFIERS"

    ### Naive Bayes
    print "\n\n\n### NAIVE BAYES ###"
    clf = GaussianNB()
    test_classifier(clf, training_test_list)


    ### Support Vector Machines
    print "\n\n\n### SUPPORT VECTOR MACHINES ###"
    # 'rbf' is the default kernel used
    print "# with 'rbf' kernel"
    clf = SVC()
    test_classifier(clf, training_test_list, scale=True)
    print '\n'

    # 'sigmoid'
    print "# with 'sigmoid' kernel"
    clf = SVC(kernel="sigmoid")
    test_classifier(clf, training_test_list, scale=True)


    ### Decision Trees
    print "\n\n\n### DECISION TREES ###"
    # 'max_features' default is None = n_features
    print "# using all features"
    clf = DecisionTreeClassifier()
    test_classifier(clf, training_test_list)
    print '\n'

    # 'sqrt'
    print "# using the square root of the features"
    clf = DecisionTreeClassifier(max_features="sqrt")
    test_classifier(clf, training_test_list)
    print '\n'

    # 'log2'
    print "# using the log2 of the features"
    clf = DecisionTreeClassifier(max_features="log2")
    test_classifier(clf, training_test_list)
    print '\n'

    # running the Decision Tree for all possible feature amounts
    for i in range(1,6):
        print "# using {0} feature(s)".format(i)
        clf = DecisionTreeClassifier(max_features=i)
        test_classifier(clf, training_test_list)
        print '\n'


    ### K-nearest Neighbours
    print "\n\n\n### K-NEAREST NEIGHBORS ###"
    # 'n_neighbors' default is 5
    print "# with k = 6 (all features)"
    neigh = KNeighborsClassifier(n_neighbors=6)
    test_classifier(neigh, training_test_list, scale=True)
    print '\n'

    # 'n_neighbors' default is 5
    print "# with k = 5"
    neigh = KNeighborsClassifier()
    test_classifier(neigh, training_test_list, scale=True)
    print '\n'

    # running kNN for some different amounts of neighbors
    for i in range(1,5):
        print "# with k =", i
        neigh = KNeighborsClassifier(n_neighbors=i)
        test_classifier(neigh, training_test_list, scale=True)
        print '\n'

# testing a variety of classifiers
test_a_lot(no_na_list)

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

def test_feature_combinations(features_list):
    from feature_format import featureFormat, targetFeatureSplit
    from sklearn.cross_validation import train_test_split
    from sklearn.tree import DecisionTreeClassifier

    ### Extract features and labels from datasets for local testing
    no_na_data = featureFormat(no_na_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(no_na_data)

    # new features filtered, NaN values removed
    features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                                labels,
                                                                                test_size=0.3,
                                                                                random_state=42)

    clf = DecisionTreeClassifier()

    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)

    print construct_CM(labels_test, pred)
    print calculate_f1(labels_test, pred)

    return clf, pred


features_list = ['poi',
            'shared_receipt_with_poi',
            'total_payments',
            'from_this_person_to_poi',
            'sent_received_ratio',
            'bonus']

clf, pred = test_feature_combinations(features_list)

# statistics
poi_count = np.sum(no_na_labels)

print "POIs", poi_count
print len(pred)
print no_na_labels_test

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

def get_most_important_features():
    # creating the overfitted tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    no_na_data = featureFormat(no_na_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(no_na_data)

    # new features filtered, NaN values removed
    features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                                    labels,
                                                                                    test_size=0.3,
                                                                                    random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc = accuracy_score(labels_test, pred)
    print "overfitted accuracy", acc

    # calculating feature importances
    feat_imp = clf.feature_importances_
    # print the most important (common) ones
    print feat_imp
    for index, feature in enumerate(feat_imp):
        if feature > 0.2:
            print "spot:", index, ":", features_list[index+1], " | value:", feature

# running the function 10 times to see which features prevail
for i in range(10):
    get_most_important_features()
    print "\n"

# truncated features_list gleaned from the results
features_list = ['poi',
            'shared_receipt_with_poi',
            'total_payments',
            'from_this_person_to_poi',
            'bonus',
            'sent_received_ratio']

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)