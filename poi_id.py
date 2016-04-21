#!/usr/bin/python

import sys
import pickle
import pprint
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

# a dict with the basic structure of the data, to keep count of occurences
count_dict = {'bonus': 0,
 'deferral_payments': 0,
 'deferred_income': 0,
 'director_fees': 0,
 'email_address': 0,
 'exercised_stock_options': 0,
 'expenses': 0,
 'from_messages': 0,
 'from_poi_to_this_person': 0,
 'from_this_person_to_poi': 0,
 'loan_advances': 0,
 'long_term_incentive': 0,
 'other': 0,
 'poi': 0,
 'restricted_stock': 0,
 'restricted_stock_deferred': 0,
 'salary': 0,
 'shared_receipt_with_poi': 0,
 'to_messages': 0,
 'total_payments': 0,
 'total_stock_value': 0}
# to see which features are present for many people, i count
for person, p_dict in data_dict.items():
    for key, value in p_dict.items():
        if value != "NaN":
            count_dict[key] += 1
# for better comparison i will use the percentage of total
amount_people = len(data_dict)
perc_dict = {}
for key, value in count_dict.items():
    perc_dict[key] = round((value * 100.0) / amount_people, 2)
#pprint.pprint(perc_dict)
# filtering for the more common features, with a treshhold of 50%
common_feature_list = []
for key, value in perc_dict.items():
    if value >= 50:
        common_feature_list.append(key)

# removing features I don't want to investigate (for justification
# see the ML_report_Enron file)
remove_list = ['to_messages', 'exercised_stock_options', 'email_address',
            'total_stock_value', 'expenses', 'from_messages', 'other', 'restricted_stock']
features_list = [feature for index, feature in enumerate(common_feature_list) if feature not in remove_list]
# ordering 'poi' to index = 0
features_list = features_list[4:] + features_list[:4]
#pprint.pprint(features_list)
"""
MY FEATURES:
['poi',
 'shared_receipt_with_poi',
 'from_poi_to_this_person',
 'salary',
 'total_payments',
 'bonus',
 'from_this_person_to_poi']
 """

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
# adding feature to features_list
features_list.append("sent_received_ratio")


###
### TRANSFORMATION
###

# two adapted datasets accounting for NaNs
enron_no_na = enron_df.dropna()
enron_median = enron_df.fillna(enron_df.median())
# reformat the pandas df to a dict, for further processing with the lesson code
no_na_dataset = enron_no_na.to_dict(orient='index')
median_dataset = enron_median.to_dict(orient='index')


# creating a dataset without NaN values (removing the rows that contain NaN)
no_na_data = featureFormat(no_na_dataset, features_list, sort_keys = True)
no_na_labels, no_na_features = targetFeatureSplit(no_na_data)

# creating a dataset with NaN replaced by the median
median_data = featureFormat(median_dataset, features_list, sort_keys = True)
median_labels, median_features = targetFeatureSplit(median_data)

from sklearn.cross_validation import train_test_split
# features filtered, NaN values removed
no_na_features_train, no_na_features_test, no_na_labels_train, no_na_labels_test = train_test_split(no_na_features, no_na_labels, test_size=0.3, random_state=42)
# features filtered, NaN replaced with the median
median_features_train, median_features_test, median_labels_train, median_labels_test = train_test_split(median_features, median_labels, test_size=0.3, random_state=42)

# creating lists:
no_na_list = [no_na_features_train, no_na_features_test, no_na_labels_train, no_na_labels_test]
median_list = [median_features_train, median_features_test, median_labels_train, median_labels_test]

# final dataset-choice extra variable for testing compatibility
my_dataset = median_dataset


###
### Task 4: Try a variety of classifiers
###

def test_classifier(classifier_obj, data):
    """Measures the time and accuracy of a given classifier.

    Takes as input a classifiert object (tuned or untuned)
    and a list containing the training and testing features and labels
    in this form:
    data = [features_train, features_test, labels_train, labels_test]
    Prints processing time and overall accuracy score.
    """
    from time import time
    from sklearn.metrics import accuracy_score

    features_train = data[0]
    features_test = data[1]
    labels_train = data[2]
    labels_test = data[3]

    # taking the time the algorithm runs
    t0 = time()
    classifier_obj.fit(features_train, labels_train)
    print "training time:", round(time()-t0, 3), "s"

    # taking time for prediction
    t1 = time()
    pred = classifier_obj.predict(features_test)
    print "predicting time:", round(time()-t1, 3), "s"

    acc = accuracy_score(labels_test, pred)
    print "accuracy:", acc

def test_a_lot(training_test_list):
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
    test_classifier(clf, training_test_list)
    print '\n'

    # 'sigmoid'
    print "# with 'sigmoid' kernel"
    clf = SVC(kernel="sigmoid")
    test_classifier(clf, training_test_list)


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

    # 1
    print "# using 1 feature"
    clf = DecisionTreeClassifier(max_features=1)
    test_classifier(clf, training_test_list)


    ### K-nearest Neighbours
    print "\n\n\n### K-NEAREST NEIGHBORS ###"
    # 'n_neighbors' defaUlt is 5
    print "# with k = 5"
    neigh = KNeighborsClassifier()
    test_classifier(neigh, training_test_list)
    print '\n'

    # 1
    print "# with k = 1"
    neigh = KNeighborsClassifier(n_neighbors=1)
    test_classifier(neigh, training_test_list)

# calling the wrapper function to measure the different accuracies
test_a_lot(no_na_list)
test_a_lot(median_list)

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.svm import SVC
clf = SVC()
clf.fit(median_features_train, median_labels_train)
pred = clf.predict(median_features_test)

# statistics
poi_count = 0
for p in pred:
    if p == 1.:
        poi_count += 1

print poi_count
print len(pred)
print median_labels_test

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# # Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)