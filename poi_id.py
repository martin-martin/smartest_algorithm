# IMPORTANT FOR REVIEWER:
# for the full description of my work: https://github.com/martin-martin/smartest_algorithm/blob/master/Enron_report.md
# for the full code in .ipynb: https://github.com/martin-martin/smartest_algorithm/blob/master/ml_enron.ipynb

#!/usr/bin/python

##########################
###### PREPARATIONS ######
##########################

# importing libraries
import sys
import pickle
import pprint
import numpy as np
import pandas as pd
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

from matplotlib import pyplot as plt

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# preparing a pandas DF for easier access
enron_df = pd.DataFrame(data_dict)
# better suited in the other format
enron_df = enron_df.transpose()

# removing the email-address column because it has no predicitve power
enron_df = enron_df.drop('email_address', 1)

# setting the initial features list
features_list = list(enron_df.columns.values)

# quick statistical overview
nr_poi = len(enron_df[enron_df.poi == 1])
nr_non_poi = len(enron_df[enron_df.poi == 0])
print "total number of data points:", len(enron_df)
print "of which {0} are POI and {1} are non-POI.".format(nr_poi, nr_non_poi)


#############################
###### OUTLIER REMOVAL ######
#############################

# quick EDA plot shows a possible outlier
plt.scatter(enron_df['bonus'], enron_df['total_payments'])

# 'NaN' are encoded as string instead of nan - I change this to work with them easier
enron_df.replace(['NaN'], [None], inplace=True)

outlier = enron_df['bonus'].max()
#print enron_df.loc[enron_df['bonus'] == outlier]
# this shows that the entry is TOTAL, and can be removed
enron_df = enron_df[enron_df.bonus != outlier]

# quick plot displays the new distribution after removal
plt.scatter(enron_df['bonus'], enron_df['total_payments'])

# repeating the steps looking for other outliers
outlier = enron_df.total_payments.max()
#print enron_df.total_payments.loc[enron_df.total_payments == outlier]
# shows that this outlier is valid and should remain in the dataset

# looking for other outliers
# (example plot; this was initially performed for more columns)
plt.scatter(enron_df.shared_receipt_with_poi, enron_df.total_payments)

def display_outlier(column_name):
    """prints the max value in a df column of the name passed as arg."""
    outlier = enron_df[column_name].max()
    print enron_df.loc[enron_df[column_name] == outlier]

#display_outlier('shared_receipt_with_poi')
# this shows that the datapoint is a valid person
# in this .py file not all outlier investigations are displayed

# shows that this manually found datapoint is not a person, therefore I remove it
#print enron_df.loc["THE TRAVEL AGENCY IN THE PARK"]
agency = "THE TRAVEL AGENCY IN THE PARK"
enron_df = enron_df.drop([agency])


#########################
###### NEW FEATURE ######
#########################

def email_perc(row):
    """calculates the ratio of emails sent to a POI vs. emails received from a POI.

    takes as input one row of a dataframe
    returns the ratio as a floating point number.
    """
    one_way = row['from_this_person_to_poi']
    the_other = row['from_poi_to_this_person']

    ratio = 0
    # calculating the ratio only if there is no 0 value involved
    if one_way != 0 and the_other != 0:
        ratio = float(one_way) / the_other
    return ratio

# generating the feature of interest
enron_df["sent_received_ratio"] = enron_df.apply(lambda row: email_perc(row), axis=1)

# visualizing the new feature, while accounting for the POI and non-POI in the dataset
# POI are displayed as red circles, non-POI as blue circles
plt.scatter(enron_df.bonus, enron_df.sent_received_ratio, c=enron_df.poi)

# updating the features list with the new feature
# and moving 'poi' to the 0th place
features_list.append("sent_received_ratio")
features_list = features_list[12:] + features_list[:12]


##########################
###### OVERSAMPLING ######
##########################

# the ratio that non-POI are more frequent in the dataset than POI
ratio_poi = nr_non_poi / nr_poi
print "non-POI are {0} times more frequent than POI".format(ratio_poi)

print 'initial amount of rows:', len(enron_df)
# gleaning the underrepresented sample
is_poi = enron_df['poi'] == 1
oversample_row = enron_df[is_poi]
# adding copies of the POI to the dataset (=oversampling)
# to reduce the imbalance of the dataset
# I chose to double the POI instances
enron_df = enron_df.append([oversample_row] * 2, ignore_index=True)
print 'oversampled amount of rows:', len(enron_df)

# shuffle the rows so that the cloned POIs are evenly spread throughout the
# dataset. this is important for the split of data in training and testing data
enron_df_os = enron_df.reindex(np.random.permutation(enron_df.index))
# uncommenting shows that the final entries are not all POI after shuffling
#enron_df_os.tail()


############################
###### MORE WRANGLING ######
############################

# the classifiers cannot handle NaN values, therefore they need to be either
# removed or imputed. due to the small size of the dataset I chose
# to impute the median values to the NaN
enron_median = enron_df.fillna(enron_df.median().to_dict())

# translating back to a dict for easier work with the lesson code
my_dataset = enron_median.to_dict(orient='index')


###############################
###### FEATURE SELECTION ######
###############################

# I'll be using an overfit Decision Tree to calculate the feature importances
def get_most_important_features(dataset, features_list):
    """Calculates the feature importances.

    Takes as input a dataset and a list of features.
    Creates an overfit Decision Tree and calculates the feature importances.
    Returns a list with the feature importances.
    """
    # creating an overfitted decision tree
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    data = featureFormat(dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    # new features filtered, NaN values removed
    features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                                    labels,
                                                                                    test_size=0.3,
                                                                                    random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    acc = accuracy_score(labels_test, pred)
    # uncomment to print the accuracy score
    #print "overfitted accuracy", acc

    # calculating feature importances
    feat_imp = clf.feature_importances_
    # uncomment to print the most important (common) ones
    #print feat_imp
    #for index, feature in enumerate(feat_imp):
    #    if feature > 0.2:
    #        print "spot:", index, ":", features_list[index+1], " | value:", feature
    return feat_imp

# collecting the values to calculate the average feature importances
feat_imp_list_list = []
for i in range(100):
    feat_imp = get_most_important_features(my_dataset, features_list)
    feat_imp_list_list.append(feat_imp)

# creating a features list while excluding 'poi'
feature_compare = features_list[1:]
# setting up a dict with an empty list for each feature
feat_imp_dict = {}
for f in feature_compare:
    feat_imp_dict[f] = []
# adding all 100 feature importance values to the respective list
for array in feat_imp_list_list:
    for index, number in enumerate(array):
        feat_imp_dict[feature_compare[index]].append(number)

top_feat_imp = {}
# I chose a threshold of 0.1, because when higher, only one feature was selected
threshold = 0.1
# calc. the mean of all runs and store the ones passing the threshold in a dict
for key, value in feat_imp_dict.items():
    mean = np.asarray(value).mean()
    print key, ":", mean
    if mean > threshold:
        top_feat_imp[key] = mean
#print top_feat_imp

# updating the features list with the top performers (those I will actually use)
# adding 'poi' back at the 0th spot
features_list = ['poi'] + top_feat_imp.keys()


#################################
###### TESTING CLASSIFIERS ######
#################################

# preparing the data into training and testing sets
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
# features after imputing median values
features_train, features_test, labels_train, labels_test = train_test_split(features,
                                                                            labels,
                                                                            test_size=0.3,
                                                                            random_state=42)
# combining the training and testing variables into a list for easier input for the function I wrote
full_feat_list = [features_train, features_test, labels_train, labels_test]

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

    ### uncomment for information about time performance and accuracy ###
    ## taking the time the algorithm runs
    #t0 = time()
    classifier_obj.fit(features_train, labels_train)
    #print "training time:", round(time()-t0, 3), "s"

    ## taking time for prediction
    #t1 = time()
    pred = classifier_obj.predict(features_test)
    #print "predicting time:", round(time()-t1, 3), "s"

    #acc = accuracy_score(labels_test, pred)
    #print "accuracy:", acc

    print construct_CM(labels_test, pred)
    print calculate_f1(labels_test, pred)

def get_CM_nums(true_labels, predictions, CM_type):
    """Calculates the number of elements in the different cells of a confusion matrix.

    Takes as input the true labels and the predictions, both as lists.
    Further a specification of the metric wanted as an abbreviated string:
    'TP' for True Positives, 'FP' for False Positives
    'TN' for True Negatives, 'FN' for False Negatives
    Returns the amount of the specified metric.
    """
    import numpy as np

    if CM_type == "TP" or CM_type == "TN":
        if CM_type == "TP":
            CM_type = 1
        elif CM_type == "TN":
            CM_type = 0
        cpp = [1 for j in zip(true_labels, predictions) if j[0] == j[1] and j[1] == CM_type]
    elif CM_type == "FP" or CM_type == "FN":
        if CM_type == "FP":
            CM_type = 1
        elif CM_type == "FN":
            CM_type = 0
        cpp = [1 for j in zip(true_labels, predictions) if j[0] != j[1] and j[1] == CM_type]
    else:
        print "error, please enter TP, TN, FP, or FN."
    num_cpp = np.sum(cpp)
    return int(num_cpp)

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

def test_a_lot(training_test_list):
    """Wrapper function that calculates performance results for different classifiers and prints the results.

    Takes as input a list of test and training data in the following form:
    'training_test_list = [features_train, features_test, labels_train, labels_test]'
    Calls the functions test_classifier() on Naive Bayes, SVM, Decision Trees,
    and K-nearest neighbors (each with the default settings).
    Prints all results in formatted output.
    """
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier

    print "PERFORMANCE RESULTS OF DIFFERENT CLASSIFIERS"
    ### Naive Bayes
    print "\n\n### NAIVE BAYES ###"
    clf = GaussianNB()
    test_classifier(clf, training_test_list)

    ### Support Vector Machines
    print "\n\n### SUPPORT VECTOR MACHINES ###"
    clf = SVC()
    test_classifier(clf, training_test_list, scale=True)

    ### Decision Trees
    print "\n\n### DECISION TREES ###"
    clf = DecisionTreeClassifier()
    test_classifier(clf, training_test_list)

    ### K-nearest Neighbours
    print "\n\n### K-NEAREST NEIGHBORS ###"
    neigh = KNeighborsClassifier()
    test_classifier(neigh, training_test_list, scale=True)

# running the test suite
test_a_lot(full_feat_list)


################################
###### TUNING CLASSIFIERS ######
################################

# checking the best tuning setting for Decistion Trees
from sklearn.tree import DecisionTreeClassifier

parameters = {'max_depth' : [None, 10, 5, 2],
              'min_samples_split' : [2, 10, 5, 1],
              'min_samples_leaf' : [1, 5, 2],
              'min_weight_fraction_leaf' : [0, 0.25, 0.5]
             }

dt = DecisionTreeClassifier()
clf = GridSearchCV(dt, parameters)
clf.fit(features_train, labels_train)
print clf.best_params_

print "tuned DECISION TREES"
# min_samples_leaf=1 and max_depth=None are both default values
clf = DecisionTreeClassifier(min_samples_split=1, min_weight_fraction_leaf=0.5)
test_classifier(clf, full_feat_list)


# checking the best tuning setting for KNN
from sklearn.neighbors import KNeighborsClassifier

parameters = {'n_neighbors' : [5, 10, 3, 2],
              'weights' : ['uniform', 'distance'],
              'algorithm' : ['ball_tree', 'kd_tree', 'brute'],
              'leaf_size' : [30, 10, 60, 100]
             }

knn = KNeighborsClassifier()
clf = GridSearchCV(knn, parameters)
clf.fit(features_train, labels_train)
print clf.best_params_

print "tuned K-NEAREST NEIGHBORS"
# leaf_size=30 is the default value
clf = KNeighborsClassifier(n_neighbors=2, weights='distance', algorithm='ball_tree')
test_classifier(clf, full_feat_list, scale=True)


##########################################
###### FINAL CHOICE AND COMPUTATION ######
##########################################

clf = KNeighborsClassifier(n_neighbors=2, weights='distance', algorithm='ball_tree')
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)


#################################
###### OUTPUT FOR CHECKING ######
#################################

dump_classifier_and_data(clf, my_dataset, features_list)