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