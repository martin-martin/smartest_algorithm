# Looking for the _smartest algorithm in the room_

Machine Learning practice with the **Enron** Email Corpus and Financial Dataset.

---

Katie's enthusiasm is infectious, so I went ahead to learn a bit about the topic of the Enron fraud at large. Watching the documentary _Enron: The Smartest Guys in the Room_ allowed me to learn quite some new things about that ugly side of the American Dream.

Here's to some more involvement through a different path.

## Introduction
Using Machine Learning techniques, I will try to analyze the two combined datasets to try to predict who of the people that are part of the dataset might be persons of interest (POI).
I define a POI to be someone who might be worth further investigation, because he or she might be more closely involved in the Enron fraud case than other people who worked at the company at that time.

One feature that these predictions will be based on, is the label `poi`, that marks people as `1` (= POI) or `0` (non-POI). This feature was created by Katie Malone when she manually gathered data online about the fraud case. (Source: http://usatoday30.usatoday.com/money/industries/energy/2005-12-28-enron-participants_x.htm)
`poi` will allow me to train a classifier to recognize patterns that POIs might share, and therefore represents a key feature for the analysis. Of the **146** data points available in the initial dataset, **18** are labeled as POI. This represents a highly imbalanced dataset, which I will be addressing later on.

## Outlier Removal

When analyzing my dataset for outliers, I identified one row called `TOTAL`. This was most probably simply the total of all the money a person received or spent. This should therefore be removed from the dataset.

I had some trouble when trying to find the `max()` value of the columns I had plotted. My code only returned `NaN` as the max. After a while I figured that this is because the `'NaN'` values were actually **strings**, not `None` type. So I went ahead to transform them into actual `NaN`, to finally find and remove the row containing the outlier.

Further, I also removed by hand a row called `THE TRAVEL AGENCY IN THE PARK`, that was a bogus firm that some of the people involved in the fraud were using for booking travel costs. I've found out about this data point through inspecting the `enron61702insiderpay.pdf` file manually.

## Overfitting an imbalanced dataset
As gleaned from the investigation above, the dataset is highly imbalanced. There are initially only 18 POI and after outlier removal 126 non-POI. This comes to a ratio of 1:7 in favor of non-POI.

Because it is difficult to gain valuable results from such an imbalanced dataset, the underrepresented sample needs to be **overfitted**. To this end I duplicated the POI rows and shuffled the dataset, so that there would not be a problem when later splitting the data into training and testing samples.

## Features

Before diving into further analysis, I removed the column `email_address`. It contained unique strings for each person and therefore won't have any predictive power.

### A new feature
Wondering about the **ratio** of emails sent to a poi vs. received from a poi, as mentioned in the course, made me create this new feature: `sent_received_ratio`.
The idea behind this feature was, that POIs might be more likely to send a higher amount fo emails among each other than towards non-POIs.

### Selecting features
Then, for choosing the features for the analysis, I used an overfit Decision Tree from which I calculated the feature importances.

I ran the calculation 100 times and collected all the importance scores, then I took the mean importance for each feature.
I filtered my results with a treshold of 0.1 to come up with the following result:

```
{'from_poi_to_this_person': 0.11607985097841694,
 'other': 0.14604877019076973,
 'restricted_stock': 0.37501469730149045}
 ```

Therefore my **final set of 4 features** was:

```
['poi', 'from_poi_to_this_person', 'other', 'restricted_stock']
```

The **new feature** I had generated earlier fared not too well in the feature importance test, yielding with 0.0160556220096 not much predictive power over 100 runs. Therefore I did not include it in the final analysis.


## Choice of Algorithm
To test the performance of various classifiers, I wrote a test suite called `test_a_lot()`. The wrapper function ran tests for **Naive Bayes**, **SVM**s, **Decision Trees** and **K-nearest Neighbors**.

The test suite employs **feature scaling** for SVM and K-NN. This is necessary because both rely on a measure of distance, which gets distorted if the different features used are not re-adapted on a scale from 0 to 1.
Further the test suite calculates and prints the confusion matrix as well as precision, recall, and the F1-score for each classifier at its default settings.

I used this approach to get a quick overview on the results the different classifiers yield, before deciding which ones to further invest into tuning for even better results.

Some of the classifiers I employed returned very low scores for precision and recall. Only Decision Trees and K-Nearest-Neighbors yielded results that were above 0.3, while SVM and Naive Bayes did not return useful predictions.

The two classifiers that yielded better F1-scores were taken forward. Both **Decision Trees** and **KNN** might have potential for improvement through parameter tuning.

## Tuning the Classifier
Parameter tuning is important for Machine Learning, because different settings can yield quite different results. The performance of the classifiers can differ wildly, and only an optimum balance of tuned parameters will bring the best results.

For this project I used `GridSearchCV` to look for the optimum parameter combination for the two classifiers that brought the best results in their default setting: Decision Trees and K-Nearest Neighbors.
I did this also because both their F1-scores were relatively high, and I was interested whether parameter tuning might bring one or the other to a clearly better predictive result.

From a defined list of settings I tried, GridSearchCV returned the following best parameter combinations for **Decision Trees**:

```
{'min_samples_split': 1, 'min_weight_fraction_leaf': 0.5, 'max_depth': None, 'min_samples_leaf': 1}
```
effectively tuning `min_samples_split` and `min_weight_fraction_leaf` differently than the default.


and the following for **K-Nearest-Neighbors**:

```
{'n_neighbors': 2, 'weights': 'distance', 'leaf_size': 30, 'algorithm': 'ball_tree'}
```
effectively tweaking `n_neighbors`, `weights`, and the optional `algorithm` argument, while `leaf-size` honed in the default value.

My results were astonishing: While the non-tuned Decision Tree returned a F1-score of ~0.67 and the non-tuned KNN only ~0.41, after tuning the KNN algorithm clearly outperformed the Decision Tree with a F1-score of `0.838709677419` vs. a only slightly improved score of ~0.68 for the Decision Tree.

While both algorithms performed similarly well in _recall_, the Decision Trees returned substantially more False Positives than KNN, lowering its _precision_ significantly. Since however precision is what I defined to be even more important than recall, because I don't want to mark innocent people as POI, KNN is definitely the better choice.

This is a great example to illustrate how important parameter tuning can be in Machine Learning. Without attempting to tune the KNN algorithm, my choice would have fallen on Decision Trees and I might have ended up flagging three times more people wrongly as POI than with employing the tuned KNN.

## Validation
It is important to **avoid overfitting** the learning algorithm with the data provided. This would happen when learning and testing the the parameters of a function on the same data.

To avoid this mistake, I used **cross validation**. This means that a part of the inital data is kept aside and not used while learning the parameter settings. Instead it is used to test the learned algorithm on, which will allow us to check how the model's predictive power performs on a previously unseen dataset.

For this I was using the scikit-learn `sklearn.cross_validation.train_test_split()` helper function. My choice fell on retaining 30% of the data aside for testing purposes. The `random_state` variable was set to 42, to allow consistent results and make it possible for other to check my work.


## Evaluation Method
Evaluating the results is an extremely important aspect of Machine Learning. I will try to make this clear using this example situation. When I was building my analysis functions, I initially calculated only **accuracy**. Accuracy of the classifiers can however differ sometimes very much from other, more useful evaluation metrics, such as **precision** and **recall**.
Taking a look at the **confusion matrix** for the classifier in my output shows, that sometimes not a single True Positive had been identified, however the _accuracy_ of the classifier would still be very high.
Therefore accuracy not the right metric to evaluate the performance of my classifiers and instead I resorted to calculate and consider precision, recall, and the F1-score.

High **precision** means that the people I identify as POI are actually POIs, however I might not be detecting some of them.

>Precision = TP / FP + TP

High **recall** means that I'm very likely to find all of the POIs, however I might also flag some innocent people falsely as POIs.

>Recall =  TP / FN + TP

The **F1-score** can be seen as a weighted average of precision and recall, balancing the trade-off between these two metrics. It denotes a value between 0 and 1, where 1 is the more favorable outcome.

For the sake of _guilty until proven different_ I would consider **precision** as the more important metric in this analysis, however I primarily decided to focus on achieving a high F1-score. In the final decision for which algorithm to use, I kept in mind to value precision higher than recall.


## Discussion
### Handling missing Values
I was not able to run the classifiers with the `NaN` missing values in my data. I could think of two different ways of working around this issue:

1. Replacing all the missing `NaN`s in all features with the **median** of the respective feature
2. Removing all the rows that include at least one `NaN` in any feature column

#### Replacing NaN with the median
My reasoning for this was, that a row containing e.g. one `NaN` in one of the features might have a very good predictive effect. Therefore when it gets removed, this information is lost.
Replacing the `NaN` with the median allows the row to remain part of the analysis and potentially increase the accuracy of the prediction.

I decided for the **median** over the mean, because **outliers** are present and important to the data and the prediction. They represent the already identified pois, who are essential for our prediction.
When using the mean on the now-empty values, these outliers would severly alter the distribution. Therefore I chose the median, as it is resisting outlier-influence.

#### Removing all rows containing NaN
Intuitively, this feels like a nice way to approach the issue, since I would prefer to deduce my predictions from _actual complete data_. The larger amount of data entering the algorithms, the better. And incomplete instances might skew the predictions and are therfore rather removed.

However, _the more data the better_ has of course also this other aspect. Removing all the rows that contain _any_ `NaN`s leaves me with an empty dataset. Effectively I would end up using at least less than half of the original rows that were available for analysis.

This is clearly a problem because it might exclude people that could be POIs or simply reduce the predictive power of the analysis.

### Algorithm Performance

The tuned K-Nearest-Neighbor algorithm that I ended up using for my final analysis was trained on the training set that was initially split using CrossValidation. 30% of the data were kept aside for testing purposes. The results from the tests on the testing portion of the dataset were:

```               
              predicted class
              _Yes_|__No_
actual | Yes |  13 |  3
class  | No  |  2  |  36

precision: 0.866666666667
recall:    0.8125
f1_score:  0.838709677419
```

Running it in `tester.py`, and thereby on a much larger dataset, gave me the following results:

```
Accuracy: 0.90839 | Precision: 0.75383 | Recall: 0.99520 | F1: 0.85786 | F2: 0.93530
```

The values are quite high which means that the algorithm performs well in predicting POI. With recall being higher than precision in the larger testing set, it means that I am more likely to mis-identify someone as a POI than that I will be missing a real POI.
This is not exactly in line with what I consider more important in this case. However, both scores are still rather high in this tuned setting so I think that it is still okay.


## Resources
### python
- http://stackoverflow.com/questions/18837607/remove-multiple-items-from-list-in-python
- http://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
- http://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
- http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html
- http://stackoverflow.com/questions/15772009/shuffling-permutating-a-dataframe-in-pandas
- http://stackoverflow.com/questions/24029659/python-pandas-replicate-rows-in-dataframe

### machine learning
- https://discussions.udacity.com/t/does-imputation-of-missing-features-cause-data-leakage/39739/2
- https://discussions.udacity.com/t/mistake-in-the-way-email-poi-features-are-engineered-in-the-course/4841/9
- https://discussions.udacity.com/t/does-imputation-of-missing-features-cause-data-leakage/39739/8
- https://en.wikipedia.org/wiki/Precision_and_recall
- https://en.wikipedia.org/wiki/Cross-validation_(statistics)
- http://scikit-learn.org/stable/modules/cross_validation.html
- http://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
- https://www.quora.com/In-classification-how-do-you-handle-an-unbalanced-training-set