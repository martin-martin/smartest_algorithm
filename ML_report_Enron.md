# Looking for the _smartest algorithm in the room_

Machine Learning practice with the **Enron** Email Corpus and Financial Dataset.

---

Katie's enthusiasm is infectious, so I went ahead to learn a bit about the topic of the Enron fraud at large. Watching the documentary allowed me to learn quite some new things about that ugly side of the _American Dream_.

Here's to some more involvement through a different path.

## What is this about?
Using Machine Learning techniques, I will try to analyze the two combined datasets to try to predict who of the people that are part of the dataset might be persons of interest (POI).
I define a POI to be someone who might be worth further investigation, because he or she might be more closely involved in the Enron fraud case than other people who worked at the company at that time.

One feature that these predictions will be based on, is the label `poi`, that marks people as `1` (= POI) or `0` (non-POI). This feature was created by Katie Malone when she manually gathered data online about the fraud case. (Source: http://usatoday30.usatoday.com/money/industries/energy/2005-12-28-enron-participants_x.htm)
`poi` will allow me to train a classifier to recognize patterns that POIs might share, and therefore represents a key feature for the analysis. Of the **146** data points available in the initial dataset, **18** are labeled as POI.


## Selecting Features
### Selection Mechanism
At first I wanted to know how often do I actually have data for the respective features, so I coded this up.
Initially I wanted to consider only features that are shared by at least 60% of the people in the dataset (where `feature != 'NaN'`). However, this did not include any data of the Email Corpus, so I reduced the treshold to 50%.

Of the result:

```
['salary', 'to_messages', 'total_payments', 'exercised_stock_options', 'bonus', 'email_address', 'total_stock_value', 'expenses', 'from_messages', 'other', 'from_this_person_to_poi', 'poi', 'shared_receipt_with_poi', 'restricted_stock', 'from_poi_to_this_person']
```

I selected

```
['poi', 'salary', 'bonus', 'total_payments', 'shared_receipt_with_poi', 'from_poi_to_this_person', 'from_this_person_to_poi' ]
```


#### features removed
'email_address' is a unique string for each person and therefore won't have any predictive power. 'to_messages' and 'from_messages' seems better encoded in 'from_poi_to_this_person' and 'from_this_person_to_poi', considering that I will explore the data in correlation to pois (persons of interest).
I don't even know what 'other' is, and neither am I familiar with 'restricted_stock' and 'exercised_stock_options'.
I can probably understand 'total_stock_value', however I know that high-ranking pois were selling off their stocks of the company towards the end, and I just generally don't know enough about the stock market to handle these values well. I also am unsure about what 'expenses' are logged in the so-named variable, so for this analysis I kept it aside.


#### features retained
What I therefore chose to further investigate and train my classifiers on, is 'salary', 'bonus' and 'total_payments', for the more obvious financial-related features. Further I chose 'from_poi_to_this_person' and 'from_this_person_to_poi' from the email corpus data, to investigate a possible relationship indicated of pois sending an increased amount of emails between each other than towards non-pois.
And finally I also chose 'shared_receipt_with_poi' from the financial dataset. This is another more exploratoy feature, but it got me interested to look for trends here (and to properly understand what does this mean).

### Other Considerations
I decided not to employ a Decision Tree algorithm for selecting my features. My main reason for this was, that even though some now-discarded feature might do a good job in predicting pois, I found it dangerous to base this on features that have less than half of the data involved.
With 146 people, the dataset is anyways not very large, therefore I wanted to make sure to train and test my algorithms at least on a somewhat representable size.

### A new feature
Wondering about the **ratio** of emails sent to a poi vs. received from a poi, as mentioned in the course, made me create this new feature: `sent_received_ratio`.
The interesting idea behind this feature was, that POIs might be more  likely to send a higher amount fo emails among each other than towards non-POIs.

Therefore my **final set of 8 features** was:

```
['poi', 'salary', 'bonus', 'total_payments', 'shared_receipt_with_poi', 'from_poi_to_this_person', 'from_this_person_to_poi', 'sent_received_ratio']
```

## Outliers and other Wrangling

### outliers

When analyzing my features for outliers, I identified one called `TOTAL` in the `salary` column. This was most probably simply the total of all the salaries, and should therefore be removed from the dataset.

I had a little trouble here when trying to find the `max()` value of the columns I had plotted. My code only returned `NaN` as the max. After a while I figured that this is because the `'NaN'` values were actually **strings**, not `None` type. So I went ahead to transform them into actual `NaN`, to finally find and remove the row containing the outlier.

Further, I also removed by hand the obvious "outlier" `THE TRAVEL AGENCY IN THE PARK` that was a bogus firm that some of the people involved in the fraud were using for booking travel costs. I've found out about this data point through inspecting the `enron61702insiderpay.pdf` file manually.

### (near) empty rows
Because rows with scarce data are probably not very useful for a realistic prediction, I decided to remove rows that have nearly all data of my chosen features missing.

Since the dataset is however rather small, I wanted to take a cautious approach and still keep as many rows as possible. I therefore removed only rows that have 2 or less features containing actual values. My choice for the threshold fell on 2, because **all** rows have an entry for `poi` (`1`/`0`). Therefore I was essentially only removing rows that have **no additional feature** containing a value.

This approach prevented me from discarding too many of the anyway scarce rows, yet also get rid of entries that are more likely to distract than aid the prediction power of my algorithm-to-be.

## Choice of Algorithm
To test the performance of various classifiers, I wrote a test suite called `test_a_lot()`. The wrapper function ran tests for **Naive Bayes**, **SVM**s (iterating on the _kernels_), **Decision Trees** (with different settings for _n_features_) and **K-nearest Neighbors** (with different settings for _n_neighbors_).

The test suite employs **feature scaling** for SVM and K-NN. This is necessary because both rely on a measure of distance, which gets distorted if the different features used are not re-adapted on a scale from 0 to 1.
Further the test suite calculates and prints the confusion matrix as well as precision, recall, and the F1-score for each classifier and its tuning.

Further, I've run the test suite with two different inputs, one time effectively removing all rows containing `NaN` values, and one time substituting them for the median of each column (see _Discussion_ further down).

Some of the classifiers I employed returned very bad results for precision and recall. This was especailly true for the dataset with the imputed medians. There only Decision Trees yielded results that were above 0. The dataset with the `NaN`s removed showed better results, and additionally to Decision Trees, also Naive Bayes returned somewhat useful figures. Both SVM and KNN did not return useful predictions.

The classifier that yielded the best results (gleaned from the F1-score) was a **Decision Tree** - with the amounts of features used varying over different runs.

## Tuning the Classifier
Finally, in my tuning phase I wrote a bunch of tries to find a good feature combination for my Decision Tree.

I now went to employ an overfit Decision Tree to estimate the feature importance, ran my calculation 10 times and selected all features that in any run passed the threshold of 0.2.

Those were:

```
features = ['poi',
            'shared_receipt_with_poi',
            'total_payments',
            'from_this_person_to_poi',
            'sent_received_ratio',
            'bonus']
```

Using these features for my Decision Tree classifier (effectively tuning it to use 6 features), produces recurrently stable F1-scores and consistently keeps precision and recall above 0.3.


## Validation

Validation is an extremely important aspect of Machine Learning. I will try to make this clear using this example situation. When I was building my analysis functions, I initially calculated only **accuracy**. Accuracy of the classifiers can however differ sometimes very much from other, more useful validation metrics, such as precision and recall.
Taking a look at the **confusion matrix** for the classifier in my output shows, that sometimes not a single True Positive had been identified, however the _accuracy_ of the classifier would still be very high.
Therefore accuracy not the right metric to evaluate the performance of my classifiers and instead I resorted to calculate and consider precision, recall, and the F1-score.

_Which of the two will I favor? What does each or the other mean exactly?_

High **precision** means that the people I identify as POI are actually POIs, however I might not be detecting some of them.

>Precision = TP / FP + TP

High **recall** means that I'm very likely to find all of the POIs, however I might also flag some innocent people falsely as POIs.

>Recall =  TP / FN + TP

The **F1-score** can be seen as a weighted average of precision and recall, balancing the trade-off between these two metrics. It denotes a value between 0 and 1, where 1 is the more favorable outcome.

For the sake of _guilty until proven different_ I would consider **precision** as the more important metric in this analysis, however I simply decided to focus on achieving a high F1-score.


## Discussion
### Handling missing Values
I was not able to run the classifiers with the `NaN` missing values in my data. Therefore I chose two different ways of working around this issue:

1. Replacing all the missing `NaN`s in all features with the **median** of the respective feature
2. Removing all the rows that include at least one `NaN` in any feature column

#### Replacing NaN with the median
My reasoning for this was, that a row containing e.g. one `NaN` in one of the features might have a very good predictive effect. Therefore when it gets removed, this information is lost.
Replacing the `NaN` with the median allows the row to remain part of the analysis and potentially increase the accuracy of the prediction.

I decided for the **median** over the mean, because **outliers** are present and important to the data and the prediction. They represent the already identified pois, who are essential for our prediction.
When using the mean on the now-empty values, these outliers would severly alter the distribution. Therefore I chose the median, as it is resisting outlier-influence.

#### Removing all rows containing NaN
This approach was the one that eventually turned out to provide me with better F1-scores in using the classifiers.

Intuitively it feels like a nice way to approach the issue, since I would prefer to deduce my predictions from _actual data_. The larger amount of data entering the algorithms, the better. And incomplete instances might skew the predictions and are therfore rather removed.

However, _the more data the better_ has of course also this other aspect. Removing all the rows that contain _any_ `NaN`s leaves the data points at only 61 people. Effectively I end up using less than half of the original rows that were available for analysis.

This is clearly a problem because it might exclude people that could be POIs or simply reduce the predictive power of the analysis.

### Algorithm Performance
The Decision Tree that I ended up using for my final analysis returned a mean of `0.394642857142` for the **precision scores** and a mean of `0.54` for the **recall scores**.

Generally this means that I am more likely to mis-identify someone as a POI than that I will be missing a real POI.
This is not exactly in line with what I was considering more important initially. However, both scores are rather high in this tuned setting so I think that it is still okay.


## Resources
### python
- http://stackoverflow.com/questions/18837607/remove-multiple-items-from-list-in-python
- http://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
- http://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
- - http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html

### machine learning
- https://discussions.udacity.com/t/does-imputation-of-missing-features-cause-data-leakage/39739/2
- https://discussions.udacity.com/t/mistake-in-the-way-email-poi-features-are-engineered-in-the-course/4841/9
- https://discussions.udacity.com/t/does-imputation-of-missing-features-cause-data-leakage/39739/8
- https://en.wikipedia.org/wiki/Precision_and_recall