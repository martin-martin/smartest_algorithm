# Looking for the _smartest algorithm in the room_

Machine Learning practice with the **Enron** Email Corpus and Financial Dataset.

---

Katie's enthusiasm is contractible, so I went ahead to learn a bit. Watching the documentary allowed me to learn quite some new things about that ugly side of the _American Dream_.

Here's to some more involvement through a different path.

## What is this about?
Using Machine Learning techniques, I will try to analyze the two combined datasets to try to predict who of the people that are part of the dataset might be persons of interest (poi).
I define a poi to be someone who might be worth further investigation, because he or she might be more closely involved in the Enron fraud case than other people who worked at the company at that time.

One feature that these predictions will be based on, is the label 'poi', that was manually assembled by Katie Malone from people proven to be involved in the fraud. (Source: http://usatoday30.usatoday.com/money/industries/energy/2005-12-28-enron-participants_x.htm)
It therefore represents a key feature for the analysis.


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


#### out
'email_address' is a unique string for each person and therefore won't have any predictive power. 'to_messages' and 'from_messages' seems better encoded in 'from_poi_to_this_person' and 'from_this_person_to_poi', considering that I will explore the data in correlation to pois (persons of interest).
I don't even know what 'other' is, and neither am I familiar with 'restricted_stock' and 'exercised_stock_options'.
I can probably understand 'total_stock_value', however I know that high-ranking pois were selling off their stocks of the company towards the end, and I just generally don't know enough about the stock market to handle these values well. I also am unsure about what 'expenses' are logged in the so-named variable, so for this analysis I kept it aside.


#### in
What I therefore chose to further investigate and train my classifiers on, is 'salary', 'bonus' and 'total_payments', for the more obvious financial-related features. Further I chose 'from_poi_to_this_person' and 'from_this_person_to_poi' from the email corpus data, to investigate a possible relationship indicated of pois sending an increased amount of emails between each other than towards non-pois.
And finally I also chose 'shared_receipt_with_poi' from the financial dataset. This is another more exploratoy feature, but it got me interested to look for trends here (and to properly understand what does this mean).

### Other Considerations
I decided not to employ a Decision Tree algorithm for selecting my features. My main reason for this was, that even though some now-discarded feature might do a good job in predicting pois, I found it dangerous to base this on features that have less than half of the data involved.
With 146 people, the dataset is anyways not very large, therefore I wanted to make sure to train and test my algorithms at least on a somewhat representable size.

### A new feature
Wondering about the **ratio** of emails sent to a poi vs. received from a poi, as mentioned in the course, made me create this new feature: `sent_received_ratio`.

In a try I went to remove the other email-related features, considering that their information is encoded in the new feature.

Therefore my **final set of features** was:

```
['poi', 'salary', 'bonus', 'total_payments', 'shared_receipt_with_poi', `sent_received_ratio`]
```

## Outliers

When analyzing my features for outliers, I identified one called `TOTAL` in the `salary` column. This was most probably simply the total of all the salaries, and should therefore be removed from the dataset.

I had a little trouble here when trying to find the `max()` value of the columns I had plotted. My code only returned `NaN` as the max. After a while I figured that this is because the `'NaN'` values were actually **strings**, not `None` type. So I went ahead to transform them into actual `NaN`, to finally find and remove the row containing the outlier.

## Validation
### Precision
Precision = TP / FP + TP

### Recall
Recall =  TP / FN + TP


## Discussion
### Replacing NaN with the median
I was not able to run the classifiers with the `NaN` missing values in my data. Therefore I chose two different ways of working around this issue:

1. Removing all the rows that include at least one `NaN` in any feature column
2. Replacing all the missing `NaN`s in all features with the **median** of the respective feature

My reasoning for this is, that a row containing e.g. one `NaN` in one of the features might have a very good predictive effect. Therefore when it gets removed, this information is lost.
Replacing the `NaN` with the median allows the row to remain part of the analysis and potentially increase the accuracy of the prediction.

I decided for the **median** over the mean, because **outliers** are present and important to the data and the prediction. They represent the already identified pois, who are essential for our prediction.
When using the mean on the now-empty values, these outliers would severly alter the distribution. Therefore I chose the median, as it is resisting outlier-influence.

The resulting accuracy of some of the classifiers is very high when using the dataset that includes the median values. I am still unsure about this result. I guess I should go and do validation!


## Resources
### python
- http://stackoverflow.com/questions/18837607/remove-multiple-items-from-list-in-python
- http://stackoverflow.com/questions/18172851/deleting-dataframe-row-in-pandas-based-on-column-value
- http://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values-in-a-column-in-pandas
- - http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.dropna.html