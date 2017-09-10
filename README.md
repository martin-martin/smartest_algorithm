# The smartest _algorithm_ in the room
Machine Learning analysis of the Enron public dataset to identify people involved in the fraud case (Udacity DAND P5)

## Discussion
### Handling missing Values
I was not able to run the classifiers with the NaN missing values in my data. I could think of two different ways of working around this issue:

1. Replacing all the missing NaNs in all features with the median of the respective feature
2. Removing all the rows that include at least one NaN in any feature column

### Replacing NaN with the median
My reasoning for this was, that a row containing e.g. one NaN in one of the features might have a very good predictive effect. Therefore when it gets removed, this information is lost. Replacing the NaN with the median allows the row to remain part of the analysis and potentially increase the accuracy of the prediction.

I decided for the median over the mean, because outliers are present and important to the data and the prediction. They represent the already identified pois, who are essential for our prediction. When using the mean on the now-empty values, these outliers would severly alter the distribution. Therefore I chose the median, as it is resisting outlier-influence.

### Removing all rows containing NaN
Intuitively, this feels like a nice way to approach the issue, since I would prefer to deduce my predictions from actual complete data. The larger amount of data entering the algorithms, the better. And incomplete instances might skew the predictions and are therfore rather removed.

However, the more data the better has of course also this other aspect. Removing all the rows that contain any NaNs leaves me with an empty dataset. Effectively I would end up using at least less than half of the original rows that were available for analysis.

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

Running it in tester.py, and thereby on a much larger dataset, gave me the following results:

```
Accuracy: 0.90839 | Precision: 0.75383 | Recall: 0.99520 | F1: 0.85786 | F2: 0.93530
```

The values are quite high which means that the algorithm performs well in predicting POI. With recall being higher than precision in the larger testing set, it means that I am more likely to mis-identify someone as a POI than that I will be missing a real POI. This is not exactly in line with what I consider more important in this case. However, both scores are still rather high in this tuned setting so I think that it is still okay.
