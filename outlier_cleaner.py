#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (errorerence between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """
    # creating lists
    cleaned_data, error_list = [], []
    # populating the error list with the difference
    # between actual and predicted net worth
    for index, worth in enumerate(net_worths):
        pred = predictions[index]
        error = abs(worth - pred)
        error_list.append(error)
    # get number of items to be kept (discard the top 10%)
    item_n = len(net_worths) * 0.9
    # create list of tuples:
    cleaned_data = zip(ages, net_worths, error_list)
    # remove top errors
    while len(cleaned_data) > item_n:
        cleaned_data.remove(max(cleaned_data, key=lambda x:x[2]))

    print cleaned_data[:5]
    return cleaned_data

