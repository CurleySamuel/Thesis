import pandas as pd
import numpy as np
import sys
from sklearn import metrics
import matplotlib.pyplot as plt


def main():
    data = pd.read_csv(
        sys.argv[1],
        index_col="MLSNUM",
        parse_dates=["LISTDATE", "SOLDDATE", "EXPIREDDATE"],
    )
    if len(sys.argv) > 2:
        for f in sys.argv[2:]:
            new_data = pd.read_csv(
                f,
                index_col="MLSNUM",
                parse_dates=["LISTDATE", "SOLDDATE", "EXPIREDDATE"],
            )
            data = data.append(new_data)
    data = data.drop(['Unnamed: 0', 'EXPIREDDATE', 'COOLING',
                      'AREA', "SHOWINGINSTRUCTIONS", "OFFICEPHONE", "STATUS"], 1)

    data.loc[data['BATHS'].isnull(), 'BATHS'] = data['BEDS'] / 2

    import ipdb
    ipdb.set_trace()
    for x in ["LISTDATE", "SOLDDATE"]:
        data[x] = (
            data[x] - data[x].min()).astype('timedelta64[M]').astype(int)

    for var in ["PROPTYPE", "STYLE", "HEATING", "CITY", "LEVEL"]:
        new_data = data[var].str.get_dummies(sep=', ')
        new_data.rename(
            columns=lambda x: "({}) ".format(var) + x, inplace=True)
        data = data.merge(
            new_data,
            left_index=True,
            right_index=True
        )
        data.drop(var, 1)

    msk = np.random.rand(len(data)) < 0.95
    training_set = data[msk]
    testing_set = data[~msk]
    print "Size of training set: {}\nSize of testing set: {}".format(len(training_set), len(testing_set))

    score_1, predictions_1 = sqft_sold_price_univariate_linear_regression(
        training_set, testing_set)
    score_2, predictions_2 = location_sold_price_random_forest(
        training_set, testing_set)
    score_3, predictions_3 = numeric_data_sold_price_random_forest(
        training_set, testing_set)
    score_4, predictions_4 = dummie_columns_random_forest(
        training_set, testing_set)
    score_5, predictions_5 = dummie_columns_extra_trees(
        training_set, testing_set)
    score_7, predictions_7 = dummie_columns_gradient_boosting(
        training_set, testing_set)

    # sample_predictions(
    # testing_set, predictions_1, predictions_2, predictions_3, predictions_4)


def sqft_sold_price_univariate_linear_regression(train, test):
    from sklearn.linear_model import LinearRegression
    print "-- {} --".format("Linear Regression using SQFT")
    lr = LinearRegression()
    lr.fit(train[["SQFT"]], train["SOLDPRICE"])
    score = lr.score(test[["SQFT"]], test["SOLDPRICE"])
    predictions = lr.predict(test[["SQFT"]])
    sample_predictions(test, predictions)
    print "Accuracy: {}\n".format(score)
    return score, predictions


def location_sold_price_random_forest(train, test):
    from sklearn.ensemble import RandomForestRegressor
    print "-- {} --".format("Random Forest Regression using LAT/LNG")
    predicting_columns = ["lat", "lng"]
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    rf.fit(train[predicting_columns], train["SOLDPRICE"])
    score = rf.score(test[predicting_columns], test["SOLDPRICE"])
    predictions = rf.predict(test[predicting_columns])
    sample_predictions(test, predictions)
    print "Accuracy: {}\n".format(score)
    return score, predictions


def numeric_data_sold_price_random_forest(train, test):
    from sklearn.ensemble import RandomForestRegressor
    print "-- {} --".format("Random Forest Regression using LAT/LNG/AGE/SQFT/BEDS/BATHS/GARAGE/LOTSIZE")
    predicting_columns = [
        "AGE", "lng", "SQFT", "BEDS", "BATHS", "lat", "GARAGE", "LOTSIZE"]  # LEVEL had a B in it
    rf = RandomForestRegressor(n_estimators=100, n_jobs=-1)
    rf.fit(train[predicting_columns], train["SOLDPRICE"])
    score = rf.score(test[predicting_columns], test["SOLDPRICE"])
    predictions = rf.predict(test[predicting_columns])
    sample_predictions(test, predictions)
    print "Accuracy: {}\n".format(score)
    return score, predictions


def dummie_columns_random_forest(train, test):
    from sklearn.ensemble import RandomForestRegressor
    print "-- {} --".format("Random Forest Regression using all but remarks")
    predicting_columns = list(train._get_numeric_data().columns.values)
    predicting_columns.remove("LISTPRICE")
    predicting_columns.remove("SOLDPRICE")
    predicting_columns.remove("DTO")
    predicting_columns.remove("DOM")
    rf = RandomForestRegressor(
        n_estimators=300, n_jobs=-1)
    rf.fit(train[predicting_columns], train["SOLDPRICE"])
    score = rf.score(test[predicting_columns], test["SOLDPRICE"])
    predictions = rf.predict(test[predicting_columns])
    sample_predictions(test, predictions)
    print "-- Feature Importance --"
    for x in range(len(rf.feature_importances_)):
        print predicting_columns[x], rf.feature_importances_[x]
    print "Accuracy: {}\n".format(score)
    return score, predictions


def dummie_columns_extra_trees(train, test):
    from sklearn.ensemble import ExtraTreesRegressor
    print "-- {} --".format("Extremely Randomized Trees Regression using all but remarks")
    predicting_columns = list(train._get_numeric_data().columns.values)
    predicting_columns.remove("LISTPRICE")
    predicting_columns.remove("SOLDPRICE")
    predicting_columns.remove("DTO")
    predicting_columns.remove("DOM")
    rf = ExtraTreesRegressor(
        n_estimators=300, n_jobs=-1)
    rf.fit(train[predicting_columns], train["SOLDPRICE"])
    score = rf.score(test[predicting_columns], test["SOLDPRICE"])
    predictions = rf.predict(test[predicting_columns])
    sample_predictions(test, predictions)
    print "Accuracy: {}\n".format(score)
    return score, predictions


def dummie_columns_gradient_boosting(train, test):
    from sklearn.ensemble import GradientBoostingRegressor
    print "-- {} --".format("Gradient Boosting Regression using all but remarks")
    predicting_columns = list(train._get_numeric_data().columns.values)
    predicting_columns.remove("LISTPRICE")
    predicting_columns.remove("SOLDPRICE")
    predicting_columns.remove("DTO")
    predicting_columns.remove("DOM")
    svr = GradientBoostingRegressor(n_estimators=300)
    svr.fit(train[predicting_columns], train["SOLDPRICE"])
    score = svr.score(test[predicting_columns], test["SOLDPRICE"])
    predictions = svr.predict(test[predicting_columns])
    sample_predictions(test, predictions)
    print "Accuracy: {}\n".format(score)
    return score, predictions


def sample_predictions(actual, *args):
    sample_size = 20
    samples = np.random.randint(0, high=len(actual), size=sample_size)
    for idx in range(len(args)):
        print '{:^30}'.format(idx),
    print '{:^30}'.format("List Price"),
    print '{:^30}\n'.format("Sale Price")
    for sample in samples:
        for predictions in args:
            print '{:^30,}'.format(int(predictions[sample])),
        print '{:^30,}'.format(int(actual.iloc[[sample]]["LISTPRICE"].values[0])),
        print '{:^30,}'.format(int(actual.iloc[[sample]]["SOLDPRICE"].values[0]))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print "Error: Missing input file arguments."
