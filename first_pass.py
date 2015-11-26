import pandas as pd
import numpy as np
import sys
from sklearn import metrics
import matplotlib.pyplot as plt

to_predict_mls = []


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
    data = data.drop(['Unnamed: 0', 'EXPIREDDATE', 'COOLING', 'AREA', "SHOWINGINSTRUCTIONS", "OFFICEPHONE", "STATUS",
                      "OFFICENAME", "HOUSENUM2", "HOUSENUM1", "DTO", "DOM", "JUNIORHIGHSCHOOL", "AGENTNAME", "HIGHSCHOOL", "STREETNAME", "PHOTOURL", "HIGHSCHOOL", "ELEMENTARYSCHOOL"], 1)

    data.loc[data['BATHS'].isnull(), 'BATHS'] = data['BEDS'] / 2

    for x in ["LISTDATE", "SOLDDATE"]:
        data[x] = (
            data[x] - data[x].min()).astype('timedelta64[M]').astype(int)

    #data = mix_in_to_predict_data(data)

    for var in ["PROPTYPE", "STYLE", "HEATING", "CITY", "LEVEL"]:
        if var != "LEVEL":
            data[var] = data[var].str.lower()
        new_data = data[var].str.get_dummies(sep=', ')
        new_data.rename(
            columns=lambda x: "({}) ".format(var) + x, inplace=True)
        data = data.merge(
            new_data,
            left_index=True,
            right_index=True
        )
        data.drop(var, 1)

    #data, to_predict_data = mix_out_to_predict_data(data)

    msk = np.random.rand(len(data)) < 0.8
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
    score_6, predictions_6 = dummie_columns_gradient_boosting(
        training_set, testing_set)

    # sample_predictions(
    # testing_set, predictions_1, predictions_2, predictions_3, predictions_4)


def mix_in_to_predict_data(data):
    to_predict_data = pd.read_csv(
        "predictions/to_predict.csv"
    )
    to_predict_data = to_predict_data.drop(["ACREAGE", "AGE_YEAR"], 1)
    to_predict_data["SOLDDATE"] = to_predict_data["LISTDATE"]
    global to_predict_mls
    to_predict_mls = to_predict_data.index.values
    data = data.append(to_predict_data)
    return data


def mix_out_to_predict_data(data):
    to_predict_data = data.tail(len(to_predict_mls))
    data = data.drop(to_predict_data.index)
    return data, to_predict_data


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
    predicting_columns.remove("SQFT")
    rf = RandomForestRegressor(
        n_estimators=300, n_jobs=-1)
    rf.fit(train[predicting_columns], train["SOLDPRICE"])
    score = rf.score(test[predicting_columns], test["SOLDPRICE"])
    predictions = rf.predict(test[predicting_columns])
    sample_predictions(test, predictions)
    # print "-- Feature Importance --"
    # for x in range(len(rf.feature_importances_)):
    #    print predicting_columns[x], rf.feature_importances_[x]
    feature_importance = rf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, test[predicting_columns].columns.values[sorted_idx], fontsize=6)
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
    print "Accuracy: {}\n".format(score)
    return score, predictions


def dummie_columns_extra_trees(train, test):
    from sklearn.ensemble import ExtraTreesRegressor
    print "-- {} --".format("Extremely Randomized Trees Regression using all but remarks")
    predicting_columns = list(train._get_numeric_data().columns.values)
    predicting_columns.remove("LISTPRICE")
    predicting_columns.remove("SOLDPRICE")
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
