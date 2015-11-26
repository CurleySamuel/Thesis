import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from time import time
from operator import itemgetter
from scipy.stats import randint as sp_randint


def main():
    data = get_data()
    data = normalize_data(data)
    data, data_original = scalar_data(data)
    data_train_x, data_test_x, data_train_y, data_test_y = split_data(data)
    print "\nSize of training set: {}\nSize of testing set: {}\nNumber of features: {}\n".format(len(data_train_y), len(data_test_y), len(data_test_x.columns.values))
    """
    simple_random_forest(
        data_train_x, data_test_x, data_train_y, data_test_y)
    simple_extremely_random_trees(
        data_train_x, data_test_x, data_train_y, data_test_y)
    """
    simple_gradient_boosting(
        data_train_x, data_test_x, data_train_y, data_test_y)
    # fine_tune_gradient_boosting_hyper_params(
    #    data_train_x, data_test_x, data_train_y, data_test_y)


def simple_random_forest(data_train_x, data_test_x, data_train_y, data_test_y):
    from sklearn.ensemble import RandomForestRegressor
    print "-- {} --".format("Random Forest Regression using all but remarks")
    rf = RandomForestRegressor(
        n_estimators=300,
        n_jobs=-1
    )
    rf.fit(data_train_x, data_train_y)
    sample_predictions(rf.predict(data_test_x), data_test_y)
    score = rf.score(data_test_x, data_test_y)
    cross_validated_scores = cross_val_score(
        rf, data_test_x, data_test_y, cv=5)
    print "MSE Accuracy: {}".format(score)
    print "MSE Across 5 Folds: {}".format(cross_validated_scores)
    print "95%% Confidence Interval: %0.3f (+/- %0.3f)\n" % (cross_validated_scores.mean(), cross_validated_scores.std() * 1.96)


def simple_extremely_random_trees(data_train_x, data_test_x, data_train_y, data_test_y):
    from sklearn.ensemble import ExtraTreesRegressor
    print "-- {} --".format("Extremely Randomized Trees Regression using all but remarks")
    rf = ExtraTreesRegressor(
        n_estimators=300,
        n_jobs=-1
    )
    rf.fit(data_train_x, data_train_y)
    sample_predictions(rf.predict(data_test_x), data_test_y)
    score = rf.score(data_test_x, data_test_y)
    cross_validated_scores = cross_val_score(
        rf, data_test_x, data_test_y, cv=5)
    print "MSE Accuracy: {}".format(score)
    print "MSE Across 5 Folds: {}".format(cross_validated_scores)
    print "95%% Confidence Interval: %0.3f (+/- %0.3f)\n" % (cross_validated_scores.mean(), cross_validated_scores.std() * 1.96)


def simple_gradient_boosting(data_train_x, data_test_x, data_train_y, data_test_y):
    from sklearn.ensemble import GradientBoostingRegressor
    print "-- {} --".format("Gradient Boosting Regression using all but remarks")
    rf = GradientBoostingRegressor(
        n_estimators=500,
        subsample=0.8,
        learning_rate=0.09,
        min_samples_leaf=7,
        min_samples_split=9,
        max_features=10,
        max_depth=8
    )
    rf.fit(data_train_x, data_train_y)
    sample_predictions(rf.predict(data_test_x), data_test_y)
    score = rf.score(data_test_x, data_test_y)
    cross_validated_scores = cross_val_score(
        rf, data_test_x, data_test_y, cv=5)
    print "MSE Accuracy: {}".format(score)
    print "MSE Across 5 Folds: {}".format(cross_validated_scores)
    print "95%% Confidence Interval: %0.3f (+/- %0.3f)\n" % (cross_validated_scores.mean(), cross_validated_scores.std() * 1.96)

    """
    test_score = np.zeros((1000,), dtype=np.float64)

    for i, y_pred in enumerate(rf.staged_predict(data_test_x)):
        test_score[i] = rf.loss_(data_test_y, y_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(1000) + 1, rf.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(1000) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    plt.show()
    """


def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


def fine_tune_gradient_boosting_hyper_params(data_train_x, data_test_x, data_train_y, data_test_y):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.grid_search import RandomizedSearchCV
    print "-- {} --".format("Fine-tuning Gradient Boosting Regression")
    rf = GradientBoostingRegressor(
        n_estimators=500
    )
    param_dist = {
        "learning_rate": [0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.15, 0.2],
        "max_depth": sp_randint(1, 11),
        "min_samples_split": sp_randint(1, 11),
        "min_samples_leaf": sp_randint(1, 11),
        "subsample": [0.2, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "max_features": sp_randint(1, 11)
    }
    n_iter_search = 100
    random_search = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        n_jobs=-1,
        cv=5,
        verbose=1
    )

    start = time()
    random_search.fit(data_train_x, data_train_y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.grid_scores_)


def get_data():
    # Read the first argument, csv -> DataFrame
    data = pd.read_csv(
        sys.argv[1],
        index_col="MLSNUM",
        parse_dates=["LISTDATE", "SOLDDATE", "EXPIREDDATE"],
    )
    if len(sys.argv) > 2:
        # Read subsequent arguments, appending them into the same DataFrame.
        for f in sys.argv[2:]:
            new_data = pd.read_csv(
                f,
                index_col="MLSNUM",
                parse_dates=["LISTDATE", "SOLDDATE", "EXPIREDDATE"],
            )
            data = data.append(new_data)
    return data


def normalize_data(data):
    # Drop all the columns that we don't want.
    data = data.drop(['Unnamed: 0', 'EXPIREDDATE', 'COOLING', 'AREA', "SHOWINGINSTRUCTIONS", "OFFICEPHONE", "STATUS",
                      "OFFICENAME", "HOUSENUM2", "HOUSENUM1", "DTO", "DOM", "JUNIORHIGHSCHOOL", "AGENTNAME", "HIGHSCHOOL", "STREETNAME", "PHOTOURL", "HIGHSCHOOL", "ELEMENTARYSCHOOL", "ADDRESS", "LISTPRICE"], 1)
    # If missing data on number of baths, set it to number of beds / 2.
    data.loc[data['BATHS'].isnull(), 'BATHS'] = data['BEDS'] / 2
    # Convert dates into number of days since the latest date.
    for x in ["LISTDATE", "SOLDDATE"]:
        data[x] = (
            data[x] - data[x].min()).astype('timedelta64[M]').astype(int)
    return data


def scalar_data(data):
    old_data = data.copy()

    # Column 'OTHERFEATURES' contains a semicolon seperated
    # string of feature:value pairs. We need to parse those for
    # every row and seperate them into their own columns.
    sub_columns = ['Basement', 'Fireplaces', 'Roof', 'Floor', 'Appliances', 'Foundation', 'Construction',
                   'Exterior', 'Exterior Features', 'Insulation', 'Electric', 'Interior Features', 'Hot Water']
    for sub_column in sub_columns:
        data[sub_column] = data['OTHERFEATURES'].str.extract(
            "{}:(.*?);".format(sub_column))
    data = data.drop('OTHERFEATURES', 1)
    # Take these unstandardized fields and create 'dummy columns' from them
    # which have a 1 or 0 for each row. The number of dummy columns is equal
    # to the number of distinct possible answers for each column.
    #
    # i.e. PROPTYPE will get split up into (PROPTYPE) SF and (PROPTYPE) MF
    sub_columns.extend(
        ["PROPTYPE", "STYLE", "HEATING", "CITY", "LEVEL", "STATE"])
    for var in sub_columns:
        if var == "LEVEL":
            # let the hate flow through you young padawan.
            data[var] = data[var].fillna(0.0).replace(
                to_replace='B', value=1.0).astype(int).astype(str)
        else:
            # Calling .lower() on an int makes it None.
            data[var] = data[var].str.lower()
        new_data = data[var].str.get_dummies(sep=', ')
        new_data.rename(
            columns=lambda x: "({}) ".format(var) + x, inplace=True)
        data = data.merge(
            new_data,
            left_index=True,
            right_index=True
        )
        data = data.drop(var, 1)
    return data._get_numeric_data(), old_data


def split_data(data):
    from sklearn.cross_validation import train_test_split
    x = data.drop('SOLDPRICE', 1)
    y = data['SOLDPRICE']
    return train_test_split(x, y, test_size=0.25)


def sample_predictions(predicted, actual):
    sample_size = 20
    samples = np.random.randint(0, high=len(actual), size=sample_size)
    print '{:^30}'.format("Predicted"),
    print '{:^30}\n'.format("Sale Price")
    for sample in samples:
        print '{:^30,}'.format(int(predicted[sample])),
        print '{:^30,}'.format(int(actual.iloc[[sample]].values[0]))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print "Error: Missing input file arguments."
