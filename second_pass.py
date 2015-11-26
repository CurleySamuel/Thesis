import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score


def main():
    data = get_data()
    data = normalize_data(data)
    data, data_original = scalar_data(data)
    data_train_x, data_test_x, data_train_y, data_test_y = split_data(data)
    print "\nSize of training set: {}\nSize of testing set: {}\nNumber of features: {}\n".format(len(data_train_y), len(data_test_y), len(data_test_x.columns.values))
    #simple_random_forest(data_train_x, data_test_x, data_train_y, data_test_y)
    # simple_extremely_random_trees(
    #    data_train_x, data_test_x, data_train_y, data_test_y)
    simple_gradient_boosting(
        data_train_x, data_test_x, data_train_y, data_test_y)


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
        learning_rate=0.05
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
