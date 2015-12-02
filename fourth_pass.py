import pandas as pd
import numpy as np
import sys
from sklearn.cross_validation import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LarsCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.cluster import AffinityPropagation


def main():
    # Pulls in all the housing data from geocoded files.
    data = get_data()
    # Drop a bunch of unwanted columns and parse dates.
    data = normalize_data(data)
    # Take categorical columns and generate dummy features with binary cels.
    data = binarize_categorical_data(data)
    # Split my data into the 4 sets of training and test.
    data_train_x, data_test_x, data_train_y, data_test_y = split_data(data)
    print "\nSize of training set: {}\nSize of testing set: {}\nNumber of features: {}\n".format(
        len(data_train_y), len(data_test_y), len(data_test_x.columns.values)
    )
    """
        We want -
            1. It's cluster derived from location, size, etc.
            2. An estimate derived from remarks.
            3. An estimate derived from everything but remarks.
            4. An estimate derived from the previous two estimates.
    """
    final_pipeline = Pipeline([
        ('remarks_split', FeatureUnion([
            ('remarks_pipe', Pipeline([
                ('peel_remarks', PeelRemarks()),
                ('fill_nans', FillNaNs()),
                ('tfidf', TfidfVectorizer(
                    ngram_range=(1, 4),
                    max_df=0.7,
                    min_df=5,
                    sublinear_tf=True
                )),
                ('truncatedsvd', TruncatedSVD(
                    n_components=500,
                    algorithm='arpack'
                )),
                ('larscv', RegressorWrapper(LarsCV(
                    max_iter=500,
                    max_n_alphas=750,
                    normalize=False,
                    cv=5,
                    verbose=True
                )))
            ])),
            ('not_remarks_pipe', Pipeline([
                ('cluster', ClustererWrapper(AffinityPropagation(
                    verbose=True
                ))),
                ('main_model', GradientBoostingRegressor(
                    loss='huber',
                    n_estimators=500,
                    subsample=0.6,
                    learning_rate=0.08,
                    min_samples_leaf=3,
                    min_samples_split=1,
                    max_features='auto',
                    max_depth=5,
                    alpha=0.9,
                    min_weight_fraction_leaf=0.0,
                    verbose=1
                ))
            ])),
        ])),
        ('final_model', RandomForestRegressor(
            n_estimators=500,
            n_jobs=-1,
            verbose=1
        ))
    ])

    final_pipeline.fit(data_train_x, data_train_y)
    report_accuracy(
        final_pipeline, data_test_x, data_test_y, name='combined model')


def get_data():
    # Read the first argument, csv -> DataFrame
    data = pd.read_csv(
        sys.argv[1],
        index_col="MLSNUM",
        parse_dates=["LISTDATE", "SOLDDATE", "EXPIREDDATE"],
        encoding="utf-8"
    )
    if len(sys.argv) > 2:
        # Read subsequent arguments, appending them into the same DataFrame.
        for f in sys.argv[2:]:
            new_data = pd.read_csv(
                f,
                index_col="MLSNUM",
                parse_dates=["LISTDATE", "SOLDDATE", "EXPIREDDATE"],
                encoding="utf-8"
            )
            data = data.append(new_data)
    return data


def normalize_data(data):
    # Drop all the columns that we don't want.
    data = data.drop(['Unnamed: 0', 'EXPIREDDATE', 'COOLING', 'AREA', "SHOWINGINSTRUCTIONS", "OFFICEPHONE", "STATUS", "OFFICENAME", "HOUSENUM2", "HOUSENUM1",
                      "DTO", "DOM", "JUNIORHIGHSCHOOL", "AGENTNAME", "HIGHSCHOOL", "STREETNAME", "PHOTOURL", "HIGHSCHOOL", "ELEMENTARYSCHOOL", "ADDRESS", "LISTPRICE"], 1)
    # Convert dates into number of days since the latest date.
    for x in ["LISTDATE", "SOLDDATE"]:
        data[x] = (
            data[x] - data[x].min()).astype('timedelta64[M]').astype(int)
    return data


def binarize_categorical_data(data):
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
        data = pd.concat([data, new_data], axis=1, join='inner')
        data = data.drop(var, 1)
    return data


def split_data(data):
    from sklearn.cross_validation import train_test_split
    x = data.drop('SOLDPRICE', 1)
    y = data['SOLDPRICE']
    return train_test_split(x, y, test_size=0.25)


def report_accuracy(model, data_test_x, data_test_y, name="model"):
    score = model.score(data_test_x, data_test_y)
    cross_validated_scores = cross_val_score(
        model, data_test_x, data_test_y, cv=5)
    data_predicted_y = model.predict(data_test_x)
    print "{:-^60}".format(name.upper() + " ACCURACY")
    print "MAPE: {}".format(mean_absolute_percentage_error(data_test_y, data_predicted_y))
    print "MSE Accuracy: {}".format(score)
    print "MSE Across 5 Folds: {}".format(cross_validated_scores)
    print "95%% Confidence Interval: %0.3f (+/- %0.3f)\n" % (cross_validated_scores.mean(), cross_validated_scores.std() * 1.96)
    sample_predictions(model.predict(data_test_x), data_test_y)


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true))


def sample_predictions(predicted, actual):
    sample_size = 25
    samples = np.random.randint(0, high=len(actual), size=sample_size)
    print '{:^30}'.format("Predicted"),
    print '{:^30}\n'.format("Actual")
    for sample in samples:
        print '{:^30,}'.format(int(predicted[sample])),
        print '{:^30,}'.format(int(actual.iloc[[sample]].values[0]))


class PeelRemarks:
    # Input: Full data block.
    # Fit: Pass.
    # Transform: Return only remarks column.

    def fit(self, x, y):
        return self

    def transform(self, x):
        return x['REMARKS']

class FillNaNs:
    # Input: Remarks column.
    # Fit: Pass.
    # Transform: Remarks column with all np.nan's replaced.

    def fit(self, x, y):
        return self

    def transform(self, x):
        return x.fillna('')


class ClustererWrapper:
    # Input: Full data block.
    # Fit: Train cluster alg on subset of data.
    # Transform: Predict based on subset. Append predictions to data. Return
    # data.

    def __init__(self, model):
        self.cluster_model = model

    def fit(self, x, y):
        subset = self.get_subset(x)
        self.cluster_model.fit(subset, y)
        return self

    def transform(self, x):
        subset = self.get_subset(x)
        x['cluster'] = self.cluster_model.predict(subset)
        return x

    def get_subset(self, x):
        # 'AGE'?
        return x[['lat', 'lng', 'SQFT', 'BEDS', 'BATHS', 'ZIP', 'GARAGE', 'LOTSIZE']].fillna(0)


class RegressorWrapper:
    # Input: Output from TruncatedSVD
    # Fit: Fit the internal model.
    # Transform: Return predictions only.

    def __init__(self, model):
        self.regressor_model = model

    def fit(self, x, y):
        self.regressor_model.fit(x, y)
        return self

    def transform(self, x):
        return self.regressor_model.predict(x)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        print "Error: Missing input file arguments."
