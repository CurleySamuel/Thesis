import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

def main():
    data = get_data()
    data = normalize_data(data)
    data, data_original = dummy_data(data)
    data_train_x, data_test_x, data_train_y, data_test_y = split_data(data)
    print "Size of training set: {}\nSize of testing set: {}".format(len(data_train_y), len(data_test_y))


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
                      "OFFICENAME", "HOUSENUM2", "HOUSENUM1", "DTO", "DOM", "JUNIORHIGHSCHOOL", "AGENTNAME", "HIGHSCHOOL", "STREETNAME", "PHOTOURL", "HIGHSCHOOL", "ELEMENTARYSCHOOL"], 1)

    # If missing data on number of baths, set it to number of beds / 2.
    data.loc[data['BATHS'].isnull(), 'BATHS'] = data['BEDS'] / 2

    # Convert dates into number of days since the latest date.
    for x in ["LISTDATE", "SOLDDATE"]:
        data[x] = (
            data[x] - data[x].min()).astype('timedelta64[M]').astype(int)
    return data


def dummy_data(data):
    old_data = data.copy()
    # Take these unstandardize fields and create 'dummy columns' from them
    # which have a 1 or 0 for each row. The number of dummy columns is equal
    # to the number of distinct possible answers for each column.
    #
    # i.e. PROPTYPE will get split up into (PROPTYPE) SF and (PROPTYPE) MF
    for var in ["PROPTYPE", "STYLE", "HEATING", "CITY", "LEVEL"]:
        if var != "LEVEL":
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
        data.drop(var, 1)
    return data, old_data



def split_data(data):
    from sklearn.cross_validation import train_test_split
    x = data.drop('SOLDPRICE', 1)
    y = data['SOLDPRICE']
    return train_test_split(x, y, test_size=0.25)



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
