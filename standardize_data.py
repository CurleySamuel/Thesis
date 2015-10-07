import csv
import os
from termcolor import colored

unwanted_fields = ["''", "SHOWINGINSTRUCTIONS", "OFFICEPHONE", "STATUS", "AGENTNAME",
                   "EXPIREDATE", "PHOTOURL", "OFFICENAME", "HOUSENUM1", "HOUSENUM2", "STREETNAME"]


def main():
    files = filter(lambda x: x.endswith(".csv"), os.listdir(os.getcwd()))
    records = []
    if "cleaned.csv" in files:
        print colored("Detected preprocessed data. Importing.", "green")
        with open("cleaned.csv") as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row in reader:
                records.append(row)
    else:
        print colored("No preprocessed data found. Importing.", "green")
        for in_file in files:
            print colored("\tImporting " + in_file, "yellow")
            with open(in_file) as csvfile:
                reader = csv.DictReader(csvfile, delimiter='\t')
                for row in reader:
                    records.append(row)
        print colored(str(len(records)) + " records imported. Processing.", "green")
        print colored("\tPurging unwanted fields", "yellow")
        for record in records:
            for field in unwanted_fields:
                record.pop(field, None)
        print colored("\tSeparating OTHERFEATURES", "yellow")
        for record in records:
            other = record.pop("OTHERFEATURES", None)
            if other is None:
                pass
            other = other.split(';')
            for field in other:
                try:
                    split = field.split(':')
                    record[split[0]] = split[1]
                except IndexError:
                    pass
        print colored("\tGeocoding records", "yellow")
        for x in range(len(records)):
            print colored("\t\t{}/{} complete".format(x, len(records)), "yellow") + "\r",

    import ipdb
    ipdb.set_trace()


if __name__ == "__main__":
    main()
