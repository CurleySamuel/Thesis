from termcolor import colored
from datetime import datetime
import csv

unwanted_fields = ["''", "SHOWINGINSTRUCTIONS", "OFFICEPHONE", "STATUS", "AGENTNAME",
                   "EXPIREDATE", "PHOTOURL", "OFFICENAME", "HOUSENUM1", "HOUSENUM2", "STREETNAME"]


class Record:

    # -- Predictive metrics --
    # int
    age = "AGE"
    # string ("Range, Dishwasher, Disposal")
    appliances = "Appliances"
    # boolean
    basement = "Basement"
    # float
    baths = "BATHS"
    # int
    beds = "BEDS"
    # string
    city = "CITY"
    # string ("Frame, Block")
    construction = "Construction"
    # string
    cooling = "COOLING"
    # int
    days_on_market = "DOM"
    # int
    days_til_offer = "DTO"
    # string ("Fuses, 60 Amps/Less")
    electric = "Electric"
    # string ("Bowman")
    element_school = "ELEMENTARYSCHOOL"
    # string ("Vinyl, Brick")
    exterior = "Exterior"
    # string ("--")
    exterior_feats = "Exterior Features"
    # int
    fireplaces = "Fireplaces"
    # string ("Wood, Concrete")
    floor = "Floor"
    # string ("Poured concrete")
    foundation = "Foundation"
    # int
    garage = "GARAGE"
    # string ("Hot water baseboard, Oil")
    heating = "HEATING"
    # string ("Lexington")
    high_school = "HIGHSCHOOL"
    # string ("Tankless")
    hot_water = "Hot Water"
    # string ("--")
    insulation = "Insulation"
    # string ("Security System")
    interior_feats = "Interior Features"
    # string ("Clarke")
    junior_high = "JUNIORHIGHSCHOOL"
    # int
    level = "LEVEL"
    # int
    list_price = "LISTPRICE"
    # int
    lot_size = "LOTSIZE"
    # string ("SF")
    prop_type = "PROPTYPE"
    # string ("Nestled on a quite country lane...")
    remarks = "REMARKS"
    # string ("Asphalt/Fiberglass Shingles")
    roof = "Roof"
    # datetime
    sold_date = "SOLDDATE"
    # int
    sold_price = "SOLDPRICE"
    # string ("Ranch")
    style = "STYLE"
    # int
    sqft = "SQFT"

    # -- Non-predictive metrics --
    # string
    address = "ADDRESS"
    # datetime
    expire_date = "EXPIREDDATE"
    # datetime
    list_date = "LISTDATE"
    # int
    mls_num = "MLSNUM"
    # string
    state = "STATE"
    # int
    zip_code = "ZIP"

    def import_database_row(self, input_dict):
        member_vars = [attr for attr in dir(self) if not callable(
            getattr(self, attr)) and not attr.startswith("__")]
        for var in member_vars:
            setattr(self, var, input_dict[var])
        for var in ["age", "beds", "days_on_market", "days_til_offer", "fireplaces", "garage", "level", "list_price", "lot_size", "sold_price", "sqft", "mls_num", "zip_code"]:
            setattr(self, var, int(float(getattr(self, var))))
        for var in ["baths"]:
            setattr(self, var, float(getattr(self, var)))
        for var in ["basement"]:
            setattr(self, var, getattr(self, var) == "True")
        for var in ["sold_date", "expire_date", "list_date"]:
            setattr(self, var, datetime.strptime(
                getattr(self, var), "%Y-%m-%d %H:%M:%S"))
        return self

    def import_file_row(self, input_dict):
        # Purge unwanted fields
        for field in unwanted_fields:
            input_dict.pop(field, None)

        # Pull secondary "other features" into primary fields
        other = input_dict.pop("OTHERFEATURES", None)
        if other is not None:
            other = other.split(';')
            for field in other:
                try:
                    split = field.split(':')
                    input_dict[split[0]] = split[1]
                except IndexError:
                    pass

        # Geocode the address

        # Set all member variables
        member_vars = [attr for attr in dir(self) if not callable(
            getattr(self, attr)) and not attr.startswith("__")]
        for var in member_vars:
            setattr(self, var, input_dict[getattr(self, var)])
        for var in ["age", "beds", "days_on_market", "days_til_offer", "fireplaces", "garage", "level", "list_price", "lot_size", "sold_price", "sqft", "mls_num", "zip_code"]:
            # Convert appropriate variables to ints
            try:
                setattr(self, var, int(float(getattr(self, var))))
            except Exception:
                if var == "level":
                    setattr(self, var, 1)
                else:
                    print colored("Failed int conversion. Variable: {}, Value: {}".format(var, getattr(self, var)), "red")
        for var in ["baths"]:
            # Convert appropriate variables to floats
            try:
                setattr(self, var, float(getattr(self, var)))
            except Exception:
                print colored("Failed float conversion. Variable: {}, Value: {}".format(var, getattr(self, var)), "red")
        for var in ["basement"]:
            # Convert appropriate variables to booleans
            try:
                setattr(self, var, getattr(self, var) == "Yes")
            except Exception:
                print colored("Failed boolean conversion. Variable: {}, Value: {}".format(var, getattr(self, var)), "red")
        for var in ["sold_date", "expire_date", "list_date"]:
            try:
                setattr(
                    self, var, datetime.strptime(getattr(self, var), "%m/%d/%Y"))
            except Exception:
                if var == "expire_date" and getattr(self, var) == "":
                    setattr(self, var, datetime(2016, 1, 1))
                else:
                    print colored("Failed datetime conversion. Variable: {}, Value: {}".format(var, getattr(self, var)), "red")
        # Return Record object
        return self

    def to_dict(self):
        temp = {}
        field_names = [attr for attr in dir(self) if not callable(
            getattr(self, attr)) and not attr.startswith("__")]
        for var in field_names:
            temp[var] = getattr(self, var)
        return temp


class Records:

    records = []

    def __init__(self, database_file="database.csv"):
        try:
            with open(database_file) as f:
                print colored("Discovered database file", "green")
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    self.records.append(Record().import_database_row(row))
                print colored("\tSuccessfully imported {} records".format(len(self.records)), "green")
        except IOError:
            print colored("No database file found", "red")

    def import_from_file(self, input_file):
        try:
            with open(input_file) as f:
                print colored("Importing records from file {}".format(input_file), "green")
                old_record_count = len(self.records)
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    print colored("\tProcessing record {}".format(len(self.records) - old_record_count), "green") + "\r",
                    self.records.append(Record().import_file_row(row))
                print colored("Successfully processed {} items.".format(len(self.records) - old_record_count, "green"))
        except IOError:
            print colored("File {} does not exist.".format(input_file), "red")

    def export_to_file(self, database_file="database.csv"):
        print colored("Writing {} records to database.".format(len(self.records)), "green")
        fieldnames = [attr for attr in dir(Record) if not callable(
            getattr(Record, attr)) and not attr.startswith("__")]
        with open(database_file, 'w') as db:
            writer = csv.DictWriter(db, fieldnames=fieldnames, delimiter='\t')
            writer.writeheader()
            for record in self.records:
                writer.writerow(record.to_dict())
        print colored("Records written.", "green")

    def close(self):
        self.export_to_file()
