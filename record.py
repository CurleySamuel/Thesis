from termcolor import colored
import csv


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
    hot_water = "Hot water"
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
    expire_date = "EXPIREDATE"
    # datetime
    list_date = "LISTDATE"
    # int
    mls_num = "MLSNUM"
    # string
    state = "STATE"
    # int
    zip_code = "ZIP"

    def import_database_row(self, input_dict):
        pass

    def import_file_row(self, input_dict):
        pass


class Records:

    records = []

    def __init__(self, database_file="database.json"):
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
            print colored("No such file exists", "red")

    def export_to_file(self, overwrite=False, database_file="database.json"):
        pass

    def close(self):
        self.export_to_file()
