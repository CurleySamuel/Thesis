from datetime import datetime
import csv
import pickle
import sys
from geopy.geocoders import Nominatim
from time import sleep

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

    # -- Calculated metrics --

    lat = None
    lng = None
    full_address = None
    bounding_box = None
    housing_type = None

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

    def __init__(self, input_dict):
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

        # Set all member variables
        member_vars = [attr for attr in dir(self) if not callable(
            getattr(self, attr)) and not attr.startswith("__") and getattr(self, attr) is not None]
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
                    print "Failed int conversion. Variable: {}, Value: {}".format(var, getattr(self, var))
        for var in ["baths"]:
            # Convert appropriate variables to floats
            try:
                setattr(self, var, float(getattr(self, var)))
            except Exception:
                print "Failed float conversion. Variable: {}, Value: {}".format(var, getattr(self, var))
        for var in ["basement"]:
            # Convert appropriate variables to booleans
            try:
                setattr(self, var, getattr(self, var) == "Yes")
            except Exception:
                print "Failed boolean conversion. Variable: {}, Value: {}".format(var, getattr(self, var))
        for var in ["sold_date", "expire_date", "list_date"]:
            try:
                setattr(
                    self, var, datetime.strptime(getattr(self, var), "%m/%d/%Y"))
            except Exception:
                if var == "expire_date" and getattr(self, var) == "":
                    setattr(self, var, datetime(2016, 1, 1))
                else:
                    print "Failed datetime conversion. Variable: {}, Value: {}".format(var, getattr(self, var))

    def to_dict(self):
        temp = {}
        field_names = [attr for attr in dir(self) if not callable(
            getattr(self, attr)) and not attr.startswith("__")]
        for var in field_names:
            temp[var] = getattr(self, var)
        return temp


class Records:

    records = {}
    geolocator = Nominatim()

    def __init__(self, store_file="data.dat"):
        try:
            with open(store_file, 'rb') as f:
                self.records = pickle.load(f)
                print "{} records imported from store file '{}'.".format(len(self.records), store_file)
        except IOError:
            print "No store file found."

    def import_from_file(self, input_file):
        try:
            with open(input_file) as f:
                print "Importing records from file '{}'".format(input_file)
                old_record_count = len(self.records)
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    new_record = Record(row)
                    self.records[new_record.mls_num] = new_record
                new_record_count = len(self.records) - old_record_count
                print "\tImported {} new records.".format(new_record_count)
                self._get_existing_geocoding()
                self._geocode_new_records()
        except IOError:
            print "File '{}' does not exist.".format(input_file)

    def _get_existing_geocoding(self):
        file_name = "property_geocodes.csv"
        try:
            with open(file_name, 'r') as f:
                reader = csv.DictReader(f)
                counter = 0
                for row in reader:
                    mls = int(row["MLS_NUM"])
                    lat = float(row["ADDR_LAT"])
                    lng = float(row["ADDR_LON"])
                    if mls in self.records.keys() and self.records[mls].lat is None:
                        self.records[mls].lat = lat
                        self.records[mls].lng = lng
        except IOError:
            print "File '{}' does not exist.".format(file_name)
        except ValueError as e:
            print "Bad input data:", e

    def _geocode_new_records(self):
        records_to_process = [
            mls_num for mls_num in self.records.keys() if self.records[mls_num].lat is None]
        print "\tGeocoding {} new records.".format(len(records_to_process))
        try:
            for itr in range(len(records_to_process)):
                print "\t\t{} / {}".format(itr, len(records_to_process))
                sys.stdout.write("\033[F")
                mls_num = records_to_process[itr]
                """
                constructed_addr = "{}, {} {}".format(
                    self.records[mls_num].address, self.records[mls_num].city, self.records[mls_num].state)
                location = self.geolocator.geocode(constructed_addr)
                try:
                    self.records[mls_num].lat = location.latitude
                    self.records[mls_num].lng = location.longitude
                    self.records[mls_num].full_address = location.address
                    self.records[mls_num].bounding_box = [
                        float(elem) for elem in location.raw["boundingbox"]]
                    self.records[mls_num].housing_type = location.raw["type"]
                except Exception:
                    import ipdb
                    ipdb.set_trace()
                """
        except Exception as e:
            print "Geocoding hit the exception '{}'\nSaving records and exiting.".format(e)
            self.export_to_file()
            sys.exit()

    def export_to_file(self, store_file="data.dat"):
        try:
            with open(store_file, 'wb') as f:
                pickle.dump(self.records, f)
                print "{} records written to store file '{}'.".format(len(self.records), store_file)
        except IOError:
            if store_file != "backup_data.dat":
                print "Failed to write to store file '{}'. Attempting write to backup location."
                self.export_to_file(store_file="backup_data.dat")
            else:
                print "Failed to write to backup location."
                import ipdb
                ipdb.set_trace()

    def close(self):
        self.export_to_file()
