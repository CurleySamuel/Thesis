import csv
import sys
from collections import defaultdict
from os import listdir, getcwd, path
import time
from datetime import timedelta
from geopy.geocoders import ArcGIS

start = time.time()
output_directory = "geocoded_data/"

# Bring in all the known geocode data (keyed on MLS_NUM).
print "Bringing in known geocode data"
known_geocode_file = "known_geocodes/geocodes.csv"
known_geocodes = {}
with open(known_geocode_file, 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        try:
            mls = int(row["MLS_NUM"])
            lat = float(row["ADDR_LAT"])
            lng = float(row["ADDR_LON"])
        except ValueError:
            continue
        known_geocodes[mls] = {"lat": lat, "lng": lng}

for outfile in filter(lambda x: x.endswith(".csv"), listdir(getcwd() + "/" + output_directory)):
    with open(output_directory + outfile, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                mls = int(row["MLSNUM"])
                lat = float(row["lat"])
                lng = float(row["lng"])
            except ValueError:
                continue
            known_geocodes[mls] = {"lat": lat, "lng": lng}


# Bring in all the raw data (keyed on origin_file + MLS_NUM).
print "Bringing in raw data"
records = defaultdict(dict)
for input_file in sys.argv[1:]:
    try:
        with open(input_file, 'r') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            for row in reader:
                try:
                    mls = int(row["MLSNUM"])
                except ValueError:
                    continue
                records[input_file][mls] = row
    except IOError:
        print "File does not exist:", input_file
        continue


# Match data on MLS_NUM. Any matches are processed and added to final_records.
print "Doing initial matching"
final_records = defaultdict(dict)
for input_file in records.keys():
    for raw_key in records[input_file].keys():
        if raw_key in known_geocodes.keys():
            records[input_file][raw_key][
                "lat"] = known_geocodes[raw_key]["lat"]
            records[input_file][raw_key][
                "lng"] = known_geocodes[raw_key]["lng"]
            final_records[input_file][raw_key] = records[input_file][raw_key]
            records[input_file].pop(raw_key, None)


# Begin geocoding. Every 100 records purge everything to files.
def print_status(records, final_records):
    print "\033[2J -- Status -- "
    print "Time elapsed:", str(timedelta(seconds=(time.time() - start)))
    for input_file in records.keys():
        print "\n" + input_file + ":"
        print "\tCompleted:", len(final_records[input_file])
        print "\tRemaining:", len(records[input_file])
    print "\n\n"


def write_to_file(final_records):
    print "Writing results to file"
    fieldnames = next(
        final_records[final_records.keys()[0]].itervalues()).keys()
    for input_file in final_records.keys():
        input_file_name = path.basename(input_file)
        output_file = output_directory + input_file_name
        with open(output_file, 'w') as out_file:
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)
            writer.writeheader()
            for mls in final_records[input_file].keys():
                writer.writerow(final_records[input_file][mls])


geolocator = ArcGIS(timeout=None)


def geocode(record):
    if record["ADDRESS"] == "Lot C Fern Street":
        record["ADDRESS"] = "Fern Street"
    elif record["ADDRESS"] == "Lot A Fitzgerald Ave":
        record["ADDRESS"] = "Fitzgerald Ave"
    constructed_addr = "{}, {} {}".format(
        record["ADDRESS"], record["CITY"], record["STATE"])
    location = geolocator.geocode(constructed_addr)
    if location is None:
        print "Failed to geocode:", constructed_addr
        return None, None
    print constructed_addr, "\n\t-> ({}, {})".format(location.latitude, location.longitude)
    return location.latitude, location.longitude


try:
    counter = 0
    total_count = reduce(
        lambda x, y: x + y, [len(records[x]) for x in records.keys()])
    for input_file in records.keys():
        for raw_key in records[input_file].keys():
            if counter % 100 == 0:
                write_to_file(final_records)
                print_status(records, final_records)
            print "Geocoding record {} / {}".format(counter, total_count)
            lat, lng = geocode(records[input_file][raw_key])
            if lat is None:
                # Can't be geocoded
                records[input_file].pop(raw_key, None)
                counter += 1
                continue
            records[input_file][raw_key]["lat"] = lat
            records[input_file][raw_key]["lng"] = lng
            final_records[input_file][raw_key] = records[input_file][raw_key]
            records[input_file].pop(raw_key, None)
            counter += 1
            time.sleep(1.0)
except (Exception, KeyboardInterrupt) as e:
    # ANY exception we need to purge all results to file
    # before quitting.
    # This prevents doing recomputation.
    print "Encountered exception:", e
    write_to_file(final_records)
    sys.exit()

print "Geocoding complete!"
write_to_file(final_records)
print_status(records, final_records)
