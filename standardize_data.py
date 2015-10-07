import os
import sys
import record
from termcolor import colored


def main():
    records = record.Records()
    for f in sys.argv[1:]:
        records.import_from_file(f)
    records.close()


if __name__ == "__main__":
    main()
