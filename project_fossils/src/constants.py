import os
from pathlib import Path

# Paths
DATA_FOLDER = Path(os.path.dirname(os.path.abspath(__file__))).parent / "data"
ORIGINAL_DB_PATH = DATA_FOLDER / "now_db.csv"
COUNTRY_DATA_PATH = DATA_FOLDER / "110m_cultural.zip"

# constants
SEPARATOR = "\t"
AREA_HALF_EDGE = 5

# Common DataFrame fields
SPECIMEN_ID = "SPECIMEN_ID"
ALL_OCCURRENCES = "ALL_OCCURRENCES"
FIRST_OCCURRENCES = "FIRST_OCCURRENCES"
TIME_UNIT = "TIME_UNIT"
LIDNUM = "LIDNUM"
