"""
The constants module contains all constants which are used by the other modules
"""

# Dict of all available features and their corresponding index
FEATURES = {
    'PermitNum': 0,
    'PermitClass': 1,
    'PermitClassMapped': 2,
    'PermitTypeMapped': 3,
    'PermitTypeDesc': 4,
    'Description': 5,
    'HousingUnits': 6,
    'HousingUnitsRemoved': 7,
    'HousingUnitsAdded': 8,
    'EstProjectCost': 9,
    'AppliedDate': 10,
    'IssuedDate': 11,
    'ExpiresDate': 12,
    'CompletedDate': 13,
    'StatusCurrent': 14,
    'OriginalAddress1': 15,
    'OriginalCity': 16,
    'OriginalState': 17,
    'OriginalZip': 18,
    'ContractorCompanyName': 19,
    'Link': 20,
    'Latitude': 21,
    'Longitude': 22,
    'Location_1': 23,
    'nHood': 24
}

# List of all filtered feature names
FILTERED_FEATURES = [
    'PermitClass',
    'PermitClassMapped',
    'PermitTypeMapped',
    'HousingUnits',
    'AppliedDate',
    'nHood',
]

# Features which have categorical values
CATEGORICAL_FEATURES = [0, 1, 2, 5]

# Features which have numeric values
NUMERIC_FEATURES = [3, 4]

# Statuses which are interpreted as issued
STATUS_ISSUED = ['Completed', 'Issued']

# Statuses which are interpreted as not issued
STATUS_NOT_ISSUED = ['Canceled', 'Revoked', 'Expired']

# Folder where all data is stored
DATA_FOLDER = './data/'

# Seattle building permits dataset file
PERMITS_INPUT_FILE = DATA_FOLDER + 'building_permits.csv'

# Seattle neighborhoods geoJSON file
NHOODS_FILE = DATA_FOLDER + 'neighborhoods.geojson'

# Preprocessed data file
DATA_FILE = DATA_FOLDER + 'data.csv'

# Preprocessed labels file
LABELS_FILE = DATA_FOLDER + 'labels.csv'

# Cached neighborhoods for next run of pre-processing
CACHED_NHOODS_FILE = DATA_FOLDER + 'cached_nhoods.csv'

# Folder where all pickle files are stored
PICKLE_FOLDER = './pickles/'

# File where encoder object is stored
ENCODER_PICKLE_FILE = PICKLE_FOLDER + 'encoder.pickle'

# File where pca object is stored
PCA_PICKLE_FILE = PICKLE_FOLDER + 'pca.pickle'

# File where trained model is stored
MODEL_PICKLE_FILE = PICKLE_FOLDER + 'clf.pickle'
