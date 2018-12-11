"""
The preprocessing module is responsible of performing all pre-processing steps.

It requires the Seattle building permits dataset and the Seattle neighborhoods
geoJSON file as an input and produces the data and labels csv files which
contain the pre-processed data.

Note that the calculated neighborhoods are cached in a csv file. This file has
to be deleted when changes were applied to the module.
"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from random import random
from imblearn.over_sampling import SMOTE
from data_io import read_data, read_nhoods, write_data, store_object
from constants import *


def main():
    # Read input data
    data = read_data(PERMITS_INPUT_FILE)
    nhoods = read_nhoods(NHOODS_FILE)

    # Preprocess data
    preprocessor = Preprocessor(data, nhoods)
    preprocessor.preprocess_data()

    # Write data
    write_data(DATA_FILE, preprocessor.data)
    write_data(LABELS_FILE, preprocessor.labels)


class Preprocessor:
    """ The preprocessor class is responsible for coordinating all pre-processing tasks """

    def __init__(self, data, nhoods):
        self.data = data
        self.labels = []
        self.nhoods = nhoods

    def preprocess_data(self):
        """ Executes all pre-processing tasks """
        self.filter_data()
        self.adjust_values()
        self.set_labels()
        self.add_nhoods()
        self.filter_features()
        self.encode_categories()
        self.apply_pca()
        self.balance_data()

    def filter_data(self):
        """
        Filters irrelevant/useless data entries

        Permit applications which are on-going are filtered out as they provide
        no information about whether or not the permit was issued. In addition
        all entries with no application date are filtered out.
        """
        self.data = np.array([entry for entry in self.data if entry[FEATURES['AppliedDate']] and
                              (entry[FEATURES['StatusCurrent']] in STATUS_ISSUED or
                               entry[FEATURES['StatusCurrent']] in STATUS_NOT_ISSUED)])

    def adjust_values(self):
        """ Adjusts certain values of every permit application

        All housing units features are aggregated into one feature by adding
        them up. The application date is adjusted to contain only the year of
        the date. Additionally, it maps all possible application statuses to
        either issued or not issued.
        """
        for entry in self.data:
            entry[FEATURES['HousingUnits']] = to_float(entry[FEATURES['HousingUnitsRemoved']], 0) + \
                                              to_float(entry[FEATURES['HousingUnitsAdded']], 0)

            if entry[FEATURES['StatusCurrent']] in STATUS_ISSUED:
                entry[FEATURES['StatusCurrent']] = 1
            elif entry[FEATURES['StatusCurrent']] in STATUS_NOT_ISSUED:
                entry[FEATURES['StatusCurrent']] = 0
            else:
                raise ValueError('Invalid value for current status')

            # Only use year information from applied date
            entry[FEATURES['AppliedDate']] = entry[FEATURES['AppliedDate']][:4]

    def set_labels(self):
        """ Sets labels for every application with 0=Not Issued and 1=Issued """
        for entry in self.data:
            self.labels.append([int(entry[FEATURES['StatusCurrent']])])

    def balance_data(self):
        """ Balances dataset by under-sampling majority class and over-sampling
        minority class using SMOTE """
        num_issued = sum(x[0] for x in self.labels if x[0] == 1)
        num_not_issued = len(self.labels) - num_issued

        # Under-sample majority class until the difference is 20,000
        entries_to_delete = set()
        while len(entries_to_delete) + 20000 < abs(num_issued - num_not_issued):
            random_index = int(random() * (num_issued + num_not_issued - 1))

            if self.labels[random_index][0] == int(num_issued > num_not_issued):
                entries_to_delete.add(random_index)

        self.data = np.delete(self.data, list(entries_to_delete), axis=0)
        self.labels = np.delete(self.labels, list(entries_to_delete), axis=0)

        # Over-sample minority class
        self.data, self.labels = SMOTE().fit_resample(self.data, self.labels)
        self.labels = self.labels.reshape(len(self.labels), 1)

        # Shuffle data
        indices = np.arange(self.labels.shape[0])
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.labels = self.labels[indices]

    def add_nhoods(self):
        """ Adds neighborhood information to every application

        Note that this method can take a long time if no cached neighborhoods
        are found.
        """
        a = np.full((len(self.data), 1), 'N/A', dtype=object)
        self.data = np.append(self.data, a, axis=1)

        try:
            # Check if there are any cached neighborhoods
            neighborhoods = read_data(CACHED_NHOODS_FILE)
            for i in range(len(neighborhoods)):
                self.data[i][FEATURES['nHood']] = neighborhoods[i][0]
        except IOError:
            neighborhoods = []
            for entry in self.data:
                neighborhoods.append('N/A')
                try:
                    lat, long = float(entry[FEATURES['Latitude']]), float(entry[FEATURES['Longitude']])
                    for nhood in self.nhoods:
                        if nhood.contains(lat, long):
                            neighborhoods[-1] = nhood.name
                            entry[FEATURES['nHood']] = nhood.name
                            break
                except ValueError:
                    pass

                # Cache neighborhoods for next run
                neighborhoods = np.array(neighborhoods)
                neighborhoods = neighborhoods.reshape(len(neighborhoods), 1)
                write_data(CACHED_NHOODS_FILE, neighborhoods)

    def filter_features(self):
        """ Perform feature selection """
        self.data = self.data[:, [FEATURES[x] for x in FILTERED_FEATURES]]

    def encode_categories(self):
        """ Encode categories using One-Hot encoding and store encoder """
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(self.data[:, CATEGORICAL_FEATURES])

        self.data = np.concatenate((encoder.transform(self.data[:, CATEGORICAL_FEATURES]).toarray(),
                                    self.data[:, NUMERIC_FEATURES].astype(float)), axis=1)

        store_object(encoder, ENCODER_PICKLE_FILE)

    def apply_pca(self):
        """ Applies and stores PCA to the dataset with 10 principle components """
        pca = PCA(n_components=10)
        pca.fit(self.data)
        self.data = pca.transform(self.data)

        store_object(pca, PCA_PICKLE_FILE)


def to_float(n, default):
    """ Casts value to float if possible and returns default otherwise """
    try:
        return float(n)
    except ValueError:
        return default


if __name__ == '__main__':
    main()
