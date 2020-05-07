"""

The data_io module is responsible for all reading and writing to the file
system 
"""

import json
import csv
import numpy as np
import pickle


def store_object(obj, file_name):
    """ Stores object as a pickle file"""
    with open(file_name, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def restore_object(file_name):
    """ Restores object from pickle file"""
    with open(file_name, 'rb') as handle:
        return pickle.load(handle)


def read_data(file_name, flatten=False, to_float=False):
    """ Reads data from input file to numpy array

    :param file_name: Input file name
    :param flatten: Array is flattened if True (Default is False)
    :param to_float: Casts values to float if True (Default is False)
    """
    with open(file_name) as file:
        if to_float:
            res = np.array(list(csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)))
        else:
            res = np.array(list(csv.reader(file, delimiter=',')))

        return res.flatten() if flatten else res


def read_nhoods(file_name):
    """ Reads NHood objects from geoJSON input file """
    with open(file_name) as file:
        data = json.load(file)
        nhoods = []
        for feature in data['features']:
            name = feature['properties']['nhood']
            coordinates = feature['geometry']['coordinates']
            if coordinates and len(coordinates[0]) > 2:
                nhoods.append(NHood(name, [(lat, long) for long, lat in coordinates[0]]))

        return nhoods


def write_data(file_name, data, append=False):
    """ Writes given data to file """
    with open(file_name, mode='a' if append else 'w') as file:
        csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for entry in data:
            csv_writer.writerow(entry)


class NHood:
    """ Represents a neighborhood """

    def __init__(self, name, coordinates):
        self.name = name
        self.coordinates = coordinates

    def contains(self, x, y):
        """Return True if (x, y) lies within self.coordinates and False otherwise."""

        n = len(self.coordinates)
        inside = False

        p1x, p1y = self.coordinates[0]
        for i in range(n + 1):
            p2x, p2y = self.coordinates[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xints = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside
