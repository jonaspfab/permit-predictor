"""
The training module is responsible for training and evaluating the performance
of several classifiers.
"""

from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from data_io import read_data, store_object
from constants import DATA_FILE, LABELS_FILE, MODEL_PICKLE_FILE


# List of all classifiers which are trained
clfs = [
    tree.DecisionTreeClassifier(criterion='entropy'),
    LogisticRegression(solver='lbfgs'),
    KNeighborsClassifier(n_neighbors=5),
    SVC(kernel='rbf', gamma='scale'),
    MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10, 8, 5, 2), random_state=1)
]


def main():
    # Read input data
    data = DataPartition(read_data(DATA_FILE, to_float=True), portion_test=0.1)
    labels = DataPartition(read_data(LABELS_FILE, to_float=True, flatten=True), portion_test=0.1)

    best_f_score = 0
    best_clf = None
    for clf in clfs:
        # Train classifier
        clf.fit(data.training, labels.training)

        # Evaluate classifier
        evaluator = ModelEvaluator(clf)
        evaluator.evaluate_model(data.test, labels.test)
        print(evaluator.get_report())

        # Update best performing classifier
        if evaluator.f_score > best_f_score:
            best_clf = clf

    # Store clf to be used by permit predictor
    store_object(best_clf, MODEL_PICKLE_FILE)


class DataPartition:
    """ Stores partition of data into training and test """

    def __init__(self, data, portion_test):
        split = int(len(data) * portion_test)
        self.training = data[:-split]
        self.test = data[-split:]


class ModelEvaluator:
    """ Evaluates performance of the classifier """

    def __init__(self, clf):
        self.clf = clf
        self.accuracy = 0
        self.confusion_matrix = []
        self.precision = 0
        self.recall = 0
        self.f_score = 0

    def evaluate_model(self, data, labels):
        """ Evaluates performance by predicting labels with given data """
        predicted_labels = self.clf.predict(data)
        self.set_accuracy(predicted_labels, labels)
        self.set_confusion_matrix(predicted_labels, labels)
        self.set_precision()
        self.set_recall()
        self.set_f_score()

    def set_accuracy(self, predicted_labels, true_labels):
        """ Calculates accuracy of the classifier """
        n = len(true_labels)
        self.accuracy = sum(1 for i in range(n) if predicted_labels[i] == true_labels[i]) / n

    def set_confusion_matrix(self, predicted_labels, true_labels):
        """ Calculates confusion matrix for the classifier """
        m = [[0] * 2 for _ in range(2)]
        for i in range(len(predicted_labels)):
            m[int(true_labels[i])][int(predicted_labels[i])] += 1

        self.confusion_matrix = m

    def set_precision(self):
        """ Calculates precision of the classifier """
        self.precision = self.confusion_matrix[1][1] / (self.confusion_matrix[1][1] + self.confusion_matrix[0][1])

    def set_recall(self):
        """ Calculates recall of the classifier """
        self.recall = self.confusion_matrix[1][1] / (self.confusion_matrix[1][1] + self.confusion_matrix[1][0])

    def set_f_score(self):
        """ Calculates F-Score of the classifier """
        self.f_score = (2 * self.precision * self.recall) / (self.precision + self.recall)

    def get_report(self):
        """ Creates report for the performance of the classifier """
        report = '\n---------------------------------------'
        report += '\nReport for ' + str(type(self.clf).__name__)
        report += '\nAccuracy = ' + str(self.accuracy)
        report += '\nConfusion Matrix: \n' + '\n'.join([str(j) for j in self.confusion_matrix])
        report += '\nPrecision = ' + str(self.precision)
        report += '\nRecall = ' + str(self.recall)
        report += '\nF-Score = ' + str(self.f_score)
        report += '\n---------------------------------------\n'

        return report


if __name__ == '__main__':
    main()
