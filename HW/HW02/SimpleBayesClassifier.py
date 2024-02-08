import random as rnd
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


class SimpleBayesClassifier:

    def __init__(self, n_pos, n_neg):
        """
        Initializes the SimpleBayesClassifier with prior probabilities.

        Parameters:
        n_pos (int): The number of positive samples.
        n_neg (int): The number of negative samples.

        Returns:
        None: This method does not return anything as it is a constructor.
        """

        self.n_pos = n_pos
        self.n_neg = n_neg
        self.prior_pos = n_pos / (n_pos + n_neg)
        self.prior_neg = n_neg / (n_pos + n_neg)

    def fit_params(self, x, y, n_bins=10):
        """
        Computes histogram-based parameters for each feature in the dataset.

        Parameters:
        x (np.ndarray): The feature matrix, where rows are samples and columns are features.
        y (np.ndarray): The target array, where each element corresponds to the label of a sample.
        n_bins (int): Number of bins to use for histogram calculation.

        Returns:
        (stay_params, leave_params): A tuple containing two lists of tuples,
        one for 'stay' parameters and one for 'leave' parameters.
        Each tuple in the list contains the bins and edges of the histogram for a feature.
        """

        self.stay_params = [(None, None) for _ in range(x.shape[1])]
        self.leave_params = [(None, None) for _ in range(x.shape[1])]

        # INSERT CODE HERE
        # fit by count each bin for each feature
        # tuple of (bins, edges) for each feature
        # print(x[y == 0, 0].shape[0] == x[(np.isnan(x[:, 0])) & (y == 0), 0].shape[0] + x[(~np.isnan(x[:, 0])) & (y == 0), 0].shape[0])

        self.stay_params = [
            np.histogram(
                x[(~np.isnan(x[:, i])) & (y == 0), i], bins=n_bins, density=True
            )
            for i in range(x.shape[1])
        ]
        self.leave_params = [
            np.histogram(
                x[(~np.isnan(x[:, i])) & (y == 1), i], bins=n_bins, density=True
            )
            for i in range(x.shape[1])
        ]

        return self.stay_params, self.leave_params

    def predict(self, x, thresh=0):
        """
        Predicts the class labels for the given samples using the non-parametric model.

        Parameters:
        x (np.ndarray): The feature matrix for which predictions are to be made.
        thresh (float): The threshold for log probability to decide between classes.

        Returns:
        result (list): A list of predicted class labels (0 or 1) for each sample in the feature matrix.
        """

        y_pred = []

        # INSERT CODE HERE
        # predict by add new sample to the bin and calcualte log probability
        """
        TODO: 
        - 0 probability -> floor to small value (1e-20)
        - MAP (alpha) + (1-alpha)
        - add one smoothing
        """
        for i in range(x.shape[0]):
            pos_log_prob = np.log(self.prior_pos)
            neg_log_prob = np.log(self.prior_neg)
            for j in range(x.shape[1]):
                if not np.isnan(x[i, j]):
                    """
                    print(x[i, j])
                    print(self.stay_params[j][0])
                    print(self.stay_params[j][1])
                    print(np.digitize(x[i, j], self.stay_params[j][1]))
                    """

                    if (
                        np.digitize(x[i, j], self.stay_params[j][1])
                        > self.stay_params[j][0].shape[0]
                        or np.digitize(x[i, j], self.leave_params[j][1])
                        > self.leave_params[j][0].shape[0]
                    ):
                        continue

                    else:
                        pos_log_prob += np.log(
                            self.leave_params[j][0][
                                np.digitize(x[i, j], self.leave_params[j][1]) - 1
                            ]
                        )
                        neg_log_prob += np.log(
                            self.stay_params[j][0][
                                np.digitize(x[i, j], self.stay_params[j][1]) - 1
                            ]
                        )
            y_pred.append(1 if pos_log_prob - neg_log_prob > thresh else 0)

        return y_pred

    def fit_gaussian_params(self, x, y):
        """
        Computes mean and standard deviation for each feature in the dataset.

        Parameters:
        x (np.ndarray): The feature matrix, where rows are samples and columns are features.
        y (np.ndarray): The target array, where each element corresponds to the label of a sample.

        Returns:
        (gaussian_stay_params, gaussian_leave_params): A tuple containing two lists of tuples,
        one for 'stay' parameters and one for 'leave' parameters.
        Each tuple in the list contains the mean and standard deviation for a feature.
        """

        self.gaussian_stay_params = [(0, 0) for _ in range(x.shape[1])]
        self.gaussian_leave_params = [(0, 0) for _ in range(x.shape[1])]

        # INSERT CODE HERE
        self.gaussian_stay_params = [
            (
                np.mean(x[(~np.isnan(x[:, i])) & (y == 0), i]),
                np.std(x[(~np.isnan(x[:, i])) & (y == 0), i]),
            )
            for i in range(x.shape[1])
        ]
        self.gaussian_leave_params = [
            (
                np.mean(x[(~np.isnan(x[:, i])) & (y == 1), i]),
                np.std(x[(~np.isnan(x[:, i])) & (y == 1), i]),
            )
            for i in range(x.shape[1])
        ]

        return self.gaussian_stay_params, self.gaussian_leave_params

    def gaussian_predict(self, x, thresh=0):
        """
        Predicts the class labels for the given samples using the parametric model.

        Parameters:
        x (np.ndarray): The feature matrix for which predictions are to be made.
        thresh (float): The threshold for log probability to decide between classes.

        Returns:
        result (list): A list of predicted class labels (0 or 1) for each sample in the feature matrix.
        """

        y_pred = []

        # INSERT CODE HERE
        for i in range(x.shape[0]):
            pos_log_prob = np.log(self.prior_pos)
            neg_log_prob = np.log(self.prior_neg)
            for j in range(x.shape[1]):
                if not np.isnan(x[i, j]):
                    pos_log_prob += np.log(
                        stats.norm.pdf(
                            x[i, j],
                            self.gaussian_leave_params[j][0],
                            self.gaussian_leave_params[j][1],
                        )
                    )
                    neg_log_prob += np.log(
                        stats.norm.pdf(
                            x[i, j],
                            self.gaussian_stay_params[j][0],
                            self.gaussian_stay_params[j][1],
                        )
                    )
            y_pred.append(1 if pos_log_prob - neg_log_prob > thresh else 0)

        return y_pred
