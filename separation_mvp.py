from scipy.optimize import minimize
from sklearn.metrics import roc_curve
import numpy as np

from utils import check_lengths, check_binaries


class SeparatedClassifier:
    def __init__(self, y_train, R_train, A_train):
        check_lengths(y_train, R_train, A_train)
        check_binaries(y_train)
        self.y_train = np.array(y_train)
        self.R_train = np.array(R_train)
        self.A_train = np.array(A_train)

    def fit(self, goal_fpr, goal_tpr):
        groups = np.unique(self.A_train)
        self.randomized_thresholds = dict()
        for g in groups:
            y_true_g = self.y_train[self.A_train == g]
            R_g = self.R_train[self.A_train == g]
            self.randomized_thresholds[g] = self._find_thresholds_and_probas(
                y_true_g, R_g, goal_fpr, goal_tpr
            )

    def fair_predict(self, R_test, A_test):
        groups = np.unique(A_test)
        y_pred = []
        for r, g in zip(R_test, A_test):
            *a, p = self.randomized_thresholds[g]
            t = np.random.choice(a, p=[1 - p, p])
            y_pred.append(1 if r >= t else 0)

        return np.array(y_pred)

    def _find_thresholds_and_probas(self, y_true, scores, goal_fpr, goal_tpr):
        fpr, tpr, thresholds = roc_curve(y_true, scores, drop_intermediate=False)

        def distance(d3array):
            t0, t1, p = d3array
            i = np.argmin(abs(t0 - thresholds))
            j = np.argmin(abs(t1 - thresholds))
            x = fpr[i] + p * (fpr[j] - fpr[i])
            y = tpr[i] + p * (tpr[j] - tpr[i])
            return np.hypot(x - goal_fpr, y - goal_tpr)

        initial_guess = np.array([0.3, 0.5, 0.5])
        res = minimize(
            distance, initial_guess, method="nelder-mead", options={"xtol": 1e-10}
        )
        t0, t1, p = res["x"]
        return t0, t1, p
