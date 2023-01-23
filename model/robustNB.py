import numpy as np
from scipy.special import xlogy
from sklearn.base import BaseEstimator


class RobustNaiveBayesClassifierPercentage(BaseEstimator):

    def __init__(self, percentage_noise=0.0, debug=False):

        # assert 0 <= percentage_noise <= 100, "Please enter a valid percentage between 0 and 100 inclusive"

        self.percentage_noise = percentage_noise

        self.kappa = None

        self.pos_prior_probability = None
        self.neg_prior_probability = None
        self.theta_pos = None
        self.theta_neg = None
        self.f_pos = None
        self.f_neg = None
        self.pos_indices = None
        self.neg_indices = None

        self._has_fit = False

        self.debug = debug

    def _solve_subproblem(self, f, kappa):
        '''
        Solves the subproblem:
        max_{v > 0} F(v) = kappa * log v + \sum f_i log max(f_i, v) - max(f_i, v) 

        where kappa is self.percentage_noise * sum of word counts in text corpus training
        Returns the optimal value of the dual variable theta*
        '''
        if self.debug:
            print("Internal kappa percentage: " + str(kappa))

        if (len(f.shape) == 2):
            n = f.shape[1]
        else:
            n = f.shape[0]

        f_l1 = np.sum(f)

        f_int = (-np.sort(-f))
        f_int = np.insert(f_int, 0, 9999999)
        f_new = np.insert(f_int, n+1, 0.)

        if (len(f_new.shape) < 2):
            f_new = np.reshape(f_new, (1, f_new.shape[0]))

        rho_curr = kappa + f_l1
        h_curr = 0
        v_curr = max(f_new[0, 1], rho_curr/n)
        F_curr = rho_curr * np.log(v_curr) - n * v_curr

        v_star = v_curr
        F_star = F_curr
        k_star = 0

        for k in range(1, n + 1):
            if k == n:
                h_curr = h_curr + f_new[0, k] * \
                    np.log(f_new[0, k]) - f_new[0, k]
                F_curr = h_curr + kappa * np.log(f_new[0, k])
                v_curr = f_new[0, k]
            else:
                rho_curr = rho_curr - f_new[0, k]
                h_curr = h_curr + f_new[0, k] * \
                    np.log(f_new[0, k]) - f_new[0, k]
                v_curr = min(f_new[0, k], max(
                    f_new[0, k + 1], rho_curr/(n - k)))
                F_curr = h_curr + rho_curr * np.log(v_curr) - (n - k) * v_curr

            if F_curr > F_star:
                F_star = F_curr
                v_star = v_curr
                k_star = k

        theta = (1./(kappa + f_l1)) * np.maximum(f, v_star * np.ones(f.shape))

        return theta

    def fit(self, X, y):
        self.pos_indices = np.where(y == 1)[0]
        self.neg_indices = np.where(y == 0)[0]
        self.f_pos = np.sum(X[self.pos_indices], axis=0)
        self.f_neg = np.sum(X[self.neg_indices], axis=0)

        self.f_pos = 1 + self.f_pos
        self.f_neg = 1 + self.f_neg

        self.kappa_pos = (self.percentage_noise/100) * np.sum(self.f_pos)
        self.kappa_neg = (self.percentage_noise/100) * np.sum(self.f_neg)

        self.theta_pos = self._solve_subproblem(self.f_pos, self.kappa_pos)
        self.theta_neg = self._solve_subproblem(self.f_neg, self.kappa_neg)

        if (len(self.theta_pos.shape) < 2):
            self.theta_pos = np.reshape(
                self.theta_pos, (1, self.theta_pos.shape[0]))

        if (len(self.theta_neg.shape) < 2):
            self.theta_neg = np.reshape(
                self.theta_neg, (1, self.theta_neg.shape[0]))

        self.pos_prior_probability = len(self.pos_indices)/X.shape[0]
        self.neg_prior_probability = len(self.neg_indices)/X.shape[0]

        self._has_fit = True

    def predict(self, X):
        if not self._has_fit:
            print("Please call fit() before you start predicting")
            return None

        if self.theta_pos.shape[1] != X.shape[1]:
            print("Shape mismatch. Please train with proper dimensions")
            return None
        pos_prob = np.log(self.pos_prior_probability) + \
            np.sum(X@np.log(self.theta_pos).T, axis=-1)
        neg_prob = np.log(self.neg_prior_probability) + \
            np.sum(X@np.log(self.theta_neg).T, axis=-1)

        predictions = (pos_prob >= neg_prob).astype(int)

        return predictions

    def predict_proba(self, X):
        if not self._has_fit:
            print("Please call fit() before you start predicting")
            return None

        if self.theta_pos.shape[1] != X.shape[1]:
            print("Shape mismatch. Please train with proper dimensions")
            return None

        pos_prob = np.log(self.pos_prior_probability) + \
            np.sum(X@np.log(self.theta_pos).T, axis=-1)
        neg_prob = np.log(self.neg_prior_probability) + \
            np.sum(X@np.log(self.theta_neg).T, axis=-1)

        exp_pos = np.expand_dims(pos_prob, axis=-1)
        exp_neg = np.expand_dims(neg_prob, axis=-1)

        output = np.concatenate((exp_neg, exp_pos), axis=1)
        if len(output.shape)>2:
            output = np.squeeze(np.array(output),axis=-1)
        B = np.exp(output - np.amax(output,axis=-1)[:, np.newaxis])
        C = np.sum(B,axis=-1)
        return B/C[:, np.newaxis]


class RobustNaiveBayesMultiClassifierPercentage(BaseEstimator):

    def __init__(self, percentage_noise=0.0, debug=False, num_classes=2):

        # assert 0 <= percentage_noise <= 100, "Please enter a valid percentage between 0 and 100 inclusive"

        self.percentage_noise = percentage_noise

        self.kappa = []
        self.num_classes = num_classes
        self.prior_probability = []
        self.theta = []
        self.f = []
        self.indices = []

        self._has_fit = False

        self.debug = debug

    def _solve_subproblem(self, f, kappa):
        '''
        Solves the subproblem:
        max_{v > 0} F(v) = kappa * log v + \sum f_i log max(f_i, v) - max(f_i, v) 

        where kappa is self.percentage_noise * sum of word counts in text corpus training
        Returns the optimal value of the dual variable theta*
        '''
        if self.debug:
            print("Internal kappa percentage: " + str(kappa))

        if (len(f.shape) == 2):
            n = f.shape[1]
        else:
            n = f.shape[0]

        f_l1 = np.sum(f)

        f_int = (-np.sort(-f))
        f_int = np.insert(f_int, 0, 9999999)
        f_new = np.insert(f_int, n+1, 0.)

        if (len(f_new.shape) < 2):
            f_new = np.reshape(f_new, (1, f_new.shape[0]))

        rho_curr = kappa + f_l1
        h_curr = 0
        v_curr = max(f_new[0, 1], rho_curr/n)
        F_curr = rho_curr * np.log(v_curr) - n * v_curr

        v_star = v_curr
        F_star = F_curr
        k_star = 0

        for k in range(1, n + 1):
            if k == n:
                h_curr = h_curr + f_new[0, k] * \
                    np.log(f_new[0, k]) - f_new[0, k]
                F_curr = h_curr + kappa * np.log(f_new[0, k])
                v_curr = f_new[0, k]
            else:
                rho_curr = rho_curr - f_new[0, k]
                h_curr = h_curr + f_new[0, k] * \
                    np.log(f_new[0, k]) - f_new[0, k]
                v_curr = min(f_new[0, k], max(
                    f_new[0, k + 1], rho_curr/(n - k)))
                F_curr = h_curr + rho_curr * np.log(v_curr) - (n - k) * v_curr

            if F_curr > F_star:
                F_star = F_curr
                v_star = v_curr
                k_star = k

        theta = (1./(kappa + f_l1)) * np.maximum(f, v_star * np.ones(f.shape))

        return theta

    def fit(self, X, y):
        for i in range(self.num_classes):
            self.indices.append(np.where(y == i)[0])

            self.f.append(np.sum(X[self.indices[-1]], axis=0))

            self.f[-1] += 1

            self.kappa.append((self.percentage_noise/100) * np.sum(self.f[-1]))

            self.theta.append(self._solve_subproblem(
                self.f[-1], self.kappa[-1]))

            if (len(self.theta[-1].shape) < 2):
                self.theta[-1] = np.reshape(self.theta[-1],
                                            (1, self.theta[-1].shape[0]))

            self.prior_probability.append(len(self.indices[-1])/X.shape[0])

        self._has_fit = True

    def predict(self, X):
        if not self._has_fit:
            print("Please call fit() before you start predicting")
            return None

        if self.theta[-1].shape[1] != X.shape[1]:
            print("Shape mismatch. Please train with proper dimensions")
            return None

        result = []
        for i in range(self.num_classes):
            result.append(
                np.log(self.prior_probability[i]) + np.sum(X@np.log(self.theta[i]).T, axis=-1))

        result = np.array(result)

        return np.argmax(result, axis=0)

    def predict_proba(self, X):
        if not self._has_fit:
            print("Please call fit() before you start predicting")
            return None

        if self.theta[-1].shape[1] != X.shape[1]:
            print("Shape mismatch. Please train with proper dimensions")
            return None

        result = []
        for i in range(self.num_classes):
            result.append(np.expand_dims(np.log(
                self.prior_probability[i]) + np.sum(X@np.log(self.theta[i]).T, axis=-1), axis=-1))

        output = np.concatenate(result, axis=1)
        B = np.exp(output - np.amax(output,axis=-1)[:, np.newaxis])
        C = np.sum(B,axis=-1)
        return B/C[:, np.newaxis]
