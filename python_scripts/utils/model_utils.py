import numpy as np
import pandas as pd
from scipy import optimize

from tqdm.notebook import tqdm


def rlp_func(x, m, n, abs_lp=True):
    """
    Compute recent LP in x. The resulting value depends on sizes of 2 windows equal to `m` and `n`.
    LP is equal to the absolute difference between average score over the first and the second window.
    The first window spans a subsequence of x from the beginning of x to m, i.e. x[:m]
    The second windon spans a subsequence of x from the end of x to -n, i.e.  x[-n:]
    """
    diff = np.mean(x[:m]) - np.mean(x[-n:])
    return np.abs(diff) if abs_lp else diff


def get_baseline_aic(n, k):
    """
    Log likelihood of random choice model is equal to log-likelihood
    of random choice probability log(p=1/k), multiplied by the number of trials n.
    To convert it to AIC, we simply multiply by 2 and don't penalize by number of parameters,
    since this model does not have any parameters.
    """
    # Get baseline AIC
    neg_log_lik = -np.sum(np.full(n, np.log(1/k)))
    return 2*neg_log_lik + 2*0


def softmax_1d(x, shift=True):
    """
    Shifting prevents overflow
    """
    z = x - np.max(x) if shift else x
    a = np.exp(z)
    return a / np.sum(a)


def softmax_2d(x, shift=True):
    """Shifting prevents overflow"""
    z = x - np.max(x, axis=1)[:, np.newaxis] if shift else x
    a = np.exp(z)
    return a / np.sum(a, axis=1)[:, np.newaxis]


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def standardize(x):
    return (x - x.mean()) / x.std()


def neg_log_likelihood(params, *args):
    coefs = params[:-1]
    inps = args[:-1]
    u = sum([coef * inp for coef, inp in zip(coefs, inps)])
    p = softmax_2d(u * params[-1])
    p_choices = p[args[-1].astype(bool)]
    loglik_trials = np.log(p_choices)
    loglik_sum = np.sum(loglik_trials, axis=0)
    return -loglik_sum


class SoftmaxChoiceModel(object):
    def __init__(self, objective, data, init_dict, hist=False):
        self.objective = objective
        tau_specs = init_dict.pop('tau')
        self.tau_range = tau_specs[0]
        self.components = list(init_dict.keys())
        self.init_ranges = [i[0] for i in init_dict.values()]
        self.bounds = [i[0] if i[1] else [None, None] for i in init_dict.values()]
        self.bounds.append(self.tau_range if tau_specs[1] else [None, None])

        # Note that we "shift" data and choice data relative to each other (cutting one trial from each)
        # This is done so that the model tries to predict the *next* choice, not the current one
        self.data = [data.loc[:, c + '1':c + '4'].values[:-1, :] for c in self.components]
        self.choice_data = data.loc[:, 'ch1':'ch4'].values[1:, :]
        self.fit_data = self.data + [self.choice_data]

        self.params = None
        self.negloglik = None
        self.hist = [] if hist else None

    def initialize_params(self):
        return np.array([np.random.uniform(l, u) for l, u in self.init_ranges + [self.tau_range]])

    def transform_inp_data(self, func):
        self.fit_data = [func(data) for data in self.data] + [self.choice_data]

    def fit_l_bfgs_b(self, init_guess, disp=False):
        params, likelihood, _ = optimize.fmin_l_bfgs_b(
            func=self.objective,
            x0=init_guess,
            args=self.fit_data,
            bounds=self.bounds,
            approx_grad=True,
            disp=disp
        )
        return params, likelihood

    def best_of_n(self, n, progbar=False):
        if progbar:
            iter_ = tqdm(range(n), desc='Seed', leave=False)
        else:
            iter_ = range(n)
        for i in iter_:
            fitted_params, loss = self.fit_l_bfgs_b(self.initialize_params())
            loss = np.around(loss, 5)
            if self.hist is not None:
                self.hist.append([i, fitted_params, loss])
            if self.negloglik is None:
                self.negloglik = loss
                self.params = fitted_params
            else:
                if loss < self.negloglik:
                    self.negloglik = loss
                    self.params = fitted_params

    def n_best_stop(self, n_stop, max_iter, show_progress=False):
        if show_progress:
            progbar = tqdm(range(n_stop), desc='Seed', leave=False)
        n_min = 1
        for i in range(max_iter):
            fitted_params, loss = self.fit_l_bfgs_b(self.initialize_params())
            loss = np.around(loss, 5)
            if self.hist is not None:
                self.hist.append([i, *fitted_params, loss])
            if self.negloglik is None:
                self.negloglik = loss
                self.params = fitted_params
                if show_progress:
                    progbar.update()
            else:
                if loss < self.negloglik:
                    self.negloglik = loss
                    self.params = fitted_params
                    n_min = 1
                    if show_progress:
                        progbar.reset()
                        progbar.refresh()
                elif loss == self.negloglik:
                    n_min += 1
                    if show_progress:
                        progbar.update()
            if n_min == n_stop:
                if show_progress:
                    progbar.close()
                break

    def get_predictions(self):
        if self.params is not None:
            u = sum([i * p for i, p in zip(self.data, self.params[:-1])])
            p = softmax_2d(u * self.params[-1])
            return u, p

    def get_aic(self):
        # We add 1 to the number of params to account for the temperature parameter
        if self.negloglik:
            return 2 * self.negloglik + 2 * (len(self.params) + 1)
        else:
            return None

    def get_param_csv(self):
        if self.params is not None:
            return ','.join(['{:.5f}'.format(p) for p in self.params])
        else:
            return None

    def get_hist_df(self):
        return pd.DataFrame(self.hist, columns=['iter'] + self.components + ['tau', 'loss']).set_index('iter')

    def clip_data(self, trial):
        self.data = [data[:trial, :] for data in self.data]
        self.fit_data = [data[:trial, :] for data in self.fit_data]
        self.choice_data = self.choice_data[:trial, :]
