import dask
from dask.base import tokenize
from dask.imperative import Value
from sklearn.metrics import accuracy_score
from operator import getitem

from sklearn.pipeline import Pipeline
from sklearn.base import clone


def fit_transform(est, X, y):
    est = clone(est)
    data = est.fit_transform(X, y)
    return est, data


def fit(est, X, y):
    est = clone(est)
    est.fit(X, y)
    return est


def transform(est, X):
    return est.transform(X)

def predict(est, X):
    return est.predict(X)


class DaskPipeline(Pipeline):
    def __init__(self, pairs):
        self.steps = pairs
        self.dask = dict()

    @property
    def named_steps(self):
        return dict(self.steps)

    @property
    def name(self):
        return tokenize('pre-transform', self, X, y, self.params)

    @property
    def _estimator_keys(self):
        return [(getitem, (self.name, i), 0) for i in range(len(self.pairs))]

    def get_fit_keys(self, X, y):
        name, est = self.steps[0]
        L = [('fit', name, tokenize(type(est), est.get_params(), X, y))]
        for name, est in self.steps[1:]:
            L.append(('fit', name, tokenize(type(est), est.get_params(), X, y, L[-1])))
        return L

    def get_predict_keys(self, X):
        L = [('transform', self.steps[0][0], tokenize(self._fit_estimators[0], X))]
        for (name, est), fit in zip(self.steps[1:], self._fit_estimators[1:]):
            L.append(('transform', name, tokenize(fit, X)))
        L[-1] = ('predict', name, tokenize(fit, X))
        return L

    def fit(self, X, y=None):
        names = self.get_fit_keys(X, y)
        self._fit_estimators = [(k[0] + '-estimator',) + k[1:] for k in names]
        self._fit_data = [(k[0] + '-data',) + k[1:] for k in names[:-1]]
        dsk = dict()

        est = self.steps[0][1]
        dsk = {names[0]: (fit_transform, est, X, y)}

        for old_data, name, (_, est) in list(zip(self._fit_data[:-1],
                                                 names[1:-1],
                                                 self.steps[1:-1])):
            dsk[name] = (fit_transform, est, old_data, y)

        dsk2 = {(k[0] + '-estimator',) + k[1:]: (getitem, k, 0)
                for k in dsk}
        dsk3 = {(k[0] + '-data',) + k[1:]: (getitem, k, 1)
                for k in dsk}

        dsk[names[-1]] = (fit, self.steps[-1][1], self._fit_data[-1], y)
        self._fit_estimators.append(names[-1])

        self.dask = dict()
        self.dask.update(dsk)
        self.dask.update(dsk2)
        self.dask.update(dsk3)

        return self

    def predict(self, X):
        names = self.get_predict_keys(X)

        self._predict_keys = names

        dsk = {names[0]: (transform, self._fit_estimators[0], X)}
        for old_name, name, est in zip(names[:-1], names[1:], self._fit_estimators[1:]):
            dsk[name] = (transform, est, old_name)

        dsk[names[-1]] = (predict, self._fit_estimators[-1], names[-2])

        self.dask.update(dsk)

        return Value(names[-1], [self.dask])

    def score(self, X, y):
        self.predict(X)
        names = self.get_predict_keys(X)
        name = ('score', tokenize(names[-1], y))
        dsk = {name: (accuracy_score, names[-1], y)}

        return Value(name, [dsk, self.dask.copy()])
