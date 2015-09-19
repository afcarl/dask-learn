from play import DaskPipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import make_blobs, fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_selection import SelectFdr
from sklearn.feature_extraction.text import CountVectorizer
from dask.async import get_sync
from sklearn.pipeline import Pipeline
import numpy as np

results = []
pipeline = DaskPipeline([("count", CountVectorizer()),
                         ("select_fdr", SelectFdr()),
                         ("svm", LinearSVC())])

# X, y = make_blobs()
categories = [
    'alt.atheism',
    'talk.religion.misc',
]

data_train = fetch_20newsgroups(subset='train', categories=categories)
data_test = fetch_20newsgroups(subset='test', categories=categories)
X_train, y_train = data_train.data, data_train.target
X_test, y_test = data_test.data, data_test.target

for fdr in [0.05, 0.01, 0.1, 0.2]:
    for C in np.logspace(-3, 2, 3):
        pipeline.set_params(select_fdr__alpha=fdr, svm__C=C)
        pipeline.fit(X_train, y_train)
        results.append(pipeline.score(X_test, y_test))


"""
from dask.diagnostics import ProgressBar
ProgressBar().register()
"""

from dask.imperative import compute, value
value(results).visualize('dask.pdf')
results2 = compute(results, get=get_sync)
print results2

