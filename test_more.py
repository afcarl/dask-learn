from play import DaskPipeline
from sklearn.svm import LinearSVC
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_selection import SelectFdr
from sklearn.feature_extraction.text import CountVectorizer
from grid_search import GridSearchCV
#from sklearn.pipeline import Pipeline
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

param_grid = {'select_fdr__alpha': [0.05, 0.01, 0.1, 0.2],
              'svm__C': np.logspace(-3, 2, 3)}

dask_grid = GridSearchCV(pipeline, param_grid)
dask_grid.fit(X_train, y_train)
dask_grid._dask_value.visualize("dask_grid.pdf")

print(dask_grid.score(X_test, y_test))
