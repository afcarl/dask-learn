
from play import DaskPipeline
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from grid_search import GridSearchCV
import numpy as np

results = []
pipeline = DaskPipeline([("pca", PCA()),
                         ("select_k", SelectKBest()),
                         ("svm", LinearSVC())])

iris = load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target)

param_grid = {'select_k__k': [1, 2, 3, 4],
              'svm__C': np.logspace(-3, 2, 3)}
import dask
dask.set_options(get=dask.async.get_sync)
from dask.diagnostics import ProgressBar
with ProgressBar():
    dask_grid = GridSearchCV(pipeline, param_grid)
    dask_grid.fit(X_train, y_train)
    dask_grid._dask_value.visualize("dask_grid_iris.pdf")

print(dask_grid.score(X_test, y_test))
