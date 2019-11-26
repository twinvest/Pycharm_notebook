from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np

data = datasets.load_breast_cancer()
x = data['data']
y = data['target']
print(x.shape)
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, random_state=0)

n_estimators = [10,20,30,40,50]
max_featrues = [2,3,4]
bootstrap = [True, False]

param_grid = [{'n_estimators' : n_estimators, 'max_features': max_featrues
              ,'bootstrap': bootstrap}]

rf = RandomForestRegressor()

grid_search = GridSearchCV(rf, param_grid=param_grid, cv = 4,
                          scoring='neg_mean_squared_error')

grid_search.fit(X_train, Y_train)

print(grid_search.best_params_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)