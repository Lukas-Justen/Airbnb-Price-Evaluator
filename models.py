import pandas

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.model_selection import cross_validate

data = pandas.read_csv('data/listings_first_concat_clean.csv')
data = data.sample(frac=1)
data = data.loc[data['price'] < 699]

y = data['price']
X = data.drop(['price'], axis=1)
X = VarianceThreshold(0.1).fit_transform(X)

folds = 5

print("Random Forest")
clf = RandomForestRegressor(n_estimators=300)
results = cross_validate(clf, X, y, cv=folds, return_train_score=True)
avg_train = sum(results['train_score']) / folds
avg_test = sum(results['test_score']) / folds
print(avg_train, avg_test)

print("Linear Regression")
reg = LinearRegression(normalize=True, n_jobs=-1)
results = cross_validate(reg, X, y, cv=folds, return_train_score=True)
avg_train = sum(results['train_score']) / folds
avg_test = sum(results['test_score']) / folds
print(avg_train, avg_test)

print("Elastic Net")
r_reg = ElasticNet(l1_ratio=0.7, random_state=42)
results = cross_validate(r_reg, X, y, cv=folds, return_train_score=True)
avg_train = sum(results['train_score']) / folds
avg_test = sum(results['test_score']) / folds
print(avg_train, avg_test)

print("Ridge Regressor")
ridge = Ridge(alpha=1.0)
results = cross_validate(ridge, X, y, cv=folds, return_train_score=True)
avg_train = sum(results['train_score']) / folds
avg_test = sum(results['test_score']) / folds
print(avg_train, avg_test)

print("Lasso Regressor")
lasso = Lasso(alpha=0.05, copy_X=True, fit_intercept=True, max_iter=500,
              normalize=True, positive=False, precompute=False, random_state=None,
              selection='cyclic', tol=0.0001, warm_start=False)
results = cross_validate(lasso, X, y, cv=folds, return_train_score=True)
avg_train = sum(results['train_score']) / folds
avg_test = sum(results['test_score']) / folds
print(avg_train, avg_test)

print("Gradient Boosted Regressor")
gbr = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=12, random_state=42, loss='ls')
results = cross_validate(gbr, X, y, cv=folds, return_train_score=True)
avg_train = sum(results['train_score']) / folds
avg_test = sum(results['test_score']) / folds
print(avg_train, avg_test)

