import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_validate

data = pd.read_csv('./data/listings_first_concat_clean.csv')
data = data.loc[data['price'] < 699]
data = data.sample(frac=1)

y = data['price']
X = data.drop(['price'], axis=1)
selector = VarianceThreshold(0.04)
print(X.shape)
X = selector.fit_transform(X)
print(X.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)

print("Random Forest")
clf = RandomForestRegressor(n_estimators=500, max_features=14, max_depth=40)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

# cv_results = cross_validate(clf, X, y, cv=3, return_train_score=True)
# print(sorted(cv_results.keys()))
# print(cv_results['test_score'])
# print(cv_results['train_score'])

# print("Ridge Regression")
# ridge = Ridge(alpha=1.0)
# ridge.fit(X,y)
# print(ridge.score(X_train,y_train))
# print(ridge.score(X_test,y_test))

# print("ElasticNet Regression")
# elast = ElasticNet(random_state=0)
# elast.fit(X_train, y_train)
# print(elast.score(X_train, y_train))
# print(elast.score(X_test, y_test))

# print("Polynomial Regression")
# poly = PolynomialFeatures(2)
# poly_X = poly.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(poly_X, y, test_size=0.3, random_state=42, shuffle=True)
# lr = LinearRegression()
# lr.fit(X_train,y_train)
# print(lr.score(X_train[:100], y_train[:100]))
# print(lr.score(X_test, y_test))

# lr = LinearRegression().fit(X,y)
# data = data.loc[data['price'] < 699]
# prices = np.array(data["price"])
# prices = prices.reshape(-1, 1)
# data["bin"] = KMeans(n_clusters=15).fit_predict(prices)
#
# ranges = {}
# for cluster in set(data["bin"]):
#     df = data.loc[data['bin'] == cluster]
#     ranges[cluster] = (min(df["price"]), max(df["price"]), len(df))
#     print(cluster, ranges[cluster])
