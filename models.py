from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.kernel_ridge import KernelRidge
from parse import *

data = get_processed_data()
clf = RandomForestRegressor(n_estimators=300)

y = data['price']
X = data.drop(['price'], axis=1)

print("Random Forest")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print (X_train, X_test, y_train, y_test)
clf.fit(X_train, y_train)
print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))
