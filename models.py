from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from parse import *

data = get_processed_data()
clf = RandomForestRegressor()

y = data['price']
X = data.drop(['price'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)

clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
