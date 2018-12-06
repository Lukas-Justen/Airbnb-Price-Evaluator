from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np

from parse import *

class ModelRunner:

    def rfreg(self):
        data = get_processed_data()
        sel = VarianceThreshold(threshold=0)

        y = data['price']
        X = data.drop(['price'], axis=1)
        print (X.columns)
        X = sel.fit_transform(X)
        print (X[0].shape)


        print("Random Forest")
        clf = RandomForestRegressor(n_estimators=300, criterion='mae',
                                          max_features=14, max_depth=None,
                                          n_jobs=4, oob_score=True)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)
        clf.fit(X_train, y_train)
        print(clf.score(X_train, y_train))
        print(clf.score(X_test, y_test))

    def rfclass(self):

        data = get_processed_data()
        sel = VarianceThreshold(threshold=0.05)

        # print (X.columns)

        prices = np.array(data["price"])
        prices = prices.reshape(-1, 1)
        data["bin"] = KMeans(n_clusters=6).fit_predict(prices)

        ranges = {}
        for cluster in set(data["bin"]):
            df = data.loc[data['bin'] == cluster]
            ranges[cluster] = (min(df["price"]),max(df["price"]))
            print(cluster, ranges[cluster])

        y = data['bin']
        print (Counter(y))
        X = data.drop(['price','bin'], axis=1)
        X = sel.fit_transform(X)
        print (X[0].shape)

        print (data.columns)
        print("Random Forest")
        clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
        clf.fit(X_train, y_train)
        print(clf.score(X_train, y_train))
        print(clf.score(X_test, y_test))



mr = ModelRunner()
mr.rfclass()
