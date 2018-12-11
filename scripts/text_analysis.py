import pandas as pd
import re
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer


data = pd.read_csv('data/seattle/3/listings_texts.csv')

corpus = data['description']
y = data['price']
X=[]
for i,line in enumerate(corpus):
    clear = [x for x in re.sub(r'[^\w\'\s]', '',line.lower()).split() if x not in stopwords.words('english')]
    X.append(' '.join(clear))
    if i%100 == 0:
        print("Progress : ", i)
    if i == 4000:
        break

print("Moving on!")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

net = Sequential()
net.add(Dense(200, input_dim=X[0].shape[1], kernel_initializer='normal',activation='relu'))
net.add(Dense(100, input_dim=200, kernel_initializer='normal',activation='relu'))
net.add(Dense(1, input_dim=100, kernel_initializer='normal'))
net.compile(loss='mean_squared_error', optimizer='adam')
net.fit(X[:3000],y[:3000], epochs=70, batch_size=100)

print(net.evaluate(X[3001:] ,y[3001:]))
for i in range(50):
    print(net.predict(X[3001+i]), y[3001+i])
