from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras import utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from pymongo import MongoClient

import numpy as np

client = MongoClient()
db = client.bigdata
collection=db.bigdata

num_words = 10000

xs = []
ys = []
x_test = []
x_test_id = []

ycount=collection.count({"y": {"$exists": "true"}})

for k in range(ycount):
    xs.append([]) 
    x_test.append([])

yi = 0
N=20

for everyy in collection.find({"y": {"$exists": "true"}}):  
    x_test_id.append(everyy["_id"])
    i=0
    ys.append(everyy["y"])
    xname = "x" + str(i)
    while i!=N:
        xname = "x" + str(i) 
        everyx = everyy[xname]
        xs[yi].append(everyx)
        i=int(i)
        i+=1
    yi+=1

yi=0 

for everynoy in collection.find({"y":{"$exists":"false"}}):
    i=0
    xname = "x" + str(i) 
    while i!=N:
        xname = "x" + str(i) 
        everyx = everyy[xname]
        x_test[yi].append(everyx)
        i=int(i)
        i+=1
    yi+=1
 

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(ys)
tokenizer.word_index

sequences = tokenizer.texts_to_sequences(ys)
y_train = pad_sequences(sequences)

x_train = xs

model = Sequential()
model.add(Embedding(num_words, 64, input_length = 20))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(x_train, y_train, epochs=5, batch_size=1, verbose=2)

pred = model.predict(x_test)
print(tokenizer.word_index[pred[0]])
print(pred[1][0])
 

for id in x_test_id:
    item = collection.find({"_id":id})
    print(item)

plt.plot(history.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
plt.xlabel('Эпоха обучения')
plt.ylabel('Доля верных ответов')
plt.legend()
plt.show()
