from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Activation
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

ycount=collection.count({"y": {"$exists": True}})
noycount = collection.count({"y": {"$exists": False}})

for k in range(ycount):
    xs.append([]) 

for k in range(noycount):
    x_test.append([])

count=0
for everyy in collection.find({"y":{"$exists":True}}):
    j=0
    ys.append(everyy["y"])
    xname = 'x' + str(j)
    while xname in everyy:
        x_value = everyy[xname]
        xs[count].append(x_value)
        j = int(j)
        j+=1
        xname = 'x' + str(j)
    count+=1

count=0
for everynoy in collection.find({"y":{"$exists":False}}):
    x_test_id.append(everynoy['_id'])
    j=0
    xname = 'x' + str(j)
    while xname in everynoy:
        x_value = everynoy[xname]
        x_test[count].append(x_value)
        j = int(j)
        j+=1
        xname = 'x' + str(j)
    count+=1

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(ys)
tokenizer.word_index

sequences = tokenizer.texts_to_sequences(ys)
indices_to_char = dict((i+1, c) for i, c in enumerate(tokenizer.word_index)) 

y_train = pad_sequences(sequences)

x_train = np.array(xs)

model = Sequential()
model.add(Embedding(num_words, 10, input_length = 20))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(x_train, y_train, epochs=3, batch_size=1, verbose=2)

pred = model.predict(x_test)

def int_r(num):
    num = int(num + (0.5 if num > 0 else -0.5))
    return num

y_arr=[]
for y_item in pred:
    for y in y_item:
        y = int_r(y)
        if (y<=0):
            y=1
        if (y>=6):
            y=5
        y = indices_to_char[y]
        y_arr.append(y)

k=0
while k!=len(x_test_id):
    f = collection.update_one({"_id":x_test_id[k]}, {"$set":{"y":y_arr[k]}})
    k+=1
 
