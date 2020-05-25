from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten
from keras import utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
from pymongo import MongoClient
import numpy as np

client = MongoClient()
db = client.bigdata
collection=db.bigdata

num_words = 10000

xs = []
ys = []

ycount=collection.count({"y": {"$exists": "true"}})

for k in range(ycount):
    xs.append([]) 

yi = 0
N=20

for everyy in collection.find({"y": {"$exists": "true"}}):
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

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(ys)
tokenizer.word_index

sequences = tokenizer.texts_to_sequences(ys)
y_train = pad_sequences(sequences, 20)

x_train = np.zeros(len(xs[0]), len(xs), 1)

model = Sequential()
model.add(Embedding(num_words, 64, input_length = 20))
model.add(Dense(128, activation='relu',input_shape = (x_train.shape, )))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)

mse, mae = model.evaluate(x_test, y_test, verbose=0)

print("Средняя абсолютная ошибка (тысяч долларов):", mae)

pred = model.predict(x_test)

print("Предсказанная стоимость:", pred[1][0], ", правильная стоимость:", y_test[1])
