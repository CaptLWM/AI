# LSTM과 GRU 셀
## LSTM 신경망 훈련하기
from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = imdb.load_data(
    num_words=500)

train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

from tensorflow.keras.preprocessing.sequence import pad_sequences

train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)

from tensorflow import keras

model = keras.Sequential()

model.add(keras.layers.Embedding(500, 16, input_length=100))
model.add(keras.layers.LSTM(8))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.summary()
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding (Embedding)       (None, 100, 16)           8000

 lstm (LSTM)                 (None, 8)                 800

 dense (Dense)               (None, 1)                 9

=================================================================
Total params: 8,809
Trainable params: 8,809
Non-trainable params: 0
_________________________________________________________________
'''

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', 
              metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model.fit(train_seq, train_target, epochs=100, batch_size=64,
                    validation_data=(val_seq, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
'''
Epoch 1/100
313/313 [==============================] - 11s 30ms/step - loss: 0.6920 - accuracy: 0.5511 - val_loss: 0.6904 - val_accu
Epoch 2/100
313/313 [==============================] - 10s 31ms/step - loss: 0.6877 - accuracy: 0.6221 - val_loss: 0.6838 - val_accu
...(생략)...
Epoch 28/100
313/313 [==============================] - 12s 38ms/step - loss: 0.4178 - accuracy: 0.8115 - val_loss: 0.4421 - val_accuracy: 0.7960
'''

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()


## 순환 층에 드롭아웃 적용하기
model2 = keras.Sequential()

model2.add(keras.layers.Embedding(500, 16, input_length=100))
model2.add(keras.layers.LSTM(8, dropout=0.3))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy', 
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-dropout-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model2.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()


## 2개의 층을 연결하기
model3 = keras.Sequential()

model3.add(keras.layers.Embedding(500, 16, input_length=100))
model3.add(keras.layers.LSTM(8, dropout=0.3, return_sequences=True))
model3.add(keras.layers.LSTM(8, dropout=0.3))
model3.add(keras.layers.Dense(1, activation='sigmoid'))

model3.summary()
'''
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 embedding_2 (Embedding)     (None, 100, 16)           8000

 lstm_2 (LSTM)               (None, 100, 8)            800

 lstm_3 (LSTM)               (None, 8)                 544

 dense_2 (Dense)             (None, 1)                 9

=================================================================
Total params: 9,353
Trainable params: 9,353
Non-trainable params: 0
_________________________________________________________________
'''

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model3.compile(optimizer=rmsprop, loss='binary_crossentropy', 
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-2rnn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model3.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])
'''
Epoch 1/100
313/313 [==============================] - 21s 57ms/step - loss: 0.6922 - accuracy: 0.5451 - val_loss: 0.6908 - val_accuracy: 0.5596
Epoch 2/100
313/313 [==============================] - 18s 57ms/step - loss: 0.6851 - accuracy: 0.6169 - val_loss: 0.6753 - val_accuracy: 0.6532
...(생략)...
Epoch 35/100
313/313 [==============================] - 17s 55ms/step - loss: 0.4221 - accuracy: 0.8072 - val_loss: 0.4396 - val_accuracy: 0.7990
'''

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()


## GRU 신경망 훈련하기
model4 = keras.Sequential()

model4.add(keras.layers.Embedding(500, 16, input_length=100))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1, activation='sigmoid'))

model4.summary()
'''
Model: "sequential_3"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 embedding_3 (Embedding)     (None, 100, 16)           8000

 gru (GRU)                   (None, 8)                 624

 dense_3 (Dense)             (None, 1)                 9

=================================================================
Total params: 8,633
Trainable params: 8,633
Non-trainable params: 0
'''

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model4.compile(optimizer=rmsprop, loss='binary_crossentropy', 
               metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-gru-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3,
                                                  restore_best_weights=True)

history = model4.fit(train_seq, train_target, epochs=100, batch_size=64,
                     validation_data=(val_seq, val_target),
                     callbacks=[checkpoint_cb, early_stopping_cb])
'''
_________________________________________________________________
Epoch 1/100
313/313 [==============================] - 13s 34ms/step - loss: 0.6926 - accuracy: 0.5218 - val_loss: 0.6917 - val_accuracy: 0.5496
Epoch 2/100
313/313 [==============================] - 11s 36ms/step - loss: 0.6907 - accuracy: 0.5608 - val_loss: 0.6896 - val_accuracy: 0.5738
...(생략)...
Epoch 53/100
313/313 [==============================] - 10s 31ms/step - loss: 0.4042 - accuracy: 0.8197 - val_loss: 0.4333 - val_accuracy: 0.7996
'''

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()


## 마무리
test_seq = pad_sequences(test_input, maxlen=100)

rnn_model = keras.models.load_model('best-2rnn-model.h5')
# 텐서플로 2.3에서는 버그(https://github.com/tensorflow/tensorflow/issues/42890) 때문에 compile() 메서드를 호출해야 합니다.
# rnn_model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics='accuracy')

rnn_model.evaluate(test_seq, test_target)
# 782/782 [==============================] - 8s 10ms/step - loss: 0.4330 - accuracy: 0.7968
