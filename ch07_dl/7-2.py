# 심층 신경망
# 2개의 층
from tensorflow import keras

(train_input, train_target), (test_input,
                              test_target) = keras.datasets.fashion_mnist.load_data()

from sklearn.model_selection import train_test_split

train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

dense1 = keras.layers.Dense(100, activation='sigmoid', input_shape=(784,))
dense2 = keras.layers.Dense(10, activation='softmax')


# 심층 신경만 만들기
model = keras.Sequential([dense1, dense2])

model.summary()
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 dense (Dense)               (None, 100)               78500

 dense_1 (Dense)             (None, 10)                1010

=================================================================
'''


# 층을 추가하는 다른 방법
model = keras.Sequential([
    keras.layers.Dense(100, activation='sigmoid',
                       input_shape=(784,), name='hidden'),
    keras.layers.Dense(10, activation='softmax', name='output')
], name='패션 MNIST 모델')

model.summary()
'''
Model: "패션 MNIST 모델"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 hidden (Dense)              (None, 100)               78500

 output (Dense)              (None, 10)                1010

=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
'''

model = keras.Sequential()
model.add(keras.layers.Dense(100, activation='sigmoid', input_shape=(784,)))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
'''
Model: "sequential_1"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 dense_2 (Dense)             (None, 100)               78500

 dense_3 (Dense)             (None, 10)                1010

=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

model.fit(train_scaled, train_target, epochs=5)
'''
Epoch 1/5
1500/1500 [==============================] - 4s 2ms/step - loss: 0.5605 - accuracy: 0.8110
Epoch 2/5
1500/1500 [==============================] - 3s 2ms/step - loss: 0.4075 - accuracy: 0.8527
Epoch 3/5
1500/1500 [==============================] - 3s 2ms/step - loss: 0.3747 - accuracy: 0.8649
Epoch 4/5
1500/1500 [==============================] - 3s 2ms/step - loss: 0.3511 - accuracy: 0.8728
Epoch 5/5
1500/1500 [==============================] - 3s 2ms/step - loss: 0.3346 - accuracy: 0.8799
'''


## 렐루 활성화 함수
model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
'''
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 flatten (Flatten)           (None, 784)               0

 dense_4 (Dense)             (None, 100)               78500

 dense_5 (Dense)             (None, 10)                1010

=================================================================
Total params: 79,510
Trainable params: 79,510
Non-trainable params: 0
_________________________________________________________________
'''

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

model.fit(train_scaled, train_target, epochs=5)
'''
Epoch 1/5
1500/1500 [==============================] - 8s 5ms/step - loss: 0.5359 - accuracy: 0.8119
Epoch 2/5
1500/1500 [==============================] - 5s 4ms/step - loss: 0.3950 - accuracy: 0.8569
Epoch 3/5
1500/1500 [==============================] - 5s 3ms/step - loss: 0.3572 - accuracy: 0.8711
Epoch 4/5
1500/1500 [==============================] - 5s 3ms/step - loss: 0.3336 - accuracy: 0.8811
Epoch 5/5
1500/1500 [==============================] - 4s 3ms/step - loss: 0.3191 - accuracy: 0.8856
'''

model.evaluate(val_scaled, val_target)
# 375/375 [==============================] - 1s 2ms/step - loss: 0.3552 - accuracy: 0.8788


## 옵티마이저
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics='accuracy')

sgd = keras.optimizers.SGD()
model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics='accuracy')

sgd = keras.optimizers.SGD(learning_rate=0.1)

sgd = keras.optimizers.SGD(momentum=0.9, nesterov=True)

adagrad = keras.optimizers.Adagrad()
model.compile(optimizer=adagrad, loss='sparse_categorical_crossentropy', metrics='accuracy')

rmsprop = keras.optimizers.RMSprop()
model.compile(optimizer=rmsprop, loss='sparse_categorical_crossentropy', metrics='accuracy')

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

model.fit(train_scaled, train_target, epochs=5)
'''
Epoch 1/5
1500/1500 [==============================] - 4s 2ms/step - loss: 0.5247 - accuracy: 0.8176
Epoch 2/5
1500/1500 [==============================] - 4s 2ms/step - loss: 0.3957 - accuracy: 0.8597
Epoch 3/5
1500/1500 [==============================] - 4s 3ms/step - loss: 0.3575 - accuracy: 0.8704
Epoch 4/5
1500/1500 [==============================] - 4s 3ms/step - loss: 0.3280 - accuracy: 0.8810
Epoch 5/5
1500/1500 [==============================] - 4s 3ms/step - loss: 0.3100 - accuracy: 0.8847
'''

model.evaluate(val_scaled, val_target)
# 375/375 [==============================] - 1s 2ms/step - loss: 0.3748 - accuracy: 0.8622
