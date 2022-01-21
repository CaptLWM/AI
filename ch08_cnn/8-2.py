# 합성곱 신경망을 사용한 이미지 분류
## 패션 MNIST 데이터 불러오기
from tensorflow import keras
from sklearn.model_selection import train_test_split

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)


## 합성곱 신경망 만들기
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu', padding='same', 
                              input_shape=(28,28,1)))

model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dropout(0.4))
model.add(keras.layers.Dense(10, activation='softmax'))

model.summary()
'''
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 28, 28, 32)        320

 max_pooling2d (MaxPooling2D  (None, 14, 14, 32)       0
 )

 conv2d_1 (Conv2D)           (None, 14, 14, 64)        18496

 max_pooling2d_1 (MaxPooling  (None, 7, 7, 64)         0
 2D)

 flatten (Flatten)           (None, 3136)              0

 dense (Dense)               (None, 100)               313700

 dropout (Dropout)           (None, 100)               0

 dense_1 (Dense)             (None, 10)                1010

=================================================================
Total params: 333,526
Trainable params: 333,526
Non-trainable params: 0
_________________________________________________________________
'''

# keras.utils.plot_model(model)

# keras.utils.plot_model(model, show_shapes=True, to_file='cnn-architecture.png', dpi=300)


## 모델 컴파일과 훈련
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics='accuracy')

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5')
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2,
                                                  restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20,
                    validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])
'''
Epoch 1/20
1500/1500 [==============================] - 30s 20ms/step - loss: 0.5261 - accuracy: 0.8099 - val_loss: 0.3240 - val_accuracy: 0.8790
Epoch 2/20
1500/1500 [==============================] - 31s 20ms/step - loss: 0.3502 - accuracy: 0.8752 - val_loss: 0.2779 - val_accuracy: 0.8969
Epoch 3/20
1500/1500 [==============================] - 31s 20ms/step - loss: 0.3036 - accuracy: 0.8905 - val_loss: 0.2774 - val_accuracy: 0.8985
Epoch 4/20
1500/1500 [==============================] - 31s 21ms/step - loss: 0.2712 - accuracy: 0.9015 - val_loss: 0.2477 - val_accuracy: 0.9076
Epoch 5/20
1500/1500 [==============================] - 31s 21ms/step - loss: 0.2470 - accuracy: 0.9106 - val_loss: 0.2369 - val_accuracy: 0.9154
Epoch 6/20
1500/1500 [==============================] - 33s 22ms/step - loss: 0.2271 - accuracy: 0.9168 - val_loss: 0.2404 - val_accuracy: 0.9142
Epoch 7/20
1500/1500 [==============================] - 33s 22ms/step - loss: 0.2098 - accuracy: 0.9233 - val_loss: 0.2301 - val_accuracy: 0.9181
Epoch 8/20
1500/1500 [==============================] - 32s 21ms/step - loss: 0.1962 - accuracy: 0.9274 - val_loss: 0.2336 - val_accuracy: 0.9153
Epoch 9/20
1500/1500 [==============================] - 32s 21ms/step - loss: 0.1807 - accuracy: 0.9319 - val_loss: 0.2225 - val_accuracy: 0.9219
Epoch 10/20
1500/1500 [==============================] - 33s 22ms/step - loss: 0.1676 - accuracy: 0.9360 - val_loss: 0.2233 - val_accuracy: 0.9237
Epoch 11/20
1500/1500 [==============================] - 34s 23ms/step - loss: 0.1563 - accuracy: 0.9407 - val_loss: 0.2290 - val_accuracy: 0.9233
'''

import matplotlib.pyplot as plt

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

model.evaluate(val_scaled, val_target)

plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()

preds = model.predict(val_scaled[0:1])
print(preds)
'''
[[2.9993918e-19 3.5966895e-23 6.8053086e-20 3.4029983e-19 3.1868540e-16
  2.1523373e-14 2.3810946e-17 8.9157977e-14 1.0000000e+00 1.4259664e-17]]
'''

plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

classes = ['티셔츠', '바지', '스웨터', '드레스', '코트',
           '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']

import numpy as np
print(classes[np.argmax(preds)])
# 가방

test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0

model.evaluate(test_scaled, test_target)
# 313/313 [==============================] - 2s 6ms/step - loss: 0.2437 - accuracy: 0.9142


