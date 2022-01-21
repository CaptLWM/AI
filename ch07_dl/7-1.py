# 인공 신경망
# 패션 MNIST
from tensorflow import keras

(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()

print(train_input.shape, train_target.shape)
# (60000, 28, 28) (60000,)

print(test_input.shape, test_target.shape)
# (10000, 28, 28) (10000,)


import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 10, figsize=(10,10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')
    axs[i].axis('off')
# plt.show()

print([train_target[i] for i in range(10)])
# [9, 0, 0, 3, 0, 2, 7, 2, 5, 5]


import numpy as np
print(np.unique(train_target, return_counts=True))
'''
(array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],
      dtype=int64))
'''


## 로지스틱 회귀로 패션 아이템 분류하기
train_scaled = train_input / 255.0
train_scaled = train_scaled.reshape(-1, 28*28)

print(train_scaled.shape)
# (60000, 784)

from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier

sc = SGDClassifier(loss='log', max_iter=5, random_state=42)

scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)
print(np.mean(scores['test_score']))
# 0.8192833333333333


## 인공신경망
### 텐서플로와 케라스
import tensorflow as tf
from tensorflow import keras


## 인공신경망으로 모델 만들기
from sklearn.model_selection import train_test_split

train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)

print(train_scaled.shape, train_target.shape)
# (48000, 784) (48000,)

print(val_scaled.shape, val_target.shape)
# (12000, 784) (12000,)

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
model = keras.Sequential(dense)


## 인공신경망으로 패션 아이템 분류하기
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')
print(train_target[:10])
# [7 3 5 8 6 9 3 3 9 9]

model.fit(train_scaled, train_target, epochs=5)
'''
Epoch 1/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.6088 - accuracy: 0.7927
Epoch 2/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.4776 - accuracy: 0.8401
Epoch 3/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.4554 - accuracy: 0.8474
Epoch 4/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.4441 - accuracy: 0.8535
Epoch 5/5
1500/1500 [==============================] - 2s 1ms/step - loss: 0.4366 - accuracy: 0.8557
'''

model.evaluate(val_scaled, val_target)
'''
375/375 [==============================] - 0s 877us/step - loss: 0.4570 - accuracy: 0.8448
'''
