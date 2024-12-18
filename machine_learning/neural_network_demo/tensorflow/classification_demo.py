import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def train_softmax_model():
    # 数据准备
    data = load_iris()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    enc = OneHotEncoder(sparse_output=False)
    y = enc.fit_transform(y.reshape(-1, 1))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(type(X_train[0]))

    # 模型定义
    model = Sequential()
    model.add(Dense(10, input_dim=4, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 训练模型
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, verbose=1)

    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Accuracy on test set: {accuracy:.2f}')
    return history


def train_linear_model():
    # 数据准备
    data = load_iris()
    X, y = data.data, data.target
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(type(X_train[0]))

    # 模型定义
    model = Sequential()
    model.add(Dense(10, input_dim=4, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(3, activation='linear'))
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 训练模型
    history = model.fit(X_train, y_train, epochs=200, batch_size=16, validation_split=0.2, verbose=1)

    # 评估模型
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f'Accuracy on test set: {accuracy:.2f}')
    return history


def showHistory(history):
    # 可视化训练过程
    plt.figure(figsize=(12, 4))

    # 绘制训练和验证的损失值
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    # 绘制训练和验证的准确率
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()


showHistory(train_linear_model())
