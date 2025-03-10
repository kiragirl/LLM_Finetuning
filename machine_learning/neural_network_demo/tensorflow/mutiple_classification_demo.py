import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 超参数
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# 将标签转换为 one-hot 编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# 定义 CNN 模型
def create_cnn_model():
    model = models.Sequential()

    # 卷积层 1
    model.add(layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1), padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # 卷积层 2
    model.add(layers.Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2)))

    # 展平层
    model.add(layers.Flatten())

    # 全连接层 1
    model.add(layers.Dense(1024, activation='relu'))

    # 输出层（全连接层 2）
    model.add(layers.Dense(10, activation='softmax'))

    return model


# 创建模型
model = create_cnn_model()

# 打印模型结构
model.summary()

# 编译模型
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])
#
# # 训练模型
# history = model.fit(train_images, train_labels,
#                     epochs=num_epochs,
#                     batch_size=batch_size,
#                     validation_data=(test_images, test_labels))
#
# # 测试模型
# test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
# print(f'Test Accuracy: {test_acc * 100:.2f}%')
#
# # 绘制训练和验证的损失曲线
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Training and Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# # 绘制训练和验证的准确率曲线
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Training and Validation Accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
#
#
# # 可视化预测结果
# def visualize_predictions(model, test_images, test_labels, num_samples=5):
#     indices = tf.random.uniform([num_samples], minval=0, maxval=len(test_images), dtype=tf.int32)
#     sample_images = tf.gather(test_images, indices)
#     sample_labels = tf.gather(test_labels, indices)
#
#     predictions = model.predict(sample_images)
#     predicted_classes = tf.argmax(predictions, axis=1)
#     true_classes = tf.argmax(sample_labels, axis=1)
#
#     fig, axes = plt.subplots(1, num_samples, figsize=(10, 2))
#     for i in range(num_samples):
#         ax = axes[i]
#         ax.imshow(sample_images[i].squeeze(), cmap='gray')
#         ax.set_title(f'Pred: {predicted_classes[i]}, True: {true_classes[i]}')
#         ax.axis('off')
#
#     plt.show()
#
#
# # 调用可视化函数
# visualize_predictions(model, test_images, test_labels)