import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

print(tf.__version__)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()  # 获取训练数据和测试数据的值与标签
x_train = x_train.reshape((-1, 28 * 28)) / 255.0  # 将训练数据重构为28×28格式，并将值归并到0~1
x_test = x_test.reshape((-1, 28 * 28)) / 255.0  # 将测试数据重构为28×28格式，并将值归并到0~1

print("There have %s sample_train" % x_train.shape[0])  # 60000训练
print("There have %s sample_test" % x_test.shape[0])  # 10000测试

target = 7
tx = x_train[y_train == target]  # 找出标签为7的训练数据
length = tx.shape[0]

code_dim = 64  # 隐藏层第一层节点数
code_dim2 = 16  # 隐藏层第二层节点数
inputs = layers.Input(shape=(x_train.shape[1],), name='inputs')  # 构建网络的第一层
code1 = layers.Dense(code_dim, activation='sigmoid', name='code1')(inputs)  # 全连接
code2 = layers.Dense(code_dim2, activation='sigmoid', name='code2')(code1)
code3 = layers.Dense(code_dim, activation='sigmoid', name='code3')(code2)
outputs = layers.Dense(x_train.shape[1], activation='sigmoid', name='outputs')(code3)

auto_encoder = keras.Model(inputs, outputs)  # 构建从输入层到输出层的整体模型
auto_encoder.summary()  # 输出参数状况-节点数，连接数

encoder = keras.Model(inputs, code2)  # 构建输入层到中间层模型-编码器
decoder_input = keras.Input((code_dim2,))  # 构建中间层解码器
decoder_output1 = auto_encoder.layers[-2](decoder_input)
decoder_output = auto_encoder.layers[-1](decoder_output1)
decoder = keras.Model(decoder_input, decoder_output)  # 构建中间层到输出层模型-解码器

auto_encoder.compile(optimizer='Adam',  # adam优化器
                     loss='binary_crossentropy')  # 损失函数：交叉熵

history = auto_encoder.fit(tx, tx, batch_size=64, epochs=230, validation_split=0.1)
auto_encoder.save('auto_encoder.h5')

encoded = encoder.predict(x_test)
decoded = decoder.predict(encoded)

inputs2 = layers.Input(shape=(x_train.shape[1],), name='inputs2')  # 构建识别正确目标概率网络
code12 = layers.Dense(28, activation='sigmoid', name='code12')(inputs2)
output2 = layers.Dense(1, activation='sigmoid', name='output2')(code12)
model2 = keras.Model(inputs2, output2)
model2.compile(optimizer='adam', loss='binary_crossentropy')
his = model2.fit(x_train, (y_train == target).astype('int'), batch_size=1024, epochs=20, class_weight={1: 9, 0: 1},
                 validation_split=0.1)
model2.save('model2.h5')

Y = model2.predict(decoded)
[print("%.2f" % i, end=' ') for i in np.concatenate(Y)[15:30]]
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# cm = confusion_matrix(y_test==7,np.where(Y>0.5,1,0))
# print(Y.sum(),'\n',cm)
for item in his.history:
    plt.plot(his.history[item])
    plt.title(item)
    plt.show()

plt.hist(Y)
plt.show()
plt.figure(figsize=(10, 4))

n = 15
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[15+i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, n + i + 1)
    plt.imshow(decoded[15+i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
