import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences

# 数据集目录
dataset_directory = r"data1\sketch_datas"

# 加载手绘图像序列数据集
classes = ['apple', 'bear', 'bird', 'bicycle', 'bus', 'ambulance', 'pig', 'owl', 'cat', 'foot']
data = []

for class_name in classes:
    class_directory = os.path.join(dataset_directory, class_name + '.npy')
    sequences = np.load(class_directory)
    labels = [classes.index(class_name)] * len(sequences)
    data.extend(list(zip(sequences, labels)))

# 打乱数据集
np.random.shuffle(data)

# 划分训练集和测试集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_sequences, train_labels = zip(*train_data)
test_sequences, test_labels = zip(*test_data)

# 使用pad_sequences填充序列
train_sequences = pad_sequences(train_sequences, padding='post', dtype='float32')
test_sequences = pad_sequences(test_sequences, padding='post', dtype='float32')

# 构建双向LSTM 模型
model = Sequential()
model.add(Bidirectional(LSTM(128, input_shape=(None, 5))))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
history = model.fit(train_sequences, np.array(train_labels),
                    epochs=2, validation_data=(test_sequences, np.array(test_labels)))

# 保存模型
model.save("LSTM_hand_drawn_model2.h5")

# 绘制训练曲线
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 绘制准确率曲线
plt.plot(history.history['accuracy'], label='train_accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 在每类数据的测试集中随机抽取10个序列进行预测
for class_name in classes:
    class_directory = os.path.join(dataset_directory, class_name + '.npy')
    sequences = np.load(class_directory)
    class_indices = [i for i, label in enumerate(test_labels) if label == classes.index(class_name)]
    sample_indices = np.random.choice(class_indices, 10, replace=False)

    for i, index in enumerate(sample_indices):
        test_sequence = pad_sequences([test_sequences[index]], padding='post', dtype='float32')
        true_label = test_labels[index]

        # 预测
        predictions = model.predict(test_sequence)

        # 获取预测概率分布
        predicted_label = np.argmax(predictions)
        probability_distribution = predictions[0]

        # 输出预测结果
        print(f"True Label: {class_name}, Predicted Label: {classes[predicted_label]}")

        # 绘制预测概率分布的柱状图
        plt.bar(classes, probability_distribution)
        plt.title('Probability Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Probability')

        plt.show()
