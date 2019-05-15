##### 导入cifar-10
```
#导入数据
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
import time


def unpickle(filename):
    with open(filename, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        return d

def onehot(labels):
    '''
    one-hot 编码
    '''
    n_sample = len(labels)
    n_class = max(labels) + 1
    onehot_labels = np.zeros((n_sample, n_class))
    onehot_labels[np.arange(n_sample), labels] = 1
    return onehot_labels

# 训练数据集
data1 = unpickle('data_batch_1')
data2 = unpickle('data_batch_2')
data3 = unpickle('data_batch_3')
data4 = unpickle('data_batch_4')
data5 = unpickle('data_batch_5')
X_train = np.concatenate((data1['data'], data2['data'], data3['data'], data4['data'], data5['data']), axis=0)
y_train = np.concatenate((data1['labels'], data2['labels'], data3['labels'], data4['labels'], data5['labels']), axis=0)
y_train = onehot(y_train)

# 测试数据集
test = unpickle('test_batch')
X_test = test['data'][:5000, :]
y_test = onehot(test['labels'])[:5000, :]

print('Training dataset shape:', X_train.shape)
print('Training labels shape:', y_train.shape)
print('Testing dataset shape:', X_test.shape)
print('Testing labels shape:', y_test.shape)
```
##### output:
```
Training dataset shape: (50000, 3072)
Training labels shape: (50000, 10)
Testing dataset shape: (5000, 3072)
Testing labels shape: (5000, 10)
```

##### 模型构建：
```
# 模型参数
learning_rate = 1e-3
train_iters = 200
batch_size = 50


# 构建模型
x = tf.placeholder(tf.float32, [None, 3072])

# 成本函数
y = tf.placeholder(tf.float32, [None, 10])

W_conv = {
    'conv1': tf.Variable(tf.truncated_normal([5, 5, 3, 32], stddev=0.0001)),
    'conv2': tf.Variable(tf.truncated_normal([5, 5, 32, 64],stddev=0.01)),
    'fc3': tf.Variable(tf.truncated_normal([8*8*64, 384], stddev=0.1)),
    'fc4': tf.Variable(tf.truncated_normal([384, 192], stddev=0.1)),
    'fc5': tf.Variable(tf.truncated_normal([192, 10], stddev=0.1))
}
b_conv = {
    'conv1': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[32])),
    'conv2': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[64])),
    'fc3': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[384])),
    'fc4': tf.Variable(tf.constant(0.1, dtype=tf.float32, shape=[192])),
    'fc5': tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[10]))
}

X_image = tf.reshape(x, [-1, 32, 32, 3])
# 第一层卷积
conv1 = tf.nn.conv2d(X_image, W_conv['conv1'], strides=[1, 1, 1, 1], padding='SAME')
conv1 = tf.nn.bias_add(conv1, b_conv['conv1'])
conv1 = tf.nn.relu(conv1)
# 第一层池化
pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# 第一LRN层，Local Response Normalization
norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
# 第二层卷积
conv2 = tf.nn.conv2d(norm1, W_conv['conv2'], strides=[1, 1, 1, 1], padding='SAME')
conv2 = tf.nn.bias_add(conv2, b_conv['conv2'])
conv2 = tf.nn.relu(conv2)
# 第二层LRN，Local Response Normalization
norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
# 第二层池化
pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
reshape = tf.reshape(pool2, [-1, 8*8*64])

#第三层 全连接
fc3 = tf.add(tf.matmul(reshape, W_conv['fc3']), b_conv['fc3'])
fc3 = tf.nn.relu(fc3)


#第四层全连接层
fc4 = tf.add(tf.matmul(fc3, W_conv['fc4']), b_conv['fc4'])
fc4 = tf.nn.relu(fc4)


# 第五层全连接层
fc5 = tf.nn.softmax(tf.add(tf.matmul(fc4, W_conv['fc5']), b_conv['fc5']))

# 定义损失
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc5, labels=y))
optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
#optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 评估模型
correct_prediction = tf.equal(tf.argmax(fc5, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    a = []
    total_batch = int(X_train.shape[0] / batch_size)
    start_time = time.time()
    for i in range(train_iters):
        for batch in range(total_batch):
            batch_x = X_train[batch*batch_size : (batch+1)*batch_size, :]
            batch_y = y_train[batch*batch_size : (batch+1)*batch_size, :]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        acc = accuracy.eval(session=sess, feed_dict={x: batch_x, y: batch_y})
        a.append(acc)
        end_time = time.time()
        print("step = %d, train_acc = %g, time = %g"%(i, acc, (end_time - start_time)))
        start_time = end_time
    print("Optimization Finished!")
    # Test
    test_acc = accuracy.eval(session=sess, feed_dict={x: X_test, y: y_test})
    print("Testing Accuracy:", test_acc)
    plt.plot(a)
    plt.xlabel('Iter')
    plt.ylabel('Acc')
    plt.title('lr = %f, ti = %d, bs = %d, acc = %f' % (learning_rate, train_iters, batch_size, test_acc))
    plt.tight_layout()
```

##### output

```
step = 183, train_acc = 0.64, time = 14.4998
step = 185, train_acc = 0.66, time = 14.4916
step = 186, train_acc = 0.68, time = 14.5461
step = 187, train_acc = 0.64, time = 14.5041
step = 188, train_acc = 0.64, time = 14.4781
step = 189, train_acc = 0.66, time = 14.4776
step = 190, train_acc = 0.64, time = 14.4871
step = 191, train_acc = 0.62, time = 14.5355
step = 192, train_acc = 0.64, time = 14.488
step = 193, train_acc = 0.64, time = 14.4715
step = 194, train_acc = 0.64, time = 14.4992
step = 195, train_acc = 0.64, time = 14.5334
step = 196, train_acc = 0.64, time = 14.4865
step = 197, train_acc = 0.68, time = 14.5028
step = 198, train_acc = 0.68, time = 14.4849
step = 199, train_acc = 0.64, time = 14.5161
Optimization Finished!
Testing Accuracy: 0.4966
```
