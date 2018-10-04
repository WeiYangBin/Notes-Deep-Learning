
```
import tensorflow as tf

#权重初始化
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev = 0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape = shape)
  return tf.Variable(initial)

#卷积， 池化
def conv2d(x, W):
  '''
  x ：4D的tensor，[batch,height,width,channels]
  W ：4D的tensor，[height,width,in_channel,out_channel]
  strides ： 步长，D的tensor，[1,stride,stride,1]，一般第一维和第四维都是1，因为很少有对batch和channel进行卷积计算
  '''
  return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME') 


def max_pool_2x2(x):
  '''
  value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
  ksize：池化窗口的大小，4D的tensor,[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
  strides：和卷积类似，窗口在每一个维度上滑动的步长，[1, stride,stride, 1]
  padding：和卷积类似，可以取’VALID’ 或者’SAME’，返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式。
  '''
  return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

```

```
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST.data/', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

#第一层卷积
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_conv1 = max_pool_2x2(h_conv1)

#第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_pool2 =  max_pool_2x2(h_conv2)

#全连接层
W_fc3 = weight_variable([7 * 7 * 64, 1024])
b_fc3 = bias_variable([1024])

h_pool3_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc3 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc3))

#随机失活dropout
keep_prob = tf.placeholder(tf.float32)
h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)

#输出层
W_fc4 = weight_variable([1024, 10])
b_fc4 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)

#成本函数
y_ = tf.placeholder(tf.float32, [None, 10])

#训练，评估
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))          
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

sess.run(tf.global_variables_initializer())
for i in range(2000):
  batch = mnist.train.next_batch(50)
  if i % 100 == 0:
    train_accuracy = accuracy.eval(session=sess, feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(session=sess, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("test accuracy %g"%accuracy.eval(session=sess, feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```

```
输出结果如下：
Extracting MNIST.data/train-images-idx3-ubyte.gz
Extracting MNIST.data/train-labels-idx1-ubyte.gz
Extracting MNIST.data/t10k-images-idx3-ubyte.gz
Extracting MNIST.data/t10k-labels-idx1-ubyte.gz
step 0, training accuracy 0.08
step 100, training accuracy 0.84
step 200, training accuracy 0.86
step 300, training accuracy 0.98
step 400, training accuracy 0.94
step 500, training accuracy 0.9
step 600, training accuracy 1
step 700, training accuracy 0.98
step 800, training accuracy 0.96
step 900, training accuracy 0.94
step 1000, training accuracy 0.94
step 1100, training accuracy 0.96
step 1200, training accuracy 0.98
step 1300, training accuracy 0.96
step 1400, training accuracy 0.96
step 1500, training accuracy 0.98
step 1600, training accuracy 1
step 1700, training accuracy 0.98
step 1800, training accuracy 0.98
step 1900, training accuracy 0.98
test accuracy 0.9763
```
如果range改为20000，最后我们的test accuracy会达到99.28