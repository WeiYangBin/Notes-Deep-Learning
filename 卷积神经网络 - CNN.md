## <center>卷积神经网络 - CNN</center>
###### <center>2018, OCT 4</center>

[AlexNet原文 ](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

之前有看过一些自己方向的论文，但是总体还是属于一种蒙的状态，然后觉得还是有必要把一些Base paper拿出来看看，毕竟基础很重要嘛， 写写随笔谈谈收获。 

![image](https://github.com/WeiYangBin/Notes-Deep-Learning/blob/master/Picture/Alexnet.png)

上面这个就是CNN的网络架构，跑的数据集是ImageNet

那么我们现在就来“剖析”一下，把deep Learning推上热门的CNN,因为Alex团队用了两个GPU跑，所以图片中的架构被分为两半，而我们下面放在一起进行讨论

##### Step 1: 
输入图片数据(227 * 227 * 3),可能有人会有疑问上图中的大小是（224 * 224 * 3 ）上图中的(224 *  224 * 3)应该是错的，为什么错呢，因为不符合公式，不符合逻辑，不符合常理，没道理呀，哈哈哈，所以也就是
```
Input : Image_size = [227 * 227 * 3], [ height, weight, channel]
```


计算卷积后的图片大小的公式如下
```math
[(h + 2p - f )/ s + 1  ,  (w + 2p - f )/ s + 1]

h :height  

w:weight

p:padding

f:filter

s = strides
```

##### Step 2:

对输入的图片进行**第一层卷积**

如论文中的图片，利用[11 * 11 * 3]的核对图片进行卷积,之后对卷积后的图片进行activation，normalization,和pooling,
activation,normalization不会改变图像大小故不变。
```
Convolution：filter/kernel = [11 * 11 * 3], [ height, weight, channel] 
filter/kernel 个数： 96
strides = 4
padding = 'Valid'  Valid 也就是无填充 padding = 0

#利用上面公式计算(227 + 2 * 0 - 11) / 4 + 1  = 55 
conv1 = [55 * 55 * 96]
conv1 = relu(conv1) #size = [55 * 55 * 96]
conv1 = lrn(conv1)  #size = [55 * 55 * 96]

Max_pool : kernel_size = (3 * 3), strides = 2
#利用上面公式计算(55 - 3) / 2 + 1  = 27 
Max_pool1 = [27 * 27 * 96]
```

##### Step 3:

第一层的池化输出进行**第二层卷积**

采用了
'SAME'填充的方式，'SAME'也就是保存图片大小不变进行卷积
```
Convolution：filter/kernel = [5 * 5 * 96], [ height, weight, channel] 
filter/kernel 个数： 256
strides = 1
padding = 'SAME'   padding = 2

#利用上面公式计算(27 + 2 * 2 - 5) / 1 + 1  = 55 
conv2 = [27 * 27 * 256]
conv2 = relu(conv2) #size = [27 * 27 * 256]
conv2 = lrn(conv2)  #size = [27 * 27 * 256]

Max_pool : kernel_size = (3 * 3), strides = 2
#利用上面公式计算(27 + 2 * 0 - 3) / 2 + 1  = 13 
Max_pool2 = [13 * 13 * 256]
```
##### Step 3:

第二层的池化输出进行**第三层卷积**

第三层卷积不再进行池化

```
Convolution：filter/kernel = [3 * 3 * 256], [ height, weight, channel] 
filter/kernel 个数： 384
strides = 1
padding = 'SAME'   padding = 1

#利用上面公式计算(13 + 2 * 1 - 3) / 1 + 1  = 13 
conv3 = [13 * 13 * 384]
conv3 = relu(conv3) #size = [13 * 13 * 384]

```
##### Step 4:

第三层的激活输出进行**第四层卷积**

第四层卷积同样不再进行池化

```
Convolution：filter/kernel = [3 * 3 * 384], [ height, weight, channel] 
filter/kernel 个数： 384
strides = 1
padding = 'SAME'   padding = 1

#利用上面公式计算(13 + 2 * 1 - 3) / 1 + 1  = 13 
conv4 = [13 * 13 * 384]
conv4 = relu(conv4) #size = [13 * 13 * 384]
```

##### Step 5:

第四层的激活输出进行**第五层卷积**


```
Convolution：filter/kernel = [3 * 3 * 384], [ height, weight, channel] 
filter/kernel 个数： 256
strides = 1
padding = 'SAME'   padding = 1

#利用上面公式计算(13 + 2 * 1 - 3) / 1 + 1  = 13
conv4 = [13 * 13 * 256]
conv4 = relu(conv4) #size = [13 * 13 * 256]

Max_pool : kernel_size = (3 * 3), strides = 2
#利用上面公式计算(13 + 2 * 0 - 3) / 2 + 1  = 6 
Max_pool4 = [6 * 6 * 256]
```
##### Step 6:

第五层的池化层输出进行**第六层全连接**

平铺开我们第五层的池化层与第六层的4096个神经元进行全连接
```
全连接
fully_connect6 = [6 * 6 * 256, 4096]
fully_connect6 = relu(fully_connect6 )
dropout(fully_connect6)   #dropout随机失活，防止过拟合的一种方式
```

##### Step 7:

第六层的全连接输出进行**第七层全连接**

第六层4096个神经元与第七层的4096个神经元进行全连接
```
全连接
fully_connect7 = [4096, 4096]
fully_connect7 = relu(fully_connect7 )
dropout(fully_connect7)   #dropout随机失活，防止过拟合的一种方式
```
##### Step 8:

第七层的全连接输出进行**第八层softmax输出层**

第七层4096个神经元与第八层的1000个神经元进行全连接
```
输出层
out_put = [4096, 1000]
```
因为有1000类所以输出为1000，利用softmax进行评分。

本人基于tensorflow搭了一个拙劣的cnn 跑了MNIST和cifar-10，Accuracy不高，只为通过代码更好的理解CNN，有兴趣的话点击下方查看代码：

[MNIST手写体数字识别代码](https://github.com/WeiYangBin/Notes-Deep-Learning/blob/master/MNIST%E6%89%8B%E5%86%99%E4%BD%93%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E4%BB%A3%E7%A0%81.md)

[cifar-10图像识别代码](https://github.com/WeiYangBin/Notes-Deep-Learning/blob/master/cifar-10%E5%9B%BE%E5%83%8F%E8%AF%86%E5%88%AB%E4%BB%A3%E7%A0%81.md)
