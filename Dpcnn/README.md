# DPCNN文本分类
超参数：

num_filters = 250

knenel_sice = 3

## 1.region embedding
对embedding进行一次卷积(valid)(3gram)，后产生的结果。此操作和TEXTCnn底层一致。

区别在于，此部分有两种方式，一种保留词序的卷积，一种是混乱词序(可能有助于减少过拟合)

## 2.卷积层
卷积属于等长卷积(same)

1.对region embedding结果进行卷积，产生conv1；

2.对conv1进行卷积，产生conv2；

3.将conv2 + region embedding进行残差连接(shortcut connection),得到conv3；

4.对conv3进行max_pooling(1/2),得到结果pool;

5.对pool进行卷积，得到pool1；

6.对pool1进行卷积，得到pool2；

7.将pool2 + pool进行残差连接，得到pool3；

8.重复4-7，产生pooling

9.对pooling进行max_pooling(1/2)，得到pooling；

10.对pooling进行全连接，然后再接softmax层。

## 3.max_pooling
每次max_pooling都是strides = 2,因此，随着深度增加，每次卷积的计算量都是减半的，网络层看着形似‘金字塔’;文章中称这种行为叫Downsamoling。

## 4.shortcut connection
随着网络层加深，每层的输入很容易接近0，主要作用是为了缓解梯度消失问题

feature maps number:
继续增加数量，并不能增加准确度，反而会增加计算量。



# 代码解读
最近刚学习tensorflow2.0 拿来练练手。
数据处理阶段：主要是分词，去停用词后取词频靠前的9999词

使用预训练的词嵌入
```
self.embedding = tf.keras.layers.Embedding(10000, 100,
                                           embeddings_initializer=tf.constant_initializer(embeddings),
                                           trainable=False)
```
下面就是主模型，跟上面介绍的 卷积层 过程一致。

# 总结
总得来说，Dpcnn模型并不复杂，效果很棒
数据使用的是以前的老数据，所有预训练词向量也用的是以前的。
tensorflow2.0真的是简化了很多，还有很多小细节，我需要继续学习。