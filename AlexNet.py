import tensorflow as tf


def conv_op(input, name, kernel_h, kernel_w, out_channels, strides, padding):
    # 卷积层
    in_channels = input.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(name=scope+"w", shape=[kernel_h, kernel_w, in_channels, out_channels],
                             dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input=input, filter=kernel, strides=(1, strides, strides, 1), padding=padding)
        biases = tf.Variable(tf.constant(0.0, shape=[out_channels], dtype=tf.float32), name="biases")
        z = tf.nn.bias_add(conv, biases)
        activation = tf.nn.relu(z, name=scope)
        return activation


def pooling_op(input, name, pooling_size, pooling_strides):
        output = tf.nn.max_pool(value=input, ksize=[1, pooling_size, pooling_size, 1],
                                strides=[1, pooling_strides, pooling_strides, 1], padding="VALID")
        return output


def LRN_layer(x,  name, R, alpha, beta, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=R, alpha=alpha, beta=beta, name=name, bias=bias)

def AlexNet(input, label, is_training):
    # conv1
    conv1 = conv_op(input, "conv1", 11, 11, 96, 4, "VALID")   # 55*55*96
    norm1 = LRN_layer(conv1, 'norm1', 2, 1e-4, 0.75, )
    pooling1 = pooling_op(norm1, "pooling1", 3, 2)    #27*27*96

    conv2 = conv_op(pooling1, "conv2", 5, 5, 256, 1, "SAME") #27*27*256
    norm2 = LRN_layer(conv2, 'norm2', 2, 1e-4, 0.75, )
    pooling2 = pooling_op(norm2, "pooling2", 3, 2)  #13*13*256

    conv3 = conv_op(pooling2, "conv3", 3, 3, 384, 1, "SAME")
    conv4 = conv_op(conv3, "conv4", 3, 3, 384, 1, "SAME")
    conv5 = conv_op(conv4, "conv5", 3, 3, 256, 1, "SAME")
    pooling2 = pooling_op(conv5, "pooling5", 3, 2)  # 13*13*2566
    p1 = tf.layers.flatten(pooling2)
    if is_training:
        tf.nn.dropout(p1, keep_prob=0.5)
    p2 = tf.layers.dense(inputs=p1, units=1024, activation=tf.nn.relu)
    if is_training:
        tf.nn.dropout(p2, keep_prob=0.5)
    p3 = tf.layers.dense(inputs=p2, units=1024, activation=tf.nn.relu)
    p4 = tf.layers.dense(inputs=p3, units=10, activation=None)
    # y_pre = tf.arg_max(tf.nn.softmax(logits=p4), 1, name="y_label")
    # label 是一个one-hot类型的数据
    # tf.nn.softmax_cross_entropy_with_logits()函数把神经网络模型的输出[batch, num_class]的数据和label做一个交叉熵损失
    # reduce_sum是交叉熵损失，reduce_mean是一个向量的平均
    cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=p4))

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(p4, -1), label), dtype=tf.float32))
    train_op = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    return cross_entropy, accuracy, train_op



