import tensorflow as tf
from AlexNet import AlexNet
from data_process import make_batch
batch_size = 128
epoch = 2
train_path = "CIFAR-10-data\\train\\"
input = tf.placeholder(dtype=tf.float32, shape=[None, 227, 227, 3])
label = tf.placeholder(dtype=tf.int64, shape=[None])
loss, accuracy, train_op = AlexNet(input, label, is_training=True)
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epoch):
        data = make_batch(filepath=train_path, batch_size=batch_size)
        for train_X, train_y in data:
            cost, acc, _ = sess.run([loss, accuracy, train_op], feed_dict={input: train_X, label: train_y})
            print("epoch = %d---------loss=%f---------accuracy=%f" % (i, cost, acc))
        saver.save(sess, "model/AlexNet-model.ckpt", global_step=epoch)


