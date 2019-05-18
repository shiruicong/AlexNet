import tensorflow as tf
from AlexNet import AlexNet
from data_process import make_batch
batch_size = 128

test_path = "CIFAR-10-data\\test\\"
input = tf.placeholder(dtype=tf.float32, shape=[None, 227, 227, 3])
label = tf.placeholder(dtype=tf.int64, shape=[None])
loss, accuracy, train_op = AlexNet(input, label, is_training=False)
saver = tf.train.Saver()
model_file = tf.train.latest_checkpoint("model/")
with tf.Session() as sess:
    saver.restore(sess, model_file)
    data = make_batch(filepath=test_path, batch_size=batch_size)
    total_loss = 0
    total_acc = 0
    count = 0
    for train_X, train_y in data:
        cost, acc = sess.run([loss, accuracy], feed_dict={input: train_X, label: train_y})
        total_loss += cost
        total_acc += acc
        count = count + 1
        print("loss=%f---------accuracy=%f" % (cost, acc))
    print("avarage_loss:%f ---------  avarage_acc:%f" % (total_loss/count, total_acc/count))


