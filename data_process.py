import os
import numpy as np
import cv2
import random


def make_data(filepath ):
    fo = open(os.path.join(filepath, "lables.txt"), "w", encoding="UTF-8")
    for i in range(0, 10):
        picture_names = os.listdir(os.path.join(filepath, str(i)))
        for name in picture_names:
            fo.write(name + " " + str(i) + "\n")
    fo.close()

def make_batch(filepath ,batch_size):
    fi = open(os.path.join(filepath, 'lables.txt'), "r")
    lines = fi.readlines()
    indexes = np.arange(len(lines))
    random.shuffle(indexes)
    batch_X = []
    batch_y = []
    for index in indexes:
        sp = lines[index].strip().split()
        assert len(sp) == 2
        image = cv2.imread(os.path.join(filepath, sp[1], sp[0]))
        image = cv2.resize(image, dsize=(227, 227))/255
        batch_X.append(image)
        batch_y.append(int(sp[1]))
        if batch_size == len(batch_X):
            yield np.array(batch_X), np.array(batch_y)
            batch_X = []
            batch_y = []
    if len(batch_X) > 0:
        yield np.array(batch_X), np.array(batch_y)


# if __name__ == "__main__":
#     train_path = "./CIFAR-10-data/train"
#     test_path = "./CIFAR-10-data/test"
#     make_data(test_path)   # 对训练数据或者测试数据生成一个label.txt的文件，该文档存储的是：图片名称+图片分类标签

