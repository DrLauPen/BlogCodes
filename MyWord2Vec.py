import tensorflow as tf
import jieba
import warnings
import os
import numpy as np
import pandas as pd
from tensorflow.python.framework import graph_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

class Word2Vec:
    def __init__(self, data, label, n_iters, vocab_size):
        embed_size = 300  # 嵌入的维数是300
        x = tf.placeholder(tf.float32, shape=(None, vocab_size))
        y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

        # 隐藏层部分
        w = tf.Variable(tf.random_normal([vocab_size, embed_size]))  # 最后的word2vec权重矩阵,每一行对应一个词语呢
        b = tf.Variable(tf.random_normal([embed_size]))
        hiddenlayers = tf.add(tf.matmul(x, w), b)

        # 输出层部分
        w2 = tf.Variable(tf.random_normal([embed_size, vocab_size]))
        b2 = tf.Variable(tf.random_normal([vocab_size]))
        y = tf.add(tf.matmul(hiddenlayers, w2), b2)
        prediction = tf.nn.softmax(y)  # 交叉熵计算的结果有可能log0得出nan要小心，采用。clip_by_value()的方法解决

        loss = tf.reduce_mean(
            -tf.reduce_sum(y_label * tf.log(tf.clip_by_value(prediction, 1e-8, 1.0)), reduction_indices=1))

        trainstep = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(n_iters):
                sess.run(trainstep, feed_dict={x: data, y_label: label})
                if i % 1 == 0:
                    theloss = sess.run(loss, feed_dict={x: data, y_label: label})
                    print("loss:%s" % theloss)
            Weight = sess.run(w, feed_dict={x: data, y_label: label})
            np.savetxt("w.txt", Weight, delimiter=',')
            # graph_def = tf.get_default_graph().as_graph_def()
            # output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['w'])



def OntHotEncode(mystr):
    """将语料库转换成对应的skip-grams的训练独热码语料"""
    mywords = mystr.split(' ')
    words_bag = set(mywords)
    vocab_size = len(words_bag)  # 词袋的大小
    w2n_vocab = {}  # 从文字转换到数字
    for i, item in enumerate(words_bag):
        if item not in w2n_vocab.keys():
            w2n_vocab[item] = i

    for i, item in enumerate(mywords):  # 用独热码OneHot代替原来的词语
        vec = np.zeros(vocab_size)
        vec[w2n_vocab[item]] = 1
        mywords[i] = vec

    window = [-2, -1, 1, 2]  # 设定窗口的值,当前默认是2-grams
    data = []
    label = []
    for i, item in enumerate(mywords):  # 从当前的词语中提取出对应的skip-grams训练集合
        for step in window:  # 遍历窗口的元素
            if 0 < i + step < len(mywords):  # 对应添加元素
                data.append(item)
                label.append(mywords[i + step])

    return data, label, vocab_size


if __name__ == '__main__':
    mystr = "the quick brown fox jumps over the lazy dog"
    data, label, vocab_size = OntHotEncode(mystr)
    data = np.array(data, dtype='float32')
    label = np.array(label, dtype='float32')
    myw2v = Word2Vec(data, label, 20, vocab_size)

    #读取对应的嵌入权重（尚未还原哦）
    # weight = []
    # with open('w.txt') as f:
    #     for i in f:
    #         weight.append(i)


