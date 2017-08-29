# coding=utf-8
"""
@ license: Apache Licence
@ github: invoker4zoo
@ author: invoker/cc
@ wechart: whatshowlove
@ software: PyCharm
@ file: fastText_model.py
@ time: $17-8-29 下午3:13
"""

import tensorflow as tf
import numpy as np

class fastTextM(object):
    """
    fastText为一种极简的文本处理方式，原理是将句子中所有的词向量进行平均，然后直接介入softmax层进行标签分类
    适用于简单的任务，没有非线性变化
    """
    def __init__(self, label_size, learning_rate, batch_size, decay_steps, decay_rate, num_sampled, sentence_len,
                 vocab_size, embed_size, is_training):

        # init variable
        self.label_size = label_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.sentence_len = sentence_len
        # vocab_size 词id的数量，词库大小， embed_size 词向量的大小
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.decay_steps, self.decay_rate = decay_steps, decay_rate

        # init tf graph
        self.sentence = tf.placeholder(tf.int32, shape=[None, self.sentence_len], name="sentence")  # X
        self.labels = tf.placeholder(tf.int32, shape=[None], name="Labels")  # y
        self.global_step = tf.Variable(0, trainable=False, name="Global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.epoch_step = tf.Variable(0, trainable=False, name="Epoch_Step")
        # network params
        self.Embedding = tf.get_variable("Embedding", [self.vocab_size, self.embed_size])
        self.W = tf.get_variable("W", [self.embed_size, self.label_size])
        self.b = tf.get_variable("b", [self.label_size])
        self.logits = self.build_logits()

        if not is_training:
            print 'not for training, skip build training graph'
            return
        # build training graph
        self.loss = self.build_loss()
        self.train_op = self.build_train()
        # build output and calculate accuracy
        self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32),
                                      self.labels)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")


    def build_logits(self):
        """
        initial input embedding
        building hidden layer
        embedding_lookup:根据train_inputs中的id，寻找embeddings中的对应元素。比如，train_inputs=[1,3,5]，则找出embeddings中下标为1,3,5的向量组成一个矩阵返回
        Embedding 为词id与词向量的字典，为提前训练的字典，多使用word2vector
        """
        sentence_embeddings = tf.nn.embedding_lookup(self.Embedding, self.sentence)
        self.sentence_embeddings = tf.reduce_mean(sentence_embeddings, axis=1)
        logits = tf.matmul(self.sentence_embeddings, self.W) + self.b
        return logits


    def build_loss(self):
        """
        build loss function
        use nce_loss if is_training else use cross_entropy
        nce_loss 的计算是在每个batch随机采样负样本，
        采样的频率和word2vector的类别编号有关
        在word2vector 中，类别编号越大说明出现频率越高
        在[0, range_max)中采样出一个整数k
        概率P(k) = (log(k + 2) - log(k + 1)) / log(range_max + 1) k越大越难被采到
        """
        if self.is_training:
            labels=tf.reshape(self.labels,[-1])               #[batch_size,1]------>[batch_size,]
            labels=tf.expand_dims(labels,1)                   #[batch_size,]----->[batch_size,1]
            loss = tf.reduce_mean(
                        tf.nn.nce_loss(weights=tf.transpose(self.W),
                                       biases=self.b,
                                       labels=labels,
                                       inputs=self.sentence_embeddings,
                                       num_sampled=self.num_sampled,
                                       num_classes=self.label_size,partition_strategy="div")
                    )
        else:
            labels_one_hot = tf.one_hot(self.labels, self.label_size) #[batch_size]---->[batch_size,label_size]
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot,logits=self.logits)
            print("loss0:", loss)
            loss = tf.reduce_sum(loss, axis=1)
            print("loss1:",loss)

        return loss

    def train(self):
        """
        exponential_decay:退化学习率，对学习率进行指数衰退
        :return:
        """
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps,self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,learning_rate=learning_rate, optimizer="Adam")
        return train_op


def class_test():
    num_classes = 19
    learning_rate = 0.01
    batch_size = 8
    decay_steps = 1000
    decay_rate = 0.9
    sequence_length = 5
    vocab_size = 10000
    embed_size = 100
    is_training = True
    dropout_keep_prob = 1
    fastText = fastTextM(num_classes, learning_rate, batch_size, decay_steps, decay_rate,5,sequence_length,vocab_size,embed_size,is_training)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            input_x = np.zeros((batch_size,sequence_length),dtype=np.int32)
            input_y = np.array([1,0,1,1,1,2,1,1],dtype=np.int32)
            loss, acc, predict, _ = sess.run([fastText.loss_val,fastText.accuracy,fastText.predictions,fastText.train_op],
                                        feed_dict={fastText.sentence:input_x,fastText.labels:input_y})
            print("loss:",loss,"acc:",acc,"label:",input_y,"prediction:",predict)


if __name__=='__main__':
    class_test()