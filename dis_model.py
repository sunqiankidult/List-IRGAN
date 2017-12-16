import tensorflow as tf
import cPickle
import numpy as np

class DIS():
    M=7
    N=5

    def __init__(self, itemNum, userNum, emb_dim, lamda, param=None, initdelta=1, learning_rate=0.05):
        self.itemNum = itemNum
        self.userNum = userNum
        self.emb_dim = emb_dim
        self.lamda = lamda  # regularization parameters
        self.param = param
        self.initdelta = initdelta
        self.learning_rate = learning_rate
        self.d_params = []
        self.curUserNum = -1

        with tf.variable_scope('discriminator'):
            if self.param == None:
                self.user_embeddings = tf.Variable(
                    tf.random_uniform([self.userNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_embeddings = tf.Variable(
                    tf.random_uniform([self.itemNum, self.emb_dim], minval=-self.initdelta, maxval=self.initdelta,
                                      dtype=tf.float32))
                self.item_bias = tf.Variable(tf.zeros([self.itemNum]))
            else:
                self.user_embeddings = tf.Variable(self.param[0])
                self.item_embeddings = tf.Variable(self.param[1])
                self.item_bias = tf.Variable(self.param[2])

        self.d_params = [self.user_embeddings, self.item_embeddings, self.item_bias]

        # placeholder definition
        self.u = tf.placeholder(tf.int32)
        self.i = tf.placeholder(tf.int32)
        self.label = tf.placeholder(tf.float32)

        self.u_embedding = tf.nn.embedding_lookup(self.user_embeddings, self.u)
        self.i_embedding = tf.nn.embedding_lookup(self.item_embeddings, self.i)
        self.i_bias = tf.gather(self.item_bias, self.i)

        # self.pre_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias
        #
        # self.pre_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label,
        #                                                          logits=self.pre_logits) + self.lamda * (
        #     tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias)
        # )
        #
        # d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        # self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)
        #
        # self.reward_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding),
        #                                    1) + self.i_bias
        # self.reward = 2 * (tf.sigmoid(self.reward_logits) - 0.5)
        # ------------

        self.fui = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias
        self.objectiveR = self.fui
        def dcg_u(x):
            rui = tf.reduce_sum(tf.sigmoid(self.M *
                                           tf.map_fn(lambda y:y-x[0],self.fui)))
            return tf.sigmoid(self.N - rui) * (tf.pow(2.0, x[1]) - 1.) / tf.log(rui + 2.)

        # self.dcgu = dcg_u((self.fui[0],self.label[0]))
        # for x in range(1,self.curUserNum):
        #     # print x
        #     self.dcgu = tf.concat([self.dcgu,dcg_u((self.fui[x],self.label[x]))],0)
        self.dcgu = tf.map_fn(dcg_u,(self.fui,self.label),dtype=tf.float32)
        self.objectiveF = tf.reduce_sum(self.dcgu)

        self.pre_loss= -tf.reduce_sum(self.dcgu) + self.lamda * (
            tf.nn.l2_loss(self.u_embedding) + tf.nn.l2_loss(self.i_embedding) + tf.nn.l2_loss(self.i_bias)
        )

        d_opt = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.d_updates = d_opt.minimize(self.pre_loss, var_list=self.d_params)

        self.reward_fui = tf.reduce_sum(tf.multiply(self.u_embedding, self.i_embedding), 1) + self.i_bias

        # self.reward = dcg_u((self.reward_fui[0], self.label[0]))
        # for x in range(1,self.curUserNum):
        #     self.reward = tf.concat([self.reward,dcg_u((self.reward_fui[x],self.label[x]))],0)
        self.reward = tf.map_fn(dcg_u, (self.reward_fui, self.label), dtype=tf.float32)

        # ------------
        # for test stage, self.u: [batch_size]
        self.all_rating = tf.matmul(self.u_embedding, self.item_embeddings, transpose_a=False,
                                    transpose_b=True) + self.item_bias

        self.all_logits = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias
        self.NLL = -tf.reduce_mean(tf.log(
            tf.gather(tf.reshape(tf.nn.softmax(tf.reshape(self.all_logits, [1, -1])), [-1]), self.i))
        )
        # for dns sample
        self.dns_rating = tf.reduce_sum(tf.multiply(self.u_embedding, self.item_embeddings), 1) + self.item_bias

    def save_model(self, sess, filename):
        param = sess.run(self.d_params)
        cPickle.dump(param, open(filename, 'w'))

    def setCurUserNum(self,cur):
        self.curUserNum = cur