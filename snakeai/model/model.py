import tensorflow as tf
import os
import numpy as np

from tensorflow.python.training.adam import AdamOptimizer


class Model:
    """ Represents a Snake agent powered by DQN with experience replay. """

    def __init__(self, inputs, network, check_point="dqn.ckpt"):
        self.saver = tf.train.Saver()
        self.summary_writer = tf.summary.FileWriter("/tmp/dqn")
        self.inputs = inputs
        self.network = network
        self.targets = tf.placeholder(tf.float32, shape=(None, self.output_shape[1]))

        summary_names = ["actions", "fruits_eaten", "timesteps_survived"]

        summary_variables = [tf.Variable(name=name + "_variable", dtype=tf.int8,
                                         initial_value=[0] * self.output_shape[1])
                             for name in summary_names if name == "actions"]

        summary_variables += [tf.Variable(name=name + "_variable", dtype=tf.int8, initial_value=0) for name in
                              summary_names if name != "actions"]
        summary_ops = [tf.assign(summary_variables[i],
                                 tf.placeholder(dtype=summary_variables[i].dtype))
                       for i in range(len(summary_names))]

        summary = [tf.summary.histogram(summary_names[i], summary_variables[i]) for i in range(1)]
        summary += [tf.summary.scalar(summary_names[i], summary_variables[i]) for i in range(1, len(summary_names))]

        self.summary_ops = summary_ops + [tf.summary.merge_all()]

        global_step = tf.train.create_global_step()
        self.loss = tf.losses.mean_squared_error(self.network, self.targets)
        optimizer = AdamOptimizer()
        self.train_step = optimizer.minimize(loss=self.loss, global_step=global_step)
        self.sess = tf.Session()

        with self.sess.as_default():
            tf.global_variables_initializer().run()

        if os.path.exists(check_point):
            self.saver.restore(self.sess, check_point)

    @property
    def input_shape(self):
        return tuple(self.inputs.get_shape().as_list())

    @property
    def output_shape(self):
        return tuple(self.network.get_shape().as_list())

    def train_on_batch(self, samples, targets, summary):
        global_step = tf.train.get_global_step()
        summary[0], _ = np.histogram(summary[0], bins=range(self.output_shape[1] + 1))
        feed_dict = {self.summary_ops[i]: summary[i] for i in range(len(summary))}
        feed_dict[self.inputs] = samples
        feed_dict[self.targets] = targets
        run_result = self.sess.run(fetches=dict(summary=self.summary_ops, loss=self.loss, train_step=self.train_step),
                                   feed_dict=feed_dict)
        self.summary_writer.add_summary(run_result["summary"][-1], tf.train.global_step(self.sess, global_step))
        return run_result["loss"]

    def predict(self, samples):
        return self.sess.run(fetches=self.network, feed_dict={self.inputs: samples})

    def save(self, file_name):
        self.saver.save(self.sess, "/tmp/dqn" + file_name)
