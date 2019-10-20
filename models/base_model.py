import os
import re
import sys
import time

import numpy as np
import tensorflow as tf



class BaseModel(object):

	global_t = tf.placeholder(dtype=tf.int32, name="global_t")
	learning_rate = None
	scope = None

	@staticmethod
	def print_model_stats(tvars):
		total_parameters = 0
		for variable in tvars:
			shape = variable.get_shape()
			variable_parametes = 1
			for dim in shape:
				variable_parametes *= dim.value
			print("Trainable %s with %d parameters" % (variable.name, variable_parametes))
			total_parameters += variable_parametes
		print("Total number of trainable parameters is %d" % total_parameters)


	@staticmethod
	def create_rnn_cell(cell_size, cell_type="gru", keep_prob=1.0, num_layer=1):
		cells = []
		for _ in range(num_layer):
			if cell_type == "gru":
				cell = tf.nn.rnn_cell.GRUCell(cell_size)
			else:
				cell = tf.nn.rnn_cell.LSTMCell(cell_size, use_peepholes=False, forget_bias=1.0)
			if keep_prob < 1.0:
				cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
			cells.append(cell)
		if num_layer == 1:
			cell = cells[0]
		else:
			cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
		return cell


	@staticmethod
	def print_loss(prefix, loss_names, losses, postfix):
		template = "%s "
		for name in loss_names:
			template += "%s " % name
			template += " %f "
		template += "%s"
		template = re.sub(' +', ' ', template)
		avg_losses = []
		values = [prefix]

		for loss in losses:
			values.append(np.mean(loss))
			avg_losses.append(np.mean(loss))
		values.append(postfix)

		print(template % tuple(values))
		return avg_losses


	def train(self, global_t, sess, train_feed):
		raise NotImplementedError("Train function needs to be implemented")


	def valid(self, *args, **kwargs):
		raise NotImplementedError("Valid function needs to be implemented")


	def test(self, *args, **kwargs):
		raise NotImplementedError("Test function needs to be implemented")


	def batch_2_feed(self, *args, **kwargs):
		raise NotImplementedError("Implement how to unpack the back")


	def optimize(self, sess, config, loss, log_dir):
		if log_dir is None:
			return
		if self.scope is None:
			tvars = tf.trainable_variables()
		else:
			tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.scope)
		grads = tf.gradients(loss, tvars)

		if config.grad_clip is not None:
			grads, _ = tf.clip_by_global_norm(grads, tf.constant(config.grad_clip))
		"""
		if config.grad_noise > 0:
			grad_std = tf.sqrt(config.grad_noise / tf.pow(1.0 + tf.to_float(self.global_t), 0.55))
			grads = [g + tf.truncated_normal(tf.shape(g), mean=0.0, stddev=grad_std) for g in grads]
		"""
		if config.op == "adam":
			print("Use Adam")
			optimizer = tf.train.AdamOptimizer(config.init_lr)
		elif config.op == "rmsprop":
			print("Use RMSProp")
			optimizer = tf.train.RMSPropOptimizer(config.init_lr)
		else:
			print("Use SGD")
			optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

		self.train_ops = optimizer.apply_gradients(zip(grads, tvars))
		self.print_model_stats(tvars)
		train_log_dir = os.path.join(log_dir, "checkpoints")
		print("Save summary to %s" % log_dir)
		self.train_summary_writer = tf.summary.FileWriter(train_log_dir, sess.graph)