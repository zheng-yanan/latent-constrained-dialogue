import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope
import numpy as np


def output_projection_layer(num_units, num_symbols, name="output_projection"):

	def output_fn(outputs):

		with variable_scope.variable_scope('%s' % name):

			weights1 = tf.get_variable("weights1", [num_units, num_symbols])
			bias1 = tf.get_variable("biases1", [num_symbols])
			result_final = tf.nn.softmax(tf.matmul(outputs, weights1) + bias1)

		return result_final


	def sampled_sequence_loss(outputs, targets, masks):

		with variable_scope.variable_scope('decoder/%s' % name):

			batch_size = tf.shape(targets)[0]
			seq_len = tf.shape(targets)[1]

			weights1 = tf.get_variable("weights1", [num_units, num_symbols])
			bias1 = tf.get_variable("biases1", [num_symbols])

			# extra_information = tf.reshape(tf.tile(extra_information, [1, seq_len]), [batch_size*seq_len, -1])
			local_outputs = tf.reshape(outputs, [-1, num_units])
			# local_outputs = tf.concat([local_outputs, extra_information], -1)
			

			result_symbol_1 = tf.nn.softmax(tf.matmul(local_outputs, weights1) + bias1)
			result_final = tf.reshape(result_symbol_1, [batch_size, seq_len, -1])

			onehot_symbol_labels = tf.one_hot(targets, num_symbols)

			local_loss = -tf.reduce_sum(onehot_symbol_labels * tf.log(result_final), axis=-1)
			local_loss = local_loss * masks


			local_loss = tf.reduce_sum(local_loss, -1)

			# ppl_loss = tf.reduce_sum(local_loss) / (tf.reduce_sum(masks)+1e-12)

			return local_loss# , ppl_loss

	return output_fn, sampled_sequence_loss

