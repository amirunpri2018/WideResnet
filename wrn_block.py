import tensorflow as tf
import numpy as np

class WRNBlock(object):
	def __init__(self, input_channel, output_channel,name,  downsize = False, first= False):
		self.input_channel = input_channel
		self.downsize = downsize
		self.first = first
		self.variables = {}
		self.name = name
		self.initializer = tf.contrib.layers.xavier_initializer_conv2d()
		self.regularizer = tf.contrib.layers.l2_regularizer(0.0005)
		self.output_channel =output_channel
		with tf.variable_scope(self.name) as scope:
			self.variables["kernel_1_weights"] = tf.get_variable("kernel_1_weights", [3,3,input_channel,output_channel], regularizer = self.regularizer, initializer = self.initializer, dtype = tf.float32)
			self.variables["kernel_1_bias"] = tf.get_variable("kernel_1_bias", [output_channel], regularizer = self.regularizer,initializer =self.initializer)
			if self.downsize:
				self.variables["kernel_downsize_weights"] = tf.get_variable("kernel_downsize_weights", [1,1,input_channel,output_channel], regularizer = self.regularizer,initializer = self.initializer, dtype = tf.float32)
				self.variables["kernel_downsize_bias"] = tf.get_variable("kernel_downsize_bias", [output_channel], regularizer = self.regularizer,initializer =self.initializer)
			self.variables["kernel_2_weights"] = tf.get_variable("kernel_2_weights", [3,3,output_channel,output_channel], regularizer = self.regularizer,initializer = self.initializer, dtype = tf.float32)
			self.variables["kernel_2_bias"] = tf.get_variable("kernel_2_bias", [output_channel], regularizer = self.regularizer,initializer =self.initializer)

	def forwardprop(self, input_tensor, reuse_var = True):
		with tf.variable_scope(self.name, reuse = reuse_var):
			pre_conv_1 = tf.nn.relu(tf.contrib.layers.batch_norm(input_tensor))
			if self.downsize:
				if self.first:
					conv1 = tf.nn.conv2d(pre_conv_1, self.variables["kernel_1_weights"] , [1,1,1,1], 'SAME')
					conv_down = tf.nn.conv2d(pre_conv_1, self.variables["kernel_downsize_weights"], [1,1,1,1], 'SAME')
				else:
					conv1 = tf.nn.conv2d(pre_conv_1, self.variables["kernel_1_weights"] , [1,2,2,1], 'SAME')
					conv_down = tf.nn.conv2d(pre_conv_1, self.variables["kernel_downsize_weights"],[1,2,2,1], "VALID")
				conv_down_out = tf.nn.bias_add(conv_down, self.variables["kernel_downsize_bias"])
			else:
				conv1 = tf.nn.conv2d(pre_conv_1, self.variables["kernel_1_weights"] , [1,1,1,1], 'SAME')
			conv1_out = tf.nn.bias_add(conv1, self.variables["kernel_1_bias"])
			pre_conv_2 = tf.nn.relu(tf.contrib.layers.batch_norm(conv1_out))
			conv2 = tf.nn.conv2d(pre_conv_2, self.variables["kernel_2_weights"], [1,1,1,1], 'SAME')
			conv2_out = tf.nn.bias_add(conv2, self.variables["kernel_2_bias"])
			if self.downsize:
				block_output = tf.add(conv2_out, conv_down_out)
			else:
				block_output = tf.add(conv2_out, input_tensor)

		return block_output
