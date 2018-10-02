import tensorflow as tf
import numpy as np
from wrn_block import WRNBlock
import os
import datetime

#depth_per_down = (n-4)/6 for WRN-n-w where w is widening_factor

# WRN-16-1 : Model Test Error : 16.86%


class WRN(object):

	def __init__(self, input_dim, n_classes, initial_lr, depth_per_down = 2, widening_factor = 1):
		self.learning_rate = tf.placeholder(tf.float32, shape = [])
		self.lr_value = initial_lr
		self.input_dim = input_dim
		self.n_classes = n_classes
		self.widening_factor = widening_factor
		self.depth_per_down = depth_per_down
		self.name = "WRU-{}-{}".format((self.depth_per_down*6)+4,self.widening_factor)
		self.regularizer = tf.contrib.layers.l2_regularizer(0.0005)
		self.input_img = tf.placeholder(tf.float32, shape = [None] + list(self.input_dim))
		self.input_labels = tf.placeholder(tf.int32, [None, ])
		input_labels_1h = tf.one_hot(self.input_labels,self.n_classes, on_value=1.0, off_value = 0.0)
		self.initlize_vars()
		self.final_logits = self.forwardprop(self.input_img, reuse_var = False)
		# reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES,self.name)
		# reg_term = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(0.0005), reg_variables)
		self.prediction = tf.argmax(tf.nn.softmax(self.final_logits, axis = -1), axis = -1)
		self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.final_logits, labels=input_labels_1h)) #+ reg_term
		self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)

		tf.summary.scalar(name="Model_X_entropy Loss", tensor=self.loss)
		self.summary_op = tf.summary.merge_all()

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)
		self.saver = tf.train.Saver()
		self.results_path = "./Results"


	def initlize_vars(self):
		self.model_vars = {}
		with tf.variable_scope(self.name) as scope:
			self.model_vars["conv1_kernel_weights"] = tf.get_variable("kernel_1_weights", [3,3,self.input_dim[-1],16], regularizer = self.regularizer, initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
			self.model_vars["conv1_kernel_bias"] = tf.get_variable("kernel_1_bias", [16], regularizer = self.regularizer, initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
			input_channel = 16
			for i in range(2,5):
				for j in range(1,self.depth_per_down +1):
					if j == 1:
						downsize = True
						if i == 2:
							output_channel = input_channel*self.widening_factor
							first = True
						else:
							output_channel = input_channel*2
							first = False
					else:
						output_channel = input_channel
						downsize = False
						first = False
					self.model_vars["conv{}_block{}".format(i,j)] = WRNBlock(input_channel, output_channel, "conv{}_block{}".format(i,j), downsize = downsize, first =first)
					input_channel = output_channel
			self.model_vars["final_dense_weights"] = tf.get_variable("final_dense_weights",[64*self.widening_factor,self.n_classes], regularizer = self.regularizer, initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
			self.model_vars["final_dense_bias"] = tf.get_variable("final_dense_bias",[self.n_classes], regularizer = self.regularizer, initializer = tf.contrib.layers.xavier_initializer(), dtype = tf.float32)

	def forwardprop(self, input_tensor,reuse_var =True):
		with tf.variable_scope(self.name, reuse = reuse_var):
			conv_init = tf.nn.conv2d(input_tensor, self.model_vars["conv1_kernel_weights"],[1,1,1,1],"SAME")
			out_tensor = tf.nn.bias_add(conv_init,self.model_vars["conv1_kernel_bias"])
			for i in range(2,5):
				for j in range(1,self.depth_per_down + 1):
					out_tensor = self.model_vars["conv{}_block{}".format(i,j)].forwardprop(out_tensor,reuse_var = reuse_var)
			out_tensor = tf.nn.relu(tf.contrib.layers.batch_norm(out_tensor))
			out_tensor = tf.layers.average_pooling2d(out_tensor, (self.input_dim[0]/4,self.input_dim[1]/4), 1)
			out_tensor = tf.contrib.layers.flatten(out_tensor)
			final_logits = tf.matmul(out_tensor, self.model_vars["final_dense_weights"]) + self.model_vars["final_dense_bias"]
		return final_logits

	def create_checkpoint_folders(self, batch_size, n_epochs):
		folder_name = "/{0}_{1}_{2}_wrn-{3}-{4}".format(
			datetime.datetime.now(),
			batch_size,
			n_epochs, (self.depth_per_down*6)+4,self.widening_factor).replace(':', '-')
		tensorboard_path = self.results_path + folder_name + '/tensorboard'
		saved_model_path = self.results_path + folder_name + '/saved_models/'
		log_path = self.results_path + folder_name + '/log'
		if not os.path.exists(self.results_path + folder_name):
			os.mkdir(self.results_path + folder_name)
			os.mkdir(tensorboard_path)
			os.mkdir(saved_model_path)
			os.mkdir(log_path)
		return tensorboard_path, saved_model_path, log_path

	def load(self, modelpath):
		self.saver.restore(self.sess, save_path=tf.train.latest_checkpoint(modelpath))
		return None

	def get_learing_rate(self,epoch):
		decrease_lr = [60,120,160]
		if epoch in decrease_lr:
			self.lr_value *= 0.8
		return self.lr_value

	def test(self, dataset):
		total = len(dataset.test_x)
		test_batch_size = int(total/10)
		test_label = dataset.test_y.reshape((total,))
		predictions = []
		for i in range(10):
			batch_x = dataset.test_x[test_batch_size*i : test_batch_size* (i + 1)]
			test_label_batch = test_label[test_batch_size*i : test_batch_size* (i + 1)]
			prediction = self.sess.run(self.prediction, feed_dict = {self.input_img : batch_x, self.input_labels : test_label_batch})
			predictions.append(prediction)
		predictions = np.array(predictions).reshape((total,))
		error = 0
		for ind, prediction in enumerate(predictions):
			if prediction != dataset.test_y[ind]:
				error += 1
		test_error = float(error/total) * 100
		print( "Model Test Error : {}%".format(test_error))

	def train(self, dataset, batch_size = 128, epochs = 100):
		if not os.path.exists(self.results_path):
			os.mkdir(self.results_path)

		self.step = 0
		self.tensorboard_path, self.saved_model_path, self.log_path = self.create_checkpoint_folders(batch_size, epochs)
		self.writer = tf.summary.FileWriter(logdir=self.tensorboard_path, graph=self.sess.graph)

		batch_p_epoch = int(dataset.size/batch_size)

		for epoch in range(1, epochs + 1):
			print("------------------Epoch {}/{}------------------".format(epoch, epochs))

			for batch in range(batch_p_epoch):
				batch_train, batch_label = dataset.get_batch(batch_size)

				self.sess.run(self.optimizer, feed_dict={self.input_img: batch_train, self.input_labels: batch_label, self.learning_rate : self.get_learing_rate(epoch)})

				# Print log and write to log.txt every 50 batches
				if batch % 50 == 0:
					loss = self.sess.run(self.loss, feed_dict = {self.input_img: batch_train, self.input_labels: batch_label})
					summary = self.sess.run(self.summary_op, feed_dict = {self.input_img: batch_train, self.input_labels: batch_label})
					self.writer.add_summary(summary, global_step=self.step)
					print("Epoch: {}, iteration: {}".format(epoch, batch))
					print("Model Loss: {}".format(loss))
					with open(self.log_path + '/log.txt', 'a') as log:
						log.write("Epoch: {}, iteration: {}\n".format(epoch, batch))
						log.write("Loss: {}\n".format(loss))

				self.step += 1
			self.saver.save(self.sess, save_path=self.saved_model_path, global_step=self.step)
		return None
