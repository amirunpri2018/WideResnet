from wrn import WRN
from keras.datasets import cifar10
import numpy as np
from numpy.random import RandomState


class CIFARDataset(object):
	def __init__(self):
		(self.train_x, self.train_y) , (self.test_x, self.test_y) = cifar10.load_data()

		self.train_x = self.train_x.astype('float32')
		self.train_x = (self.train_x - self.train_x.mean(axis=0)) / (self.train_x.std(axis=0))
		self.test_x = self.test_x.astype('float32')
		self.test_x = (self.test_x - self.test_x.mean(axis=0)) / (self.test_x.std(axis=0))
		self.size = len(self.train_x)

		self.train_y = self.train_y.reshape((self.size,))
		self.idx = 0

	def get_batch(self,batch_size):
		start = self.idx
		end = self.idx + batch_size
		if end >= self.size:
			end -= self.size
			batch_raw_data = np.array(self.train_x[start:])
			self.train_x = self.shuffle(self.train_x)
			try:
				batch_raw_data = np.concatenate((batch_raw_data,np.array(self.train_x[:end])),axis=0)
			except ValueError:
				print(batch_raw_data.shape)
				print(np.array(self.train_x[:end]).shape)
			batch_labels = self.train_y[start:]
			self.train_y = self.shuffle(self.train_y)
			batch_labels = np.concatenate((batch_labels,self.train_y[:end]),axis=0)
		else:
			batch_raw_data = np.array(self.train_x[start:end])
			batch_labels = self.train_y[start:end]
		self.idx = end
		return batch_raw_data, batch_labels

	def shuffle(self,data, random_seed=1):
		data_copy = np.copy(data).tolist()
		rand = RandomState(random_seed)
		rand.shuffle(data_copy)
		return data_copy

def main():
	model = WRN([32,32,3], 10, 0.1, depth_per_down=4,widening_factor=10)
	dataset = CIFARDataset()
	#model.load("./Results/2018-09-25 16-17-12.105585_100_200_wrn-28-10/saved_models/")
	#model.test(dataset)
	#model.train(dataset,epochs = 200, batch_size = 100)

if __name__ == "__main__":
	main()
