'''Thanks to https://github.com/GuyHacohen/curriculum_learning for the open-sourced code framework.
we use the same scripts used for "On The Power of Curriculum Learning in Training Deep Networks" by Hacohen et al. (ICML, 2019)'''

import numpy as np 
import torchvision
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import models
import skimage.measure

class model():

	def __init__(self, dataset, optimizer, device, name=None):
		
		self.lr = 0.001
		self.loss = np.inf
		self.accuracy = 0
		self.val_acc = 0
		
		if name == None or name == 'shallow_cnn':
			self.net = models.shallow_cnn(dataset)

		elif name == 'perceptron':
			self.net = models.perceptron(dataset)

		elif name == 'deep_cnn':
			self.net = models.deep_cnn(dataset)

		elif name == 'linear_perceptron':
			self.net = models.linear_perceptron(dataset)

		else:
			assert False, 'Invalid model name\n'

		if optimizer == 'sgd':
			self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, weight_decay=1e-3)

		elif optimizer == 'adam':
			self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-3)

		else:
			assert False, 'Invalid optimizer\n'

		self.criterion = nn.CrossEntropyLoss()
		self.device = device
		self.net = self.net.to(device)

		self.history = {'val_acc':[], 'val_loss':[],\
						 'train_acc':[], 'train_loss':[]}

	def set_lr(self, lr):		
		for param in self.optimizer.param_groups:
			param['lr'] = lr

	def train(self, x, y):
		
		x = torch.tensor(x, dtype=torch.float32).to(self.device)
		y = torch.tensor(y, dtype=torch.long).to(self.device)
		self.optimizer.zero_grad()
		self.outputs = self.net(x)

		self.loss = self.criterion(self.outputs, y)
		self.loss.backward()
		self.optimizer.step()

		_, predicted = torch.max(self.outputs.data, 1)
		self.accuracy = (predicted == y).sum().item()/y.size(0)
		self.history['train_acc'].append(self.accuracy)
		self.history['train_loss'].append(self.loss.item())

	def validate(self, dataset, return_acc=False):

		X = torch.tensor(dataset.x_test, dtype=torch.float32)
		y = torch.tensor(dataset.y_test, dtype=torch.long)
		batch_size = 500
		loss = 0
		total, correct = 0, 0

		with torch.no_grad():
			for i in range(0, len(X), batch_size):

				x_ = X[i: i+batch_size].to(self.device)
				y_ = y[i: i+batch_size].to(self.device)
				outputs = self.net(x_)
				loss += self.criterion(outputs, y_).item()
				_, predicted = torch.max(outputs.data, 1)
				total += y_.size(0)
				correct += (predicted == y_).sum().item()

		self.val_acc = correct/total
		self.history['val_acc'].append(self.val_acc)
		self.history['val_loss'].append(loss/(len(X)/batch_size))

		if return_acc:
			return self.val_acc

	def get_history(self):
		return self.history 

class load_data():

	def __init__(self, dataset, noise=0):

		assert dataset in ['cifar10', 'cifar100', 'mnist', \
		'fmnist', 'cifar100_aquatic_mammals',\
		'cifar100_small_mammals', 'synthetic'], 'Dataset unavailable\n'

		self.name = dataset 
		path = 'dataset/'
		
		if dataset == 'cifar10':
			get = torchvision.datasets.CIFAR10

		elif dataset.startswith('cifar100'):
			get = torchvision.datasets.CIFAR100

		elif dataset == 'mnist':
			get = torchvision.datasets.MNIST

		elif dataset == 'fmnist':
			get = torchvision.datasets.FashionMNIST

		transform = transforms.Compose([transforms.ToTensor(), AddGaussianNoise(0, noise)])
		
		trainset = get(root=path, train=True, download=True, transform=transform)
		classes = trainset.class_to_idx
		trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=2)
		trainset = list(trainloader)[0]
		x_train = trainset[0]
		y_train = trainset[1]
		
		testset = get(root=path, train=False, download=True, transform=transform)
		testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=2)
		testset = list(testloader)[0]
		x_test = testset[0]
		y_test = testset[1]

		del trainset, testset

		if dataset.startswith('cifar100_'):

			if dataset == 'cifar100_small_mammals':
				labels = ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel']
			elif dataset == 'cifar100_aquatic_mammals':
				labels = ['beaver', 'dolphin', 'otter', 'seal', 'whale']

			labels_idx = [classes[i] for i in labels]
			inds = []
			for i in labels_idx:
				inds += np.where(y_train == i)[0].tolist()
			self.x_train = x_train[inds]
			self.y_train = y_train[inds] - 100
			for i in range(len(labels_idx)):
				self.y_train[self.y_train == labels_idx[i]-100] = i

			inds = []
			for i in labels_idx:
				inds += np.where(y_test == i)[0].tolist()
			self.x_test = x_test[inds]
			self.y_test= y_test[inds] - 100
			for i in range(len(labels_idx)):
				self.y_test[self.y_test == labels_idx[i]-100] = i

		else:
			self.x_train = x_train
			self.y_train = y_train 
			self.x_test = x_test
			self.y_test = y_test

	def normalize(self):

		print('Normalizing dataset...')

		if len(self.x_train.shape) == 4:
			for i in range(self.x_train.shape[-1]):
				mean = self.x_train[:,:,:,i].mean()
				std = self.x_train[:,:,:,i].std()
				self.x_train = (self.x_train-mean)/std
				self.x_test = (self.x_test-mean)/std

		else:
			mean = self.x_train.mean()
			std = self.x_train.std()
			self.x_train = (self.x_train-mean)/std
			self.x_test = (self.x_test-mean)/std

	def sort(self, order):

		self.x_train = self.x_train[order]
		self.y_train = self.y_train[order]
		self.order = order

def exp_lr(start_lr, lr_decay, min_lr, lr_step):

	def function(batch_no):
		return max(min_lr, start_lr/lr_decay**(batch_no//lr_step))
	
	return function

def exp_pace_function(pace_start_fraction, pace_step, pace_exp_increase):

	def function(batch_no):
		try: # if value overflows
			return min(1, pace_start_fraction*pace_exp_increase**(batch_no//pace_step)) 
		except:
			return 1

	return function

def get_batch(dataset, expose_size, batch_size):

	assert expose_size >= batch_size, 'Batch size is too big {}/{}\n'.format(expose_size, batch_size)
	X = dataset.x_train
	y = dataset.y_train
	X_ = X[: expose_size]
	y_ = y[: expose_size]
	inds = np.arange(expose_size)
	np.random.shuffle(inds)
	inds = inds[: batch_size]
	return X_[inds], y_[inds]

def balance_order(order, dataset) :

	classes = np.unique(dataset.y_train, axis=0)
	order_class = {i.item():[] for i in classes}

	for i in range(len(order)):
		order_class[dataset.y_train[order[i]].item()].append(order[i])

	balanced_order = []
	counter, i = 0, 0

	while counter < len(order):
		for key in list(order_class.keys()):
			try:
				balanced_order.append(order_class[key][i])
				counter += 1
			except:
				pass
		i += 1

	return np.stack(balanced_order)

def get_order(dataset, curriculum, order, device):

	if curriculum == 'vanilla':
		return balance_order(np.arange(len(dataset.x_train)), dataset)

	elif curriculum == 'random':
		ind = np.arange(len(dataset.x_train))
		np.random.shuffle(ind)
		return balance_order(ind, dataset)

	else:
		if order == 'std':
			X = dataset.x_train.numpy()
			X = X.reshape(len(X), -1)
			X = X-X.mean(0)
			temp = X.std(1)
			ind = np.argsort(temp)

		elif order == 'entropy':
			X = dataset.x_train
			X = X.reshape(len(X), -1)
			X = X-X.mean(0)
			temp = []
			for i in range(len(X)):
			    temp.append(skimage.measure.shannon_entropy(X[i]))
			ind = np.argsort(temp)

		if curriculum == 'anticurriculum':
			return balance_order(ind[::-1], dataset)

		elif curriculum == 'curriculum':
			return balance_order(ind, dataset)

class AddGaussianNoise(object):
	# Reference: \
	# https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
    
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)