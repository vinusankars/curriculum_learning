import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import autograd_hacks
import numpy as np
from copy import copy
np.random.seed(0)

# creates FCN-M where M = self.num_units
class perceptron(nn.Module):
	def __init__(self, x, y, hidden_layers=1):
		self.N = x.shape[0]
		self.num_classes = len(y.unique())
		self.D = int(len(x.flatten())/self.N)
		self.hidden_layers = hidden_layers
		self.num_units = 10
		super(perceptron, self).__init__()		
		S = self.D
		i = 0
		for i in range(hidden_layers):
			exec('self.fc' + str(i) + ' = nn.Linear(S, self.num_units, bias=False)')
			S = self.num_units
		exec('self.fc' + str(i+1) + ' = nn.Linear(S, self.num_classes, bias=False)')

	def forward(self, x):
		self.x = x.view(-1, self.D)
		i = 0
		for i in range(self.hidden_layers):
			exec('self.x = self.fc' + str(i) + '(self.x)')
			self.x = F.elu(self.x)
		exec('self.x = self.fc' + str(i+1) + '(self.x)')
		return self.x 

# helpful for learning rate decay
def set_lr(lr):
	for param in optimizer.param_groups:
		param['lr'] = lr

# for representing the weight parameters of the network as a vector
def getw():
	w = []
	for param in net.parameters():
		w.append(np.stack([param.detach().cpu().numpy().reshape(-1)]))
	w = np.concatenate((w[0], w[1]), axis=1)[0]
	return w

# gets the gradient information of the current pass
def getgrad():
	w = []
	for param in net.parameters():
		w.append(param.grad1.detach().cpu().numpy().reshape(len(x), -1))
	w = np.concatenate((w[0], w[1]), axis=1)
	return w

# This is the DCL+ model
def best():
	w_ = np.load('optimal_mnist.npy') # this is \tilde{w}
	Loss = []
	autograd_hacks.add_hooks(net)
	itr = 0
	lr = LR
	k = int(kk*len(x))

	for i in range(batches//(k//batch_size)+1):

		autograd_hacks.clear_backprops(net)
		axis = w_ - getw()
		axis = axis/np.linalg.norm(axis)
		optimizer.zero_grad()
		output = net(x)
		loss = criterion(output, y)
		loss.backward(retain_graph=True) 
		autograd_hacks.compute_grad1(net)
		grad = getgrad()
		dot = np.abs((grad*axis).sum(axis=1))
		order = np.argsort(dot)
		order = np.flip(order)

		for j in range(0, k//batch_size*batch_size, batch_size): 

			ind = copy(order[j: j+batch_size]) 
			x_ = x[ind]
			y_ = y[ind]
			optimizer.zero_grad()
			outputs = net(x_)
			loss = criterion(outputs, y_)
			loss.backward()
			optimizer.step()
			outputs = net(x_test)
			_, predicted = torch.max(outputs.data, 1)
			Loss.append([criterion(outputs, y_test).item(), (predicted==y_test).sum().item()/len(y_test)]) 
			itr += 1

			if itr%100 == 0:  # lr decay
				lr = lr/1.2 
				set_lr(lr)

	autograd_hacks.disable_hooks()
	return Loss

# This is the DCL- model
def reverse_best():
	w_ = np.load('optimal_mnist.npy') # this is \tilde{w}
	Loss = []
	autograd_hacks.add_hooks(net)
	itr = 0
	lr = LR
	k = int(kk*len(x))

	for i in range(batches//(k//batch_size)+1):

		autograd_hacks.clear_backprops(net)
		axis = w_ - getw()
		axis = axis/np.linalg.norm(axis)
		optimizer.zero_grad()
		output = net(x)
		loss = criterion(output, y)
		loss.backward(retain_graph=True) 
		autograd_hacks.compute_grad1(net)
		grad = getgrad()
		dot = np.abs((grad*axis).sum(axis=1))
		order = np.argsort(dot)
		order = np.flip(order)

		for j in list(range(0, k//batch_size*batch_size, batch_size))[::-1]: 
		# The mini-batch sequence within an epoch is reversed

			ind = copy(order[j: j+batch_size]) 
			x_ = x[ind]
			y_ = y[ind]
			optimizer.zero_grad()
			outputs = net(x_)
			loss = criterion(outputs, y_)
			loss.backward()
			optimizer.step()
			outputs = net(x_test)
			_, predicted = torch.max(outputs.data, 1)
			Loss.append([criterion(outputs, y_test).item(), (predicted==y_test).sum().item()/len(y_test)]) 
			itr += 1

			if itr%100 == 0:  # lr decay
				lr = lr/1.2 
				set_lr(lr)

	autograd_hacks.disable_hooks()
	return Loss

# the vanilla model
def vanilla():
	Loss = []
	lr = LR

	for i in range(batches):
		correct, total = 0, 0
		ind = np.arange(len(x))
		np.random.shuffle(ind)
		ind = ind[: batch_size]
		x_ = x[ind]
		y_ = y[ind]
		optimizer.zero_grad()
		outputs = net(x_)
		loss = criterion(outputs, y_)
		loss.backward()
		optimizer.step()
		outputs = net(x_test)
		L = criterion(outputs, y_test).item()
		_, predicted = torch.max(outputs.data, 1)
		Loss.append([L, (predicted==y_test).sum().item()/len(y_test)])
		if (i+1)%100 == 0: # lr decay
			lr = lr/1.2
			set_lr(lr)
	return Loss

# Loading dataset
get = torchvision.datasets.MNIST
transform = transforms.Compose([transforms.ToTensor(),])
trainset = get(root='.', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset), shuffle=False, num_workers=2)
trainset = list(trainloader)[0]
x_train = trainset[0]
y_train = trainset[1]
ind = (y_train<=1)
x = (x_train[ind])
y = (y_train[ind])

testset = get(root='.', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset), shuffle=False, num_workers=2)
testset = list(testloader)[0]
x_test = testset[0]
y_test = testset[1]
ind = (y_test<=1)
x_test = x_test[ind]
y_test = y_test[ind]

# normalize data
x_test = (x_test-x.mean())/x.std()
x = (x-x.mean())/x.std()
del x_train, y_train, testset, trainset

batches = 1000
batch_size = 50
kk = 0.9 # the k parameter in the manuscript, controls pacing function
LR = 1e-1 # initial learning rate
trials = 1 # number of independent experiments

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
x = x.to(device)
y = y.to(device)
x_test = x_test.to(device)
y_test = y_test.to(device)

for tr in range(trials):

	print(tr)

	net = perceptron(x, y, 1) # Loads FCN-128
	optimizer = optim.SGD(net.parameters(), lr=LR)
	criterion = nn.CrossEntropyLoss()
	net = net.to(device)
	Loss = best() # The DCL+ model
	np.save('result/best_loss_'+str(tr), Loss)
	print('best')

	net = perceptron(x, y, 1) 
	optimizer = optim.SGD(net.parameters(), lr=LR)
	criterion = nn.CrossEntropyLoss()
	net = net.to(device)
	Loss = reverse_best() # The DCL- model
	np.save('result/reverse_best_loss_'+str(tr), Loss)
	print('reverse_best')

	net = perceptron(x, y, 1)
	optimizer = optim.SGD(net.parameters(), lr=LR)
	criterion = nn.CrossEntropyLoss()
	net = net.to(device)
	Loss = vanilla() # The vanilla model
	np.save('result/vanilla_loss_'+str(tr), Loss) 
	print('vanilla')