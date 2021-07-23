import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

# generates FCN-512
class perceptron(nn.Module):

	def __init__(self, dataset, hidden_layers=1):

		self.N = dataset.x_train.shape[0]
		self.num_classes = len(dataset.y_train.unique())
		self.D = int(len(dataset.x_train.flatten())/self.N)
		self.hidden_layers = hidden_layers
		self.num_units = 512
		
		super(perceptron, self).__init__()		
		S = self.D
		i = 0
		for i in range(hidden_layers):
			exec('self.fc' + str(i) + ' = nn.Linear(S, self.num_units)')
			S = self.num_units
		exec('self.fc' + str(i+1) + ' = nn.Linear(S, self.num_classes)')
		self.fc_dropout = torch.nn.Dropout(0.5)

	def forward(self, x):

		self.x = x.view(-1, self.D)
		i = 0
		for i in range(self.hidden_layers):
			exec('self.x = self.fc' + str(i) + '(self.x)')
			self.x = F.elu(self.x)
			if i%2 == 0:
				self.x = self.fc_dropout(self.x)
		exec('self.x = self.fc' + str(i+1) + '(self.x)')
		return self.x 