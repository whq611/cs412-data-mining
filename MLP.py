import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

class MLP(nn.Module):
	def __init__(self, input_dim, hidden_dim, num_classes):
		super(MLP, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(inplace=True),
			nn.Linear(hidden_dim, num_classes)
		)

	def forward(self, x):
		x = x.view(x.size()[0], -1) # flatten
		out = self.fc(x)
		return out