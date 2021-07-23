'''Thanks to https://github.com/GuyHacohen/curriculum_learning for the open-sourced code framework.
we use the same scripts used for "On The Power of Curriculum Learning in Training Deep Networks" by Hacohen et al. (ICML, 2019)'''

from main import main
from argparse import Namespace
import numpy as np
import torch
torch.manual_seed(0)
np.random.seed(0)

def fmnist_perceptron_vanilla(): # this is the vanilla model

	args = Namespace(dataset='fmnist',
					curriculum='vanilla',
					order='std',
					num_epochs=40,
					batch_size=100,
					model='perceptron',
					optimizer='sgd',
					start_lr=0.05,
					lr_decay=1.1,
					min_lr=1e-4,
					lr_step=500,
					pace_step=100,
					pace_start_fraction=0.04,
					pace_exp_increase=1.9,
					num_trials=1,
					test_step=25,
					noise_std=0,
					output_path="")

	main(args)

def fmnist_perceptron_anticurriculum(): # this is the stddev- model

	args = Namespace(dataset='fmnist',
					curriculum='anticurriculum',
					order='std',
					num_epochs=40,
					batch_size=100,
					model='perceptron',
					optimizer='sgd',
					start_lr=0.05,
					lr_decay=1.1,
					min_lr=1e-4,
					lr_step=500,
					pace_step=75,
					pace_start_fraction=0.04,
					pace_exp_increase=1.3,
					num_trials=1,
					test_step=25,
					noise_std=0,
					output_path="")

	main(args)

def fmnist_perceptron_curriculum(): # this is the stddev+ model

	args = Namespace(dataset='mnist',
					curriculum='curriculum',
					order='std',
					num_epochs=40,
					batch_size=100,
					model='perceptron',
					optimizer='sgd',
					start_lr=0.002,
					lr_decay=1.1,
					min_lr=1e-4,
					lr_step=500,
					pace_step=20,
					pace_start_fraction=0.04,
					pace_exp_increase=1.5,
					num_trials=1,
					test_step=25,
					noise_std=0,
					output_path="")

	main(args)


def mnist_perceptron_vanilla(): # this is the vanilla model

	args = Namespace(dataset='mnist',
					curriculum='vanilla',
					order='std',
					num_epochs=8,
					batch_size=100,
					model='perceptron',
					optimizer='sgd',
					start_lr=0.002,
					lr_decay=1.1,
					min_lr=1e-4,
					lr_step=400,
					pace_step=20,
					pace_start_fraction=0.04,
					pace_exp_increase=1.7,
					num_trials=1,
					test_step=25,
					noise_std=0,
					output_path="")

	main(args)

def mnist_perceptron_anticurriculum(): # this is the stddev- model

	args = Namespace(dataset='mnist',
					curriculum='anticurriculum',
					order='std',
					num_epochs=8,
					batch_size=100,
					model='perceptron',
					optimizer='sgd',
					start_lr=0.002,
					lr_decay=1.1,
					min_lr=1e-4,
					lr_step=400,
					pace_step=20,
					pace_start_fraction=0.04,
					pace_exp_increase=1.7,
					num_trials=1,
					test_step=25,
					noise_std=0,
					output_path="")

	main(args)

def mnist_perceptron_curriculum(): # this is the stddev+ model

	args = Namespace(dataset='mnist',
					curriculum='curriculum',
					order='std',
					num_epochs=8,
					batch_size=100,
					model='perceptron',
					optimizer='sgd',
					start_lr=0.002,
					lr_decay=1.1,
					min_lr=1e-4,
					lr_step=400,
					pace_step=20,
					pace_start_fraction=0.04,
					pace_exp_increase=1.5,
					num_trials=1,
					test_step=25,
					noise_std=0,
					output_path="")

	main(args)


if __name__ == '__main__':

	mnist_perceptron_vanilla() # this is the vanilla model
	mnist_perceptron_anticurriculum() # this is the stddev- model
	mnist_perceptron_curriculum() # this is the stddev+ model

	fmnist_perceptron_vanilla() # this is the vanilla model
	fmnist_perceptron_anticurriculum() # this is the stddev- model
	fmnist_perceptron_curriculum() # this is the stddev+ model