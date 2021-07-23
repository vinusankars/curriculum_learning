'''Thanks to https://github.com/GuyHacohen/curriculum_learning for the open-sourced code framework.
we use the same scripts used for "On The Power of Curriculum Learning in Training Deep Networks" by Hacohen et al. (ICML, 2019)'''

import argparse
import helper
import numpy as np
import torch
import multiprocessing as mp 
import pickle
from time import time

torch.manual_seed(0)
np.random.seed(0)

def main(args):

	global model
	print('Dataset is', args.dataset)
	print('Model is', args.model, args.optimizer)

	if args.output_path == "":
		output_path =  'dump/' + '_'.join([args.curriculum, args.order, args.model, args.dataset])
	else:
		output_path = args.output_path

	print(output_path)
	dataset = helper.load_data(args.dataset, args.noise_std)
	train_size = len(dataset.x_train)
	num_batches = (train_size*args.num_epochs)//args.batch_size

	lr_scheduler = helper.exp_lr(args.start_lr, args.lr_decay, args.min_lr, args.lr_step)
	
	pace_function = helper.exp_pace_function(args.pace_start_fraction, args.pace_step, args.pace_exp_increase)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	# device = torch.device("cpu")
	print('Device is', device)

	assert args.curriculum in ['curriculum', 'anticurriculum', 'vanilla', 'random'], \
								'Invalid arg for curriculum passed\n'
	
	assert args.order in ['std', 'dense', 'entropy'], \
								'Invalid arg for order passed\n'

	order = helper.get_order(dataset, args.curriculum, args.order, device)
	dataset.normalize()
	dataset.sort(order)
	history, result = [], []
	lr = args.start_lr

	start = time()
	for trial in range(args.num_trials):

		print('\nStarting trial #{}'.format(trial+1))
		model = helper.model(dataset, args.optimizer, device, name=args.model) 
		for batch in range(num_batches):

			if args.curriculum != 'vanilla':
				expose_size = pace_function(batch)
			else:
				expose_size = 1.0
			lr = lr_scheduler(batch)
			if model.lr != lr and args.optimizer != 'adam':
				model.set_lr(lr)

			print('{}/{} lr={:.5f} loss={:.5f} tr_acc={:.5f} val_acc={:.5f} expose={:d}'.format(batch+1, num_batches,\
			lr, model.loss, model.accuracy, model.val_acc, int(train_size*expose_size)), flush=True, end='\r')
			
			x, y = helper.get_batch(dataset, int(train_size*expose_size), args.batch_size)
			model.train(x, y)

			if (batch+1)%args.test_step == 0:
				model.validate(dataset)

			if batch >= num_batches-5:
				result.append(model.validate(dataset, True))

		history.append(model.get_history())

	History = {}
	for i in list(history[0].keys()):
		a = []
		for j in range(len(history)):
			a.append(history[j][i])
		a = np.stack(a)
		History[i] = a.mean(0)
		History[i+'_std'] = a.std(0)


	print('\nTime taken = {} seconds\n'.format(time()-start))

	if output_path != '':
		with open(output_path+'_history', 'wb') as f:
			pickle.dump(History, f)
		print('Training acc:', History['train_acc'][-1])
		print('Test acc:', History['val_acc'][-1])
		# torch.save(model.net.state_dict(), output_path+'_model.pth')

	else:
		print(result)
		return np.stack(result).mean()

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset', default='fmnist')
	# parser.add_argument('--output_path', default='dump/curriculum_svm_shallow_perceptron_adam_cifar100_small_mammals_noise000')
	parser.add_argument('--curriculum', default='curriculum')
	parser.add_argument('--order', default='std')
	parser.add_argument('--num_epochs', default=150, type=int)
	parser.add_argument('--batch_size',default=100, type=int)
	parser.add_argument('--model', default='perceptron')
	parser.add_argument('--optimizer', default='sgd')
	parser.add_argument('--start_lr', default=0.1, type=float)
	parser.add_argument('--lr_decay', default=1.1, type=float)
	parser.add_argument('--min_lr', default=10**-4, type=float)
	parser.add_argument('--lr_step', default=200, type=int)
	parser.add_argument('--pace_step', default=25, type=int)
	parser.add_argument('--pace_start_fraction', default=0.04, type=float)
	parser.add_argument('--pace_exp_increase', default=1.9, type=float)
	parser.add_argument('--num_trials', default=1, type=int)
	parser.add_argument('--test_step', default=50, type=int)
	parser.add_argument('--noise_std', default=0, type=float)
	parser.add_argument('--output_path', default="")

	args = parser.parse_args()
	model = []
	main(args)