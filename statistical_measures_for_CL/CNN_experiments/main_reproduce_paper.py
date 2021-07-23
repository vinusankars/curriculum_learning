'''Thanks to https://github.com/GuyHacohen/curriculum_learning for the open-sourced code framework.
we use the same scripts used for "On The Power of Curriculum Learning in Training Deep Networks" by Hacohen et al. (ICML, 2019)'''

from argparse import Namespace
from main_train_networks import run_expriment
from tensorflow.compat.v1 import set_random_seed
set_random_seed(42)
from numpy.random import seed 
seed(42)

def vanilla_cifar100_st_vgg(repeats, order="inception", output_path="", model="stVGG"):
    
    args = Namespace(dataset="cifar100",
                     model=model,
                     output_path=output_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="vanilla",
                     batch_size=100,
                     num_epochs=200,
                     learning_rate=0.12,
                     lr_decay_rate=1.1,
                     minimal_lr=5e-3,
                     lr_batch_size=400,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order=order,
                     test_each=200,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)

def curriculum_cifar100_st_vgg(repeats, order="inception", output_path="", model="stVGG", learning_rate=0.12):
    args = Namespace(dataset="cifar100",
                     model=model,
                     output_path=output_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=200,
                     learning_rate=learning_rate,
                     lr_decay_rate=1.1,
                     minimal_lr=5e-3,
                     lr_batch_size=400,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order=order,
                     test_each=200,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)

def anti_curriculum_cifar100_st_vgg(repeats, order="inception", output_path="", model="stVGG", learning_rate=0.12):
    args = Namespace(dataset="cifar100",
                     model=model,
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=200,
                     learning_rate=learning_rate,
                     lr_decay_rate=1.1,
                     minimal_lr=5e-3,
                     lr_batch_size=400,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order=order,
                     test_each=200,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)

def vanilla_cifar10_st_vgg(repeats, order="inception", output_path="", model="stVGG"):
    args = Namespace(dataset="cifar10",
                     model=model,
                     output_path=output_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="vanilla",
                     batch_size=100,
                     num_epochs=150,
                     learning_rate=0.12,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-3,
                     lr_batch_size=700,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order=order,
                     test_each=100,
                     repeats=repeats,
                     balance=True)
    run_expriment(args)

def curriculum_cifar10_st_vgg(repeats, order="inception", output_path="", learning_rate=0.12, model="stVGG"):
    args = Namespace(dataset="cifar10",
                     model=model,
                     output_path=output_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=150,
                     learning_rate=learning_rate,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-3,
                     lr_batch_size=700,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order=order,
                     test_each=100,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


if __name__ == "__main__":
    
    output_path = "results/"
    num_repeats = 1 # number of independent trials
	# c_c100 is curriculum with cifar100, a_c100 is anticurriculum with cifar100, v_c100 is vanilla with cifar100     
    # c_c10 is curriculum with cifar10, v_c10 is vanilla with cifar10
    choice = (input('Enter choice [v_c100, a_c100, c_c100, v_c10, c_c10]: '))

    print(choice)

    if choice == 'v_c100':
        do = vanilla_cifar100_st_vgg
    elif choice == 'c_c100':
        do = curriculum_cifar100_st_vgg
    elif choice == 'a_c100':
        do = anti_curriculum_cifar100_st_vgg
    elif choice == 'v_c10':
        do = vanilla_cifar10_st_vgg
    elif choice == 'c_c10':
        do = curriculum_cifar10_st_vgg

    else:
        print('Error')
        exit()

    order = 'std' # type of ordering, std uses stddev
    model = 'stVGG'
    name = output_path+choice+'_'+order+'_'+model
    print(name)
    do(num_repeats, order=order, output_path=name, model=model)