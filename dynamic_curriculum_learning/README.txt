Note: autograd_hacks.py belongs to https://github.com/cybertronai
Thanks to https://github.com/cybertronai/autograd-hacks for the autograd_hacks.py script
that helps in extracting the gradient information from the networks.

main_mnist.py for experiment 1 uses \bar{w} from optimal_mnist.npy (obtained from full SGD training of vanilla model).
main_sm.py for experiment 2 uses \bar{w} from optimal_sm.npy (obtained from full SGD training of vanilla model).

The dataset required will be automatically downloaded.
The results will be saved in the result folder.

In the scripts:
best() represents DCL+ model
reverse_best() represents DCL- model
vanilla() represents vanilla model

Execute:
python main_mnist.py
python main_sm.py

for running Experiment 1 and 2, respectively, as mentioned in our draft.