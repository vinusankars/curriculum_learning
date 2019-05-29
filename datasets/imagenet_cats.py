#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 15:03:40 2018

@author: stenly
"""
import imagenet_subset
import pickle

    
class Imagenet_cats(imagenet_subset.Imagenet_Subset):

    def __init__(self, normalize=True):
        subset_name = "Cats"
        self.cat_labels_path = r"/cs/labs/daphna/guy.hacohen/imagenet/ILSVRC2012_devkit_t12/data/cat_labels"
        with open(self.cat_labels_path , "rb+") as cat_file:
            subsets_idxes = pickle.load(cat_file)
        super(Imagenet_cats, self).__init__(normalize=normalize,
                                            subsets_idxes=subsets_idxes,
                                            subset_name=subset_name)
        
