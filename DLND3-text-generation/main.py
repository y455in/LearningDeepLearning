'''
Created on Mar 29, 2017

@author: Chewbacca
'''
import helper
import problem_unittests as tests

from collections import Counter
import random
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    prob_dict = [[x,ind] for ind, x in enumerate(probabilities)]
    
    prob_dict = sorted(prob_dict, reverse=True, key=lambda prob: prob[0])[:5]
    
    prob_idx = np.random.randint(0, len(prob_dict))
    
    idx = prob_dict[prob_idx][1]
    
    return int_to_vocab.get(idx)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)