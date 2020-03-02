#Implement naive bayes
from import_art_stop_allyrs_v2 import import_article
import pandas as pd
import math
import os
from nltk.corpus import stopwords
firstnwords=100

#Read the dictionary of all english words
def load_eng_words():
    """ load_eng_words function
    @returns valid_words (set of strings): the set of string in the english words file
    """
    with open('words_alpha.txt') as word_file:
        valid_words = set(word_file.read().split())
    return valid_words