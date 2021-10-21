# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 11:38:10 2021

@author: anonymous
"""

import pandas as pd
import numpy as np
import os
import math
import sys
import copy
from icecream import ic
# import copy
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import ref_index
from scipy.fftpack import fft, fftfreq


def zero_padding(data, len_pad):
    pad = np.zeros(len_pad)
    data_pad = np.insert(pad, 0, data)
    acf = (sum(np.abs(data)) / len(data)) / (sum(np.abs(data_pad)) / len(data_pad))
    return data_pad * acf


print(5)
