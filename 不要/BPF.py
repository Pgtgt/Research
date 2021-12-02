# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 02:11:54 2021

@author: wsxhi

BPF_hanning(ACF)_zeropadding(ACF)があるため，これは不要．

"""

# =============================================================================
# # ------------バンドパスフィルタ法------
# =============================================================================
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
# OPTIMIZE:tunig
# =============================================================================
# param tuning
# 最適地20210905 +1-ref
# CUT_T = 15e-12
# CUT_WIDTH = 0.00001e-12# 0.001 0.01もかわらず
# EXPNUM = 14
# 最適地　inter 20210925
# CUT_T = 8e-12  # 大きいほど，o付近をつぶ(光源の影響)
# CUT_WIDTH = 4e-13
# =============================================================================


#  cut_T cutT = 6.6 p にて2mm以下が無理になる
LIST_HYPERPARAMS = (
    # dict(cutT=6e-10, cutwidth=1e-13, expnum=14),
    # dict(cutT=6e-10, cutwidth=1e-12, expnum=14),
    # dict(cutT=6e-10, cutwidth=1e-11, expnum=14),
    # dict(cutT=6e-10, cutwidth=1e-10, expnum=14),
    # dict(cutT=6e-10, cutwidth=1e-9, expnum=14),
    # dict(cutT=6e-10, cutwidth=1e-8, expnum=14),
    # dict(cutT=6e-11, cutwidth=1e-7, expnum=14),
    # dict(cutT=6e-11, cutwidth=1e-13, expnum=14),
    # dict(cutT=6e-11, cutwidth=1e-12, expnum=14),
    # dict(cutT=6e-11, cutwidth=1e-11, expnum=14),
    # dict(cutT=6e-11, cutwidth=1e-10, expnum=14),
    # dict(cutT=6e-11, cutwidth=1e-9, expnum=14),
    # dict(cutT=6e-11, cutwidth=1e-8, expnum=14),
    # dict(cutT=10e-12, cutwidth=2e-14, expnum=14),  # Goo

    # dict(cutT=10e-12, cutwidth=2e-13, expnum=14),  # Goo
    # dict(cutT=10e-12, cutwidth=2e-12, expnum=14),  # Goo
    # dict(cutT=10e-12, cutwidth=5e-12, expnum=14),  # Goo


    dict(cutT=12e-12, cutwidth=1.6e-12, expnum=15),  # Goo
    dict(cutT=13e-12, cutwidth=1.6e-12, expnum=15),  # Goo
    dict(cutT=14e-12, cutwidth=1.6e-12, expnum=15),  # Goo
    dict(cutT=15e-12, cutwidth=1.6e-12, expnum=15),  # Goo
    dict(cutT=16e-12, cutwidth=1.6e-12, expnum=15),  # Goo
    dict(cutT=17e-12, cutwidth=1.6e-12, expnum=15),  # Goo
    dict(cutT=18e-12, cutwidth=1.6e-12, expnum=15),  # Goo
    dict(cutT=19e-12, cutwidth=1.6e-12, expnum=15),  # Goo
    dict(cutT=20e-12, cutwidth=1.6e-12, expnum=15),  # Goo

    dict(cutT=12e-12, cutwidth=3.2e-12, expnum=15),  # Goo
    dict(cutT=13e-12, cutwidth=3.2e-12, expnum=15),  # Goo
    dict(cutT=14e-12, cutwidth=3.2e-12, expnum=15),
    dict(cutT=15e-12, cutwidth=3.2e-12, expnum=15),
    dict(cutT=16e-12, cutwidth=3.2e-12, expnum=15),
    dict(cutT=17e-12, cutwidth=3.2e-12, expnum=15),
    dict(cutT=18e-12, cutwidth=3.2e-12, expnum=15),
    dict(cutT=19e-12, cutwidth=3.2e-12, expnum=15),
    dict(cutT=20e-12, cutwidth=3.2e-12, expnum=15),

    dict(cutT=12e-12, cutwidth=4.8e-12, expnum=15),
    dict(cutT=13e-12, cutwidth=4.8e-12, expnum=15),
    dict(cutT=14e-12, cutwidth=4.8e-12, expnum=15),
    dict(cutT=15e-12, cutwidth=4.8e-12, expnum=15),
    dict(cutT=16e-12, cutwidth=4.8e-12, expnum=15),
    dict(cutT=17e-12, cutwidth=4.8e-12, expnum=15),
    dict(cutT=18e-12, cutwidth=4.8e-12, expnum=15),
    dict(cutT=19e-12, cutwidth=4.8e-12, expnum=15),
    dict(cutT=20e-12, cutwidth=4.8e-12, expnum=15),


    dict(cutT=12e-12, cutwidth=6.4e-12, expnum=15),  # Goo
    dict(cutT=13e-12, cutwidth=6.4e-12, expnum=15),  # Goo
    dict(cutT=14e-12, cutwidth=6.4e-12, expnum=15),  # Goo
    dict(cutT=15e-12, cutwidth=6.4e-12, expnum=15),  # Goo
    dict(cutT=16e-12, cutwidth=6.4e-12, expnum=15),  # Goo
    dict(cutT=17e-12, cutwidth=6.4e-12, expnum=15),  # Goo
    dict(cutT=18e-12, cutwidth=6.4e-12, expnum=15),  # Goo
    dict(cutT=19e-12, cutwidth=6.4e-12, expnum=15),  # Goo
    dict(cutT=20e-12, cutwidth=6.4e-12, expnum=15),  # Goo


    dict(cutT=12e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=13e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=14e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=15e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=17e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=18e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=19e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=20e-12, cutwidth=8e-12, expnum=15),  # Goo

    dict(cutT=12e-12, cutwidth=9.6e-12, expnum=15),  # Goo
    dict(cutT=13e-12, cutwidth=9.6e-12, expnum=15),  # Goo
    dict(cutT=14e-12, cutwidth=9.6e-12, expnum=15),  # Goo
    dict(cutT=15e-12, cutwidth=9.6e-12, expnum=15),  # Goo
    dict(cutT=16e-12, cutwidth=9.6e-12, expnum=15),  # Goo
    dict(cutT=17e-12, cutwidth=9.6e-12, expnum=15),  # Goo
    dict(cutT=18e-12, cutwidth=9.6e-12, expnum=15),  # Goo
    dict(cutT=19e-12, cutwidth=9.6e-12, expnum=15),  # Goo
    dict(cutT=20e-12, cutwidth=9.6e-12, expnum=15),  # Goo


    dict(cutT=12e-12, cutwidth=10.2e-12, expnum=15),  # Goo
    dict(cutT=13e-12, cutwidth=10.2e-12, expnum=15),  # Goo
    dict(cutT=14e-12, cutwidth=10.2e-12, expnum=15),  # Goo
    dict(cutT=15e-12, cutwidth=10.2e-12, expnum=15),  # Goo
    dict(cutT=16e-12, cutwidth=10.2e-12, expnum=15),  # Goo
    dict(cutT=17e-12, cutwidth=10.2e-12, expnum=15),  # Goo
    dict(cutT=18e-12, cutwidth=10.2e-12, expnum=15),  # Goo
    dict(cutT=19e-12, cutwidth=10.2e-12, expnum=15),  # Goo
    dict(cutT=20e-12, cutwidth=10.2e-12, expnum=15),  # Goo

    dict(cutT=12e-12, cutwidth=11.8e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=13e-12, cutwidth=11.8e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=14e-12, cutwidth=11.8e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=15e-12, cutwidth=11.8e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=16e-12, cutwidth=11.8e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=17e-12, cutwidth=11.8e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=18e-12, cutwidth=11.8e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=19e-12, cutwidth=11.8e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=20e-12, cutwidth=11.8e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo

    dict(cutT=12e-12, cutwidth=13.4e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=13e-12, cutwidth=13.4e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=14e-12, cutwidth=13.4e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=15e-12, cutwidth=13.4e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=16e-12, cutwidth=13.4e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=17e-12, cutwidth=13.4e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=18e-12, cutwidth=13.4e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=19e-12, cutwidth=13.4e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=20e-12, cutwidth=13.4e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo

    dict(cutT=12e-12, cutwidth=15e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=13e-12, cutwidth=15e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=14e-12, cutwidth=15e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=15e-12, cutwidth=15e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=16e-12, cutwidth=15e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=17e-12, cutwidth=15e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=18e-12, cutwidth=15e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=19e-12, cutwidth=15e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=20e-12, cutwidth=15e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo

    dict(cutT=12e-12, cutwidth=16.6e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=13e-12, cutwidth=16.6e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=14e-12, cutwidth=16.6e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=15e-12, cutwidth=16.6e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=16e-12, cutwidth=16.6e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=17e-12, cutwidth=16.6e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=18e-12, cutwidth=16.6e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=19e-12, cutwidth=16.6e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    dict(cutT=20e-12, cutwidth=16.6e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo

    # dict(cutT=12e-12, cutwidth=18.2e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=13e-12, cutwidth=18.2e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=14e-12, cutwidth=18.2e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=15e-12, cutwidth=18.2e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=16e-12, cutwidth=18.2e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=17e-12, cutwidth=18.2e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=18e-12, cutwidth=18.2e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=19e-12, cutwidth=18.2e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=20e-12, cutwidth=18.2e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo

    # dict(cutT=12e-12, cutwidth=20e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=13e-12, cutwidth=20e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=14e-12, cutwidth=20e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=15e-12, cutwidth=20e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=16e-12, cutwidth=20e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=17e-12, cutwidth=20e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=18e-12, cutwidth=20e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=19e-12, cutwidth=20e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=20e-12, cutwidth=20e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo


    # dict(cutT=12e-12, cutwidth=22e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=13e-12, cutwidth=22e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=14e-12, cutwidth=22e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=15e-12, cutwidth=22e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=16e-12, cutwidth=22e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=17e-12, cutwidth=22e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=18e-12, cutwidth=22e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=19e-12, cutwidth=22e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=20e-12, cutwidth=22e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo

    # dict(cutT=12e-12, cutwidth=24e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=13e-12, cutwidth=24e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=14e-12, cutwidth=24e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=15e-12, cutwidth=24e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=16e-12, cutwidth=24e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=17e-12, cutwidth=24e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=18e-12, cutwidth=24e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=19e-12, cutwidth=24e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=20e-12, cutwidth=24e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo

    # dict(cutT=12e-12, cutwidth=26e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=13e-12, cutwidth=26e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=14e-12, cutwidth=26e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=15e-12, cutwidth=26e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=16e-12, cutwidth=26e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=17e-12, cutwidth=26e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=18e-12, cutwidth=26e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=19e-12, cutwidth=26e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=20e-12, cutwidth=26e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo

    # dict(cutT=12e-12, cutwidth=28e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=13e-12, cutwidth=28e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=14e-12, cutwidth=28e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=15e-12, cutwidth=28e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=16e-12, cutwidth=28e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=17e-12, cutwidth=28e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=18e-12, cutwidth=28e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=19e-12, cutwidth=28e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=20e-12, cutwidth=28e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo

    # dict(cutT=12e-12, cutwidth=30e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=13e-12, cutwidth=30e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=14e-12, cutwidth=30e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=15e-12, cutwidth=30e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=16e-12, cutwidth=30e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=17e-12, cutwidth=30e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=18e-12, cutwidth=30e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=19e-12, cutwidth=30e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo
    # dict(cutT=20e-12, cutwidth=30e-12, expnum=15),  # Goodict(cutT=16e-12, cutwidth=8e-12, expnum=15),  # Goo

    # dict(cutT=10e-12, cutwidth=12e-12, expnum=14),  # Goo
    # dict(cutT=10e-12, cutwidth=14e-12, expnum=14),  # Goo


    # dict(cutT=10e-12, cutwidth=15e-12, expnum=14),  # Goo
    # dict(cutT=10e-12, cutwidth=20e-12, expnum=14),  # Goo
    # dict(cutT=10e-12, cutwidth=22e-12, expnum=14),  # Goo
    # dict(cutT=10e-12, cutwidth=24e-12, expnum=14),  # Goo

    # dict(cutT=10e-12, cutwidth=40e-12, expnum=14),  # Goo
    # dict(cutT=1e-10, cutwidth=1e-12, expnum=14),  # Goo
    # dict(cutT=1e-10, cutwidth=1e-11, expnum=14),  # こっちのほうがいいかも　局所的にみると
    # dict(cutT=6e-12, cutwidth=1e-10, expnum=14),
    # dict(cutT=6e-12, cutwidth=1e-9, expnum=14),
    # dict(cutT=6e-12, cutwidth=1e-8, expnum=14),
    # dict(cutT=6e-12, cutwidth=1e-7, expnum=14),

)


n_air = ref_index.edlen(wave=(1554.134049+1563.862587)/2, t=27, p=101325, rh=70)
K=0.741244259


def Dialog_File(rootpath=r"C:", caption="choise"):
    """
    引数:初期ディレクトリ
    戻り値：ファイルパス
    """
    from PyQt5 import QtWidgets
    # 実行ディレクトリ取得D
    app_dialog_file = QtWidgets.QApplication(sys.argv)
    # ディレクトリ選択ダイアログを表示-
    filepath = QtWidgets.QFileDialog.getOpenFileName(
        parent=None, caption=caption, directory=rootpath)

    # sys.exit(app_dialog_file.exec_())
    return filepath[0]


def Dialog_Folder(rootpath=r"C:", caption="choise"):
    """
    引数:初期ディレクトリ
    戻り値：フォルダ(ディレクトリ)パス
    """
    from PyQt5 import QtWidgets
    import sys
    # ディレクトリ選択ダイアログを表示
    app = QtWidgets.QApplication(sys.argv)
    folderpath = QtWidgets.QFileDialog.getExistingDirectory(
        parent=None, caption=caption, directory=rootpath)
    return folderpath


class AbsoluteDistance():

    def __init__(self):
        pass

    def OSAcsvfre_In(self, filepath):  # fre-IのOSA信号(35行目から)をよむ freq domeinではデータは不当間隔データ
        self.filepath = filepath
        wholedata = pd.read_csv(self.filepath, header=None, skiprows=34).values
        # wholedataの[:,0]を取り出した後，ravel1で次元配列へ直す 単位も　Hz, Wへ変換
        Fdata = np.flipud(wholedata[:, 0].ravel())*1e+12
        Idata = np.flipud(wholedata[:, 1].ravel())*1e-3
        return Fdata, Idata

    def Inter(self, x, y, expnum=16):
        """
        不等間隔データをスプライン補完により等間隔データにする

        Args:
            x (float): signal.　補完まえ
            y (float): signal.
            expnum (int, optional): exponent. Defaults to 16.

        Returns:
            xinter (float): signal.補間後
            yinter (float): signal.
            SampNum_inter (int): sampling number .
            dx (float): sampling interval (x)

        """

        import sys
        from scipy import interpolate
        xinterstart = min(x)*(1+sys.float_info.epsilon)
        xinterend = max(x)*(1-sys.float_info.epsilon)
        SampNum_inter = 2**expnum  # 分割数
        xinter = np.linspace(xinterstart, xinterend, SampNum_inter)
        SpFun = interpolate.interp1d(x, y, kind="cubic")
        yinter = SpFun(xinter)
        dx = (xinterend-xinterstart)/(SampNum_inter-1)
        return xinter, yinter, SampNum_inter, dx

    def FFT(self, x, y):  # 等間隔データをFFT
        N = len(x)
        FF = np.fft.fft(y)
        dx = np.abs((x[-1]-x[0])/(N-1))
        freq = fftfreq(len(FF), d=dx)

        freq = np.concatenate([freq[int(N/2):], freq[:int(N/2)]])
        FF = np.concatenate([FF[int(N/2):], FF[:int(N/2)]])

        FF_abs = np.abs(FF)
        FF_abs_amp = FF_abs/(N/2)

        # f=np.linspace(0,1.0/dx,N)
        return freq, FF, FF_abs_amp

    def wrappedphase(self, e):
        """
        複素数配列eの位相を -pi/2 ~ pi/2の範囲で求める

        Args:
            e (complex): complex array.

        Returns:
            wrap (float): -pi/2 ~ pi/2.

        """

        wrap = np.empty(len(e))
        for _ in range(len(e)):
            wrap[_] = math.atan(e.imag[_]/e.real[_])

        return wrap

    def path_difference(self, F_unequal, I_unequal, cutT=10e-12, cutwidth=1e-12, expnum=16, removeT=[None, None]):
        """
        filepathから結果を分析．上の関数軍はこの関数のためのもの
        結果が欲しいときは'self.path_diff'とかで呼び出す

        C_1 + C_2*cos(phi) =>C_2/2 *exp(j*phi) => phi = a *F +b =>a
        Args:
            filepath (str): filepath.
            cutT (float, optional): DESCRIPTION. Defaults to 10e-12.
            cutwidth (float, optional): DESCRIPTION. Defaults to 1e-12.
            expnum (int, optional): DESCRIPTION. Defaults to 16.
            removeT ([float(minT), float(maxT)], optional): 邪魔な成分が出てきてしまった時，このTの範囲は０にする．マイナス成分は指定しなくてよい．．必要ないならどっちもＮｏｎｅへ．Defaults to [None,None]. この操作の後にピークサーチをする
        Returns:
            None

        """
        cutT, cutwidth, expnum = cutT, cutwidth, expnum
        # 補間 I(f_uneq) => I(f_euneq)
        F, I, SampNum_inter, dF = self.Inter(F_unequal, I_unequal, expnum)

        # =============================================================================
        # FFT:  I(f) C_1 + C_2*cos(phi(f) ) ====>   FFt(T)=C_1 + C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))
        # =============================================================================
        #
        T, FFt, FFt_abs_amp = self.FFT(F, I)
        # plt.plot(T, FFt_abs_amp)
        delta_T = (1.0/dF)/(SampNum_inter-1)

        # =============================================================================
        # Filtering: FFt(T)=C_1 + C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))
        #   ====> F2(T)=C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))
        # =============================================================================
        F2 = copy.deepcopy(FFt)
        F2[(T <= 0)] = 0  # (負の)周波数帯をカット
        F2[(T < cutT)] = 0  # カットオフ未満周波数のデータをゼロにする，光源の影響排除

        F2_abs = np.abs(F2)
        # 振幅をもとの信号に揃える
        F2_abs_amp = F2_abs / SampNum_inter * 2  # 交流成分はデータ数で割って2倍
        # plt.plot(T, F2_abs_amp)
        # plt.xlim(-300e-12, 300e-12)
        # plt.ylim(-0.2e-7, 0.2e-6)
        # plt.show()
        # =============================================================================
        # Filtering:   F2(T)=C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))
        #   ====>  F3(T)=C_2/2 exp(j*phi(T) )
        # =============================================================================

        F3 = copy.deepcopy(F2)

        if (removeT[0] == None) or (removeT[1] == None):
            pass
        else:
            F3[((removeT[0] < T) & (T < removeT[1]))] = 0  # removeT間は0にする

        peak = np.argmax(F2_abs_amp)
        # print(str(T[peak]))
        F3[((T < T[peak]-cutwidth/2) | (T[peak]+cutwidth/2 < T))] = 0  # 所望のピークだけのこす

        # IFFT   F3(T)=C_2/2 exp(j*phi(T) )  ====>  I(f)=C_2/2 exp(j*phi(f) )
        F3_ifft = np.fft.ifft(F3)  # IFFT  C_2/2 exp(j*phi(f) )
        F3_ifft_abs = np.abs(F3_ifft)
        F3_ifft_abs_amp = F3_ifft_abs / SampNum_inter * 2

        wrap = self.wrappedphase(F3_ifft)
        # wrap=F3_ifft
        # wrap_abs=np.abs(wrap)

        phi = np.unwrap(p=wrap * 2)/2

        a, b = np.polyfit(F, phi, 1)  # phi = a *F + bの1じ多項式近似
        # https://biotech-lab.org/articles/4907 R2値
        R2 = metrics.r2_score(phi, a * F + b)
        path_diff = 299792458/(2*np.pi*n_air)*a

        # a =2 pi Dd n / c
        # b = phi余り
        # Dd = path_diff

        self.F, self.T, self.FFt, self.FFt_abs_amp, self.F2_abs_amp, self.F3_ifft_abs_amp, self.wrap, self.T_peak, self.F_ifft_abs_amp_filterd = F, T, FFt, FFt_abs_amp, F2_abs_amp, F3_ifft_abs_amp, wrap, T[
            peak], F3_ifft_abs_amp
        self.cutT, self.cutwidth, self.expnum = cutT, cutwidth, expnum
        self.phi, self.a, self.b, self.R2, self.n_air, self.path_diff = phi, a, b, R2, n_air, path_diff


# =============================================================================
# 指定フォルダからCSVを探索 BPF_method (AbsoluteDinstance()のインスタンス)の引数にする絶対パスリストpaths_raw_dataを得るプロセス
# =============================================================================

print("CSVをまとめたxlsxを選択")
matomexlsxpath = Dialog_File(caption="matome XLSXえらぶ")

df_wholedata = pd.read_excel(matomexlsxpath, index_col=0, sheet_name="wholedata")
df_sort = pd.read_excel(matomexlsxpath, index_col=0, sheet_name="sort")

F_uneq = df_wholedata.index[28:].astype(float).values * 1e12
# faile nameをxlsxからListとして取得
# 辞書⇒Dataframe
# https://qiita.com/ShoheiKojima/items/30ee0925472b7b3e5d5c
names_rawdata = df_sort.loc["NAME", :].to_list()

# =============================================================================
# 計算用インスタンスBPF_method と結果格納データフレームdf_resultParams df_phiの準備
# =============================================================================

BPF_method = AbsoluteDistance()

df_resultParams = pd.DataFrame(
    index=["T_peak", "a", "b", "R2", "path_diff", "cutT", "cutwidth", "expnum"], columns=names_rawdata)
df_phi = pd.DataFrame(columns=names_rawdata)


# =============================================================================
# LIST_HYPERPARAMSのデータを適用し，分析を回す．
# =============================================================================

for idict_Params in LIST_HYPERPARAMS:
    print(idict_Params)
    # =============================================================================
    # name_rawdataのデータを全て計算する
    # =============================================================================

    for i_name in names_rawdata:

        """
        i番目のデータにたいし，分析を行って行く．
        その後ほしいパラメータをdf_resultParamsに追加していく
        """
        I_uneq = df_wholedata.loc[:, i_name][28:].astype(float).values*1e-3
        # LIST_HYPERPARAMS
        BPF_method.path_difference(F_unequal=F_uneq, I_unequal=I_uneq,
                                   cutT=idict_Params["cutT"], cutwidth=idict_Params["cutwidth"], expnum=idict_Params["expnum"])

        df_resultParams.loc["T_peak", i_name] = BPF_method.T_peak
        df_resultParams.loc["a", i_name] = BPF_method.a
        df_resultParams.loc["b", i_name] = BPF_method.b
        df_resultParams.loc["R2", i_name] = BPF_method.R2
        df_resultParams.loc["path_diff", i_name] = BPF_method.path_diff
        df_resultParams.loc["cutT", i_name] = BPF_method.cutT
        df_resultParams.loc["cutwidth", i_name] = BPF_method.cutwidth
        df_resultParams.loc["expnum", i_name] = BPF_method.expnum

        df_phi.loc[:, i_name] = BPF_method.phi
        print("\r"+i_name, end="")

    df_phi.index = BPF_method.F

    # =============================================================================
    # データ保存手続き
    # =============================================================================

    # df_resultParamsをdf_resultParamsOptimized(Dataframe)に変換・成型
    # df_resultParamsOptimized = pd.DataFrame.from_dict(df_resultParams).T
    # df_resultParamsOptimized.columns = names_rawdata

    #　matomexlsxがあるディレクトリに，分析結果格納ディレクトリnew_dir = "AnaResults"を作る．既に存在していたら作らない
    dir_Ana = os.path.join(os.path.split(matomexlsxpath)[0], "AnaResults")
    if os.path.exists(dir_Ana) == False:
        os.makedirs(dir_Ana)
    else:
        pass

    name_file_AnaResult = "Ana"+"cutT" + \
        str(idict_Params["cutT"]) + "_"+"cutwidth"+str(idict_Params["cutwidth"]
                                                       )+"_"+"expnum"+str(idict_Params["expnum"])+".xlsx"
    path_AnaResult = os.path.join(dir_Ana, name_file_AnaResult)

    df_resultParams.to_excel(path_AnaResult)

    # JUMP_FREQ = 8 #OPTIMIZE tuning
    # df_F_phi_dev8 = df_F_phi[::JUMP_FREQ]
    # df_F_phi_dev8.to_excel("f_F_phi_dev8.xlsx")

    # =============================================================================
    # 位置測定の結果をサンプリングナンバー順にプロット inlineがおすすめ
    # =============================================================================

    title="cutT="+str(idict_Params["cutT"]) +  "\n"+"cutwidth="+str(idict_Params["cutwidth"])+"\n"+"expnum="+str(idict_Params["expnum"])
    fig=plt.scatter(df_sort.loc["Posi_pls", :].astype(int),
                df_resultParams.loc["path_diff", :], s=2,label = title)
    plt.ylim(0.003,0.020)
    # plt.ylim(0.005, 0.01)
    plt.title(title)
    # plt.xlim(-20000, 20000)
    # plt.ylim(0.035, 0.055)
    plt.show()


"""
"""