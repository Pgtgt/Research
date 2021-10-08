# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 16:45:47 2021

@author: anonymous
"""

import numpy as np
import plotly.graph_objects as go

import plotly.express as px
import plotly.io as pio
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import sys
import pandas as pd

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

path_excel = Dialog_File(caption = "distance excel")

SHEETNAME = "posi-STD_X"
# SHEETNAME = "posi-AVE_X"

df_distance_info = pd.read_excel(path_excel, sheet_name = SHEETNAME, header = 0, index_col=0)