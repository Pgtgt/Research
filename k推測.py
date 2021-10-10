# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 20:59:13 2021

@author: wsxhi
"""

import numpy as np
import pandas as pd
import os
from scipy import optimize
import matplotlib.pyplot as plt
import sys

def Dialog_File(rootpath=r"C:", caption="choise"):
    """
    choose folder path by Explore

    Args:
        rootpath (str, optional): initial path of Explore. Defaults to r"C:".
        caption (str, optional): title of Explore. Defaults to "choise".

    Returns:
        folderpath (str): folder path.
    """
    from PyQt5 import QtWidgets
    # ディレクトリ選択ダイアログを表示-
    app = QtWidgets.QApplication(sys.argv)
    filepath = QtWidgets.QFileDialog.getOpenFileName(
        parent=None, caption=caption, directory=rootpath)

    return filepath[0]

# =============================================================================
# csvmatome.xlxsから，データを取り出し，
# 強度:df_intensities["ファイル名"]
# frequency:freq
# =============================================================================

path_csvmatome = Dialog_File(caption = "choose matome.xlxs")
df_wholedata = pd.read_excel(path_csvmatome, sheet_name = "wholedata",header =0, index_col =0)
df_sort = pd.read_excel(path_csvmatome, sheet_name = "sort",header =0, index_col =0)

df_intensities=df_wholedata.loc["[TRACE DATA]":].iloc[1:].astype(float)*1e-3
freq=df_wholedata.loc["[TRACE DATA]":].iloc[1:].astype(float).index *1e12

# name0 = "000060-0_OSA1@-10000pls"

# =============================================================================
# intensity = A * norm(sigma, mu)形式のフィッテイングを行う．
# A, sigma, muを返す．
# https://lastline.hatenablog.com/entry/2018/04/12/112502
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# =============================================================================

# 推定後パラメータを格納用のdfを設定
df_fit = pd.DataFrame(index = ["A","mu","sigma","r2"], columns =df_intensities.columns)

# 関数定義
def gauss(x, A=1, mu=0, sigma=1):
    return A * np.exp(-(x - mu)**2 / (2*sigma**2))

for dataname in df_intensities.columns:

    # 関数のピークから推定用の初期値を設定
    peak_num=np.argmax(df_intensities[dataname])
    init_param = [df_intensities[dataname][peak_num], freq[peak_num], 0.1e+12]

    # 最適化
    popt = 0,0,0 # default
    try:
        popt, pcov = optimize.curve_fit(gauss, freq, df_intensities[dataname], p0 = init_param)
        # https://sabopy.com/py/curve_fit/
        residuals =  df_intensities[dataname] - gauss(freq, popt[0],popt[1],popt[2])
        rss = np.sum(residuals**2)#residual sum of squares = rss
        tss = np.sum((df_intensities[dataname]-np.mean(df_intensities[dataname]))**2)#total sum of squares = tss
        r2 = 1 - (rss / tss)
        df_fit.loc["A",dataname],df_fit.loc["mu",dataname],df_fit.loc["sigma",dataname]=popt
        df_fit.loc["r2",dataname] = r2
    except:
        print(dataname+"failure")

df_sort_fit = pd.concat([df_sort, df_fit])


with pd.ExcelWriter(path_csvmatome, engine="openpyxl",mode = "a") as writer:
    #engine="openpyxl"にしないと，mode = "a"が使えない
    # https://stackoverflow.com/questions/54863238/pandas-excelwriter-valueerror-append-mode-is-not-supported-with-xlsxwriter
    df_sort_fit.to_excel(writer, sheet_name="sort_fit",)





# col = ["s1","s2","s3","s4","s5",]
# index =[1,2,3,4,5,6,7,8]
# df = pd.DataFrame(index = index, columns = col)

