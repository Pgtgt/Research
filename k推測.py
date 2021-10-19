# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 20:59:13 2021

@author: wsxhi

1st spectrumを収めたcsvmatome.xlsxのsheet_name = "wholedata","sort"から
    ・各スペクトルのピーク周波数
    ・theta
等を求める．
その後csvmaotme.xlsxにsheet_name = "sort_fit"を追加

* %matplotlib inlineを奨励
連続してplt.show()するため，
%matplotlib により実行すると固まる

"""

import numpy as np
import pandas as pd
import os
from scipy import optimize
import matplotlib.pyplot as plt
import sys
from PyQt5 import QtWidgets
import ref_index

# =============================================================================
# PARAM SETTING (n_air(refractive index) m_per_gr(grating m/groove), M(order))
# =============================================================================
n_air = ref_index.edlen(wave=(1554.134049+1563.862587)/2, t=27, p=101325, rh=70)

# Grating info
# https://www.thorlabs.co.jp/thorproduct.cfm?partnumber=GR13-0616
gr_per_mm = 600  # grating pitch /mm
mm_per_gr = 1 / gr_per_mm
m_per_gr = mm_per_gr * 1e-3

M = 1
c = 299792458


# =============================================================================
# csvmatome.xlxsから，データを取り出し，
# 強度:df_intensities["ファイル名"]
# frequency:freq
# =============================================================================
app = QtWidgets.QApplication(sys.argv)


def Dialog_File(rootpath=r"C:", caption="choise"):
    """
    choose folder path by Explore

    Args:
        rootpath (str, optional): initial path of Explore. Defaults to r"C:".
        caption (str, optional): title of Explore. Defaults to "choise".

    Returns:
        folderpath (str): folder path.
    """

    # ディレクトリ選択ダイアログを表示-
    filepath = QtWidgets.QFileDialog.getOpenFileName(
        parent=None, caption=caption, directory=rootpath)

    return filepath[0]


path_csvmatome = Dialog_File(caption="choose matome.xlxs")
df_wholedata = pd.read_excel(
    path_csvmatome, sheet_name="wholedata", header=0, index_col=0)
df_sort = pd.read_excel(path_csvmatome, sheet_name="sort", header=0, index_col=0)

df_intensities = df_wholedata.loc["[TRACE DATA]":].iloc[1:].astype(float)*1e-3
freq = df_wholedata.loc["[TRACE DATA]":].iloc[1:].astype(float).index * 1e12


# =============================================================================
# intensity = A * norm(sigma, mu)形式のフィッテイングを行う．
# A, sigma, muを返す．
# PARAM SETTINGSの値を使って，　回折格子方程式からtheta(:中心周波数)を出す．
# https://lastline.hatenablog.com/entry/2018/04/12/112502
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# =============================================================================

"""推定後パラメータを格納用のdfを設定"""
df_fit = pd.DataFrame(index=["A", "mu:f_center", "theta_rad", "sigma",
                      "r2", "n_air", "d(m/groove)"], columns=df_intensities.columns)


"""関数定義"""
def gauss(x, A=1, mu=0, sigma=1):
    return A * np.exp(-(x - mu)**2 / (2*sigma**2))


for dataname in df_intensities.columns:

    """関数のピークから推定用の初期値を設定"""
    peak_num = np.argmax(df_intensities[dataname])
    init_param = [df_intensities[dataname][peak_num], freq[peak_num], 0.1e+12]

    print("\r"+dataname, end="")
    """fittingの実行と各パラメータ計算，データ格納，プロット
    """
    try:
        """Gaussian fitting"""
        popt, pcov = optimize.curve_fit(
            gauss, freq, df_intensities[dataname], p0=init_param)


        """f_center by Grating equ."""
        f_center = popt[1]
        theta_rad = np.arcsin(M*c / (n_air*f_center*m_per_gr))

        """R2値https://sabopy.com/py/curve_fit/"""
        residuals = df_intensities[dataname] - gauss(freq, popt[0], popt[1], popt[2])
        rss = np.sum(residuals**2)  # residual sum of squares = rss
        # total sum of squares = tss
        tss = np.sum((df_intensities[dataname]-np.mean(df_intensities[dataname]))**2)
        r2 = 1 - (rss / tss)

        """結果をdf_fitへ格納"""
        df_fit.loc["A", dataname], df_fit.loc["mu:f_center",
                                              dataname], df_fit.loc["sigma", dataname] = popt
        df_fit.loc["theta_rad", dataname] = theta_rad
        df_fit.loc["r2", dataname] = r2

        """plot (%matplotlib inlineがおすすめ)"""
        plt.plot(freq, df_intensities[dataname])
        plt.plot(freq, gauss(freq, A=popt[0], mu=popt[1], sigma=popt[2]))
        plt.title(dataname)
        plt.ylim( -0.01*df_intensities.max().max(), df_intensities.max().max())
        plt.show()

    except:
        print(dataname+"failure")

df_fit.loc["n_air", df_intensities.columns[0]] = n_air
df_fit.loc["d(m/groove)", df_intensities.columns[0]] = m_per_gr
df_sort_fit = pd.concat([df_sort, df_fit])


with pd.ExcelWriter(path_csvmatome, engine="openpyxl", mode="a") as writer:
    # engine="openpyxl"にしないと，mode = "a"が使えない
    # https://stackoverflow.com/questions/54863238/pandas-excelwriter-valueerror-append-mode-is-not-supported-with-xlsxwriter
    df_sort_fit.to_excel(writer, sheet_name="sort_fit",)


# col = ["s1","s2","s3","s4","s5",]
# index =[1,2,3,4,5,6,7,8]
# df = pd.DataFrame(index = index, columns = col)
