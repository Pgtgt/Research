# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 20:59:13 2021

@author: wsxhi

1st spectrumを収めたcsvmatome.xlsxのsheet_name = "wholedata","sort"から
    ・各スペクトルのピーク周波数
    ・theta
等を求める．
その後csvmaotme.xlsxにsheet_name = "sort_fit"を追加

! %matplotlib inlineを奨励
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

# *************************
# * PARAM SETTING (n_air(refractive index) m_per_gr(grating m/groove), M(order))
# *************************

dict_nparam = dict(wave=(1554.134049+1563.862587)/2,
                   t=27, p=101325, rh=70)
n_air = ref_index.edlen(
    wave=dict_nparam["wave"], t=dict_nparam["t"], p=dict_nparam["p"], rh=dict_nparam["rh"])

# Grating info
# https://www.thorlabs.co.jp/thorproduct.cfm?partnumber=GR13-0616
gr_per_mm = 600  # grating pitch /mm
mm_per_gr = 1 / gr_per_mm
m_per_gr = mm_per_gr * 1e-3

M = 1
c = 299792458

# **************************
# * csvmatome.xlxsから，データを取り出し，
# * 強度:df_intensities["ファイル名"]
# * frequency:freq
# **************************
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
df_sort = pd.read_excel(
    path_csvmatome, sheet_name="sort", header=0, index_col=0)

df_intensities = df_wholedata.loc["[TRACE DATA]":].iloc[1:].astype(float)*1e-3
freq = df_wholedata.loc["[TRACE DATA]":].iloc[1:].astype(float).index * 1e12


# **************************
# * intensity = A * norm(sigma, mu)形式のフィッテイングを行う．
# * A, sigma, muを返す．
# * https://lastline.hatenablog.com/entry/2018/04/12/112502
# * https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
# **************************

# gfdsg
# index=["k","A", "mu:f_center", "theta_rad", "sigma","r2", "n_air", "d(m/groove)"]

df_fit = pd.DataFrame(index=["A", "mu:f_center", "sigma", "trust", "f_center", "theta_rad", "theta_deg", "k", "d(m/groove)", "n_air", "wave", "t", "p", "rh"],
                      columns=df_intensities.columns)
"""[推定後パラメータを格納用のdf_fitを設定]"""


def gauss(x, A=1, mu=0, sigma=1):
    """[summary]

    Args:
        x ([float array]): [description]
        A (int, optional): [description]. Defaults to 1.
        mu (int, optional): [description]. Defaults to 0.
        sigma (int, optional): [description]. Defaults to 1.

    Returns:
        y [float array]: [description]
    """
    y = A * np.exp(-(x - mu)**2 / (2*sigma**2))
    return y


max_intensity = df_intensities.max().max()
"""[全強度中の最大強度]"""

for dataname in df_intensities.columns:
    """関数のピークから推定用の初期値"init_param"を設定"""
    peak_num = np.argmax(df_intensities[dataname])
    init_param = [df_intensities[dataname][peak_num], freq[peak_num], 0.1e+12]

    print("\r"+dataname, end="")
    """fittingの実行と各パラメータ計算，データ格納，プロット
    """
    try:
        """Gaussian fitting"""
        popt, pcov = optimize.curve_fit(
            gauss, freq, df_intensities[dataname], p0=init_param)

        """Store Results"""
        df_fit.loc["A", dataname], df_fit.loc["mu:f_center",
                                              dataname], df_fit.loc["sigma", dataname] = popt

        # """R2値https://sabopy.com/py/curve_fit/"""
        # residuals = df_intensities[dataname] - gauss(freq, popt[0], popt[1], popt[2])
        # rss = np.sum(residuals**2)  # residual sum of squares = rss
        # # total sum of squares = tss
        # tss = np.sum((df_intensities[dataname]-np.mean(df_intensities[dataname]))**2)
        # r2 = 1 - (rss / tss)

        # ! plot (%matplotlib inlineがおすすめ)
        plt.plot(freq, df_intensities[dataname])
        plt.plot(freq, gauss(freq, A=popt[0], mu=popt[1], sigma=popt[2]))
        plt.title(dataname)
        plt.ylim(-0.01*max_intensity, max_intensity)
        plt.show()

    except:
        print(dataname+"failure")

# **************************
# * 信頼区間の選別とf_center導出．
# * 信頼区間を求め，具体的区間span_trustの表示
# * 信頼区間から各パラメータ(f_center, theta_Rad, k)等を算出
# * df_sort_fit = df_sort + df_fitを足して，データ保存
# **************************

# 1/2*max_in(強度不足)  <A <  1.5*max_in（外れ値）]
span_trust = (
    (max_intensity/2 < df_fit.loc["A", :]) & (df_fit.loc["A", :] < max_intensity*1.5))

df_fit.loc["trust", :] = span_trust
f_center = df_fit.loc["mu:f_center", span_trust].mean()
theta_rad = np.arcsin(M*c/(n_air*f_center*m_per_gr))
theta_deg = np.degrees(theta_rad)
k = 1 / (n_air * (1 + np.cos(theta_rad)))

"""結果および屈折率関係をdf_fitへ格納 最初の列のみに格納"""
df_fit.loc["f_center", df_intensities.columns[0]] = f_center
df_fit.loc["theta_rad", df_intensities.columns[0]] = theta_rad
df_fit.loc["theta_deg", df_intensities.columns[0]] = theta_deg

df_fit.loc["k", df_intensities.columns[0]] = k
df_fit.loc["d(m/groove)", df_intensities.columns[0]] = m_per_gr


df_fit.loc["n_air", df_intensities.columns[0]] = n_air
df_fit.loc["wave", df_intensities.columns[0]] = dict_nparam["wave"]
df_fit.loc["t", df_intensities.columns[0]] = dict_nparam["t"]
df_fit.loc["p", df_intensities.columns[0]] = dict_nparam["p"]
df_fit.loc["rh", df_intensities.columns[0]] = dict_nparam["rh"]

df_fit.to_excel("fit1.xlsx")

# df_sort_fit = pd.concat([df_sort, df_fit])


# with pd.ExcelWriter(path_csvmatome, engine="openpyxl", mode="a") as writer:
#     # engine="openpyxl"にしないと，mode = "a"が使えない
#     # https://stackoverflow.com/questions/54863238/pandas-excelwriter-valueerror-append-mode-is-not-supported-with-xlsxwriter
#     df_sort_fit.to_excel(writer, sheet_name="sort_fit",)

del app
# col = ["s1","s2","s3","s4","s5",]
# index =[1,2,3,4,5,6,7,8]
# df = pd.DataFrame(index = index, columns = col)
