# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 18:17:29 2021

@author: wsxhi
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
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import ref_index
from scipy.fftpack import fftfreq
from PyQt5 import QtWidgets
app = QtWidgets.QApplication(sys.argv)

n_air = ref_index.edlen(
    wave=(1554.134049+1563.862587)/2, t=27, p=101325, rh=70)
THETA_RAD = 1.214548722 #原理検証の時

K = 1/(1+np.cos(THETA_RAD))/n_air


STAGE_RSN = 0.1e-6  # m/pls ステージの分解能#TODO
LIST_HYPERPARAMS = ( #TODO
    # @ exp13,pad4
    #     BPF_method.delta_T
    # Out[26]: 8.333263889468021e-13


    dict(cutT=35e-12, cutwidth=10e-12, expnum=13, PAD_EXP=4,ANA_FREQ_START=191.65e12,ANA_FREQ_END=191.75e12),  # Goo

    # dict(cutT=10e-12, cutwidth=2e-13, expnum=14),  # Goo
    # dict(cutT=10e-12, cutwidth=2e-12, expnum=14),  # Goo
    # dict(cutT=10e-12, cutwidth=5e-12, expnum=14),  # Goo
    # dict(cutT=2e-12, cutwidth=0.5e-12, expnum=13, PAD_EXP=4),  # Goo0
    # dict(cutT=2e-12, cutwidth=0.01e-12, expnum=13, PAD_EXP=4),  # Goo1


    # dict(cutT=2e-12, cutwidth=0.5e-12, expnum=13, PAD_EXP=4),  # Goo4
    # dict(cutT=2e-12, cutwidth=1.5e-12, expnum=13, PAD_EXP=4),  # Goo5
    # dict(cutT=2e-12, cutwidth=0.5e-12, expnum=13, PAD_EXP=5),  # Goo6
    # dict(cutT=2e-12, cutwidth=0.01e-12, expnum=13, PAD_EXP=5),  # Goo7
    # dict(cutT=2e-12, cutwidth=0.05e-12, expnum=13, PAD_EXP=5),  # Goo8
    # dict(cutT=2e-12, cutwidth=0.1e-12, expnum=13, PAD_EXP=5),  # Goo
    # dict(cutT=2e-12, cutwidth=0.5e-12, expnum=13, PAD_EXP=5),  # Goo
    # dict(cutT=2e-12, cutwidth=1.5e-12, expnum=13, PAD_EXP=5),  # Goo
    # dict(cutT=2e-12, cutwidth=0.005e-12, expnum=13, PAD_EXP=4),  # Goo
    # dict(cutT=2e-12, cutwidth=0.01e-12, expnum=13, PAD_EXP=4),  # Goo
    # dict(cutT=2e-12, cutwidth=0.05e-12, expnum=13, PAD_EXP=4),  # Goo
    # dict(cutT=2e-12, cutwidth=0.1e-12, expnum=13, PAD_EXP=4),  # Goo
    # dict(cutT=2e-12, cutwidth=0.5e-12, expnum=13, PAD_EXP=4),  # Goo

)

def Dialog_File(rootpath=r"C:", caption="choise"):
    """
    引数:初期ディレクトリ
    戻り値:ファイルパス
    """
    # from PyQt5 import QtWidgets
    # 実行ディレクトリ取得D

    # ディレクトリ選択ダイアログを表示-
    filepath = QtWidgets.QFileDialog.getOpenFileName(
        parent=None, caption=caption, directory=rootpath)

    # sys.exit(app_dialog_file.exec_())
    return filepath[0]


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
            x (float): signal.補完まえ
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

    def zero_padding(self, data, len_pad):
        pad = np.zeros(len_pad-len(data))
        data_pad = np.concatenate([data, pad])
        acf = (sum(np.abs(data)) / len(data)) / \
            (sum(np.abs(data_pad)) / len(data_pad))
        return data_pad * acf

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

    def path_difference(self, F_unequal, I_unequal, cutT=10e-12, cutwidth=1e-12, expnum=16, pad_exp=3, ana_freq_start = 191.65e12, ana_freq_end=191.75e12):
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
        """補間 I(f_uneq) => I(f_euneq)"""
        self.F_inter, self.I_inter, self.SampNum_inter, self.dF = self.Inter(
            F_unequal, I_unequal, expnum)

        """window (F,Iが非周期信号であるため)"""
        hanning_win = np.hanning(self.SampNum_inter)
        # FFT後の数値に掛ければOKの補正係数
        acf_han = 1/(sum(hanning_win)/self.SampNum_inter)
        self.I_han = self.I_inter * hanning_win

        """zero padding (サンプリング数が不十分であり，FFT周波数分解能が不足しているため)"""
        len_pad = self.SampNum_inter*pow(2, pad_exp)
        self.I_han_pad = self.zero_padding(self.I_han, len_pad)
        self.F_pad = np.linspace(
            self.F_inter[0], self.F_inter[0]+(len_pad)*self.dF, len_pad+1)[:-1]
        """FFT:  I(f) C_1 + C_2*cos(phi(f) ) ====>   FFt(T)=C_1 + C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))"""
        self.T, self.FFt, self.FFt_abs_amp = self.FFT(
            self.F_pad, self.I_han_pad)
        self.FFt, self.FFt_abs_amp = self.FFt * acf_han, self.FFt_abs_amp * acf_han

        # plt.plot(self.T, self.FFt_abs_amp)
        # plt.xlim(0, 30e-12)

        # # plt.ylim(-0.2e-7, 0.2e-6)
        # plt.show()
        # plt.plot(T, FFt_abs_amp)
        self.delta_T = (1.0/self.dF)/(self.SampNum_inter-1)

        """Filtering: FFt(T)=C_1 + C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))
          ====> F2(T)=C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))"""
        self.F2 = copy.deepcopy(self.FFt)
        self.F2[(self.T <= 0)] = 0  # (負の)周波数帯をカット
        self.F2[(self.T < cutT)] = 0  # カットオフ未満周波数のデータをゼロにする，光源の影響排除
        self.F2_abs = np.abs(self.F2)
        # 振幅をもとの信号に揃える
        self.F2_abs_amp = self.F2_abs / self.SampNum_inter * 2  # 交流成分はデータ数で割って2倍
        # plt.plot(T, F2_abs_amp)
        # plt.xlim(-300e-12, 300e-12)
        # plt.ylim(-0.2e-7, 0.2e-6)
        # plt.show()
        """Filtering:   F2(T)=C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))
          ====>  F3(T)=C_2/2 exp(j*phi(T) )"""

        self.F3 = copy.deepcopy(self.F2)


        self.Tpeak = self.T[np.argmax(self.F2_abs_amp)]  # peak T(>0)
        self.F3[((self.T < self.Tpeak-cutwidth/2) |
                 (self.Tpeak+cutwidth/2 < self.T))] = 0  # 所望のピークだけのこす

        """IFFT   F3(T)=C_2/2 exp(j*phi(T) )  ====>  I(f)=C_2/2 exp(j*phi(f) )"""
        self.F3_ifft = np.fft.ifft(self.F3)  # IFFT  C_2/2 exp(j*phi(f) )
        self.F3_ifft_abs = np.abs(self.F3_ifft)
        self.F3_ifft_abs_amp = self.F3_ifft_abs / self.SampNum_inter * 2

        self.wrap = self.wrappedphase(self.F3_ifft)
        # wrap=F3_ifft
        # wrap_abs=np.abs(wrap)

        self.phi = np.unwrap(p=self.wrap * 2)/2
        self.a, self.b = np.polyfit(
            self.F_pad, self.phi, 1)  # phi = a *F + bの1じ多項式近似

        """振動成分があるF_pad-phi領域のみを取り出して，a, bを求めるように変更"""
        self.F_pad, self.phi =self.F_pad[(ana_freq_start<self.F_pad)&(self.F_pad<ana_freq_end)], self.phi[(ana_freq_start<self.F_pad)&(self.F_pad<ana_freq_end)]
        self.a, self.b = np.polyfit(
            self.F_pad, self.phi, 1)  # phi = a *F + bの1じ多項式近似

        # https://biotech-lab.org/articles/4907 R2値
        self.R2 = metrics.r2_score(self.phi, self.a * self.F_pad + self.b)
        self.path_diff = 299792458/(2*np.pi*n_air)*self.a

path_csv = Dialog_File(caption="choise single CSV")

BPF_method = AbsoluteDistance()
F_uneq, I_uneq = BPF_method.OSAcsvfre_In(path_csv)

dict_Param =LIST_HYPERPARAMS[0]
BPF_method.path_difference(F_unequal=F_uneq,
                           I_unequal=I_uneq,
                           cutT=dict_Param["cutT"],
                           cutwidth=dict_Param["cutwidth"],
                           expnum=dict_Param["expnum"],
                           pad_exp=dict_Param["PAD_EXP"],
                           ana_freq_start=dict_Param["ANA_FREQ_START"],
                           ana_freq_end=dict_Param["ANA_FREQ_END"]
                           )