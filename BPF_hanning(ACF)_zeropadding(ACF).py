# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 02:11:54 2021

@author: wsxhi
https://rikei-fufu.com/2020/06/27/post-3237-python-fft/#index_id3
https://watlab-blog.com/2019/04/20/window-correction/

#!!! たくさんプロットされるため，inline plotがおすすめ．
"""

# *****************************
# * ------------バンドパスフィルタ法------
# *****************************
#%%
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
#%%
n_air = ref_index.edlen(
    wave=(1554.134049+1563.862587)/2, t=27, p=101325, rh=70)
THETA_RAD = 1.216774219

K = 1/(1+np.cos(THETA_RAD))/n_air


STAGE_RSN = 0.1e-6  # m/pls ステージの分解能
LIST_HYPERPARAMS = (
    # @ exp13,pad4
    #     BPF_method.delta_T
    # Out[26]: 8.333263889468021e-13


    dict(cutT=13e-12, cutwidth=40e-12, expnum=13, PAD_EXP=2,ANA_FREQ_START=191.7928e12,ANA_FREQ_END=191.9765e12), 

)

#%%

def Dialog_File(rootpath=r"C:", caption="choise"):
    """
    引数:初期ディレクトリ
    戻り値：ファイルパス
    """
    # from PyQt5 import QtWidgets
    # 実行ディレクトリ取得D

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
    # from PyQt5 import QtWidgets
    # import sys
    # # ディレクトリ選択ダイアログを表示
    # app = QtWidgets.QApplication(sys.argv)
    folderpath = QtWidgets.QFileDialog.getExistingDirectory(
        parent=None, caption=caption, directory=rootpath)
    return folderpath

# %%
class AbsoluteDistance():

    def __init__(self):
        pass

    def OSAcsvfre_In(self, filepath):  # fre-IのOSA信号(35行目から)をよむ freq domeinではデータは不当間隔データ
        self.filepath = filepath
        wholedata = pd.read_csv(self.filepath, header=None, skiprows=34).values
        # wholedataの[:,0]を取り出した後，ravel1で次元配列へ直す 単位もHz, Wへ変換
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

    def FFT(self, x, y):
        """[summary]
        等間隔データをFFT
        Args:
            x ([float]): [signal]
            y ([float]): [signal]

        Returns:
            freq ([float]): [signal]
            FF ([complex): [signal]
            FF_abs_amp ([float]): |FF|
            
        """        
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
        """[zero paddingの結果と，補正係数acfを出力]
        本サイトの補正係数，おそらく本原理においては不要．算出だけした．
        https://watlab-blog.com/2020/11/16/zero-padding-fft/ 
        Args:
            data ([float array]): [description]
            len_pad ([int]): [description]

        Returns:
            data_pad [float array]: [description]
            acf[float]: [description]
            
        """        

        pad = np.zeros(len_pad-len(data))
        data_pad = np.concatenate([data, pad])
        acf = (sum(np.abs(data)) / len(data)) / \
            (sum(np.abs(data_pad)) / len(data_pad))
        return data_pad, acf

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
        Returns:
            None

        """
        """補間 I(f_uneq) => I(f_euneq)"""
        self.F_inter, self.I_inter, self.SampNum_inter, self.dF = self.Inter(
            F_unequal, I_unequal, expnum)

        """window (F,Iが非周期信号であるため)
        https://rikei-fufu.com/2020/06/27/post-3237-python-fft/
        このサイトに従い，補正係数acf_han ももとめた
        本サイトに従い補正係数も設けたが，おそらく本原理においては不要
        """
        
        hanning_win = np.hanning(self.SampNum_inter)
        # FFT後の数値に掛ければOKの補正係数
        acf_han = 1/(sum(hanning_win)/self.SampNum_inter)
        self.I_han = self.I_inter * hanning_win

        """zero padding (サンプリング数が不十分であり，FFT周波数分解能が不足しているため)"""
        self.len_pad = self.SampNum_inter*pow(2, pad_exp)
        self.I_han_pad, self.acf_pad=self.zero_padding(self.I_han, self.len_pad)
        self.F_pad = np.linspace(
            self.F_inter[0], 
            self.F_inter[0]+(self.len_pad)*self.dF, 
            self.len_pad+1)[:-1]
        """FFT:  I(f) C_1 + C_2*cos(phi(f) ) ====>   FFt(T)=C_1 + C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))"""
        self.T, self.FFt, self.FFt_abs_amp = self.FFT(
            self.F_pad, self.I_han_pad)
        self.FFt, self.FFt_abs_amp = self.FFt, self.FFt_abs_amp

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

        """振動成分があるF_pad-phi領域のみを取り出して，a, bを求めるように変更"""
        self.F_pad_cut, self.phi_cut =self.F_pad[(ana_freq_start<self.F_pad)&(self.F_pad<ana_freq_end)], self.phi[(ana_freq_start<self.F_pad)&(self.F_pad<ana_freq_end)]
        self.a, self.b = np.polyfit(
            self.F_pad_cut, self.phi_cut, 1)  # phi = a *F + bの1じ多項式近似

        # https://biotech-lab.org/articles/4907 R2値
        self.R2 = metrics.r2_score(self.phi_cut, self.a * self.F_pad_cut + self.b)
        self.path_diff = 299792458/(2*np.pi*n_air)*self.a


# =============================================================================
# 分析準備
# 分析用データを取得．　結果保存用dfを作成，計算用インスタンスの作成
# =============================================================================
"""CSVをまとめたxlsxを選択し，分析用データを得る"""
print("CSVをまとめたxlsxを選択")
matomexlsxpath = Dialog_File(caption="matome XLSXえらぶ")

df_wholedata = pd.read_excel(
    matomexlsxpath, index_col=0, sheet_name="wholedata")
df_sort = pd.read_excel(matomexlsxpath, index_col=0, sheet_name="sort")

F_uneq = df_wholedata.index[28:].astype(float).values * 1e12
# faile nameをxlsxからListとして取得
# 辞書⇒Dataframe
# https://qiita.com/ShoheiKojima/items/30ee0925472b7b3e5d5c
names_rawdata = df_sort.loc["NAME", :].to_list()

"""計算用インスタンスBPF_method と結果格納データフレームdf_resultParams df_phiの準備"""

dict_instance = dict()
BPF_method = AbsoluteDistance()

df_resultParams = pd.DataFrame(
    index=["path_diff", "Tpeak", "a", "b", "R2",
           "cutT", "cutwidth", "expnum", "pad_exp", "k",
           "stage_m", "stage_um", "z_m", "z_um"],
    columns=names_rawdata)

df_resultParams.loc["stage_m", :] = df_sort.loc["Posi_pls", :]*STAGE_RSN
df_resultParams.loc["stage_um", :] = df_sort.loc["Posi_pls", :]*STAGE_RSN*1e6


df_phi = pd.DataFrame(columns=names_rawdata)


# =============================================================================
# LIST_HYPERPARAMSのハイパーパラメータを適用し，*1分析・計算．*2データ保存．*3プロット，
# =============================================================================

for idict_Params in LIST_HYPERPARAMS:
    print(idict_Params)

    for i_name in names_rawdata:

        """
        *1 分析・計算
        i番目のデータにたいし，分析を行って行く．
        その後ほしいパラメータをdf_resultParamsに追加していく
        """
        I_uneq = df_wholedata.loc[:, i_name][28:].astype(float).values*1e-3
        # LIST_HYPERPARAMS
        BPF_method.path_difference(F_unequal=F_uneq, I_unequal=I_uneq,
                                   cutT=idict_Params["cutT"], cutwidth=idict_Params["cutwidth"], expnum=idict_Params["expnum"], pad_exp=idict_Params["PAD_EXP"],ana_freq_start=idict_Params["ANA_FREQ_START"],ana_freq_end=idict_Params["ANA_FREQ_END"])

        df_resultParams.loc["Tpeak", i_name] = BPF_method.Tpeak
        df_resultParams.loc["a", i_name] = BPF_method.a
        df_resultParams.loc["b", i_name] = BPF_method.b
        df_resultParams.loc["R2", i_name] = BPF_method.R2
        df_resultParams.loc["path_diff", i_name] = BPF_method.path_diff
        df_resultParams.loc["z_m", i_name] = BPF_method.path_diff * K
        df_resultParams.loc["z_um", i_name] = BPF_method.path_diff * K*1e6

        # df_phi.loc[:, i_name] = BPF_method.phi
        print("\r"+i_name, end="")
    print("\n")
    print("\n")
    df_resultParams.loc["cutT", names_rawdata[0]], df_resultParams.loc["cutwidth", names_rawdata[0]], df_resultParams.loc["expnum", names_rawdata[0]], df_resultParams.loc["pad_exp", names_rawdata[0]], df_resultParams.loc["k",
                                                                                                                                                                                                                             names_rawdata[0]] = idict_Params["cutT"], idict_Params["cutwidth"], idict_Params["expnum"], idict_Params["pad_exp"], K = idict_Params["cutT"], idict_Params["cutwidth"], idict_Params["expnum"], idict_Params["PAD_EXP"], K


    """
    *2 データ保存
    """
    """matomexlsxがあるディレクトリに，分析結果格納ディレクトリnew_dir = "AnaResults"を作る．既に存在していたら作らない"""
    dir_Ana = os.path.join(os.path.split(matomexlsxpath)[0], "AnaResults_F_pad_limit_Ver")
    if os.path.exists(dir_Ana) == False:
        os.makedirs(dir_Ana)
    else:
        pass

    name_file_AnaResult = "cutT" + str(idict_Params["cutT"]) + "_" +\
        "cutw"+str(idict_Params["cutwidth"])+"_" +\
        "exp"+str(idict_Params["expnum"])+str(idict_Params["PAD_EXP"])+\
        "startend"+str(idict_Params["ANA_FREQ_START"])+"-"+str(idict_Params["ANA_FREQ_END"]) + ".xlsx"
    path_AnaResult = os.path.join(dir_Ana, name_file_AnaResult)

    df_resultParams.to_excel(path_AnaResult)
    # JUMP_FREQ = 8 #OPTIMIZE tuning
    # df_F_phi_dev8 = df_F_phi[::JUMP_FREQ]
    # df_F_phi_dev8.to_excel("f_F_phi_dev8.xlsx")

    """
    *3 プロット
    プロット位置測定の結果をサンプリングナンバー順に inlineがおすすめ
    """

    title = "cutT="+str(idict_Params["cutT"]) + "\n" \
        + "cutwidth=" + str(idict_Params["cutwidth"])+"\n" \
        + "expnum=" + str(idict_Params["expnum"])+str(idict_Params["PAD_EXP"])
    fig = plt.scatter(df_sort.loc["Posi_pls", :].astype(int),
                      df_resultParams.loc["path_diff", :], s=1, label=title)
    # plt.xlim(-20000, 0)
    # plt.ylim(0.005, 0.01)
    plt.title(title)
    plt.show()

    # plt.xlim(-20000, 20000)
    # plt.ylim(0.035, 0.055)
del app