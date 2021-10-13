# -*- coding: utf-8 -*-
"""
Created on Wed Aug 11 15:14:59 2021

@author: anonymous
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
# expnum=16
# cutT =10e-12
# cutwidth=1e-12
n_air=ref_index.edlen(wave = (1554.134049+1563.862587)/2 ,t=27, p =101325, rh=70)

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
        FFt = np.fft.fft(y)
        FFt_abs = np.abs(FFt)
        FFt_abs_amp = FFt_abs/(N/2)
        dx = np.abs((x[-1]-x[0])/(N-1))
        f = np.linspace(0, 1.0/dx, N)

        return f, FFt, FFt_abs_amp

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

    def path_difference(self, filepath, cutT=10e-12, cutwidth=1e-12, expnum=16, removeT=[None, None]):
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
        filepath, cutT, cutwidth, expnum = filepath, cutT, cutwidth, expnum
        # Read CSV
        F_uneq, I_uneq = self.OSAcsvfre_In(filepath)
        # 補間 I(f_uneq) => I(f_euneq)
        F, I, SampNum_inter, dF = self.Inter(F_uneq, I_uneq, expnum)

        # =============================================================================
        # FFT:  I(f) C_1 + C_2*cos(phi(f) ) ====>   FFt(T)=C_1 + C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))
        # =============================================================================
        #
        T, FFt, FFt_abs_amp = self.FFT(F, I)
        delta_T = (1.0/dF)/(SampNum_inter-1)

        # =============================================================================
        # Filtering: FFt(T)=C_1 + C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))
        #   ====> F2(T)=C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))
        # =============================================================================
        F2 = copy.deepcopy(FFt)
        F2[(T < cutT)] = 0  # カットオフ未満周波数のデータをゼロにする，光源の影響排除
        F2[(T > T[2**(expnum-1)])] = 0  # 1/2後半の(負の)周波数帯をカット
        F2_abs = np.abs(F2)
        # 振幅をもとの信号に揃える
        F2_abs_amp = F2_abs / SampNum_inter * 2  # 交流成分はデータ数で割って2倍

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

        F3[((T < T[peak]-cutwidth/2) | (T[peak]+cutwidth/2 < T))] = 0  # 所望のピークだけのこす

        # IFFT   F3(T)=C_2/2 exp(j*phi(T) )  ====>  I(f)=C_2/2 exp(j*phi(f) )
        F3_ifft = np.fft.ifft(F3)  # IFFT  C_2/2 exp(j*phi(f) )
        F3_ifft_abs = np.abs(F3_ifft)
        F3_ifft_abs_amp = F3_ifft_abs / SampNum_inter * 2

        wrap = self.wrappedphase(F3_ifft)
        # wrap=F3_ifft
        # wrap_abs=np.abs(wrap)

        phi = np.unwrap(p = wrap *2)/2

        a, b = np.polyfit(F, phi, 1)  # phi = a *F + bの1じ多項式近似
        # a =2 pi Dd n / c
        # b = phi余り
        # Dd = path_diff

        # https://biotech-lab.org/articles/4907 R2値
        R2 = metrics.r2_score(phi, a * F + b)

        path_diff = 299792458/(2*np.pi*n_air)*a

        self.F, self.FFt, self.FFt_abs_amp,self.F2_abs_amp, self.F3_ifft_abs_amp, self.wrap, self.T_peak, self.F_ifft_abs_amp_filterd = F, FFt, FFt_abs_amp,F2_abs_amp, F3_ifft_abs_amp, wrap, T[peak], F3_ifft_abs_amp
        self.cutT, self.cutwidth, self.expnum = cutT, cutwidth, expnum
        self.phi, self.a, self.b, self.R2, self.n_air, self.path_diff = phi, a, b, R2, n_air, path_diff



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



CUT_T = 40e-12  # 大きいほど，o付近をつぶ(光源の影響)
CUT_T = 30e-12  # 大きいほど，o付近をつぶ(光源の影響)
CUT_T = 20e-12  # 大きいほど，o付近をつぶ(光源の影響)
CUT_T = 10e-12  # 大きいほど，o付近をつぶ(光源の影響)
# CUT_T = 5e-12  # 大きいほど，o付近をつぶ(光源の影響)

# CUT_T = 1e-12  # 大きいほど，o付近をつぶ(光源の影響)
# CUT_T = 0.5e-12  # 大きいほど，o付近をつぶ(光源の影響)
# CUT_T = 0.1e-12  # 大きいほど，o付近をつぶ(光源の影響)
# CUT_T = 0.05e-12  # 大きいほど，o付近をつぶ(光源の影響)
CUT_WIDTH = 10e-12
CUT_WIDTH = 40e-12
# CUT_WIDTH = 80e-12
CUT_WIDTH = 1e-12
CUT_WIDTH = 0.5e-12
# CUT_WIDTH = 4e-11

# CUT_WIDTH = 4e-12
CUT_WIDTH = 4e-13
CUT_WIDTH = 4e-14
CUT_WIDTH = 4e-15
# CUT_WIDTH = 4e-16

CUT_T =60e-12  # 大きいほど，o付近をつぶ(光源の影響)
CUT_WIDTH = 0.1e-12
# CUT_WIDTH = 60e-12
# CUT_WIDTH = 40e-12
# CUT_WIDTH = 30e-12
# CUT_WIDTH = 20e-12
# CUT_WIDTH = 10e-12
# CUT_WIDTH = 5e-12
# CUT_WIDTH = 1e-12
# CUT_WIDTH = 0.5e-12

# CUT_WIDTH = 0.1e-12




EXPNUM = 14  # 大きくしても結果はあまり変わらず．



# =============================================================================
# 指定フォルダからCSVを探索 BPF_method (AbsoluteDinstance()のインスタンス)の引数にする絶対パスリストpaths_raw_dataを得るプロセス
# =============================================================================


# folderpath = Dialog_Folder()
# filepaths = glob.glob(os.path.join(folderpath, "*.csv"),)
# https://qiita.com/amowwee/items/e63b3610ea750f7dba1b
print("CSVをまとめたxlsxを選択")
# matomexlsxpath = Dialog_File(caption="matome XLSXえらぶ") #TODO
matomexlsxpath = r"C:\Users\anonymous\Dropbox\DATAz-axis_try_4th\inter\CSV_matome.xlsx"

df_sort = pd.read_excel(matomexlsxpath,index_col = 0, sheet_name = "sort")

# faile nameをxlsxからListとして取得
# 辞書⇒Dataframe
# https://qiita.com/ShoheiKojima/items/30ee0925472b7b3e5d5c
names_rawdata = df_sort.loc["NAME",:].to_list()


# raw dataが入ったフォルダを指定
# path_folder_RawData = Dialog_Folder(caption="CSVのフォルダを選ぶ") #TODO
path_folder_RawData =r"C:\Users\anonymous\Dropbox\DATAz-axis_try_4th\inter\inter202110091444"
paths_raw_data = [os.path.join(
    path_folder_RawData, names_rawdata[i]+".csv") for i in range(len(names_rawdata))]
# paths_raw_data = df_sorteddata.loc["filepaths",:].to_list()
# =============================================================================
# 計算用インスタンスBPF_method と結果格納データフレームdf_resultParams df_phiの準備
# =============================================================================

BPF_method = AbsoluteDistance()

df_resultParams = dict(T_peak=[], a=[], b=[], R2=[],
                       path_diff=[], cutT=[], cutwidth=[], expnum=[],)

df_phi = pd.DataFrame()


# =============================================================================
# paths_raw_dataのデータを全て計算する
# =============================================================================

calc_len =len(paths_raw_data)
for i in range(calc_len):
# for i in range(200):


    """
    i番目のファイルパスにたいし，分析を行って行く．
    その後ほしいパラメータをdf_resultParamsに追加していく
    """
    filepath = paths_raw_data[i]

    BPF_method.path_difference(
        filepath, cutT=CUT_T, cutwidth=CUT_WIDTH, expnum=EXPNUM,)

    df_resultParams["T_peak"].append(BPF_method.T_peak)
    df_resultParams["a"].append(BPF_method.a)
    df_resultParams["b"].append(BPF_method.b)
    df_resultParams["R2"].append(BPF_method.R2)
    df_resultParams["path_diff"].append(BPF_method.path_diff)
    df_resultParams["cutT"].append(BPF_method.cutT)
    df_resultParams["cutwidth"].append(BPF_method.cutwidth)
    df_resultParams["expnum"].append(BPF_method.expnum)

    df_phi_i = pd.DataFrame({names_rawdata[i]: BPF_method.phi})
    df_phi = pd.concat([df_phi, df_phi_i], axis=1)
    # print("\r"+str(100*i/calc_len)+"%" , end="")
    print("\r"+str(100*i/calc_len), end="")



df_F_phi = pd.concat([pd.DataFrame({"F": BPF_method.F}), df_phi], axis=1)
# df_F_phi.set_index("F")

# https://note.nkmk.me/python-dict-clear-pop-popitem-del/
# phiは今回は抜く．　Noneを２nd argにすると，該当のKeyがなかった時に，Noneを返す


# =============================================================================
# データ保存手続き
# =============================================================================

# df_resultParamsをdf_resultParamsOptimized(Dataframe)に変換・成型
df_resultParamsOptimized = pd.DataFrame.from_dict(df_resultParams).T
df_resultParamsOptimized.columns = names_rawdata

filepathx = os.path.join(path_folder_RawData, "Ana"+"cutT"+str(CUT_T) +
                         "_"+"cutwidth"+str(CUT_WIDTH)+"_"+"expnum"+str(EXPNUM)+".xlsx")


df_resultParamsOptimized.to_excel(filepathx)
# df_F_phi.to_excel("f_phi.xlsx")

# JUMP_FREQ = 8 #OPTIMIZE tuning
# df_F_phi_dev8 = df_F_phi[::JUMP_FREQ]
# df_F_phi_dev8.to_excel("f_F_phi_dev8.xlsx")


# df_resultParams = dict(
#     T_peak=[1.2, 4.6],
#     phi =[np.arange(2,100), 12*np.arange(2,100)],
#     a=[1.2, 4.6],
#     b=[1.2, 4.6],
#     R2=[1.2, 4.6],
#     path_diff=[1.2, 4.6],
#     )

# =============================================================================
# 位置測定の結果をサンプリングナンバー順にプロット
# =============================================================================


plt.scatter(df_sort.loc["Posi_pls",:].astype(int),df_resultParams["path_diff"],s=2)

# plt.ylim(0.005, 0.01)
plt.title("cutT="+str(CUT_T) +
          "\n"+"cutwidth="+str(CUT_WIDTH)+"\n"+"expnum="+str(EXPNUM))
plt.xlim(-20000, 20000)
plt.ylim(0.035, 0.055)
# plt.ylim(0.0035,0.006)


"""



if(plot==1):


def OSAcsvfre_In(file): #fre-IのOSA信号(35行目から)をよむ

    wholedata = pd.read_csv(file, header=None, skiprows=34).values
    Fdata=np.flipud(wholedata[:,0].ravel())*10**12#wholedataの[:,0]を取り出した後，ravel1で次元配列へ直す
    Idata=np.flipud(wholedata[:,1].ravel())*10**-3

    return Fdata,

def Inter(x,y,expnum): #x,yは信号 expnumはデータ分割数＝2**expnum/ 補間後信号return xinter,yinter,分割数SampNum_inter,分割幅dx
    import sys
    from scipy import interpolate
    xinterstart=min(x)*(1+sys.float_info.epsilon)
    xinterend=max(x)*(1-sys.float_info.epsilon)
    SampNum_inter=2**expnum
    xinter=np.linspace(xinterstart, xinterend, SampNum_inter)
    SpFun=interpolate.interp1d(x,y,kind="cubic")
    yinter=SpFun(xinter)
    dx=(xinterend-xinterstart)/(SampNum_inter-1) #サンプリング間隔

    return xinter,yinter,SampNum_inter,dx


def FFT(x,y):
    N=len(x)
    FF=np.fft.fft(y)
    FF_abs=np.abs(FF)
    FF_abs_amp=FF_abs/(N/2)
    dx=np.abs((x[-1]-x[0])/(N-1))
    f=np.linspace(0,1.0/dx,N)

    return f,FF,FF_abs_amp
"""