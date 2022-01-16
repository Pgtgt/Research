"""
スペクトルより得られたphiデータから，phiの確率分布係数(正規分布 mu, sigma)を推定
"""

import glob
import os
import pandas as pd
import re
import sys
from scipy.stats import norm
from PyQt5 import QtWidgets
import numpy as np
app = QtWidgets.QApplication(sys.argv)

# Out[23]: ['33', '4', '5', '6', '7', '1']

c=299792458
dict_LF=dict(ANA_FREQ_START=191.6e12, ANA_FREQ_END=191.883e12)
#%%

def Dialog_File(rootpath=r"C:\Users\wsxhi\Dropbox\DATAz-axis_try_5th", caption="choise"):
    """
    choose folder path by Explore

    Args:
        rootpath (str, optional): initial path of Explore. Defaults to r"C:".
        caption (str, optional): title of Explore. Defaults to "choise".

    Returns:
        folderpath (str): folder path.

    """

    # 実行ディレクトリ取得D
    # ディレクトリ選択ダイアログを表示-

    filepath = QtWidgets.QFileDialog.getOpenFileName(
        parent=None, caption=caption, directory=rootpath)

    # sys.exit(app_dialog_file.exec_())
    return filepath[0]


def Dialog_Folder(rootpath=r"C:\Users\wsxhi\Dropbox\DATAz-axis_try_5th", caption="choise"):
    """
    choose file path by Explore

    Args:
        rootpath (str, optional): initial path of Explore. Defaults to r"C:".
        caption (str, optional): title of Explore. Defaults to "choise".

    Returns:
        filepath (str): file path.

    """
    folderpath = QtWidgets.QFileDialog.getExistingDirectory(
        parent=None, caption=caption, directory=rootpath)
    return folderpath

#%%
# * step1 各周波数，トライアルにおけるphiデータ読みこみ．
path_phi = Dialog_File() # from dialog
df_phi = pd.read_excel(path_phi,index_col=0)
freqs = df_phi.index
# * step2 各周波数におけるmu, sigmaを推定
# * 使用データ:100/1000~999/1000，最初は温度差が大きいため．

df_params = pd.DataFrame(index=freqs,columns=["mu","sigma"])
for freq in freqs:
    phi_of_freq = df_phi.loc[freq,:][99:]
    
    mu, sigma = norm.fit(phi_of_freq)
    df_params.loc[freq, "mu"]=mu
    df_params.loc[freq, "sigma"]=sigma

# * step3 各周波数におけるmu, sigma情報を収めたデータフレームをExcelへ出力
dir = Dialog_Folder()
path_params = os.path.join(dir, "params.xlsx")
df_params.to_excel(path_params)

# %%
# * step4 各周波数におけるmu, sigma情報を再読み込み．
path_params = Dialog_File()
df_params = pd.read_excel(path_params,index_col=0)
freqs = df_params.index

CHAPTER_sim = [i for i in range(0,10)]
SECTION_sim = [i for i in range(0,100)]

for chapter in CHAPTER_sim:
    for section in SECTION_sim:
        # * step5 1e4回のsimulation trialを取得．
        # * phi発生および傾きa, OPDの計算，データ保存
        trials_sim = [i for i in range(0,10000)]
        df_phi_sim=pd.DataFrame(index=freqs, columns=trials_sim)

        for freq in freqs:
            """10000trialのphiデータフレームの作成"""
            phi_random_singlefreq = np.random.normal(loc = df_params.loc[freq, "mu"],
                                        scale = df_params.loc[freq, "sigma"],size=10000)
            df_phi_sim.loc[freq,:] = phi_random_singlefreq
            
        """各trialにおいて傾きa=dphi/df, OPD求める．ただし，dict_LF(Liner Fitting)の指定範囲以内に限るぢってぃんぐ"""
        ana_freq_start,ana_freq_end = dict_LF["ANA_FREQ_START"], dict_LF["ANA_FREQ_END"]
        limit = (ana_freq_start<freqs)&(freqs<ana_freq_end)    
        df_phi_sim_limit = df_phi_sim.loc[limit,:] # df_phi_simから，dict_LFの範囲のみ間引いたpd

        df_LFresult = pd.DataFrame(index=["a","b", "OPD"], columns=trials_sim)

        for trial in trials_sim:
            a, b = np.polyfit(df_phi_sim_limit.index.tolist(), 
                            df_phi_sim_limit.loc[:, trial].tolist(), 1)
            OPD = c/(2*np.pi)*a
            df_LFresult.loc["a",trial]=a
            df_LFresult.loc["b",trial]=b
            df_LFresult.loc["OPD",trial]=OPD

        # df_phi_sim.to_excel("phi_sim.xlsx")
        # df_phi_sim_limit.to_excel("phi_sim_limit.xlsx")
        df_LFresult.to_excel(str(chapter)+"-"+str(section)+"abOPD.xlsx")


# 途中でこまごま．乱数の種の確認が必要

del app
# %%
