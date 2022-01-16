"""
スペクトルより得られたfcenterデータから,fcenterの確率分布係数(正規分布 mu, sigma)を推定
"""

import glob
import os
import pandas as pd
import re
import sys
from scipy.stats import norm
from PyQt5 import QtWidgets
import numpy as np
import collections
import ref_index
app = QtWidgets.QApplication(sys.argv)

dict_nparam = dict(wave =  (1554.134049+1563.862587)/2,
               t=27, p=101325,rh=70)
n_air = ref_index.edlen(wave=dict_nparam["wave"], t=dict_nparam["t"], p=dict_nparam["p"], rh=dict_nparam["rh"])

# Grating info
# https://www.thorlabs.co.jp/thorproduct.cfm?partnumber=GR13-0616
gr_per_mm = 600  # grating pitch /mm
mm_per_gr = 1 / gr_per_mm
m_per_gr = mm_per_gr * 1e-3

M = 1
c = 299792458
# Out[23]: ['33', '4', '5', '6', '7', '1']
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
# * step1 各位置，トライアルにおけるfcenterデータ読みこみ．
# * 31posi * 201trial とってしまった．200trialのはずだった．よって，最期の31*1データは切る．
"""データの取得と並べ替え．"""
path_fcenter = Dialog_File() # from dialog
df_fcens = pd.read_excel(path_fcenter,index_col=0)
df_fcens = df_fcens.iloc[:,:-31] #31*1のカット
df_fcens = df_fcens.loc[("Posi_m","mu:f_center"), :]# fcen, posi情報のみ取得
df_fcens = df_fcens.sort_values(by=["Posi_m"], axis=1)


posi_ms = collections.Counter(df_fcens.loc["Posi_m",:]) #重複無しで取得 https://note.nkmk.me/python-collections-counter/
trials = [i for i in range(0,200)]
"""indexがposi_m,columnがtrialとなるようにdfを再配列"""
df_fcens_pt = pd.DataFrame(index=posi_ms,columns=trials)
for i in range( len(posi_ms)):
    df_fcens_pt.iloc[i,:]=df_fcens.iloc[1,200*i:200*(1+i)]


# * step2 各posiにおけるmu, sigmaを推定

df_params_fcens = pd.DataFrame(index=posi_ms,columns=["mu","sigma"])

for posi_m in posi_ms:
    
    fcens_singleposi = df_fcens_pt.loc[posi_m, :].tolist()
    
    mu, sigma = norm.fit(fcens_singleposi)
    df_params_fcens.loc[posi_m, "mu"]=mu
    df_params_fcens.loc[posi_m, "sigma"]=sigma

# * step3 各posiにおけるmu, sigma情報を収めたデータフレームをExcelへ出力
dir = Dialog_Folder()
path_params_fcens = os.path.join(dir, "params_fcenters.xlsx")
df_params_fcens.to_excel(path_params_fcens)

# %%
# * step4 各周波数におけるmu, sigma情報を再読み込み．
path_params_fcens = Dialog_File()
df_params_fcens = pd.read_excel(path_params_fcens,index_col=0)
posi_ms = df_params_fcens.index



"""
信頼区間は-0.0013 ~0.0004とする．
１回目のTrialの結果から決定した．
"""
dict_trust = dict(start = -0.0013,end = 0.0004 )
trials_sim = [i for i in range(0,10000)]
df_fcens_sim=pd.DataFrame(index=posi_ms, columns=trials_sim)
# for posi_m in posi_ms:
#     """10000trialのfcenデータフレームの作成"""
#     fcen_random_singleposi = np.random.normal(loc = df_params_fcens.loc[posi_m, "mu"],
#                                 scale = df_params_fcens.loc[posi_m, "sigma"],size=10000)
#     df_fcens_sim.loc[posi_m,:] = fcen_random_singleposi
# df_fcens_sim.to_excel("fcens_sim.xlsx")


CHAPTER_sim = [i for i in range(0,10)]
SECTION_sim = [i for i in range(0,100)]

for chapter in CHAPTER_sim:
    for section in SECTION_sim:
        # * step5 1e4回のsimulation trialを取得．
        # * f_centers発生および平均f_center, thetaの計算，データ保存
        trials_sim = [i for i in range(0,10000)]
        df_fcens_sim=pd.DataFrame(index=posi_ms, columns=trials_sim)
        df_fcens_theta_result=pd.DataFrame(index=["f_center.mean","theta_rad","theta_degees"],
                                  columns=trials_sim)

        for posi_m in posi_ms:
            """10000trialのfcenデータフレームの作成"""
            fcen_random_singleposi = np.random.normal(loc = df_params_fcens.loc[posi_m, "mu"],
                                        scale = df_params_fcens.loc[posi_m, "sigma"],size=10000)
            df_fcens_sim.loc[posi_m,:] = fcen_random_singleposi
        
        """各trialにおいて平均fcenter, theta求める．"""
        fcens_mean =df_fcens_sim.loc[ dict_trust["start"]:dict_trust["end"],:].mean()    
        theta_rad = np.arcsin(M*c/(n_air*fcens_mean*m_per_gr) )
        theta_deg = np.degrees(theta_rad)
        df_fcens_theta_result.loc["f_center",:]=fcens_mean
        df_fcens_theta_result.loc["theta_rad",:]=theta_rad
        df_fcens_theta_result.loc["tehta_degrees",:]=theta_deg

        df_fcens_theta_result.to_excel(str(chapter)+"-"+str(section)+"theta_result.xlsx")



del app
# %%
