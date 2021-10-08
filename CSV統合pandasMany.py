# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:59:42 2020

@author: wsxhi



それらを一つのＣＳＶ（ＸＬＳＸ）にまとめるためのこーど．
ただし，単位は直さない．(THz, mW(dbm)のまま)


("%06d" %sweep_num2)+"-"+str(datainfo["index2"])
                                                +"_" +datainfo["title2"]
                                                +"@"+str(datainfo["current_position_pulse"])+"pls" +".csv")

"""
import glob
import os
import pandas as pd
from icecream import ic
import re

# Out[23]: ['33', '4', '5', '6', '7', '1']

# 初期ディレクトリ取得


def Dialog_Folder(rootpath_init="C:/Users/anonymous/Dropbox"):
    """


    Args:
        rootpath_init (TYPE, optional): DESCRIPTION. Defaults to "C:/Users/anonymous/Dropbox".

    Returns:
        folderpath (TYPE): DESCRIPTION.

    """

    from PyQt5 import QtCore, QtGui, QtWidgets
    import sys
    # ディレクトリ選択ダイアログを表示
    rootpath = rootpath_init
    app = QtWidgets.QApplication(sys.argv)
    folderpath = QtWidgets.QFileDialog.getExistingDirectory(
        None, r"rootpath", rootpath)
    print(folderpath)
    return folderpath


def Dialog_File(rootpath_init="C:/Users/anonymous/Dropbox"):
    """
    Args:
        rootpath_init (str, optional): initial_directory_path. Defaults to r"C:/Users/anonymous/Dropbox".

    Returns:
        filepath (str): filepath you chose

    """
    from PyQt5 import QtCore, QtGui, QtWidgets
    import sys
    # 実行ディレクトリ取得D
    rootpath = rootpath_init
    app = QtWidgets.QApplication(sys.argv)
    # ディレクトリ選択ダイアログを表示
    filepath = QtWidgets.QFileDialog.getOpenFileName(
        None, r"rootpath", rootpath)
    print(filepath)
    return filepath

def OSAcsvlam_In(filepath):  # lam-IのOSA信号(35行目から)をよむ
    # wholedata = pd.read_csv(filepath, header=None, skiprows=34).values
    df_data = pd.read_csv(filepath, skiprows=3, header=None)
    return df_data

"""findallについて
https://niwakomablog.com/python-number-extract/#:~:text=%E4%BD%BF%E3%81%84%E6%96%B9&text=re.sub()%E3%81%AF%E3%80%81%E6%96%87%E5%AD%97,%E5%8F%96%E3%82%8A%E5%87%BA%E3%81%99%E3%81%93%E3%81%A8%E3%81%8C%E3%81%A7%E3%81%8D%E3%81%BE%E3%81%99%E3%80%82
"""


# =============================================================================
# まとめる対象のフォルダを選択し，ファイル名より情報をソートしたdf_sortframe_sortedを創る
# =============================================================================
folderpath = Dialog_Folder()

# ex)"filepath =  C:/Users/anonymous/Dropbox/pythoncode/OSAhappy/inter202109161816\OSA1_-10000pulseNo000301.csv
filepaths = glob.glob(os.path.join(folderpath, '*.csv'))
filelen = len(filepaths)  # CSVデータ数に対応
# 拡張子ありのファイル名　ex)filenameswithext=OSA1_-10000pulseNo000301.csv
filenameswithext = [os.path.split(filepaths[i])[-1] for i in range(filelen)]

# 拡張子なしのファイル名　ex)filenameswithext=OSA1_-10000pulseNo000301
filenames = [os.path.splitext(filenameswithext[i])[0] for i in range(filelen)]

"""文字列から最期の数字を抜き出す方法はこれが一番よい
https://techacademy.jp/magazine/22296
"""

regex_No = re.compile("^\d+\d+\d+\d+\d+\d")
list_No = [regex_No.findall(filenames[i])[0] for i in range(filelen)]

regex_subNo = re.compile("-+\w+_")
list_subNo = [regex_subNo.findall(filenames[i])[0][1:-1] for i in range(filelen)]

regex_title = re.compile("_\w+@")
list_title = [regex_title.findall(filenames[i])[0][1:-1] for i in range(filelen)]

sortlist = [filenames, filepaths, list_title, list_No]

df_sortframe = pd.DataFrame(sortlist, index=['NAME', "PATH", "TITLE", "No"])


df_sortframe.columns = [i for i in range(filelen)]  # colums番号を直す


# =============================================================================
# df_sortframeの順番に従い実際にcsvファイルの情報を取り込んだdf_wholedataを創る
# =============================================================================
df_index_freq = OSAcsvlam_In(df_sortframe.at["PATH", 0])[0]
df_intensity000001 = OSAcsvlam_In(df_sortframe.at["PATH", 0])[1]
df_intensities = df_intensity000001


for i in range(filelen-1):
    df_data_i = OSAcsvlam_In(df_sortframe.at["PATH", i+1])
    df_intensities = pd.concat([df_intensities, df_data_i[1]], axis=1)


df_wholedata = pd.concat([df_index_freq, df_intensities], axis=1)
df_wholedata.loc[28:] = df_wholedata.loc[28:].astype(float)
df_wholedata = df_wholedata.set_index(0)
df_wholedata.columns = df_sortframe.loc["NAME"].values.tolist()


# =============================================================================
# １つのＸＬＳＸフォルダにdf_wholedata, dfsortframe _sortedを保存する．df.sortframe_sortedは幾何光路差計算のときにファイル名を渡すために使用する
# =============================================================================
XLSXpath = os.path.join(folderpath, "CSV_matome.xlsx")
writer = pd.ExcelWriter(XLSXpath, engine="xlsxwriter",)
df_wholedata.to_excel(writer, sheet_name="wholedata",)
df_sortframe.to_excel(writer, sheet_name="sort",)
writer.save()
writer.close()


"""
# df_sortframe_sorted = df_sortframe.sort_values(by=["TITLE", "No", ], axis=1)
"""