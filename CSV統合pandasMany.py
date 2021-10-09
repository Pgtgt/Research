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
import sys

# Out[23]: ['33', '4', '5', '6', '7', '1']

# 初期ディレクトリ取得

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
    # 実行ディレクトリ取得D
    # ディレクトリ選択ダイアログを表示-
    filepath = QtWidgets.QFileDialog.getOpenFileName(
        parent=None, caption=caption, directory=rootpath)

    # sys.exit(app_dialog_file.exec_())
    return filepath[0]


def Dialog_Folder(rootpath=r"C:", caption="choise"):
    """
    choose file path by Explore

    Args:
        rootpath (str, optional): initial path of Explore. Defaults to r"C:".
        caption (str, optional): title of Explore. Defaults to "choise".

    Returns:
        filepath (str): file path.

    """
    from PyQt5 import QtWidgets
    # ディレクトリ選択ダイアログを表示
    folderpath = QtWidgets.QFileDialog.getExistingDirectory(
        parent=None, caption=caption, directory=rootpath)
    return folderpath


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
list_No = [ int(regex_No.findall(filenames[i])[0]) for i in range(filelen)]

regex_subNo = re.compile("-+\w+_")
list_subNo = [ int(regex_subNo.findall(filenames[i])[0][1:-1]) for i in range(filelen)]

regex_title = re.compile("_\w+@")
list_title = [regex_title.findall(filenames[i])[0][1:-1] for i in range(filelen)]

regex_posi = re.compile("@"+".+"+"pls")
list_posi = [int(regex_posi.findall(filenames[i])[0][1:-3]) for i in range(filelen)]

sortlist = [filenames, list_title, list_No, list_subNo, list_posi]

df_sortframe = pd.DataFrame(sortlist, index=['NAME', "TITLE", "No", "subNo", "Posi_pls"])


df_sortframe.columns = [i for i in range(filelen)]  # colums番号を直す


# =============================================================================
# df_sortframeの順番に従い実際にcsvファイルの情報を取り込んだdf_wholedataを創る
# =============================================================================
df_index_freq = OSAcsvlam_In(filepaths[0])[0]
df_intensity000001 = OSAcsvlam_In(filepaths[0])[1]
df_intensities = df_intensity000001


for i in range(filelen-1):
    df_data_i = OSAcsvlam_In(filepaths[i+1])
    df_intensities = pd.concat([df_intensities, df_data_i[1]], axis=1)


df_wholedata = pd.concat([df_index_freq, df_intensities], axis=1)
df_wholedata.loc[28:] = df_wholedata.loc[28:].astype(float)
df_wholedata = df_wholedata.set_index(0)
df_wholedata.columns = df_sortframe.loc["NAME"].values.tolist()


# =============================================================================
# １つのＸＬＳＸフォルダにdf_wholedata, dfsortframe _sortedを保存する．
# =============================================================================
XLSXpath = os.path.join( os.path.dirname(folderpath), "CSV_matome.xlsx")
writer = pd.ExcelWriter(XLSXpath, engine="xlsxwriter",)
df_wholedata.to_excel(writer, sheet_name="wholedata",)
df_sortframe.to_excel(writer, sheet_name="sort",)
writer.save()
writer.close()


"""
# df_sortframe_sorted = df_sortframe.sort_values(by=["TITLE", "No", ], axis=1)
"""