# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 20:59:42 2020

@author: wsxhi

あるフォルダ内のCSVファイル軍を１つのexcelフォルダにまとめる．
ただし，単位は直さない．(THz, mW(dbm)のまま)
また，CSVファイル名から各情報を読み取り，それも一緒にまとめる．

("%06d" %sweep_num2)+"-"+str(datainfo["index2"])
                                                +"_" +datainfo["title2"]
                                                +"@"+str(datainfo["current_position_pulse"])+"pls" +".csv")

!!! 単位はそのまま THz, mW
"""
import glob
import os
import pandas as pd
import re
import sys
from PyQt5 import QtWidgets


STAGE_RSN = 0.1e-6  # TODO m/pls ステージの分解能
# Out[23]: ['33', '4', '5', '6', '7', '1']

def Dialog_File(rootpath=r"C:\Users\wsxhi\Dropbox\DATAz-axis_try_5th", caption="choise"):
    """
    choose folder path by Explore

    Args:
        rootpath (str, optional): initial path of Explore. Defaults to r"C:".
        caption (str, optional): title of Explore. Defaults to "choise".

    Returns:
        folderpath (str): folder path.

    """
    app = QtWidgets.QApplication(sys.argv)
    # 実行ディレクトリ取得D
    # ディレクトリ選択ダイアログを表示-

    filepath = QtWidgets.QFileDialog.getOpenFileName(
        parent=None, caption=caption, directory=rootpath)
    del app
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


def OSAcsvlam_In(filepath):  # lam-IのOSA信号(35行目から)をよむ
    # wholedata = pd.read_csv(filepath, header=None, skiprows=34).values
    df_data = pd.read_csv(filepath, skiprows=3, header=None)
    return df_data


"""findallについて
https://niwakomablog.com/python-number-extract/#:~:text=%E4%BD%BF%E3%81%84%E6%96%B9&text=re.sub()%E3%81%AF%E3%80%81%E6%96%87%E5%AD%97,%E5%8F%96%E3%82%8A%E5%87%BA%E3%81%99%E3%81%93%E3%81%A8%E3%81%8C%E3%81%A7%E3%81%8D%E3%81%BE%E3%81%99%E3%80%82
"""


"""
STEP1 まとめる対象のフォルダを選択し，ファイル名より情報をソートしたdf_sortを創る
"""

folderpath = Dialog_Folder()
print(folderpath)

# ex)"filepath = "E:\uncer\inter\1st202201091602\000989-990_OSA1@-5000pls.csv"
filepaths = glob.glob(os.path.join(folderpath, '*.csv'))
filelen = len(filepaths)  # CSVデータ数に対応
# 拡張子ありのファイル名 "000989-990_OSA1@-5000pls.csv"
filenameswithext = [os.path.split(filepaths[i])[-1] for i in range(filelen)]

# 拡張子なしのファイル名 "000989-990_OSA1@-5000pls"
filenames = [os.path.splitext(filenameswithext[i])[0] for i in range(filelen)]

"""文字列から最期の数字を抜き出す方法はこれが一番よい
https://techacademy.jp/magazine/22296
"""

regex_No = re.compile("^\d+\d+\d+\d+\d+\d")
list_No = [int(regex_No.findall(filenames[i])[0]) for i in range(filelen)]

regex_subNo = re.compile("-+\w+_")
list_subNo = [int(regex_subNo.findall(filenames[i])[0][1:-1])
              for i in range(filelen)]

regex_title = re.compile("_\w+@")
list_title = [regex_title.findall(filenames[i])[0][1:-1]
              for i in range(filelen)]

regex_posi = re.compile("@"+".+"+"pls")
list_posi_pls = [int(regex_posi.findall(filenames[i])[0][1:-3])
                 for i in range(filelen)]

list_posi_m = [int(regex_posi.findall(filenames[i])[0][1:-3])*STAGE_RSN
               for i in range(filelen)]

sortlist = [filenames, list_title, list_No,
            list_subNo, list_posi_pls, list_posi_m]

df_sort = pd.DataFrame(
    sortlist, index=['NAME', "TITLE", "No", "subNo", "Posi_pls", "Posi_m"])

df_sort.columns = df_sort.loc["NAME"].values.tolist()
df_sort = df_sort.sort_index(axis="columns")

"""
STEP2 df_sortの順番に従い実際にcsvファイルの情報を取り込んだdf_wholedataを創る
"""
df_index_freq = OSAcsvlam_In(filepaths[0])[0]
df_intensity000001 = OSAcsvlam_In(filepaths[0])[1]
df_intensities = df_intensity000001


for i in range(filelen-1):
    df_data_i = OSAcsvlam_In(filepaths[i+1])
    df_intensities = pd.concat([df_intensities, df_data_i[1]], axis=1)


df_wholedata = pd.concat([df_index_freq, df_intensities], axis=1)
df_wholedata.loc[28:] = df_wholedata.loc[28:].astype(float)
df_wholedata = df_wholedata.set_index(0)
df_wholedata.columns = df_sort.loc["NAME"].values.tolist()
df_wholedata = df_wholedata.sort_index(axis="columns")

"""
STEP3 1つのxlsxフォルダにdf_wholedata, dfsortframe _sortedを保存する
"""
XLSXpath = os.path.join(os.path.dirname(folderpath), "CSV_matome_micro.xlsx")

with pd.ExcelWriter(XLSXpath,) as writer:
    df_wholedata.to_excel(writer, sheet_name="wholedata",)
    df_sort.to_excel(writer, sheet_name="sort",)
