import glob
import os
from pyexpat.errors import XML_ERROR_SYNTAX
import pandas as pd
import re
from scipy.stats import norm
import sys
from PyQt5 import QtWidgets
app = QtWidgets.QApplication(sys.argv)


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


# dir_excels = Dialog_Folder()
# paths_excel = glob.glob(os.path.join(dir_excels, '*.xlsx'))
# filelen = len(paths_excel)  # CSVデータ数に対応

dirs_excels=[
    # 'E:/uncer/1st/simu/0',
    'E:/uncer/1st/simu/1',
    'E:/uncer/1st/simu/2',
    'E:/uncer/1st/simu/3',
    'E:/uncer/1st/simu/4',
    'E:/uncer/1st/simu/5',
    'E:/uncer/1st/simu/6',
    'E:/uncer/1st/simu/7',
    'E:/uncer/1st/simu/8',
    'E:/uncer/1st/simu/9',
]

dir_excels = dirs_excels[0]
for dir_excels in dirs_excels:
    # * 指定したディレクトリないのxlsxファイル名をリストアップ
    paths_excel = glob.glob(os.path.join(dir_excels, '*.xlsx'))
    filelen = len(paths_excel)  #

    # * 各ファイルから対象データを取得し，つなげて1000000長リストに成形，ガウシアンフィッティング
    theta1000000=[]
    for path_excel in paths_excel:
        df_excel = pd.read_excel(path_excel,  index_col=0)
        theta10000 =df_excel.loc["theta_rad", :].tolist()
        theta1000000 = theta1000000 +theta10000

    # * 結果をDataFrameに格納し，各フォルダに格納
    mu, sigma = norm.fit(theta1000000)
    df_result = pd.DataFrame(index=["result"], columns =["mu", "sigma"])
    df_result.loc["result","mu"], df_result.loc["result","sigma"]=mu, sigma
    path_result = os.path.join(dir_excels,"musigma.xlsx")
    df_result.to_excel(path_result)

del app

