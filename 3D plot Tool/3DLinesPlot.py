# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 22:39:07 2021

@author: wsxhi

指定様式で書かれたexlsファイルを読み込み，3d lie plot を作成．exlsと同フォルダに結果を格納するツール
"""
from re import T
import pandas as pd
import sys
import os
import plotly
import plotly.graph_objects as go
# import plotly.express as px
import plotly.io as pio
import numpy as np
import matplotlib.cm as cm
from PyQt5 import QtWidgets



def Dialog_File(rootpath=r"C:", caption="choise"):
    """
    引数:初期ディレクトリ
    戻り値:ファイルパス
    """
    app = QtWidgets.QApplication(sys.argv)
    # 実行ディレクトリ取得D
    
    # ディレクトリ選択ダイアログを表示-
    filepath = QtWidgets.QFileDialog.getOpenFileName(
        parent=None, caption=caption, directory=rootpath)
    del app
    # sys.exit(app_dialog_file.exec_())
    return filepath[0]


"""
STEP1 データ読み込み
"""
# os.chdir(os.path.dirname(os.path.abspath(__file__)))  # ディレクトリ変更
filepath = Dialog_File(caption="choose xlsx")

df_wholedata = pd.read_excel(filepath,header = 0,index_col=0, sheet_name="data")
df_cap_color= pd.read_excel(filepath,header = 0,index_col=0, sheet_name="property")

y = df_wholedata.columns.values
x = (df_wholedata.index).values

Z = (df_wholedata.values).T
X,Y =np.meshgrid(x, y, )

NAME = ["%d" % i for i in range(len(y))]

Ax_x=df_cap_color.loc["name","X"]
Ax_y=df_cap_color.loc["name","Y"]
Ax_z=df_cap_color.loc["name","Z"]
type_grad=df_cap_color.loc["name","color"]
chng_clr = plotly.colors.convert_to_RGB_255
plot = []
"""
STEP2 プロット作成
"""
for num, (i, j, k) in enumerate(zip(X, Y, Z)):
    grad=cm.get_cmap(type_grad)
    color = f"rgb{chng_clr(grad(num / len(Y)))}"
    # color = f"rgb{chng_clr(cm.gnuplot(num / len(D)))}"

    p = go.Scatter3d(
        mode='lines',
        x=i, y=j, z=k,
        line=dict(color=color),
        name=NAME[num],

    )
    plot.append(p)
# print("3")
layout = go.Layout(
    # template=template.plotly_layout(),
    title=dict(text="Spectrum", x=0.3, y=0.9,),
    scene=dict(
        xaxis=dict(
            title=Ax_x,
            # range=[187e12, 190e12],
            ticks='outside', tickwidth=2,
        ),
        yaxis=dict(
            title=Ax_y,
            ticks='outside', tickwidth=2,
        ),
        zaxis=dict(
            title=Ax_z,
            ticks='outside', tickwidth=2,
        ),
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    legend=dict(
        title='spectrums',
        x=1, y=0.9,
        itemsizing='constant',
    ),
)

fig = go.Figure(data=plot, layout=layout)


fig.update_layout(scene_aspectmode="manual",
                  scene_aspectratio=dict(x=1, y=2, z=1))
"""
STEP3 プロット出力
"""
pio.write_html(
    fig,
    os.path.join(os.path.split(filepath)[0], "matome3D.html"),

    auto_open=True,
)

# fig.show(
#     engine="kaleido",)

