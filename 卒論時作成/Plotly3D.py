# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 22:39:07 2021

@author: wsxhi
"""
import os
import pandas as pd
import sys

import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import numpy as np
import matplotlib.cm as cm

def Dialog_File(rootpath_init =r"C:"):
    """
    引数:初期ディレクトリ
    戻り値：ファイルパス
    """
    from PyQt5 import QtWidgets
    # 実行ディレクトリ取得D
    rootpath =rootpath_init
    app_dialog_file = QtWidgets.QApplication(sys.argv)
    # ディレクトリ選択ダイアログを表示-
    filepath = QtWidgets.QFileDialog.getOpenFileName(parent = None, caption = "rootpath", directory = rootpath)

    # sys.exit(app_dialog_file.exec_())
    return filepath[0]

os.chdir(os.path.dirname(os.path.abspath(__file__)))#ディレクトリ変更
filepath = Dialog_File()
print("1")
wholedata = pd.read_csv(filepath, header=None, skiprows=29).values

d = pd.read_csv(filepath, header=None, skiprows=29).values
fdata=wholedata[:,0].ravel()*10**12
d = np.arange(0,970+1,10)
print("2")
# Idata = wholedata[:,1:]*10**-3
Idata = np.flipud(wholedata[:,1:]*10**-3)
F,D = np.meshgrid(fdata,d, )

chng_clr = plotly.colors.convert_to_RGB_255
plot = []
for num, (i, j, k) in enumerate(zip(F, D, Idata.T)):
    color = f"rgb{chng_clr(cm.jet(num / len(F)))}"
    d = go.Scatter3d(
        mode='lines',
        x=i, y=j, z=k,
        line=dict(color=color)
    )
    plot.append(d)
print("3")
layout = go.Layout(
    # template=template.plotly_layout(),
    title=dict(text="Ahiii", x=0.3, y=0.9,),
    scene=dict(
        xaxis=dict(
            title='Frequency [Hz]',
            # range=[-5, 5],
            ticks='outside', tickwidth=2,
        ),
        yaxis=dict(
            title='distance [um]',
            # range=[-5, 5],
            ticks='outside', tickwidth=2,
        ),
        zaxis=dict(
            title='Intensity [W]',
            # range=[0, 50],
            ticks='outside', tickwidth=2,
        ),
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    legend=dict(
        title='symbol | color',
        x=1, y=0.9,
        itemsizing='constant',
    ),
)
print("4")

fig = go.Figure(data=plot, layout=layout)

# fig = go.Figure(data=plot)
fig.update_layout(scene_aspectmode = "manual",
                  scene_aspectratio = dict(x=1,y=2,z=1))
# plotly.offline.iplot(fig)
plotly.offline.plot(fig)
print("5")
# 作成したグラフを保存
# pio.orca.config.executable = (
#     '/Applications/orca.app/Contents/MacOS/orca'
# )
pio.write_image(fig, "try_light_+1.png")
pio.write_html(
    fig,
    "try_light_+1.html", auto_open=True,
    # config=template.plotly_config(tf=True),
)