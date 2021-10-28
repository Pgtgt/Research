
import numpy as np
# import matplotlib.pyplot as plt
import sys
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import os
from plotly.subplots import make_subplots
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from scipy.fftpack import fft, fftfreq
from scipy import interpolate
from PyQt5 import QtWidgets
import pandas as pd

EXP_NUM = 13
PAD_EXP = 4

app = QtWidgets.QApplication(sys.argv)


def Dialog_File(rootpath_init=r"C:"):
    """
    Args:
        rootpath_init (TYPE, optional): Initial Path. Defaults to r"C:".

    Returns:
        TYPE: Path of a File.

    """
    rootpath = rootpath_init

    # ディレクトリ選択ダイアログを表示-
    filepath = QtWidgets.QFileDialog.getOpenFileName(parent=None, caption="rootpath", directory=rootpath)

    # sys.exit(app_dialog_file.exec_())
    return filepath[0]


def Inter(x, y, expnum):
    """
    spline curve fitting
    Args:
        x (array of float(int)): X (unequal).
        y (array of float(int)): Y (unequal).
        expnum (int): Ninter = pow(2, expnum).

    Returns:
        xinter (array of float): X (interpolated).
        yinter (array of float): Y (interpolated).
        Ninter (int): Ninter = pow(2, expnum) = len(xinter) = len(yinter)
        dx (float): xinter interval (xinter[i+1] - xinter[i]).

    """
    xinterstart = min(x)*(1+sys.float_info.epsilon)
    xinterend = max(x)*(1-sys.float_info.epsilon)
    Ninter = 2**expnum
    xinter = np.linspace(xinterstart, xinterend, Ninter)

    SpFun = interpolate.interp1d(x, y, kind="cubic")
    yinter = SpFun(xinter)
    dx = (xinterend-xinterstart)/(Ninter-1)
    return xinter, yinter, Ninter, dx


def OSAcsvfre_In(file):  # fre-IのOSA信号(35行目から)をよむ　また，単位をTHz =>Hz, mW => Wへ修正

    wholedata = pd.read_csv(file, header=None, skiprows=34).values
    Fdata = np.flipud(wholedata[:, 0].ravel())*1e12  # 小さい周波数から格納
    Idata = np.flipud(wholedata[:, 1].ravel())*1e-3
    return Fdata, Idata


def FFT(x, y):
    N = len(x)
    FF = np.fft.fft(y)
    dx = np.abs((x[-1]-x[0])/(N-1))
    freq = fftfreq(len(FF), d=dx)

    freq = np.concatenate([freq[int(N/2):], freq[:int(N/2)]])
    FF = np.concatenate([FF[int(N/2):], FF[:int(N/2)]])

    FF_abs = np.abs(FF)
    FF_abs_amp = FF_abs/(N/2)

    return freq, FF, FF_abs_amp


def zero_padding(data, len_pad):
    pad = np.zeros(len_pad-len(data))
    data_pad = np.concatenate([data, pad])
    acf = (sum(np.abs(data)) / len(data)) / (sum(np.abs(data_pad)) / len(data_pad))
    return data_pad * acf


# =============================================================================
# データ読み込み
# 選択したOSAのCSVファイルの強度ー周波数部分のみ読み込み
# =============================================================================
path_file = Dialog_File()
Fdata, Idata = OSAcsvfre_In(path_file)

# =============================================================================
# データ処理．
#   1．スプライン補間 (データ数Ninter = pow(2, exp_num))
#   2．ハニング窓
#   3．ゼロパディング (データ数 len_pad = Ninter*pow(2, pad_exp))
#   4.FFT
# =============================================================================
"""1．スプライン補間"""
F_inter, I_inter, Ninter, dF = Inter(x=Fdata, y=Idata, expnum=EXP_NUM)
"""2．ハニング窓"""
hanning_win = np.hanning(Ninter)
acf_han = 1/(sum(hanning_win)/Ninter)  # FFT後の数値に掛ければOKの補正係数
I_han = I_inter * hanning_win

"""3．ゼロパディング (データ数 Ninter*pow(2, pad_exp))"""
len_pad = Ninter*pow(2, PAD_EXP)
I_han_pad = zero_padding(I_han, len_pad)
F_pad = np.linspace(F_inter[0], F_inter[0]+(len_pad)*dF, len_pad+1)[:-1]

"""4.FFT"""
T, FFt, FFt_abs_amp = FFT(F_pad, I_han_pad)


# =============================================================================
# Plot 該当CSVファイルがあったディレクトリにhtmlファイルにて保存
# =============================================================================
fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    go.Scatter(x=Fdata, y=Idata,    name=os.path.split(os.path.splitext(path_file)[0])[1]),

    row=1, col=1
)


fig.add_trace(
    go.Scatter(x=T, y=FFt_abs_amp, name=("expnum:%d  pad_exp:%d" % (EXP_NUM, PAD_EXP))),
    row=1, col=2
)

path_html = (os.path.splitext(path_file)[0])+".html"


plotly.offline.plot(fig,
                    auto_open=True,
                    )

fig.write_html(path_html)


# x_label,y_label,x_unit,y_unit = "Freq","Intensity","Hz","W"


"""

# os.chdir(os.path.dirname(os.path.abspath(__file__)))#ディレクトリ変更　本pyファイルがあるディレクトリを，実行ディレクトリとする．

#QtGui.QApplication.setGraphicsSystem('raster')
app = QtGui.QApplication([])
#mw = QtGui.QMainWindow()
#mw.resize(800,800)

win = pg.GraphicsWindow(title="fft")
# win2 = pg.GraphicsWindow(title="Basic plotting examples")
win.resize(1000,600)
win.setWindowTitle("OSA_FFT")
# Enable antialiasing for prettier plots
pg.setConfigOptions(antialias=True)


# p7 = win.addPlot(title=os.path.basename(file))
# p7.addLegend()
# p7.plot(Fdata,Idata , pen=(255,0,0,200),name =os.path.basename(file))
# x_min,x_max = min(T),max(T)

# x_label,y_label,x_unit,y_unit = "Freq","Intensity","Hz","W"
# p7.showGrid(x=True, y=True)
# p7.setLabel('left', y_label, units=y_unit)
# p7.setLabel('bottom', x_label, units=x_unit)




p8 = win.addPlot(title=os.path.basename(file))
p8.addLegend()
p8.plot(T,FF_abs_amp , pen=(255,0,0,200),name =os.path.basename(file))
x_min,x_max = min(T),max(T)
lr = pg.LinearRegionItem([x_min,x_max])
# lr.setZValue(-10)

# lr2.setZValue(-10)
p8.addItem(lr)
x_label,y_label,x_unit,y_unit = "Time","Amp","second","a.u."
p8.showGrid(x=True, y=True)
p8.setLabel('left', y_label, units=y_unit)
p8.setLabel('bottom', x_label, units=x_unit)

p9 = win.addPlot(title="Zoom on selected region")



p9.plot(T,FF_abs_amp , pen=(255,0,0,200), name =os.path.basename(file))
p9.addLegend()
def updatePlot():
    p9.setXRange(*lr.getRegion(), padding=0)
    # print(p9.getViewBox().viewRange()[0])
    #　￼￼Xrange￼[190.47543223919186, 190.6183659476708]　Yrange[-8.858067009681045e-06, 0.00020566566700968105]

# def updatePlot_2():
#     print(lr2.getRegion())
#     print(lr2.getRegion()[1]-lr2.getRegion()[0])# print(p9.getViewBox().viewRange()[0])
#     #　￼￼Xrange￼[190.47543223919186, 190.6183659476708]　Yrange[-8.858067009681045e-06, 0.00020566566700968105]

def updateRegion():
    lr.setRegion(p9.getViewBox().viewRange()[0])


lr.sigRegionChanged.connect(updatePlot)

# lr2.sigRegionChanged.connect(updatePlot_2)
p9.sigXRangeChanged.connect(updateRegion)
p9.showGrid(x=True, y=True)
x_label,y_label,x_unit,y_unit = "$Time$","Amp","second","a.u."
p9.setLabel('left', y_label, units=y_unit)
p9.setLabel('bottom', x_label, units=x_unit)
updatePlot()

## Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()

        """
