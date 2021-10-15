
import numpy as np
# import matplotlib.pyplot as plt
import sys
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import os
from plotly.subplots import make_subplots
import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from scipy.fftpack import fft, fftfreq
from scipy import interpolate

def Dialog_File(rootpath_init =r"C:"):
    """


    Args:
        rootpath_init (TYPE, optional): DESCRIPTION. Defaults to r"C:".

    Returns:
        TYPE: DESCRIPTION.

    """

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

def Inter(x,y,expnum):
    """


    Args:
        x (TYPE): DESCRIPTION.
        y (TYPE): DESCRIPTION.
        expnum (TYPE): DESCRIPTION.

    Returns:
        xinter (TYPE): DESCRIPTION.
        yinter (TYPE): DESCRIPTION.
        Ninter (TYPE): DESCRIPTION.
        dx (TYPE): DESCRIPTION.

    """
    #x,yは信号 expnumはデータ分割数＝2**expnum/ 補間後信号return xinter,yinter,分割数Ninter,分割幅dx


    xinterstart=min(x)*(1+sys.float_info.epsilon)
    xinterend=max(x)*(1-sys.float_info.epsilon)
    Ninter=2**expnum
    xinter=np.linspace(xinterstart, xinterend, Ninter)

    SpFun=interpolate.interp1d(x,y,kind="cubic")
    yinter=SpFun(xinter)
    dx=(xinterend-xinterstart)/(Ninter-1) #サンプリング間隔
    return xinter,yinter,Ninter,dx

def OSAcsvfre_In(file): #fre-IのOSA信号(35行目から)をよむ　また，単位をTHz =>Hz, mW => Wへ修正
    import pandas as pd
    wholedata = pd.read_csv(file, header=None, skiprows=34).values
    Fdata=np.flipud(wholedata[:,0].ravel())*1e12#小さい周波数から格納
    Idata=np.flipud(wholedata[:,1].ravel())*1e-3
    return Fdata,Idata

def FFT(x,y):
    N=len(x)
    FF=np.fft.fft(y)
    dx=np.abs((x[-1]-x[0])/(N-1))
    freq = fftfreq(len(FF), d =dx)

    freq=np.concatenate([freq[int(Ninter/2):],freq[:int(Ninter/2)]])
    FF=np.concatenate([FF[int(Ninter/2):],FF[:int(Ninter/2)]])

    FF_abs=np.abs(FF)
    FF_abs_amp=FF_abs/(N/2)

    return freq,FF,FF_abs_amp


path_file=Dialog_File()
# file="F:\研究\Data20191018\W0166.CSV"

# path_file=r"C:/Users/anonymous/Dropbox/pythoncode/OSAhappy/inter202109161816/OSA1_-50000pulse_No000588.csv"
# Read CSV
Fdata,Idata=OSAcsvfre_In(path_file)

#補間
EXPNUM=14
F,I,Ninter,dF=Inter(x=Fdata,y=Idata,expnum=EXPNUM)

#FFTと
T,FF,FF_abs_amp=FFT(F,I)



# h = int(len(T)/2)
fig = make_subplots(rows=1, cols=2)

fig.add_trace(
    go.Scatter(x=Fdata, y=Idata,    name =os.path.split(os.path.splitext(path_file)[0])[1] ),

    row=1, col=1
)


fig.add_trace(
    go.Scatter(x=T, y=FF_abs_amp),
    row =1, col =2
    )

path_html =(os.path.splitext(path_file)[0])+".html"


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