#------------旧式バンドパス法------
import pandas as pd
import numpy as np
import os
import math
import sys
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import copy
from icecream import ic

expnum=16
cutT =10e-12
cutwidth=1e-12


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


def Dialog_Folder(rootpath_init =r"C:"):
    """
    引数:初期ディレクトリ
    戻り値：フォルダ(ディレクトリ)パス
    """
    from PyQt5 import QtCore, QtGui, QtWidgets
    import sys
    # ディレクトリ選択ダイアログを表示
    rootpath = rootpath_init
    app = QtWidgets.QApplication(sys.argv)
    folderpath = QtWidgets.QFileDialog.getExistingDirectory(None, r"rootpath", rootpath)
    print(folderpath)
    return folderpath

def unwrap(wrap):#-1.5~1.5で折り返される位相を展開か
    phi=np.empty(len(wrap))
    n=0
    phi[0]=wrap[0]
    for i in range(1,len(wrap)):
        if((wrap[i] - wrap[i-1])<-2):
            n=n+1
        phi[i]=wrap[i]+(n*math.pi)

    return phi

def Inter(x,y,expnum): #x,yは信号 expnumはデータ分割数＝2**expnum/ 補間後信号return xinter,yinter,分割数Ninter,分割幅dx
    import sys
    from scipy import interpolate
    xinterstart=min(x)*(1+sys.float_info.epsilon)
    xinterend=max(x)*(1-sys.float_info.epsilon)
    Ninter=2**expnum
    xinter=np.linspace(xinterstart, xinterend, Ninter)
    SpFun=interpolate.interp1d(x,y,kind="cubic")
    yinter=SpFun(xinter)
    dx=(xinterend-xinterstart)/(Ninter-1) #サンプリング間隔

    return xinter,yinter,Ninter,dx

def OSAcsvfre_In(file): #fre-IのOSA信号(35行目から)をよむ

    wholedata = pd.read_csv(file, header=None, skiprows=34).values
    Fdata=np.flipud(wholedata[:,0].ravel())*10**12#wholedataの[:,0]を取り出した後，ravel1で次元配列へ直す
    Idata=np.flipud(wholedata[:,1].ravel())*10**-3

    return Fdata,Idata

def FFT(x,y):
    N=len(x)
    FFt=np.fft.fft(y)
    FFt_abs=np.abs(FFt)
    FFt_abs_amp=FFt_abs/(N/2)
    dx=np.abs((x[-1]-x[0])/(N-1))
    f=np.linspace(0,1.0/dx,N)

    return f,FFt,FFt_abs_amp

def wrappedphase(e):#配列eの位相を折り返しで求める-1.5~1.5

    wrap=np.empty(len(e))
    for _ in range(len(e)):
        wrap[_]=math.atan(e.imag[_]/e.real[_])

    return wrap


file=Dialog_File()

# Read CSV
F_uneq,I_uneq=OSAcsvfre_In(file)

#補間 I(f_uneq) => I(f_euneq)
F,I,Ninter,dF=Inter(F_uneq,I_uneq,expnum)

#FFT  I(f) C_1 + C_2*cos(phi(f) )     ====>    FFt(T)=C_1 + C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))
T,FFt,FFt_abs_amp=FFT(F,I)

delta_T=(1.0/dF)/(Ninter-1)

# F2 = np.copy(FF)


# ----------------------------------------------------------------------------
# C_1 + C_2*cos(phi) =>C_2/2 *exp(j*phi) => phi = a *F +b =>a
# ----------------------------------------------------------------------------
# Filtering   FFt(T)=C_1 + C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))  ====> F2(T)=C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T))
F2=copy.deepcopy(FFt)
F2[(T <cutT )] = 0 # カットオフ未満周波数のデータをゼロにする，光源の影響排除
F2[(T >T[2**(expnum-1)] )] = 0 #1/2後半の(負の)周波数帯をカット

F2_abs = np.abs(F2)

# 振幅をもとの信号に揃える
F2_abs_amp = F2_abs / Ninter * 2 # 交流成分はデータ数で割って2倍

# Filtering   F2(T)=C_2/2 exp(j*phi(T) ) + C_2/2 exp(-j*phi(T)) ====>  F3(T)=C_2/2 exp(j*phi(T) )
F3=copy.deepcopy(F2)

peak=np.argmax(F2_abs_amp)
F3[((T<T[peak]-cutwidth/2)|(T[peak]+cutwidth/2<T))] = 0 #所望のピークだけのこす

#IFFT   F3(T)=C_2/2 exp(j*phi(T) )  ====>  I(f)=C_2/2 exp(j*phi(f) )
F3_ifft = np.fft.ifft(F3) # IFFT
F3_ifft_abs = np.abs(F3_ifft)
F3_ifft_abs_amp = F3_ifft_abs / Ninter * 2
"""
wrap

pi/2
  /l  /l  /l  /l  /l            1
 / l / l / l / l / l            1
/  l/  l/  l/  l/  l            1
-pi/2

phi = unwrap(wrap) (liner)
     /
    /
   /
  /
 /
/

"""
wrap=wrappedphase(F3_ifft)
# wrap=F3_ifft
wrap_abs=np.abs(wrap)

phi=unwrap(wrap)

a, b = np.polyfit(F, phi, 1) #phi = a *F + bの1じ多項式近似
# a =2 pi Dd n / c
# b = phi余り
# Dd = path_diff

# https://biotech-lab.org/articles/4907
import sklearn.metrics as metrics
R2 = metrics.r2_score(phi,a *F + b )

plot=1
if(plot==1):
    win = pg.GraphicsWindow(title=file)
    win.resize(1600/1.5,800/1.5)
    win.setWindowTitle(file)

    p2 = win.addPlot(title='$I\left(f\right)=C_1\left(f\right)+C_2\left(f\right)\cos{\left\{2\pi\frac{\Delta d\ N}{c}f\right\}}$')
    p2.plot(F,I)


    p3 = win.addPlot(title="phi"+file)

    x_label,y_label,x_unit,y_unit = "Freq","Intensity","Hz","W"
    p3.showGrid(x=True, y=True)
    p3.setLabel('left', y_label, units=y_unit)
    p3.setLabel('bottom', x_label, units=x_unit)
    p3.plot(F,wrap, pen=(0,0,255))
    p3.plot(F,F*a+b, pen=(100,0,0))

n_air=1
path_diff=299792458/(2*math.pi*n_air)*a
ic(R2,cutT, cutwidth, a ,path_diff)

# Start Qt event loop unless running in interactive mode or using pyside.
# if __name__ == '__main__':

#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtGui.QApplication.instance().exec_()


# fld =Dialog_Folder()
# print("outputfilename (Do not include path) like 「a.csv」")
# name=input()
# fileout=fld+"\\"+name
# # data=np.c_[T,FF_abs_amp]
# data=np.c_[F,wrap]
# np.savetxt(fileout,data,delimiter=",")


"""
dir = 'F:\研究' #このディレクトリは．初期値なのでどこでもいい．CでもDでも
fld = filedialog.askdirectory(initialdir = dir)
print("outputfilename (Do not include path) like 「a.csv」")
name=input()
fileout=fld+"\\"+name
data=np.c_[F,I]
np.savetxt(fileout,data,delimiter=",")


dir = 'F:\研究'
fld = filedialog.askdirectory(initialdir = dir)
print("outputfilename (Do not include path) like 「a.csv」")
name=input()
fileout=fld+"\\"+name
data=np.c_[F,phi,a*F+b,phi-(a*F+b)]
np.savetxt(fileout,data,delimiter=",")


p4 = win.addPlot(title="phi_zenhan"+file)
p4.plot(F[:1000],phi[:1000], pen=(0,0,255))
p4.plot(F[:1000],F[:1000]*a+b, pen=(100,0,0))
win.nextRow()
p5 = win.addPlot(title="phi_kouhan"+file)
p5.plot(F[Ninter-1000:],phi[Ninter-1000:], pen=(0,0,255))
p5.plot(F[Ninter-1000:],F[Ninter-1000:]*a+b, pen=(100,0,0))
win.nextRow()
"""