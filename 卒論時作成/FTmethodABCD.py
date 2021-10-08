import math
import sys
import os
import pyqtgraph as pg
import numpy as np
from scipy import interpolate
import pandas as pd

expnum=17
cutT =8
cutwidth=10
folder="E:\研究\DATA\Data20191205\OneU"
print("exp=",expnum,"\ncutT=",cutT,"\ncutwidth=",cutwidth,"\nFTver_ignore_n_change"+folder)
A00B00C00D00=["\\A00","\\B00","\\C00","\\D00"]

result=np.array()

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
    Fdata=np.flipud(wholedata[:,0].ravel())#小さい周波数から格納
    Idata=np.flipud(wholedata[:,1].ravel())
    return Fdata,Idata

def FFT(x,y):
    N=len(x)
    FF=np.fft.fft(y)
    FF_abs=np.abs(FF)
    FF_abs_amp=FF_abs/(N/2)    
    dx=np.abs((x[-1]-x[0])/(N-1))
    f=np.linspace(0,1.0/dx,N)
    
    return f,FF,FF_abs_amp

def wrappedphase(e):#配列eの位相を折り返しで求める-1.5~1.5

    wrap=np.empty(len(e))
    for _ in range(len(e)):
        wrap[_]=math.atan(e.imag[_]/e.real[_])       
    
    return wrap


for filecapital in range(4):
    
    print(A00B00C00D00[filecapital])

    for i in range(20):
        file=folder+A00B00C00D00[filecapital]+str(i+10)+".CSV"    
        
        if (os.path.exists(file)==False):
            break
        
        # Read CSV
        Fdata,Idata=OSAcsvfre_In(file)

        #補間
        F,I,Ninter,dF=Inter(Fdata,Idata,expnum)

        #FFTと
        T,FF,FF_abs_amp=FFT(F,I)


        delta_T=(1.0/dF)/(Ninter-1)
        
        F2 = np.copy(FF) 

        # カットオフ（周波数）２ｍｍ以上の部分で計測をお勧めする
        F2[(T <cutT )] = 0 # カットオフ未満周波数のデータをゼロにする，光源の影響排除
        F2[(T >T[2**(expnum-1)] )] = 0 #1/2後半の(負の)周波数帯をカット

        F2_abs = np.abs(F2)
        # 振幅をもとの信号に揃える
        F2_abs_amp = F2_abs / Ninter * 2 # 交流成分はデータ数で割って2倍
        print(98)
        F3 = np.copy(F2)
        peak=np.argmax(F2_abs_amp)

        F3[((T<T[peak]-cutwidth/2)|(T[peak]+cutwidth/2<T))] = 0 #1/2後半の(負の)周波数帯をカット
        #IFFT
        F3_ifft = np.fft.ifft(F3) # IFFT
        F3_ifft_abs = np.abs(F3_ifft)
        F3_ifft_abs_amp = F3_ifft_abs / Ninter * 2
        g_real = F3_ifft.real # 実数部の取得

        g_imag = F3_ifft.imag # 虚部の取得

        wrap=wrappedphase(F3_ifft)

        wrap_abs=np.abs(wrap)

        phi=unwrap(wrap)
        # グラフ（オリジナルとフィルタリングを比較）

        a, b = np.polyfit(F, phi, 1)
        # sentaku=11
        # if(sentaku==1):
        #     win = pg.GraphicsWindow(title=file)
        #     win.resize(1600,800)
        #     win.setWindowTitle(file)
            
        #     p4 = win.addPlot(title="phi_zenhan"+file)
        #     p4.plot(F[:1000],phi[:1000], pen=(0,0,255))
        #     p4.plot(F[:1000],F[:1000]*a+b, pen=(255,0,0))
        #     win.nextRow()
        #     p5 = win.addPlot(title="phi_kouhan"+file)
        #     p5.plot(F[Ninter-1000:],phi[Ninter-1000:], pen=(0,0,255))
        #     p5.plot(F[Ninter-1000:],F[Ninter-1000:]*a+b, pen=(255,0,0))
        #     win.nextRow()
        #     p3 = win.addPlot(title="phi"+file)
        #     p3.plot(F,phi, pen=(0,0,255))
        #     p3.plot(F,F*a+b, pen=(255,0,0))
        
        
      
        print(299792458/(4*math.pi)*a*10**-12)

  