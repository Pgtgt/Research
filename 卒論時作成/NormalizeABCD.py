import time
import pandas as pd
import numpy as np
from scipy import interpolate
import pyqtgraph as pg
import math
import sys
import os
from scipy import optimize

expnum=14
cutT =8
cutwidth=12
print("exp=",expnum,"\ncutT=",cutT,"\ncutwidth=",cutwidth,"\nPver_ignore_n_change")
A00B00C00D00=["\\A00","\\B00","\\C00","\\D00"]
folder="F:\研究\Data20191205\EightU"

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
    import pandas as pd
    wholedata = pd.read_csv(file, header=None, skiprows=34).values
    Fdata=np.flipud(wholedata[:,0].ravel())#小さい周波数から格納
    Idata=np.flipud(wholedata[:,1].ravel())
    return Fdata,Idata
    
def FFT(x,y): #return f, FF, FF_abs_amp
    N=len(x)
    FF=np.fft.fft(y)
    FF_abs=np.abs(FF)
    FF_abs_amp=FF_abs/(N/2)    
    dx=np.abs((x[-1]-x[0])/(N-1))
    f=np.linspace(0,1.0/dx,N)
    return f,FF,FF_abs_amp



for filecapital in range(4):
    
    # print(A00B00C00D00[filecapital])
    # s=input()
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

        #I_ideal=E^2(A^2+B^2)+E^2(2AB*cos)=I_cons+I_AM
        FF_cons=np.copy(FF)
        FF_AM=np.copy(FF)

        FF_cons[((cutT<T)&(T<max(T)-cutT))] = 0 
        FF_cons_ifft = np.fft.ifft(FF_cons) # IFFT
        I_cons = FF_cons_ifft.real # 実数部の取得
        T,FF_cons,FF_cons_abs_amp=FFT(F,I_cons)
        #2つのピークを探してAM信号を作る
        #まずconsにあたる部分を削る
        FF_AM[( (T<cutT)|(max(T)-cutT<T) )] = 0 
        FF_AM_abs = np.abs(FF_AM)
        FF_AM_abs_amp = FF_AM_abs / Ninter * 2 # 交流成分はデータ数で割って2倍

        peak1=np.argmax(FF_AM_abs_amp[:2**(expnum-1)])#前半ピーク
        peak2=np.argmax(FF_AM_abs_amp[2**(expnum-1):]) + 2**(expnum-1)#後半ピーク

        FF_AM[((T<T[peak1]-cutwidth/2) | ((T[peak1]+cutwidth/2<T) & (T<T[peak2]-cutwidth/2) ) |(T[peak2]+cutwidth/2<T))] = 0#AM信号成分だけにできた

        FF_AM_ifft = np.fft.ifft(FF_AM) # IFFT
        I_AM = FF_AM_ifft.real # 実数部の取得
        T,FF_AM,FF_AM_abs_amp=FFT(F,I_AM)

        #https://dsp.stackexchange.com/questions/46291/why-is-scipy-implementation-of-hilbert-function-different-from-matlab-implemen
        I_noioff=I_AM+I_cons

        I_normalised=I_noioff/I_cons
        
        
        # https://teratail.com/questions/13455
        def fit_func(parameter,x,y):
            a = parameter[0]
            b = parameter[1]
            T = parameter[2]
            c = parameter[3]
            phi=2*math.pi*x*T
            residual=np.empty(len(phi))
            for i in range(len(phi)):
                residual[i]=math.cos(phi[i])
            return residual

        _, _, FF_normalised_abs_amp =FFT(F,I_normalised)

        FF_normalised_abs_amp_edgecut=np.copy(FF_normalised_abs_amp)#FF_normalised_abs_ampのピークは，「定数によるピーク１＿COSのピーク１＿COSのピーク２」とならぶのでCOSピーク１だけ残す
        FF_normalised_abs_amp_edgecut[( (T<cutT)|(max(T)/2<T) )] = 0 
        a=np.argmax(FF_normalised_abs_amp_edgecut) #peak_of_normalisedsignal
        print(299792458*0.5*(T[a]*10**-12))

        # parameter0 = [np.mean(I_normalised),1.,T[a],0.]
        # result = optimize.leastsq(fit_func,parameter0,args=(F,I_normalised))
        # a_fit=result[0][0]
        # b_fit=result[0][1]
        # T_fit=result[0][2]
        # c_fit=result[0][3]

        # phi_fit=2*math.pi*F*T_fit+c_fit
        # I_normalised_fit=np.empty(len(phi_fit))
        # for i in range(len(phi_fit)):
            # I_normalised_fit[i]=a_fit+b_fit*math.cos(phi_fit[i])

        # print(299792458*0.5*(T_fit*10**-12))
