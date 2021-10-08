import time
import pandas as pd
import numpy as np
from scipy import interpolate
# from scipy.signal import butter, filtfilt, hilbert
import pyqtgraph as pg
import math
import sys
from scipy.signal import hilbert#,firwin,lfilter
import os

expnum=15
cutT =8
cutwidth=12
folder="F:\研究\Data20191205\TenU"
print("exp=",expnum,"\ncutT=",cutT,"\ncutwidth=",cutwidth,"\nPver_ignore_n_change"+folder)



#データ読み込み
def FilefromDia():
    import os, tkinter, tkinter.filedialog, tkinter.messagebox
    root = tkinter.Tk()
    root.withdraw()
    fTyp = [("","*")]
    iDir = os.path.abspath(os.path.dirname(__file__))
    file = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
    # 処理ファイル名の出力
    tkinter.messagebox.showinfo('○×プログラム',file)
    
    return file

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
    
def FFT(x,y):
    N=len(x)
    FF=np.fft.fft(y)
    FF_abs=np.abs(FF)
    FF_abs_amp=FF_abs/(N/2)    
    dx=np.abs((x[-1]-x[0])/(N-1))
    f=np.linspace(0,1.0/dx,N)
    return f,FF,FF_abs_amp

def traia_line2(F,wrap):#０－３．１４の三角波を直線に
    phi=np.empty(len(wrap))
    len_wrap=len(wrap)

    
    if (np.diff(wrap)[0]<0):#最初の傾きが負なら，ｙ＝pi/2でひっくり返してしまう
        wrap=-(wrap-(math.pi/2))+math.pi/2

    phi[0]=wrap[0]

    diff_wrap=np.diff(wrap)
    diff_2_wrap=[wrap[i+1]-wrap[i-1] for i in range(1,len_wrap-1)]
    # diff_2_wrap=np.array(diff_2_wrap)
    #山，谷それぞれ，ピークのi-1,i,i+1の挙動が2種類あるので分けて考える

    
    mountafterindex1 = [i+1 for i in range(1,len_wrap-2) \
              if ( ((diff_wrap[i-1]>=0)&(diff_wrap[i]<0))) & (diff_2_wrap[i]>=0)]
    mountafterindex2 = [i for i in range(1,len_wrap-2) \
              if ( ((diff_wrap[i-1]>=0)&(diff_wrap[i]<0))) & (diff_2_wrap[i]<0)]
    mountafterindex = mountafterindex1 + mountafterindex2  
    mountafterindex=sorted(mountafterindex) #小さいインデックスが上に
    valleyafterindex1 = [i+1 for i in range(1,len_wrap-2) \
              if ( ((diff_wrap[i-1]<0)&(diff_wrap[i]>=0))) & (diff_2_wrap[i]<=0)]
    valleyafterindex2 = [i for i in range(1,len_wrap-2) \
              if ( ((diff_wrap[i-1]<0)&(diff_wrap[i]>=0))) & (diff_2_wrap[i]>0)]
    valleyafterindex = valleyafterindex1 + valleyafterindex2
    valleyafterindex=sorted(valleyafterindex) 
    

    #最初の山に来る前はphi=wrap
    phi[:mountafterindex[0]]=wrap[:mountafterindex[0]]
    
    len_change = len(mountafterindex)#山のほうの切り替えた回数
    
    if (len(mountafterindex)==len(valleyafterindex)):#/\/\/

        #/右肩上がり部を一直線になるよう処理
        for _ in range(len_change-1):#/\/\まで処理
            phi[valleyafterindex[_]:mountafterindex[_+1]] \
            =wrap[valleyafterindex[_]:mountafterindex[_+1]]  +  2*(_+1)*math.pi
        phi[valleyafterindex[-1]:]\
            =  wrap[valleyafterindex[-1]:]+2*(len(valleyafterindex)) * math.pi#最後の/処理
        #\右肩下がり部を一直線になるよう処理
        for _ in range(len_change):#/\/\/まで一括処理
            phi[mountafterindex[_]:valleyafterindex[_]]\
            =-(wrap[mountafterindex[_]:valleyafterindex[_]]-math.pi)  +  (2*_+1)*math.pi
            
    elif (len(mountafterindex)!=len(valleyafterindex)):#/\/\
        #/右肩上がり部を一直線になるよう処理

        for _ in range(len_change-1):#/\/\まで処理
            phi[valleyafterindex[_]:mountafterindex[_+1]] \
            =wrap[valleyafterindex[_]:mountafterindex[_+1]]  +  2*(_+1)*math.pi
        #\右肩下がり部を一直線になるよう処理
        for _ in range(len_change-1):#/\/まで一括処理
            phi[mountafterindex[_]:valleyafterindex[_]]\
            =-(wrap[mountafterindex[_]:valleyafterindex[_]]-math.pi)  +  (2*_+1)*math.pi
        phi[mountafterindex[-1]:]\
        =-(wrap[mountafterindex[-1]:]-math.pi)+(2*len(valleyafterindex)+1)*math.pi#最後の処理
    
    return phi,mountafterindex,valleyafterindex

for filecapital in range(4):
    s=input()
    for i in range(10):
        file="F:\研究\Data20191205\wideakan000"+str(i)+".CSV"    
        
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
        # plt.plot(F,I_AM)
        # plt.show()
        analyticSignal = hilbert(I_AM)
        amplitudeEvelope = np.abs(analyticSignal)
        #https://dsp.stackexchange.com/questions/46291/why-is-scipy-implementation-of-hilbert-function-different-from-matlab-implemen
        I_noioff=I_AM+I_cons
        I_top=I_cons+amplitudeEvelope
        I_bottom=I_cons-amplitudeEvelope
        # plt.plot(F,I_top)
        # plt.plot(F,I_bottom)
        # plt.plot(F,I_noioff)
        # plt.show()
        bunshi=I_noioff*2-(I_top+I_bottom)
        bunbo=I_top-I_bottom
        cos_phi=bunshi/bunbo
        len_cos_phi=len(cos_phi)
        phi=np.empty(len_cos_phi)

        wrap =np.array([math.acos(cos_phi[_]) for _ in range(len_cos_phi)])



        phi,mountafterindex,valleyafterindex=traia_line2(F,wrap)

        a, b = np.polyfit(F, phi, 1)

        win = pg.GraphicsWindow(title=file)
        win.resize(1600,800)
        win.setWindowTitle(file)
        
        F_hanbetu=np.array([F[i] for i in range(len(F)) if phi[i]-(a*F[i]+b)>0])
        phi_hanbetu=np.array([phi[i] for i in range(len(F)) if phi[i]-(a*F[i]+b)>0])
        
        p3 = win.addPlot(title="phi"+file)
        p3.plot(F,phi, pen=(0,0,255))
        p3.plot(F,F*a+b, pen=(255,0,0))
        p3 = win.addPlot(title="Drawing with points")
        p3.plot(F_hanbetu,phi_hanbetu, pen=(200,200,200), symbolBrush=(255,0,0), symbolPen='w')

        p4 = win.addPlot(title="phi"+file)
        p4.plot(F,phi-(F*a+b), pen=(100,0,100))

        win.nextRow()
        p1 = win.addPlot(title="Raw"+file)
        p1.plot(F,I,pen=(255,255,0))
        
        p2 = win.addPlot(title="noioff,enve"+file)

        p2.plot(F,I_noioff, pen=(255,0,0), name="Red curve")
        p2.plot(F,I_bottom, pen=(0,255,0), name="Green curve")
        p2.plot(F,I_top, pen=(0,0,255), name="Blue curve")
        
        # s=input()

        print(299792458/(4*math.pi)*a*10**-12)
        s=input()
  



