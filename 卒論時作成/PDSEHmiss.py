
import pandas as pd
import numpy as np
import os, tkinter, tkinter.filedialog, tkinter.messagebox
from scipy import interpolate
# from scipy.signal import butter, filtfilt, hilbert
import matplotlib.pyplot as plt
import math
import sys
from scipy.signal import hilbert#,firwin,lfilter
import time 
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


def traia_linekai(F,wrap):#三角波を直線に
    
    len_wrap=len(wrap)
    phi=[0]*len_wrap
    
    if (np.diff(wrap)[0]<0):#最初の傾きが負なら，ｙ＝pi/2でひっくり返してしまう
        wrap=-(wrap-(math.pi/2))
    phi[0]=wrap[0]
    
    #山，谷それぞれ，ピークのi-1,i,i+1の挙動が2種類あるので分けて考える
    mountafterindex1 = [i+1 for i in range(len_wrap-1) \
              if ( ((np.diff(wrap)[i-1]>=0)&(np.diff(wrap)[i]<0))) & (wrap[i-1]<=wrap[i+1])]
    mountafterindex2 = [i for i in range(len_wrap-1) \
              if ( ((np.diff(wrap)[i-1]>=0)&(np.diff(wrap)[i]<0))) & (wrap[i-1]>wrap[i+1])]
    mountafterindex = mountafterindex1 + mountafterindex2  
    mountafterindex=sorted(mountafterindex) #小さいインデックスが上に
    valleyafterindex1 = [i+1 for i in range(len_wrap-1) \
              if ( ((np.diff(wrap)[i-1]<0)&(np.diff(wrap)[i]>=0))) & (wrap[i-1]>=wrap[i+1])]
    valleyafterindex2 = [i for i in range(len_wrap-1) \
              if ( ((np.diff(wrap)[i-1]<0)&(np.diff(wrap)[i]>=0))) & (wrap[i-1]<wrap[i+1])]
    valleyafterindex = valleyafterindex1 + valleyafterindex2
    valleyafterindex=sorted(valleyafterindex) 
    
    
    #最初の山に来る前はphi=wrap
    phi[:mountafterindex[0]]=wrap[:mountafterindex[0]]
    
    len_change = len(mountafterindex)#山のほうの切り替えた回数
    # wrap =np.array([math.acos(cos_phi[_]) for _ in range(len_cos_phi)])
    if (len(mountafterindex)==len(valleyafterindex)):#/\/\/パターン
        #/右肩上がり部を一直線になるよう処理
        phi[valleyafterindex[_]:mountafterindex[_+1]] \
        =[wrap[valleyafterindex[_]:mountafterindex[_+1]]  +  2*(_+1)*math.pi for _ in range(len_change-1)]
        #最後の/処理
        phi[valleyafterindex[-1]:]\
        =  wrap[valleyafterindex[-1]:]+2*(len(valleyafterindex)) * math.pi
        
        #\右肩下がり部を一直線になるよう処理  #/\/\/まで一括処理
        phi[mountafterindex[_]:valleyafterindex[_]]\
        =[-(wrap[mountafterindex[_]:valleyafterindex[_]]-math.pi)  +  (2*_+1)*math.pi for _ in range(len_change)]
      
    elif (len(mountafterindex)!=len(valleyafterindex)):#/\/\パターン
        #/右肩上がり部を一直線になるよう処理 #/\/\まで一括処理
        phi[valleyafterindex[_]:mountafterindex[_+1]] \
        =[wrap[valleyafterindex[_]:mountafterindex[_+1]]  +  2*(_+1)*math.pi for _ in range(len_change-1)]
        
        #\右肩下がり部を一直線になるよう処理
        phi[mountafterindex[_]:valleyafterindex[_]]\
        =[-(wrap[mountafterindex[_]:valleyafterindex[_]]-math.pi)  +  (2*_+1)*math.pi for _ in range(len_change-1)]

        #最後の\処理            
        phi[mountafterindex[-1]:]\
        =-(wrap[mountafterindex[-1]:]-math.pi)+(2*len(valleyafterindex)+1)*math.pi
    
    phi=np.array(phi)
    return phi,mountafterindex,valleyafterindex



# file=FilefromDia()
file="F:\研究\Data20191205\TenthreeUseven\A0001.CSV"    
expnum=14
# Read CSV
Fdata,Idata=OSAcsvfre_In(file)

#補間

F,I,Ninter,dF=Inter(Fdata,Idata,expnum)

#FFTと
T,FF,FF_abs_amp=FFT(F,I)


delta_T=(1.0/dF)/(Ninter-1)


#振幅閾値カット理想的な信号_noioff
FF_noioff=np.copy(FF)
cutAmp = 4*10**-7 # 振幅強度の閾値
FF_noioff[(FF_abs_amp < cutAmp)] = 0 # 振幅が閾値未満はゼロにする（ノイズ除去）
FF_noioff_ifft = np.fft.ifft(FF_noioff) # IFFT
I_noioff = FF_noioff_ifft.real # 実数部の取得
T,FF_noioff,FF_noiof_abs_amp=FFT(F,I_noioff)

#I_noioff=E^2(A^2+B^2)+E^2(2AB*cos)=I_noioff=cons+AM
FF_cons=np.copy(FF_noioff)
FF_AM=np.copy(FF_noioff)

cutT =10

FF_cons[((T[2**(expnum-1)]>T)&(T>cutT))] = 0 
FF_cons[((T[2**(expnum-1)]<T)&(T<max(T)-cutT))] = 0 
FF_cons_ifft = np.fft.ifft(FF_cons) # IFFT
I_cons = FF_cons_ifft.real # 実数部の取得
T,FF_cons,FF_cons_abs_amp=FFT(F,I_cons)


FF_AM[(cutT>T)] = 0 
FF_AM[(T>(max(T)-cutT))] = 0 
FF_AM_ifft = np.fft.ifft(FF_AM) # IFFT
I_AM = FF_AM_ifft.real # 実数部の取得
T,FF_AM,FF_AM_abs_amp=FFT(F,I_AM)

# def FilteredSignal(signal, fs, cutoff):
    # B, A = butter(1, cutoff / (fs / 2), btype='low')
    # filtered_signal = filtfilt(B, A, signal, axis=0)
    # return filtered_signal


#T = ? cutoffに関係サンプリング周波数 (1/delta_F)#4.152584657991986e-05 when2*18
analyticSignal = hilbert(I_AM)
amplitudeEvelope = np.abs(analyticSignal)
# cutoff =dF #よくわかってないこの値のチョイス
# amplitudeEvelope = FilteredSignal(amplitudeEvelope, delta_T, cutoff)

#https://dsp.stackexchange.com/questions/46291/why-is-scipy-implementation-of-hilbert-function-different-from-matlab-implemen

I_top=I_cons+amplitudeEvelope
I_bottom=I_cons-amplitudeEvelope
bunshi=I_noioff*2-(I_top+I_bottom)
bunbo=I_top-I_bottom
cos_phi=bunshi/bunbo
len_cos_phi=len(cos_phi)
phi=np.empty(len_cos_phi)

wrap =np.array([math.acos(cos_phi[_]) for _ in range(len_cos_phi)])

#めっちゃ時間かかるわ
# t1=time.time()
# phi,mountafterindex,valleyafterindex=traia_line(F,wrap)
# t2=time.time()
# print(t2-t1)
# a, b = np.polyfit(F, phi, 1)
# print(299792458/(4*math.pi)*a*10**-12)
phi,mountafterindex,valleyafterindexkai=traia_linekai(F,wrap)


a, b = np.polyfit(F, phi, 1)

print(299792458/(4*math.pi)*a*10**-12)
# plt.plot(F,phi)
# plt.plot(F,F*a+b)
# plt.show()

# fig2, ax2 = plt.subplots(1, 1)
# ax2.plot(F,I_top)
# ax2.plot(F,I_bottom)
# ax2.plot(F,I_noioff)
# ax2.set_xlabel('F')
# ax2.set_ylabel('Amplitud')

plt.plot(F,phi)
plt.plot(F,a*F+b)
plt.show()



