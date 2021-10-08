#------------旧式バンドパス法------
import pandas as pd
import numpy as np
import os, tkinter, tkinter.filedialog, tkinter.messagebox

import matplotlib.pyplot as plt
import cmath,math

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
    
def unwrap(wrap):#-1.5~1.5で折り返される位相を展開かつ位相が０に近い部分を
    phi=np.empty(len(wrap))
    afterchangeindex=[]
    n=0
    phi[0]=wrap[0]
    for i in range(1,len(wrap)):
        if((wrap[i] - wrap[i-1])<-2):
        # if((np.diff(wrap[i-1]))<-2):
            n=n+1
            afterchangeindex.append(i)
        phi[i]=wrap[i]+(n*math.pi)
        
    return phi,afterchangeindex
    
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

def wrappedphase(e):#配列eの位相を折り返しで求める-1.5~1.5
    wrap=np.empty(len(e))
    for _ in range(len(e)):
        wrap[_]=math.atan(e.imag[_]/e.real[_])       
    return wrap
# file=FilefromDia()
file="F:\研究\Data20191018\W0040.CSV"    

# Read CSV
Fdata,Idata=OSAcsvfre_In(file)

#補間
expnum=18
Finter,Iinter,Ninter,df=Inter(Fdata,Idata,expnum)
#FFTとフィルタリング
T,FF,FF_abs_amp=FFT(Finter,Iinter)
delta_T=(1.0/df)/(Ninter-1)

# def filter(which,f,FFT,FFT_abs_amp,cut):#whici="Low"pass,"High"pass,"Amp"litude
    # if (which=="Low"):
        # FFT[(f > cut)] = 0 # カットオフを超える周波数のデータをゼロにする（ノイズ除去）
        # FFT_abs = np.abs(FFT) # FFTの複素数結果を絶対値に変換
        # FFT_abs_amp = FFT_abs / len(f) * 2 # 振幅をもとの信号に揃える(交流成分2倍)
        # FFT_ifft = np.fft.ifft(FFT) # IFFT
        # FFT[(f>cut)] = 0
    # elif (which=="High"):
        # FFT[(f<cut)] = 0
        # FFT_abs = np.abs(FFT) # FFTの複素数結果を絶対値に変換
        # FFT_abs_amp = FFT_abs / len(f) * 2 # 振幅をもとの信号に揃える(交流成分2倍)
        # FFT_ifft = np.fft.ifft(FFT) # IFFT
    # else:# (which=="Amp"):
        # FFT[(FFT_abs_amp < cut)]
        # FFT_abs = np.abs(FFT) # FFTの複素数結果を絶対値に変換
        # FFT_abs_amp = FFT_abs / len(f) * 2 # 振幅をもとの信号に揃える(交流成分2倍)
        # FFT_ifft = np.fft.ifft(FFT) # IFFT
    # return FFT_ifft
 #実数部の取得、振幅を元スケールに戻す
# F3_ifft=filter("Low",T,(filter("High",T,filter("Amp",T,FF,FF_abs_amp,2.5*10**-9),0,2)),0,T[2**(expnum-1)])


#振幅閾値カットF3
F3 = np.copy(FF) 
# 振幅強度でフィルタリング処理and周波数かっと
cutAmp = 2.5*10**-9 # 振幅強度の閾値
F3[(FF_abs_amp < cutAmp)] = 0 # 振幅が閾値未満はゼロにする（ノイズ除去）
F4 = np.copy(F3)
cutT = 2 # カットオフ（周波数）
F3[(T <cutT )] = 0 # カットオフを超える周波数のデータをゼロにする
F3[(T >T[2**(expnum-1)] )] = 0 # カットオフを超える周波数のデータをゼロにする
F3_abs = np.abs(F3)
# 振幅をもとの信号に揃える
F3_abs_amp = F3_abs / Ninter * 2 # 交流成分はデータ数で割って2倍
#IFFT
F3_ifft = np.fft.ifft(F3) # IFFT
g_real = F3_ifft.real # 実数部の取得
g_imag = F3_ifft.imag # 虚部の取得

wrap=wrappedphase(F3_ifft)
phi,afterchangeindex=unwrap(wrap)

Ph0index_sub=np.empty(len(afterchangeindex)-1)
# Ph0index_sub = [0] * (len(afterchangeindex)-1)
wrap_abs=np.abs(wrap)
len_afterchangeindex=len(afterchangeindex)
for i in range (0,len_afterchangeindex-1):
    start=afterchangeindex[i]
    end=afterchangeindex[i+1]
    Ph0index_sub[i]=np.argmin(wrap_abs[start:end])
len_Ph0index_sub=len(Ph0index_sub)
Ph0index = [int(Ph0index_sub[_])+afterchangeindex[_] for _ in range(len_Ph0index_sub)]

#IFFT
F4_ifft = np.fft.ifft(F4) # IFFT
g_real = F4_ifft.real # 実数部の取得
plt.plot(Finter,g_real)
plt.scatter(Finter[Ph0index[::2]],Iinter[Ph0index[::2]])
plt.show()

# グラフ（オリジナルとフィルタリングを比較）
a, b = np.polyfit(Finter, phi, 1)
print(a)
print(299792458/(4*math.pi)*a*10**-12)
plt.plot(Finter,a*Finter+b)
plt.show()

# plt.plot(Finter, g_real, c="orange", linewidth=3, alpha=0.7, label='filtered')


# plt.plot(Finter,phi, label='original')

# plt.legend(loc='best')
# plt.xlabel('time(sec)', fontsize=14)
# plt.ylabel('singnal', fontsize=14)


# plt.xlabel('freqency(Hz)', fontsize=14)
# plt.ylabel('amplitude', fontsize=14)
# plt.plot(T, F3_abs_amp, c='orange')
# plt.show()

# plt.xlabel('freqency(Hz)', fontsize=14)
# plt.ylabel('amplitude', fontsize=14)
# plt.plot(T, FF_abs_amp)
# plt.show()