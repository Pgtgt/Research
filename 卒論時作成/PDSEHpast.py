#------------Phase Demodulation of the Signal Envelops in a Hilbert way-----
#import
import pandas as pd
import numpy as np
import os, tkinter, tkinter.filedialog, tkinter.messagebox
from scipy import interpolate
import matplotlib.pyplot as plt
import cmath,math
import sys
from scipy.signal import hilbert,firwin,lfilter

#データ読み込み

"""
# ファイル選択ダイアログの表示
root = tkinter.Tk()
root.withdraw()
fTyp = [("","*")]
iDir = os.path.abspath(os.path.dirname(__file__))
file = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
# file="F:/W0002_fc1.CSV"
# 処理ファイル名の出力
tkinter.messagebox.showinfo('○×プログラム',file)
"""
file="F:\研究\Data20191018/W0152.CSV"
# Read CSV
wholedata = pd.read_csv(file, header=None, skiprows=34).values
Fdata=np.flipud(wholedata[:,0].ravel())#小さい周波数から格納
Idata=np.flipud(wholedata[:,1].ravel())
#スプライン
Fstart=min(Fdata)*(1+sys.float_info.epsilon)
Fend=max(Fdata)*(1-sys.float_info.epsilon)
expnum=10
Ninter=2**expnum
F=np.linspace(Fstart, Fend, Ninter)
SpFun=interpolate.interp1d(Fdata,Idata,kind="cubic")
I=SpFun(F)

#FFTとフィルタリング
df=(Fend-Fstart)/(Ninter-1) #サンプリング間隔
T=np.linspace(0,1.0/df,Ninter) #時間軸

FF=np.fft.fft(I)
FF_abs=np.abs(FF)
FF_abs_amp=FF_abs/(Ninter/2)
#振幅閾値カットF3
F3 = np.copy(FF) 
# 振幅強度でフィルタリング処理and周波数かっと
cutAmp =1.5*10**-9 # 振幅強度の閾値
F3[(FF_abs_amp < cutAmp)] = 0 # 振幅が閾値未満はゼロにする（ノイズ除去）
F3_abs = np.abs(F3)
# 振幅をもとの信号に揃える
F3_abs_amp = F3_abs / Ninter * 2 # 交流成分はデータ数で割って2倍
#IFFT
F3_ifft = np.fft.ifft(F3) # IFFT
g_real = F3_ifft.real # 実数部の取得
g_imag = F3_ifft.imag # 虚部の取得



#phase=0のポイントを求めスプライン補間でつなぐ

Ph0F
Ph0I


#envelope作れる曲線を作る

#envelope作り，復元する




plt.plot(F, g_real, c="orange", linewidth=3, alpha=0.7, label='filtered')
plt.legend(loc='best')
plt.xlabel('time(sec)', fontsize=14)
plt.ylabel('singnal', fontsize=14)


"""
#包絡線（ヒルベルト変換）
amplitude_envelope = np.abs(hilbert(g_real))

# フィルタの設計
mabaraex=10
mabara=2**mabaraex
df_fil=df*mabara
T_fil=T[::mabara]
amplitude_envelope_fil=amplitude_envelope[::mabara]
F_fil=F[::mabara]
T_fil=np.linspace(0,1.0/df_fil,mabara) #時間軸
FF_mabara=np.fft.fft(amplitude_envelope_fil)
FF_mabara_abs=np.abs(FF_mabara)
FF_mabara_abs_amp=FF_mabara_abs/(mabara/2)

#振幅閾値カットF3
F3 = np.copy(FF_mabara) 
# 振幅強度でフィルタリング処理and周波数かっと
cutT =2 # T
F3[(T_fil>cutT)] = 0 # カットオフを超える周波数のデータをゼロにする
F3_abs = np.abs(F3)
# 振幅をもとの信号に揃える
F3_abs_amp = F3_abs / mabara * 2 # 交流成分はデータ数で割って2倍
#IFFT
F3_ifft = np.fft.ifft(F3) # IFFT
g_real = F3_ifft.real # 実数部の取得
"""

"""mabaraex=4
mabara=2**mabaraex
fn = 1/(2*(df*mabara))                   # ナイキスト周波数
Tcuthil = 1#カットオフ周波数1
Wp=Tcuthil/fn
a6 = 1
numtaps = 2**(expnum-mabaraex)
b6 = firwin(numtaps, Wp, window="hann")
Fmabara=F[::mabara]
amplitude_envelope_mabara=amplitude_envelope[::mabara]
y6 = lfilter(b6, a6, amplitude_envelope_mabara)
delay = (numtaps-1)/2*(df*mabara)
plt.plot(Fmabara-delay, y6, "y", linewidth=2, label="fir")
"""
#包絡線とデータによるarccos

analytical_signal = hilbert(g_real)
# plt.plot(F,analytical_signal.real)
# plt.plot(F,analytical_signal.imag)

amplitude_envelope = np.abs(analytical_signal)

plt.plot(F,amplitude_envelope, c="darkblue",linewidth=0.2, alpha=0.7)

plt.show()
#位相計算

#位相の傾きしらべる


