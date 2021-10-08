import numpy as np



import os, tkinter, tkinter.filedialog, tkinter.messagebox
from scipy import interpolate
import matplotlib.pyplot as plt
import pandas as pd
import sys
# ファイル選択ダイアログの表示
root = tkinter.Tk()
root.withdraw()
fTyp = [("","*")]
iDir = os.path.abspath(os.path.dirname(__file__))
file = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)
# file="F:/W0002_fc1.CSV"
# 処理ファイル名の出力
tkinter.messagebox.showinfo('○×プログラム',file)
# Read CSV
wholedata = pd.read_csv(file, header=None, skiprows=34).values
Fdata=np.flipud(wholedata[:,0].ravel())
Idata=np.flipud(wholedata[:,1].ravel())

Range_mm=14438.89171
hajime_mm=7
owari_mm=9
hajime_num=int(((2**10)*(hajime_mm/Range_mm))*1000)
owari_num=int(((2**10)*(owari_mm/Range_mm))*1000)
showend_mm=100
showend_mm=int(((2**10)*(showend_mm/Range_mm))*1000)
# print("dimen",Fdata.ndim)
# print("データかたち",Fdata.shape)
# print("データ型",type(Fdata))
# print(Fdata[0])

#interpolate
Fstart=min(Fdata)*(1+sys.float_info.epsilon)
Fend=max(Fdata)*(1-sys.float_info.epsilon)
expnum=20
Ninter=2**expnum
F=np.linspace(Fstart, Fend, Ninter)
SpFun=interpolate.interp1d(Fdata,Idata,kind="cubic")
I=SpFun(F)

#抽出　
#FFT
df=(Fend-Fstart)/Ninter #サンプリング間隔
T=np.linspace(0,1.0/df,Ninter) #時間軸
FF=np.fft.fft(I)
Amp=np.abs(FF)
print("Ninter:",Ninter,"=2**",expnum)

"""
limit=len(T)//70000
FFhalf=FF[:len(T)//2]
Thalf=T[:len(T)//2]
FFlimited=FFhalf[limit:]
Tlimited=Thalf[limit:]

a=np.argmax(FFlimited)
print("PeakT:",Tlimited[a])
"""
# limit=len(T)//70000
# Amphalf=Amp[:len(T)//2]
# Thalf=T[:len(T)//2]
# Amplimited=Amphalf[limit:]
# Tlimited=Thalf[limit:]

# a=np.argmax(Amplimited)
# print("PeakT:",Tlimited[a])



# from tkinter import filedialog
# dir = 'C:\\pg'
# fld = filedialog.askdirectory(initialdir = dir) 
# print("outputfilename (Do not include path) like 「a.csv」 spline")
# name=input()
# fileout=fld+name
# data=np.c_[F,I]
# np.savetxt(fileout,data,delimiter=",")

"""
plt.subplot(3, 1, 1)#(縦分割数、横分割数、ポジション)
plt.plot(Fdata,Idata,alpha=0.2)
plt.title('Interference signal')
plt.xlabel('Freq THz')
plt.ylabel('Intensity W')
plt.subplot(3, 1, 2)
plt.plot(F,I,"-")
plt.title('Interpolated Interference signal')
plt.xlabel('Freq THz')
plt.ylabel('Intensity W')
plt.subplot(3, 1, 3)
plt.title('The result of FFT')
plt.xlabel('Time ps')
plt.ylabel('Power')
# plt.plot(Fdata,Idata,alpha=0.2)
# plt.plot(F,I,"-")
# plt.subplot(4, 1, 4)
plt.plot(Thalf,FFhalf,alpha=0.2)
plt.tight_layout()
plt.show()
"""
Amphalf=Amp[:len(T)//2]
Thalf=T[:len(T)//2]
Amplimited=Amp[hajime_num:owari_num]
Tlimited=T[hajime_num:owari_num]
a=np.argmax(Amplimited)
print("PeakT:",Tlimited[a])
print("1:CSV, other:noCSV")
answer=input()
if bool(answer=="1")==True:
    from tkinter import filedialog
    dir = 'C:\\pg'
    fld = filedialog.askdirectory(initialdir = dir) 
    print("outputfilename (Do not include path) like 「a.csv」")
    name=input()
    fileout=fld+"/"+name
    data=np.c_[Thalf,Amphalf]
    np.savetxt(fileout,data,delimiter=",")

plt.subplot(2, 1, 1)#(縦分割数、横分割数、ポジション)
plt.plot(F,I,"-")
plt.title('Interpolated Interference signal')
plt.xlabel('Freq THz')
plt.ylabel('Intensity W')
plt.subplot(2, 1, 2)
plt.title('The result of FFT')
plt.xlabel('Time ps')
plt.ylabel('Power')
# plt.plot(Fdata,Idata,alpha=0.2)
# plt.plot(F,I,"-")
# plt.subplot(4, 1, 4)
plt.plot(Thalf,Amphalf,alpha=0.5)
plt.tight_layout()
plt.show()
#グラフ　１生データとスプライン曲線　　２FFT結果
# plt.plot(Fdata, Idata, 'ro') #Sampled data
# plt.plot(F, I, 'm') #Polynominal interpolation
#CSVへ
