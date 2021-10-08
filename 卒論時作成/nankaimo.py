import pandas as pd
import numpy as np
import os, tkinter, tkinter.filedialog, tkinter.messagebox
from scipy import interpolate
import matplotlib.pyplot as plt
import sys
# ファイル選択ダイアログの表示
# Tend_ps=96325.9171886819
# Range_mm=299792458*(Tend_ps*(10**-12))/2*(10**3)
Range_mm=14438.89171
hajime_mm=0.3
owari_mm=11
hajime_num=int(((2**10)*(hajime_mm/Range_mm))*1000)
owari_num=int(((2**10)*(owari_mm/Range_mm))*1000)
print(hajime_num,owari_num)
for number in range(10): ##0-9
    file="G:\研究\Data20191020\W008"+str(number)+".CSV"
    wholedata = pd.read_csv(file, header=None, skiprows=34).values
    Fdata=np.flipud(wholedata[:,0].ravel())
    Idata=np.flipud(wholedata[:,1].ravel())


    # print("dimen",Fdata.ndim)
    # print("データかたち",Fdata.shape)
    # print("データ型",type(Fdata))
    # print(Fdata[0])

    #interpolate
    Finterstart=min(Fdata)*(1+sys.float_info.epsilon)
    Finterend=max(Fdata)*(1-sys.float_info.epsilon)
    expnum=20
    Ninter=2**expnum
    Finter=np.linspace(Finterstart, Finterend, Ninter)
    SpFun=interpolate.interp1d(Fdata,Idata,kind="cubic")
    Iinter=SpFun(Finter)

    #抽出　
    #FFT
    df=(Finterend-Finterstart)/Ninter #サンプリング間隔
    T=np.linspace(0,1.0/df,Ninter) #時間軸
    FF=np.fft.fft(Iinter)
    Amp=np.abs(FF)
    # print("Ninter:",Ninter,"=2**",expnum)

    """
    limit=len(T)//170000
    FFhalf=FF[:len(T)//2]
    Thalf=T[:len(T)//2]
    hajimeT
    owariT
    hajime_mm=5
    owari_mm=9
    FFlimited=FFhalf[limit:]
    Tlimited=Thalf[limit:]

    a=np.argmax(FFlimited)
    # print("PeakT:",Tlimited[a],299792458/2/1000/1000000000000*Tlimited[a],"m","\n")
    print(299792458/2/1000000000000*Tlimited[a])
    """
    """
    FFhalf=FF[:len(T)//2]
    Thalf=T[:len(T)//2]

    FFlimited=FF[hajime_num:owari_num]
    Tlimited=T[hajime_num:owari_num]

    a=np.argmax(FFlimited)
    """
    Amphalf=Amp[:len(T)//2]
    Thalf=T[:len(T)//2]

    Amplimited=Amp[hajime_num:owari_num]
    Tlimited=T[hajime_num:owari_num]

    a=np.argmax(Amplimited)
    # print("PeakT:",Tlimited[a],299792458/2/1000/1000000000000*Tlimited[a],"m","\n")
    print(299792458/2/1000000000000*Tlimited[a])

    # fileout="RRRR002"+str(number)+".CSV"
    # data=np.c_[T,Amp]
    # np.savetxt(fileout,data,delimiter=",")

