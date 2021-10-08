import pandas as pd
import numpy as np
import math
# import os, tkinter, tkinter.filedialog, tkinter.messagebox
import csv
# ファイル選択ダイアログの表示
"""root = tkinter.Tk()
root.withdraw()
fTyp = [("","*")]
iDir = os.path.abspath(os.path.dirname(__file__))
file = tkinter.filedialog.askopenfilename(filetypes = fTyp,initialdir = iDir)

# 処理ファイル名の出力
tkinter.messagebox.showinfo('○×プログラム',file)"""
file="F:\研究\Data20191018/W0172.CSV"
# Read CSV
wholedata = pd.read_csv(file, header=None, skiprows=34).values
Fdata=np.flipud(wholedata[:,0].ravel())
Idata=np.flipud(wholedata[:,1].ravel())

# nout = 1*10**4#出力データの区切り数
# twopiT = np.linspace(1.7321*2*math.pi, 1.7322*2*math.pi, nout)目標

nout = (1*10**4)-1#出力データの区切り数
T= np.linspace(0.000001, 7, nout)
twopiT = T*2*math.pi
import scipy.signal as signal
pgram = signal.lombscargle(Fdata, Idata, twopiT, normalize=True)
# print(pgram)
T=twopiT/(2*math.pi)

#最初の10分の１ぐらいは最大値探索から省く
limit=len(pgram)//10
pgramlimited=pgram[limit:]
Tlimited=T[limit:]
a=np.argmax(pgramlimited)
print("T:",Tlimited[a])
print("x:",299792458*0.5*(Tlimited[a]*10**-12)*10**3,"mm")

#print(T,pgram)
#print(np.argmax(pgram))
#print(f[np.argmax(pgram)])


import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.plot(Fdata, Idata,linewidth=0.3)

plt.subplot(2, 1, 2)
plt.plot(T, pgram,linewidth=0.3)
plt.show()


