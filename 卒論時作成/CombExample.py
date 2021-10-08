# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 20:22:36 2020

@author: wsxhi
"""


import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg

fstart=190e12
fend=194e12
I_off=2e24
f=np.linspace(fstart,fend,10000)
I_comb=-(f-fstart)*(f-fend)+I_off
# DRoute=0.0008547704292455535
DRoute=0.0005714637543591736
n=1 #refraction index
c=299792458
phi=2*np.pi*n*DRoute/c*f
a=0.3
b=0.3
C1=a+b
C2=2*a*b
I=I_comb*(C1+C2*np.cos(phi))
win = pg.GraphicsWindow()
win.resize(800,600)

p4 = win.addPlot(title="CombExample")
p4.plot(f,I_comb, pen=(0,0,255))
win.nextRow() 
p5 = win.addPlot(title="SignalExample")
p5.plot(f,I)
