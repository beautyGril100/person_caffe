
import os
import sys
import struct
import numpy as np

filename = 'CDNNOutput_resnet101_5a9b9ee9N199298c7_fixPC.xls'
f = open(filename, "rb")
net_result_1 = 'net_result_1.txt'
nx =1
nz=1024
pic = np.zeros((nx, nz))
for i in range(nx):
  for j in range(nz):
     data = f.read(4)
     elem = struct.unpack("f", data)[0]
     pic[i][j] =elem
     print elem
pic_1d = np.reshape(pic,(1,-1))   
np.savetxt(net_result_1,pic_1d,fmt="%.5f", delimiter="\n")
f.close()