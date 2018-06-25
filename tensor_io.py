import tensorflow as tf
import numpy as np
import time
import sys
import os
from struct import *

def io_save_array_to_txt_file(filename,arr):
    all_size = 1
    for i in range(0,len(arr.shape)):
        all_size *=  arr.shape[i]
    print ("all_size:",all_size)   
    
    act_flat = arr.reshape(all_size)
    
    print ('save ',filename)
    filename = filename.split('.')[-2] + '.txt'
    file = open(filename, "w")
    cnt = 0
    for i in range(0,all_size):
        str = '%-12f' % act_flat[i]
        file.write(str)
        if (cnt % 64 == 0 and cnt != 0):
            file.write('\n')
        cnt = cnt + 1
    file.close()
    
def io_save_array_to_binfile(filename,data):
    
    filename = filename.split('.')[-2] + '.bin'
    file = open(filename, "wb")
    
    all_size = 1
    
    for i in range(0,len(data.shape)):
        all_size *= data.shape[i]
        
    act_flat = data.reshape(all_size)
    
    for i in range(0,act_flat.shape[0]):
        file.write(pack("f",float(act_flat[i])))
    file.close()