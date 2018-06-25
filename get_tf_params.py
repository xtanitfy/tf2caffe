import os  
import numpy as np  
import caffe
from tensor_io import io_save_array_to_txt_file,io_save_array_to_binfile

from tensorflow.python import pywrap_tensorflow  
checkpoint_path='tf_model/mobileDet_V1_025_KITTI_PERSON.ckpt-255'  
tf_model_dir = 'tf_model'
caffe_model_dir = 'caffe_model'

# print(path.getcwdu())  
print(checkpoint_path)  
#read data from checkpoint file  
reader=pywrap_tensorflow.NewCheckpointReader(checkpoint_path)  
var_to_shape_map=reader.get_variable_to_shape_map()  
  
train_keys = ['weights','depthwise_weights',
            'moving_mean','moving_variance','beta','gamma','mean','var']
          
for key in var_to_shape_map:  
    #print('tensor_name',key) 
    items = key.split('/')
    #print ('tensor_name',items[-1]) 
    if items[-1] in train_keys:
        #if 'Conv2d_1_' in items[1]:
        print ('tensor_name',key) 
        print ('items[2]:',items[2]) 
        netname = items[0]
        layername = items[1]
        blogits = False

        if 'weights' in items[2]:
            type = 'weights'
            if 'depthwise_weights' == items[2]:
                type = 'dw_weights'
        elif 'gamma' in items[3] or 'beta' in items[3] or 'moving_mean' in items[3] or 'moving_variance' in items[3]:
            type = 'batchnorm'
            sub_type = items[3]
        else:
            if layername == 'Logits':
                blogits = True   
                type = 'weights'
            else:
                print ('No this type',key)
                #raw_input('pause')
                continue
        
        ckpt_data = np.array(reader.get_tensor(key))#cast list to np arrary  \
        if type == 'weights':
            param = np.transpose(ckpt_data,(3,2,0,1))
        elif type == 'dw_weights':
            #print (ckpt_data.shape)
            param = np.transpose(ckpt_data,(2,3,0,1))
            #print ('====================')
        else:
            param = ckpt_data
        #if blogits == True:
        
        #blogits = False
            
        if type == 'weights' or type == 'dw_weights':
            
            pshape = param.shape
            param_name = netname + '-' + layername + '-' + 'weights'
            shape = '{}_{}_{}_{}'.format(pshape[0],pshape[1],pshape[2],pshape[3])
        else:
            param_name = netname + '-' + layername + '-' + 'BatchNorm' + '-' + sub_type
            shape = str(param.shape[0])
        #print ('name:{} shape:{}'.format(key,shape))
        
        filename = os.path.join(tf_model_dir,param_name+'='+shape+'.txt')
        io_save_array_to_binfile(filename,param)
  


