import numpy as np

caffe_layernames = ['conv1',
                    'conv2_1',
                    'conv2_2',
                    'conv3_1',
                    'conv3_2',
                    'conv4_1',
                    'conv4_2',
                    'conv5_1',
                    'conv5_2',
                    'conv5_3',
                    'conv5_4',
                    'conv5_5',
                    'conv5_6',
                    'conv6_1',
                    'conv6_2',
                    'conv6_3',
                    'conv7']

def write_map(f):
    for i in range(0,len(caffe_layernames)):
        dw_tf_name = 'Conv2d_'+str(i)+'_depthwise-weights'
        dw_caffe_name = caffe_layernames[i] + '/dw'
        f.write(dw_caffe_name+' '+dw_tf_name+'\n')
        
        #conv2_1/sep/bn Conv2d_1_pointwise-BatchNorm-mean Conv2d_1_pointwise-BatchNorm-var
        bn_tf_mean = 'Conv2d_'+str(i)+'_depthwise-BatchNorm-moving_mean'
        bn_tf_var = 'Conv2d_'+str(i)+'_depthwise-BatchNorm-moving_variance'
        bn_caffe = caffe_layernames[i] + '/dw/bn'
        f.write(bn_caffe+' ' + bn_tf_mean + ' ' + bn_tf_var + '\n')
        
        #conv2_1/sep/scale Conv2d_1_pointwise-BatchNorm-gamma  Conv2d_1_pointwise-BatchNorm-beta
        bn_tf_gamma = 'Conv2d_'+str(i)+'_depthwise-BatchNorm-gamma'
        bn_tf_beta = 'Conv2d_'+str(i)+'_depthwise-BatchNorm-beta'
        bn_caffe = caffe_layernames[i] + '/dw/scale'
        f.write(bn_caffe+' ' + bn_tf_gamma + ' ' + bn_tf_beta + '\n')
        
        
        pw_tf_name = 'Conv2d_'+str(i)+'_pointwise-weights'
        pw_caffe_name = caffe_layernames[i] + '/sep'
        f.write(pw_caffe_name+' '+pw_tf_name+'\n')
        
        #conv2_1/sep/bn Conv2d_1_pointwise-BatchNorm-mean Conv2d_1_pointwise-BatchNorm-var
        bn_tf_mean = 'Conv2d_'+str(i)+'_pointwise-BatchNorm-moving_mean'
        bn_tf_var = 'Conv2d_'+str(i)+'_pointwise-BatchNorm-moving_variance'
        bn_caffe = caffe_layernames[i] + '/sep/bn'
        f.write(bn_caffe+' ' + bn_tf_mean + ' ' + bn_tf_var + '\n')
        
        #conv2_1/sep/scale Conv2d_1_pointwise-BatchNorm-gamma  Conv2d_1_pointwise-BatchNorm-beta
        bn_tf_gamma = 'Conv2d_'+str(i)+'_pointwise-BatchNorm-gamma'
        bn_tf_beta = 'Conv2d_'+str(i)+'_pointwise-BatchNorm-beta'
        bn_caffe = caffe_layernames[i] + '/sep/scale'
        f.write(bn_caffe+' ' + bn_tf_gamma + ' ' + bn_tf_beta + '\n')
        
        
        #conv2_1/sep/scale Conv2d_1_pointwise-BatchNorm-gamma  Conv2d_1_pointwise-BatchNorm-beta

with open('layer_map.txt','w') as f:
    write_map(f)
    
print caffe_layernames







