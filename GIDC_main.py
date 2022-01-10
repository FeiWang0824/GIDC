# -*- coding: utf-8 -*-
"""
By Fei Wang, Jan 2022
Contact: WangFei_m@outlook.com
This code implements the ghost imaging reconstruction using deep neural network constraint (GIDC) algorithm
reported in the paper: 
Fei Wang et al. 'Far-field super-resolution ghost imaging with adeep neural network constraint'. Light Sci Appl 11, 1 (2022).  
https://doi.org/10.1038/s41377-021-00680-w
Please cite our paper if you find this code offers any help.

Inputs:
A_real: illumination patterns (pixels * pixels * pattern numbers)
y_real: single pixel measurements (pattern numbers)

Outputs:
x_out: reconstructed image by GIDC (pixels * pixels)
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import GIDC_model_Unet
from PIL import Image
import os
tf.reset_default_graph()

# load data
data = loadmat('data.mat') 
result_save_path = '.\\results\\'

# create results save path
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path) 

# define optimization parameters
img_W = 64
img_H = 64
SR = 0.1                                      # sampling rate
batch_size = 1
lr0 = 0.05                                    # learning rate
TV_strength = 1e-9                            # regularization parameter of Total Variation
num_patterns = int(np.round(img_W*img_H*SR))  # number of measurement times  
Steps = 201                                   # optimization steps

A_real = data['patterns'][:, :, 0:num_patterns]  # illumination patterns
y_real = data['measurements'][0:num_patterns]    # intensity measurements

if (num_patterns > np.shape(data['patterns'])[-1]):
    raise Exception('Please set a smaller SR')

# DGI reconstruction
print('DGI reconstruction...')
B_aver  = 0
SI_aver = 0
R_aver = 0
RI_aver = 0
count = 0
for i in range(num_patterns):    
    pattern = data['patterns'][:,:,i]
    count = count + 1
    B_r = data['measurements'][i]

    SI_aver = (SI_aver * (count -1) + pattern * B_r)/count
    B_aver  = (B_aver * (count -1) + B_r)/count
    R_aver = (R_aver * (count -1) + sum(sum(pattern)))/count
    RI_aver = (RI_aver * (count -1) + sum(sum(pattern))*pattern)/count
    DGI = SI_aver - B_aver / R_aver * RI_aver
# DGI[DGI<0] = 0
print('Finished')

with tf.variable_scope('input'):           
    inpt = tf.placeholder(tf.float32,shape=[batch_size,img_W,img_H,1],name = 'inpt')
    y = tf.placeholder(tf.float32,shape=[batch_size,1,1,num_patterns],name = 'y') 
    A = tf.placeholder(tf.float32,shape=[batch_size,img_W,img_H,num_patterns],name = 'A')                
    x = tf.placeholder(tf.float32,shape=[batch_size,img_W,img_H,1],name = 'x')   
                
    isTrain = tf.placeholder(tf.bool,name = 'isTrain')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    groable = tf.Variable(tf.constant(0))
    lrate = tf.train.exponential_decay(lr0,groable,100,0.90)

# Build the DNN structure (the physical model was embedded in the DNN) y = Ax, y:measurements(known) A:physical model(known) x:object(unknown)
x_pred,y_pred = GIDC_model_Unet.inference(inpt, A, batch_size, img_W, img_H, num_patterns, isTrain)

# define the loss function
TV_reg = TV_strength*tf.image.total_variation(tf.reshape(x_pred,[batch_size,img_W,img_H,1]))
loss_y = tf.reduce_mean(tf.square(y - y_pred))
loss = loss_y + TV_reg

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)      
with tf.variable_scope('train_step'):
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5,beta2=0.9,epsilon=1e-08).minimize(loss)
           
init_op = (tf.local_variables_initializer(),tf.global_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)  
    y_real = np.reshape(y_real,[batch_size,1,1,num_patterns])    
    A_real = np.reshape(A_real,[batch_size,img_W,img_H,num_patterns])
    DGI = np.reshape(DGI,[batch_size,img_W,img_H,1])
    
    # preprocessing
    # DGI = np.transpose(DGI) # sometimes it gives better results     
    DGI = (DGI - np.mean(DGI))/np.std(DGI)
    y_real = (y_real - np.mean(y_real))/np.std(y_real)
    A_real = (A_real - np.mean(A_real))/np.std(A_real)                
    
    # prepare for surveillance             
    DGI_temp0 = np.reshape(DGI,[img_W,img_H],order='F')
    DGI_temp = np.transpose(DGI_temp0)
    y_real_temp = np.reshape(y_real,[num_patterns])
    inpt_temp = DGI
    
    print('GIDC reconstruction...')
   
    for step in range(Steps): 
        lr_temp = sess.run(lrate,feed_dict={groable:step}) 
                                        
        if step%100 == 0: 
            train_y_loss = sess.run(loss_y, feed_dict={inpt:inpt_temp,y:y_real,A:A_real,isTrain:True,lr:lr_temp})
            print('step:%d----y loss:%f----learning rate:%f----num of patterns:%d' % (step,train_y_loss,lr_temp,num_patterns))           
                        
            [y_out,x_out] = sess.run([y_pred,x_pred],feed_dict={inpt:inpt_temp,y:y_real,A:A_real,isTrain:True,lr:lr_temp})  
            x_out = np.reshape(x_out,[img_W,img_H],order='F')            
            y_out =  np.reshape(y_out,[num_patterns],order='F')   
                               
            plt.subplot(141)
            plt.imshow(DGI_temp0)
            plt.title('DGI')
            plt.yticks([])
            
            plt.subplot(142)
            plt.imshow(x_out)
            plt.title('GIDC')
            plt.yticks([])
            
            ax1 = plt.subplot(143)
            plt.plot(y_out)
            plt.title('pred_y')
            ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')       
            plt.yticks([])
            
            ax2 = plt.subplot(144)
            plt.plot(y_real_temp)
            plt.title('real_y')
            ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
            plt.yticks([])

            plt.subplots_adjust(hspace=0.25, wspace=0.25)
            plt.show()

            x_out = x_out - np.min(x_out)
            x_out = x_out*255/np.max(np.max(x_out))
            x_out = Image.fromarray(x_out.astype('uint8')).convert('L')
            x_out.save(result_save_path + 'GIDC_%d_%d.bmp'%(num_patterns,step))
                        
        # optimize the weights in the DNN
        sess.run([train_op],feed_dict={inpt:inpt_temp,y:y_real,A:A_real,isTrain:True,lr:lr_temp})

print('Finished!')
