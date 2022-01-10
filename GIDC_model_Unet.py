# -*- coding: utf-8 -*-
"""
By Fei Wang, Sep 6, 2020
Contact: WangFei_m@outlook.com
This code implements the DNN structure and measurements process of GIDC algorithm reported in the paper: 
Fei Wang, Chenglong Wang, Wenlin Gong, Shensheng Han, and Guohai Situ. 'Ghost imaging with a deep neural network constraint,'  
Please cite our paper if you find this code offers any help.

Inputs:
inpt: DGI results (batch_size * pixels * pixels * 1)
real_A: illumination patterns (batch_size * pixels * pixels * num_patterns)
batch_size: batch_size
img_W: width of image
img_H: high of image
num_A: num_patterns

Outputs:
out_x: estimated image by GIDC (batch_size * pixels * pixels * 1)
out_y: estimated intensity measurements associated with out_x and real_A (batch_size * 1 * 1 * num_patterns)
"""
# Unet
import tensorflow as tf

def weight_variable(shape, name=None):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)
    
def leaky_relu(x, leak=0.2, name=''):
            return tf.maximum(x, x * leak, name=name)
        
def inference(inpt, real_A, batch_size, img_W, img_H, num_A, isTrain=True):
    c_size = 5
    d_size = 5

    layer_1 = tf.reshape(inpt,[batch_size,img_W,img_H,1])
    
    with tf.variable_scope('conv0'):
        W_conv0 = weight_variable([c_size, c_size, 16, 1])
        conv0 = tf.nn.conv2d_transpose(layer_1,W_conv0,output_shape=[batch_size, img_W, img_H, 16],strides=[1,1,1,1],padding="SAME")   
        conv0 = tf.layers.batch_normalization(conv0, training = isTrain)    
        conv0 = tf.nn.leaky_relu(conv0)
          
    with tf.variable_scope('conv1'):
        W_conv1 = weight_variable([c_size, c_size, 16, 16])
        conv1 = tf.nn.conv2d_transpose(conv0,W_conv1,output_shape=[batch_size, img_W, img_H, 16],strides=[1,1,1,1],padding="SAME")   
        conv1 = tf.layers.batch_normalization(conv1, training = isTrain)    
        conv1 = tf.nn.leaky_relu(conv1)

        W_conv1_1 = weight_variable([c_size, c_size, 16, 16])
        conv1_1 = tf.nn.conv2d(conv1, W_conv1_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv1_1 = tf.layers.batch_normalization(conv1_1, training = isTrain)    
        conv1_1 = tf.nn.leaky_relu(conv1_1)
    
    with tf.variable_scope('conv_pooling_1'):
        Convpool_1 = tf.layers.conv2d(conv1_1, 16, [d_size, d_size], strides=(2, 2), padding='SAME')
        Convpool_1 = tf.nn.leaky_relu(tf.layers.batch_normalization(Convpool_1, training=isTrain))
        
    with tf.variable_scope('conv2'):
        W_conv2 = weight_variable([c_size, c_size, 16, 32])
        conv2 = tf.nn.conv2d(Convpool_1, W_conv2, strides=[1, 1, 1, 1], padding="SAME")   
        conv2 = tf.layers.batch_normalization(conv2, training = isTrain)    
        conv2 = tf.nn.leaky_relu(conv2)

        W_conv2_1 = weight_variable([c_size, c_size, 32, 32])
        conv2_1 = tf.nn.conv2d(conv2, W_conv2_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv2_1 = tf.layers.batch_normalization(conv2_1, training = isTrain)    
        conv2_1 = tf.nn.leaky_relu(conv2_1)
        
    with tf.variable_scope('conv_pooling_2'):
        Convpool_2 = tf.layers.conv2d(conv2_1, 32, [d_size, d_size], strides=(2, 2), padding='SAME')
        Convpool_2 = tf.nn.leaky_relu(tf.layers.batch_normalization(Convpool_2, training=isTrain))
    
    with tf.variable_scope('conv3'):
        W_conv3 = weight_variable([c_size, c_size, 32, 64])
        conv3 = tf.nn.conv2d(Convpool_2, W_conv3, strides=[1, 1, 1, 1], padding="SAME")   
        conv3 = tf.layers.batch_normalization(conv3, training = isTrain)    
        conv3 = tf.nn.relu(conv3)

        W_conv3_1 = weight_variable([c_size, c_size, 64, 64])
        conv3_1 = tf.nn.conv2d(conv3, W_conv3_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv3_1 = tf.layers.batch_normalization(conv3_1, training = isTrain)    
        conv3_1 = tf.nn.leaky_relu(conv3_1)
         
    with tf.variable_scope('conv_pooling_3'):
        Convpool_3 = tf.layers.conv2d(conv3_1, 64, [d_size, d_size], strides=(2, 2), padding='SAME')
        Convpool_3 = tf.nn.leaky_relu(tf.layers.batch_normalization(Convpool_3, training=isTrain))
        
    with tf.variable_scope('conv4'):
        W_conv4 = weight_variable([c_size, c_size, 64, 128])
        conv4 = tf.nn.conv2d(Convpool_3, W_conv4, strides=[1, 1, 1, 1], padding="SAME")   
        conv4 = tf.layers.batch_normalization(conv4, training = isTrain)    
        conv4 = tf.nn.leaky_relu(conv4)
        
        W_conv4_1 = weight_variable([c_size, c_size, 128, 128])
        conv4_1 = tf.nn.conv2d(conv4, W_conv4_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv4_1 = tf.layers.batch_normalization(conv4_1, training = isTrain)    
        conv4_1 = tf.nn.leaky_relu(conv4_1)
         
    with tf.variable_scope('conv_pooling_4'):
        Convpool_4 = tf.layers.conv2d(conv4_1, 128, [d_size, d_size], strides=(2, 2), padding='SAME')
        Convpool_4 = tf.nn.leaky_relu(tf.layers.batch_normalization(Convpool_4, training=isTrain))
        
    with tf.variable_scope('conv5'):
        W_conv5 = weight_variable([c_size, c_size, 128, 256])
        conv5 = tf.nn.conv2d(Convpool_4, W_conv5, strides=[1, 1, 1, 1], padding="SAME")   
        conv5 = tf.layers.batch_normalization(conv5, training = isTrain)    
        conv5 = tf.nn.leaky_relu(conv5)
        
        W_conv5_1 = weight_variable([c_size, c_size, 256, 256])
        conv5_1 = tf.nn.conv2d(conv5, W_conv5_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv5_1 = tf.layers.batch_normalization(conv5_1, training = isTrain)    
        conv5_1 = tf.nn.leaky_relu(conv5_1)
         
    with tf.variable_scope('conv6'):
        W_conv6 = weight_variable([c_size, c_size, 128, 256])
        conv6 = tf.nn.conv2d_transpose(conv5_1,W_conv6,output_shape=[batch_size, int(img_W/8), int(img_W/8), 128],strides=[1,2,2,1],padding="SAME")   
        conv6 = tf.layers.batch_normalization(conv6, training = isTrain)    
        conv6 = tf.nn.leaky_relu(conv6)
        
        merge1 = tf.concat([conv4_1,conv6], axis = 3)

        W_conv6_1 = weight_variable([c_size, c_size, 256, 128])
        conv6_1 = tf.nn.conv2d(merge1, W_conv6_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv6_1 = tf.layers.batch_normalization(conv6_1, training = isTrain)    
        conv6_1 = tf.nn.leaky_relu(conv6_1)
        
        W_conv6_2 = weight_variable([c_size, c_size, 128, 128])
        conv6_2 = tf.nn.conv2d(conv6_1, W_conv6_2, strides=[1, 1, 1, 1], padding="SAME")   
        conv6_2 = tf.layers.batch_normalization(conv6_2, training = isTrain)    
        conv6_2 = tf.nn.leaky_relu(conv6_2)
        
    with tf.variable_scope('conv7'):
        W_conv7 = weight_variable([c_size, c_size, 64, 128])
        conv7 = tf.nn.conv2d_transpose(conv6_2,W_conv7,output_shape=[batch_size, int(img_W/4), int(img_H/4), 64],strides=[1,2,2,1],padding="SAME")   
        conv7 = tf.layers.batch_normalization(conv7, training = isTrain)    
        conv7 = tf.nn.leaky_relu(conv7)
        
        merge2 = tf.concat([conv3_1,conv7], axis = 3)

        W_conv7_1 = weight_variable([c_size, c_size, 128, 64])
        conv7_1 = tf.nn.conv2d(merge2, W_conv7_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv7_1 = tf.layers.batch_normalization(conv7_1, training = isTrain)    
        conv7_1 = tf.nn.leaky_relu(conv7_1)
        
        W_conv7_2 = weight_variable([c_size, c_size, 64, 64])
        conv7_2 = tf.nn.conv2d(conv7_1, W_conv7_2, strides=[1, 1, 1, 1], padding="SAME")   
        conv7_2 = tf.layers.batch_normalization(conv7_2, training = isTrain)    
        conv7_2 = tf.nn.leaky_relu(conv7_2)
        
    with tf.variable_scope('conv8'):
        W_conv8 = weight_variable([c_size, c_size, 32, 64])
        conv8 = tf.nn.conv2d_transpose(conv7_2,W_conv8,output_shape=[batch_size, int(img_W/2), int(img_H/2), 32],strides=[1,2,2,1],padding="SAME")   
        conv8 = tf.layers.batch_normalization(conv8, training = isTrain)    
        conv8 = tf.nn.leaky_relu(conv8)
        
        merge3 = tf.concat([conv2_1,conv8], axis = 3)

        W_conv8_1 = weight_variable([c_size, c_size, 64, 32])
        conv8_1 = tf.nn.conv2d(merge3, W_conv8_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv8_1 = tf.layers.batch_normalization(conv8_1, training = isTrain)    
        conv8_1 = tf.nn.leaky_relu(conv8_1)
        
        W_conv8_2 = weight_variable([c_size, c_size, 32, 32])
        conv8_2 = tf.nn.conv2d(conv8_1, W_conv8_2, strides=[1, 1, 1, 1], padding="SAME")   
        conv8_2 = tf.layers.batch_normalization(conv8_2, training = isTrain)    
        conv8_2 = tf.nn.leaky_relu(conv8_2)
        
    with tf.variable_scope('conv9'):
        W_conv9 = weight_variable([c_size, c_size, 16, 32])
        conv9 = tf.nn.conv2d_transpose(conv8_2,W_conv9,output_shape=[batch_size, img_W, img_H, 16],strides=[1,2,2,1],padding="SAME")   
        conv9 = tf.layers.batch_normalization(conv9, training = isTrain)    
        conv9 = tf.nn.leaky_relu(conv9)
        
        merge4 = tf.concat([conv1_1,conv9], axis = 3)

        W_conv9_1 = weight_variable([c_size, c_size, 32, 16])
        conv9_1 = tf.nn.conv2d(merge4, W_conv9_1, strides=[1, 1, 1, 1], padding="SAME")   
        conv9_1 = tf.layers.batch_normalization(conv9_1, training = isTrain)    
        conv9_1 = tf.nn.leaky_relu(conv9_1)
        
        W_conv9_2 = weight_variable([c_size, c_size, 16, 16])
        conv9_2 = tf.nn.conv2d(conv9_1, W_conv9_2, strides=[1, 1, 1, 1], padding="SAME")   
        conv9_2 = tf.layers.batch_normalization(conv9_2, training = isTrain)    
        conv9_2 = tf.nn.leaky_relu(conv9_2)
                
    with tf.variable_scope('conv10'):
        W_conv10 = weight_variable([c_size, c_size, 16, 1])
        conv10 = tf.nn.conv2d(conv9_2, W_conv10, strides=[1, 1, 1, 1], padding="SAME")
        conv10 = tf.layers.batch_normalization(conv10, training = isTrain)    
        conv10 = tf.nn.sigmoid(conv10)           
    
    # the measurement process of ghost imaging (physical model)
    with tf.variable_scope('measurement'): 
        out_x = tf.reshape(conv10,[batch_size,img_W,img_H,1])        
        out_x = out_x/tf.reduce_max(out_x)
                
        pattern = tf.reshape(real_A,[img_W,img_H,1,num_A])
        out_y = tf.nn.conv2d(out_x,pattern,strides=[1,1,1,1],padding='VALID')
        
        # sometime the normalization helps
        mean_x, variance_x = tf.nn.moments(out_x, [0,1,2,3])
        mean_y, variance_y = tf.nn.moments(out_y, [0,1,2,3])
        out_x = (out_x - mean_x)/tf.sqrt(variance_x)
        out_y = (out_y - mean_y)/tf.sqrt(variance_y)
        # out_x = (out_x - mean_x)
        # out_y = (out_y - mean_y)
    return out_x,out_y
        
        
        
        
        
        
        
    
