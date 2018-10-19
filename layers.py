# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf

#WEIGHTS_INIT_STDEV=0.1

#def weight_init(weights_shape,name):
 #   weights=tf.Variable(tf.truncated_normal(weights_shape,stddev=WEIGHTS_INIT_STDEV, seed=1),dtype=tf.float32,name=name)    
 #   return weights
#
def conv_layer(x,input_channel,num_filters,kernel_size,name,strides=1,pad='SAME'):
    with tf.variable_scope(name):
        
        weight_shape=[kernel_size,kernel_size,input_channel,num_filters]
        weights=tf.get_variable('weights',weight_shape,tf.float32,tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN'))
        bias_weight=tf.get_variable('bias',[num_filters],tf.float32,tf.constant_initializer(0.0))
        
        stride=[1,strides,strides,1]
        x=tf.nn.conv2d(input=x,filter=weights,strides=stride,padding=pad)
        x=tf.nn.bias_add(x,bias_weight)
        x=tf.nn.relu(x)
        return x
        
    
   
###
#def bias_var(bias_shape,name):
   # initial = tf.constant(value=0.0,shape=bias_shape)
    #return tf.Variable(initial,name=name)

def pool(x,k_size,stride,name,pad='SAME',pool='max'):
    if pool=='max':
        X=tf.nn.max_pool(x,ksize=[1,k_size,k_size,1],strides=[1,stride,stride,1],padding=pad,name=name)
    else:
        X=tf.nn.avg_pool(x,ksize=[1,k_size,k_size,1],strides=[1,stride,stride,1],padding=pad,name=name)
    return X

def fc_layer(x,num_neurons,name,activation='None'):
    input_units =x.get_shape()[-1].value
    with tf.variable_scope(name):
        
        weight_shape=[input_units,num_neurons]
        
        weights=tf.get_variable('weights',weight_shape,tf.float32,tf.contrib.layers.variance_scaling_initializer(factor=2.0,mode='FAN_IN'))
        
        bias_weight=tf.get_variable('bias',[num_neurons],tf.float32,tf.constant_initializer(0.0))
        x=tf.matmul(x,weights)
        x=tf.nn.bias_add(x,bias_weight)
        if activation=='None':
            return x
        else:
            if activation == 'Relu':
                x=tf.nn.relu(x)
            if activation =='softmax':
                x=tf.nn.softmax(x)
                           
        return x 

def dropout_layer(X,keep_prob):
    dropout = tf.nn.dropout(X, keep_prob)
    return dropout


    
    

    
    
    
    