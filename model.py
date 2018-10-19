# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 11:40:52 2018

@author: Rahul Verma
"""

import tensorflow as tf
from layers import conv_layer,pool,fc_layer,dropout_layer
import numpy as np

def BKStart_drop(x,keep_prob,reuse):
    with tf.variable_scope('BKS',reuse=reuse):
        
        n="BKStart_"    
        x=conv_layer(x,1,32,5,n+"conv_1",1,pad='SAME')
        x=pool(x,3,2,name=n+"max_pool_1",pad='SAME',pool='max')
        x=conv_layer(x,32,32,4,n+"conv_2",1,pad='SAME')
        x=pool(x,3,2,n+"avg_pool_1",pool='avg')
        x=conv_layer(x,32,64,5,n+"conv_3",1,pad='SAME')
        x=pool(x,3,2,n+"avg_pool_2",pool='avg')
        flattened_shape = np.prod([s.value for s in x.get_shape()[1:]])
        x= tf.reshape(x, [-1,flattened_shape],name=n+'flatten')
        x=fc_layer(x,3072,activation='Relu',name=n+'FC_1')
        x=dropout_layer(x,keep_prob)
        logits=fc_layer(x,7,activation='None',name=n+'FC_2')
    return logits


def BKStart(x,reuse):
    with tf.variable_scope('BKS',reuse=reuse):
        
        n="BKStart_"    
        x=conv_layer(x,1,32,5,n+"conv_1",1,pad='SAME')
        x=pool(x,3,2,name=n+"max_pool_1",pad='SAME',pool='max')
        x=conv_layer(x,32,32,4,n+"conv_2",1,pad='SAME')
        x=pool(x,3,2,n+"avg_pool_1",pool='avg')
        x=conv_layer(x,32,64,5,n+"conv_3",1,pad='SAME')
        x=pool(x,3,2,n+"avg_pool_2",pool='avg')
        flattened_shape = np.prod([s.value for s in x.get_shape()[1:]])
        x= tf.reshape(x, [-1,flattened_shape],name=n+'flatten')
        x=fc_layer(x,2048,activation='Relu',name=n+'FC_1')
        #x=dropout_layer(x,keep_prob)
        logits=fc_layer(x,7,activation='None',name=n+'FC_2')
    return logits

def BKVGG8(x,keep_prob):
    
    n="BKVGG8_"    
    x=conv_layer(x,1,32,3,n+"conv_1",1,pad='SAME')
    x=pool(x,2,2,name=n+"max_pool_1",pad='SAME',pool='max')
    x=conv_layer(x,32,64,3,n+"conv_2",1,pad='SAME')
    x=pool(x,2,2,n+"max_pool_1",pool='max')
    x=conv_layer(x,64,128,3,n+"conv_3",1,pad='SAME')
    x=pool(x,2,2,n+"max_pool_2",pool='max')
    x=conv_layer(x,128,256,3,n+"conv_4",1,pad='SAME')
    x=conv_layer(x,256,256,3,n+"conv_5",1,pad='SAME')
    flattened_shape = np.prod([s.value for s in x.get_shape()[1:]])
    x= tf.reshape(x, [-1,flattened_shape],name=n+'flatten')
    x=fc_layer(x,256,activation='Relu',name=n+'FC_1')
    x=dropout_layer(x,keep_prob)
    x=fc_layer(x,256,activation='Relu',name=n+'FC_2')
    x=dropout_layer(x,keep_prob)
    logits=fc_layer(x,7,activation='None',name=n+'FC_3')
    return logits


def BKVGG10(x,reuse):
    with tf.variable_scope('BKS',reuse=reuse):
        n="BKVGG10_"    
        x=conv_layer(x,1,32,3,n+"conv_1",1,pad='SAME')
        x=pool(x,2,2,name=n+"max_pool_1",pad='SAME',pool='max')
        x=conv_layer(x,32,64,3,n+"conv_2",1,pad='SAME')
        x=pool(x,2,2,n+"max_pool_1",pool='max')
        x=conv_layer(x,64,128,3,n+"conv_3",1,pad='SAME')
        x=conv_layer(x,128,128,3,n+"conv_",1,pad='SAME')
        x=pool(x,2,2,n+"max_pool_2",pool='max')
        x=conv_layer(x,128,256,3,n+"conv_4",1,pad='SAME')
        x=conv_layer(x,256,256,3,n+"conv_5",1,pad='SAME')
        x=conv_layer(x,256,256,3,n+"conv_6",1,pad='SAME')
        flattened_shape = np.prod([s.value for s in x.get_shape()[1:]])
        x= tf.reshape(x, [-1,flattened_shape],name=n+'flatten')
        x=fc_layer(x,256,activation='Relu',name=n+'FC_1')
    #x=dropout_layer(x,keep_prob)
        x=fc_layer(x,256,activation='Relu',name=n+'FC_2')
    #x=dropout_layer(x,keep_prob)
        logits=fc_layer(x,7,activation='None',name=n+'FC_3')
    return logits

def BKVGG12(x):
    n="BKVGG12_"    
    x=conv_layer(x,1,32,3,n+"conv_1",1,pad='SAME')
    x=conv_layer(x,32,32,3,n+"conv_2",1,pad='SAME')
    x=pool(x,2,2,name=n+"max_pool_1",pad='SAME',pool='max')
    x=conv_layer(x,32,64,3,n+"conv_3",1,pad='SAME')
    x=conv_layer(x,64,64,3,n+"conv_4",1,pad='SAME')
    x=pool(x,2,2,n+"max_pool_2",pool='max')
    x=conv_layer(x,64,128,3,n+"conv_5",1,pad='SAME')
    x=conv_layer(x,128,128,3,n+"conv_6",1,pad='SAME')
    x=pool(x,2,2,n+"max_pool_3",pool='max')
    x=conv_layer(x,128,256,3,n+"conv_7",1,pad='SAME')
    x=conv_layer(x,256,256,3,n+"conv_8",1,pad='SAME')
    x=conv_layer(x,256,256,3,n+"conv_9",1,pad='SAME')
    flattened_shape = np.prod([s.value for s in x.get_shape()[1:]])
    x= tf.reshape(x, [-1,flattened_shape],name=n+'flatten')
    x=fc_layer(x,256,activation='Relu',name=n+'FC_1')
    x=fc_layer(x,256,activation='Relu',name=n+'FC_2')
    logits=fc_layer(x,7,activation='None',name=n+'FC_3')
    return logits

def BKVGG14(x):
    n="BKVGG14_"    
    x=conv_layer(x,1,32,3,n+"conv_1",1,pad='SAME')
    x=conv_layer(x,32,32,3,n+"conv_2",1,pad='SAME')
    x=pool(x,2,2,name=n+"max_pool_1",pad='SAME',pool='max')
    x=conv_layer(x,32,64,3,n+"conv_3",1,pad='SAME')
    x=conv_layer(x,64,64,3,n+"conv_4",1,pad='SAME')
    x=pool(x,2,2,n+"max_pool_2",pool='max')
    x=conv_layer(x,64,128,3,n+"conv_5",1,pad='SAME')
    x=conv_layer(x,128,128,3,n+"conv_6",1,pad='SAME')
    x=conv_layer(x,128,128,3,n+"conv_7",1,pad='SAME')
    x=pool(x,2,2,n+"max_pool_3",pool='max')
    x=conv_layer(x,128,256,3,n+"conv_8",1,pad='SAME')
    x=conv_layer(x,256,256,3,n+"conv_9",1,pad='SAME')
    x=conv_layer(x,256,256,3,n+"conv_10",1,pad='SAME')
    x=conv_layer(x,256,256,3,n+"conv_11",1,pad='SAME')
    flattened_shape = np.prod([s.value for s in x.get_shape()[1:]])
    x= tf.reshape(x, [-1,flattened_shape],name=n+'flatten')
    x=fc_layer(x,256,activation='Relu',name=n+'FC_1')
    x=fc_layer(x,256,activation='Relu',name=n+'FC_2')
    logits=fc_layer(x,7,activation='None',name=n+'FC_3')
    return logits

