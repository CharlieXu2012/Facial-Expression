# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 12:34:45 2018

@author: Rahul Verma
"""
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import model
from data import create_preprocess_data
from batchup import data_source

train_path="C:\\Users\\Rahul Verma\\Desktop\\Facial Expression\\Train_img\\"
public_test_path="C:\\Users\\Rahul Verma\\Desktop\\Facial Expression\\Test_img\\public\\"
private_test_path="C:\\Users\\Rahul Verma\\Desktop\\Facial Expression\\Test_img\\private\\"



X,Y=create_preprocess_data(train_path)
from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val=train_test_split(X,Y,test_size=0.1)
#X_test,Y_test=create_preprocess_data(public_test_path)


BATCH_SIZE=256


def loss(pred,targets):
    lossess = -tf.reduce_sum(targets * tf.log(pred + 1e-9))
    
    return lossess

def build_model(input_data_tensor,input_label_tensor):
    logits_train=model.BKVGG10(input_data_tensor,reuse=False)
    probs=tf.nn.softmax(logits_train)
    predict=tf.argmax(logits_train,1)
    #y_true=tf.arg_max(input_label_tensor,1)
    #onehot_labels = tf.one_hot(indices=tf.cast(input_label_tensor, tf.int32), depth=45)
    lossess=loss(probs,input_label_tensor)
    correct_prediction=tf.equal(predict,tf.argmax(input_label_tensor,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
   
    return lossess,accuracy,predict




BATCH_SIZE=256
#batch_size = tf.placeholder(tf.int64)
#keep_prob =tf.placeholder(tf.float32)
input_data_tensor=tf.placeholder(tf.float32,[None,42,42,1])
input_label_tensor=tf.placeholder(tf.float32,[None,7])



losess,accuracy,predict=build_model(input_data_tensor,input_label_tensor)
vars  = tf.trainable_variables() 
lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars
                    if 'bias' not in v.name ]) * 0.001

reg_loss=losess+lossL2
global_step = tf.Variable(0, trainable=False)
#learn_rate = tf.train.exponential_decay(1e-3,global_step, 1000, 0.96, staircase=False)
train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(reg_loss)

config = tf.ConfigProto(allow_soft_placement = True)

sess = tf.Session(config = config)

t_loss=[]
v_loss=[]
t_acc=[]
v_acc=[]
pred_cls=[]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    
    for epoch in range(100):
        total_loss=0
        acc_train=0
        b_t=0
        b_v=0
        #sess.run(train_init_op,feed_dict={input_data_tensor:X_train,input_label_tensor:Y_train,batch_size:BATCH_SIZE,keep_prob:0.5})
        ds = data_source.ArrayDataSource([X_train,Y_train])
        for (batch_X, batch_y) in ds.batch_iterator(batch_size=256, shuffle=np.random.RandomState(15)):
            b_t+=1
            _,loss_value,accuracy_train=sess.run([train_step,reg_loss,accuracy],feed_dict={input_data_tensor:batch_X,input_label_tensor:batch_y})
            total_loss+=loss_value
            acc_train+=accuracy_train  
        t_loss.append(total_loss)
        t_acc.append(acc_train)
        Val_loss=0
        Val_acc=0
        
        
        ds1 = data_source.ArrayDataSource([X_val,Y_val])
        for (batch_X1, batch_y1) in ds1.batch_iterator(batch_size=256,shuffle=np.random.RandomState(145)):
            
            val_loss,val_acc=sess.run([reg_loss,accuracy],feed_dict={input_data_tensor:batch_X1,input_label_tensor:batch_y1})
            Val_loss+=val_loss
            Val_acc+=val_acc
            b_v+=1
        v_loss.append(Val_loss)
        v_acc.append(Val_acc)
                
        print("Iter:{},train_Loss:{:.4f},accuracy_train:{:.4f}".format(epoch,total_loss/b_t,acc_train/b_t))
        print("Iter:{},val_Loss:{:.4f},accuracy_val:{:.4f}".format(epoch,Val_loss/b_v,Val_acc/b_v))
        #print("Iter:{},accuracy_test:{:.4f}".format(epoch,Test_acc/n_batches_test))
        #Test_acc=0
        #for k in range(n_batches_test):
         #   sess.run(iter.initializer,feed_dict={input_data_tensor:X_test,input_label_tensor:Y_test,batch_size:BATCH_SIZE})
          #  _,pred,test_acc=sess.run([train_step,predict,accuracy])
            
           # Test_acc+=test_acc
    
  #  print("accuracy_test:{:.4f}".format(Test_acc/n_batches_test))

   # sess.run(iter.initializer,feed_dict={input_data_tensor:X_v,input_label_tensor:Y_v,batch_size:X_v.shape[0]})
    
   # _,pred=sess.run([train_step,predict])
    #pred_cls.append(pred)
    
    saver = tf.train.Saver()
    save_model_path = "C:\\Users\\Rahul Verma\\Desktop\\Facial Expression\\"
    save_path = saver.save(sess=sess, save_path=save_model_path+"BKS_model.ckpt")
    
    
    print("Training finished")
    
    
#pred_cls[0].shape   

#x_range=[x for x in range(0,100,1)]    
#plt.plot(x_range,t_loss,'-b', label='Training')
#plt.plot(x_range,v_loss,'-g', label='Validation')
#plt.legend(loc='lower right', frameon=False)
#plt.ylim(ymax = 1.0, ymin = 0.0)
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.show()    
#sess=tf.Session()
#pred_cls
#sess.run(tf.argmax(Y_v,1))


