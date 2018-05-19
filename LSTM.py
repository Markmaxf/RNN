
# coding: utf-8

# In[1]:


import numpy as np 
import tensorflow as tf


# In[2]:


from tensorflow.examples.tutorials.mnist import input_data


# In[3]:


# set random seed for comparing the two result calculations
tf.set_random_seed(1)

# this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# In[4]:


# the lstm have two outputs
# hidden layer h and state c, they have the same units

h_units=64

# the input data have shape [None,28*28], convert into time series data as 28 step (column of image ) 
# and each step have 28 pixel( one time step) for input  

batch_size=128             # number of sample for training 
n_step=28
n_size=28

n_class=10                 # number of output class

xs=tf.placeholder(dtype=tf.float32,shape=[batch_size,n_step*n_size])
ys=tf.placeholder(dtype=tf.float32,shape=[batch_size,n_class])


# reshape xs into time series data 
x_inputs=tf.reshape(xs,shape=[-1,n_step,n_size])             # the input size is [batch_size,max_time,,,]

lstm_cell=tf.nn.rnn_cell.BasicLSTMCell(h_units)              # number of unit in cell, which is equal to c_state_size 
                                                             # hidden units size
_init_lstm=lstm_cell.zero_state(batch_size,dtype=tf.float32)
lstm_outputs,states=tf.nn.dynamic_rnn(lstm_cell,x_inputs,initial_state=_init_lstm,time_major=False)


# In[5]:


outpus=tf.unstack(lstm_outputs,axis=1)   # the actual ouputsize is [batch_size,max_time,hidden__units]
                                         # and convert it into [max_time,batch_size,hidden_units]
prediction=outpus[-1]                    # choose last time step results as outputs  

# prediction shape is batch,hidden_units --> convert into batch_size,nclass
weight=tf.Variable(tf.random_normal([h_units,n_class]))  
bais=tf.constant(0.01,shape=[n_class])

prediction_f=tf.matmul(prediction,weight)+bais


# In[6]:


# the loss function for prediction 
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys,logits=prediction_f))


# In[7]:


train=tf.train.AdamOptimizer(1e-4).minimize(loss)


# In[8]:


# define accuray score 
accu=tf.equal(tf.argmax(ys,1),tf.argmax(prediction_f,1))
score=tf.reduce_mean(tf.cast(accu,tf.float32))


# In[9]:


sess=tf.Session()
sess.run(tf.global_variables_initializer())


# In[10]:


for i in range(2000):
    batch_xs,batch_ys=mnist.train.next_batch(batch_size)
    sess.run(train,feed_dict={xs:batch_xs,ys:batch_ys})
    if i %200==0:
        print(sess.run(score,feed_dict={xs:batch_xs,ys:batch_ys}))


# In[11]:


sess.run(tf.shape(ys))


# In[ ]:




