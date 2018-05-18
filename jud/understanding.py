
# coding: utf-8

# In[4]:


import tensorflow as tf

######################################
# 1D condition 
a = tf.constant(2.)
# 2.0^3=8.0
b = tf.pow(a, 3)
# (a^3)' = 3a^2 = 3*2.0^2 = 12.0 
# grad 就是 d(y=x^3)/dx = 3x^2 
# 当x=a=2.0时候的取值 
grad_1 = tf.gradients(ys=b, xs=a) # 一阶导
grad_2 = tf.gradients(ys=grad_1[0], xs=a) # 二阶导
grad_3 = tf.gradients(ys=grad_2[0], xs=a) # 三阶导
grad_4 = tf.gradients(ys=grad_3[0], xs=a) # 四阶导

# 因此tf.gradients() 返回的是一个导数值
######################################



######################################
# 2D condition 
# w1 = [a1, a2]
w1 = tf.Variable([[1,2]])
# 1x2 * 2x1 is a number 
res = tf.matmul(w1, [[2],[1]]) 
# $helllo$

grads = tf.gradients(res, [w1])
grads_0 = tf.gradients(res, [w1])[0]
######################################





init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)  
    print("1D condition: ")
    ###########################
    # 2.0^3=8.0
    print((sess.run(b)))
    # (a^3)' = 3a^2 = 3*2.0^2 = 12.0
    print(sess.run(grad_1))
    print(sess.run(grad_2))
    print(sess.run(grad_3))
    print(sess.run(grad_4))
    
    print("2D condition: ")
    ###########################
    print(sess.run(res))
    print(sess.run(grads))
    print(sess.run(grads_0))


# In[ ]:




