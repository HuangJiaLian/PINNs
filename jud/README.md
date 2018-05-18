## Using Neural Network to solve equation

#### Understanding `tf.gradients()`  

```python
import tensorflow as tf

######################################
# 1D condition 
a = tf.constant(2.)
# 2.0^3=8.0
b = tf.pow(a, 3)
# (a^3)' = 3a^2 = 3*2.0^2 = 12.0 
# grad equals to d(y=x^3)/dx = 3x^2 
# when x=a=2.0  
grad_1 = tf.gradients(ys=b, xs=a) # fist order derivatives
grad_2 = tf.gradients(ys=grad_1[0], xs=a) # second order derivatives 
grad_3 = tf.gradients(ys=grad_2[0], xs=a) # 
grad_4 = tf.gradients(ys=grad_3[0], xs=a) # 

######################################
```

``` python 

######################################
# 2D condition 
# w1 = [a1, a2]
w1 = tf.Variable([[1,2]])
# 1x2 * 2x1 is a number 
res = tf.matmul(w1, [[2],[1]]) 

grads = tf.gradients(res, [w1])
grads_0 = tf.gradients(res, [w1])[0]
######################################
```

$$
w1 = \{  a_1,a_2 \}
\\

f(a_1,a_2) = 
\left\{\begin{matrix}
a_1 & a_2 
\end{matrix}
\right\} 

\left\{\begin{matrix}
2 \\
1
\end{matrix}
\right\}
= 2a_1 + a_2
$$
`grads = tf.gradients(res, [w1])`: 
$$
\left[\frac{\partial f(a_1,a_2)}{\partial a_1}  , \frac{\partial f(a_1,a_2)}{\partial a_2}\right] = [2, 1]
$$
``` python 

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
```

