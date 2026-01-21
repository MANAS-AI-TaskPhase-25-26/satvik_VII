#   xnor using neural nets 
#   architecture 
#   2 inputs --> 2 neurons --> 1 neuron --> neuron 
#   using arrays 

import random
import math
import numpy as np

def sigmoid(x):
    return  1 / (1 + math.exp(-x))

def render_layer(x,w,b):
    return  sigmoid(np.dot(x,y)+b)
    
# training data
X = [[0,0],[0,1],[1,0],[1,1]]
Y = [0,1,1,0]

wl1 =[[random.uniform(-1,1),random.uniform(-1,1)],
      [random.uniform(-1,1),random.uniform(-1,1)]]

wl2 =[[random.uniform(-1,1),random.uniform(-1,1)],]

bl1=[random.uniform(-1,1),random.uniform(-1,1)]
bl2=random.uniform(-1,1)
x=[]
l1_output=[2]
for epoc in range():
    for i in range (4):
        x=X[i]
        y=Y[i]

        #neuron activation
        activation1_1=render_layer(x,wl1[1],bl1[1])
        activation1_2=render_layer(x,wl1[2],bl1[2])

        xl2=[activation1_1,activation1_2]
        activation2_1=render_layer(xl2,wl2,bl2)

        err= activation2_1-
        #calculating slope 
        d_out   =
        dh1     =
        dh2     =


        #updating weights 
        #testing weight accuracy 