# xnor using neural nets 
#architecture 
#   2 inputs --> 2 neurons --> 1 neuron --> neuron 



import random
import math

# activation
def sig(x):
    return 1 / (1 + math.exp(-x))

# training data
X = [[0,0],[0,1],[1,0],[1,1]]
Y = [0,1,1,0]

# weights (random)
w1 = random.uniform(-1,1)
w2 = random.uniform(-1,1)

w3 = random.uniform(-1,1)
w4 = random.uniform(-1,1)

w5 = random.uniform(-1,1)
w6 = random.uniform(-1,1)

b1 = random.uniform(-1,1)
b2 = random.uniform(-1,1)
b3 = random.uniform(-1,1)

lr = 0.5

# training
for epoch in range(10000):
    for i in range(4):
        x1, x2 = X[i]
        y = Y[i]

        # forward
        h1 = sig(x1*w1 + x2*w2 + b1)
        h2 = sig(x1*w3 + x2*w4 + b2)
        out = sig(h1*w5 + h2*w6 + b3)

        # error
        err = y - out

        # backprop (explicit)
        d_out = err * out * (1 - out) #out * (1 - out), derivative of simoid funciton ]] important 
        # dL/dw = dL/dout × dout/dz × dz/dw


        d_h1 = d_out * w5 * h1 * (1 - h1)#extended differentiation for that particular neuron
        d_h2 = d_out * w6 * h2 * (1 - h2)

        # update
        # for l2 neuron 1 
        w5 += lr * d_out * h1
        w6 += lr * d_out * h2
        b3 += lr * d_out
        # for l1 neuron 1 
        w1 += lr * d_h1 * x1
        w2 += lr * d_h1 * x2
        b1 += lr * d_h1
        # for l2 neuron 2 
        w3 += lr * d_h2 * x1
        w4 += lr * d_h2 * x2
        b2 += lr * d_h2

# test
for x1, x2 in X:
    h1 = sig(x1*w1 + x2*w2 + b1)
    h2 = sig(x1*w3 + x2*w4 + b2)
    out = sig(h1*w5 + h2*w6 + b3)
    print(x1, x2, "->", round(out, 3))
