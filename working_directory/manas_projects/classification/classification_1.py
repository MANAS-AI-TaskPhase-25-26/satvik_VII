import pandas as panda
import numpy as np


def inspect(var, name="variable"):
    print(f"Name : {name}")
    print(f"Type : {type(var)}")

    if hasattr(var, "shape"):
        print(f"Shape: {var.shape}")
    else:
        print("Shape: N/A")

    print("Value:")
    print(var)
    print("-" * 40)


def scaler(x):#gpt
    
    x_scaled = np.zeros_like(x, dtype=float)

    for j in range(x.shape[1]):   # loop over features
        mean = np.mean(x[:, j])
        std = np.std(x[:, j])

        for i in range(x.shape[0]):   # loop over samples
            x_scaled[i, j] = (x[i, j] - mean) / std
    return x_scaled


def generate(z,x):
    i=0
    while(i<=2):
        r=np.random.randint(0,209)
        z[i]=x[r]
        i+=1
    return z

def assign(z,x):

    y= np.zeros(210,dtype=int)
    count_x=0

    while(count_x<210):

        z0=np.linalg.norm(z[0]-x[count_x])
        z1=np.linalg.norm(z[1]-x[count_x])
        z2=np.linalg.norm(z[2]-x[count_x])

        if((z1<z2)&(z1<z0)):
            y[count_x]=1
        elif((z0<z2)&(z0<z1)):
            y[count_x]=0
        elif((z2<z0)&(z2<z1)):
            y[count_x]=2
        count_x+=1
    return y

def cluster_centroid(x,y):

    cluster_0 = np.array([], dtype=int)
    cluster_1 = np.array([], dtype=int)
    cluster_2 = np.array([], dtype=int)


    counter=0
    while(counter<210):
        if(y[counter]==0):
            cluster_0=np.append(cluster_0,counter)
        elif(y[counter]==1):
            cluster_1=np.append(cluster_1,counter)
        elif(y[counter]==2):
            cluster_2=np.append(cluster_2,counter)
        counter+=1

    m0=np.mean(x[cluster_0],axis=0)
    m1=np.mean(x[cluster_1],axis=0)
    m2=np.mean(x[cluster_2],axis=0)

    z=np.array([m0,m1,m2])
    return z

data = panda.read_csv("/Users/satviksingh/Documents/manas_projects/classification/seeds_data.csv")

x_loader = data.drop(['Class'],axis=1)
y_loader = data['Class']

x = x_loader.values
y_checker = y_loader.values
y = np.zeros((y_checker.shape))
z = np.zeros((3,7))
x=scaler(x)
z=generate(z,x)
counter=0

while(counter<=10):
    y=assign(z,x)
    z=cluster_centroid(x,y)
    counter+=1

## gpt generated code below
# y           -> K-means cluster labels (0,1,2)
# y_checker   -> true labels (1,2,3)

y_relabel = np.zeros_like(y)

for cluster_id in [0, 1, 2]:

    # collect true labels belonging to this cluster
    labels_in_cluster = y_checker[y == cluster_id]

    # manual majority count
    count_1 = 0
    count_2 = 0
    count_3 = 0

    for label in labels_in_cluster:
        if label == 1:
            count_1 += 1
        elif label == 2:
            count_2 += 1
        elif label == 3:
            count_3 += 1

    # find majority label
    if count_1 >= count_2 and count_1 >= count_3:
        majority_label = 1
    elif count_2 >= count_1 and count_2 >= count_3:
        majority_label = 2
    else:
        majority_label = 3

    # relabel cluster
    for i in range(len(y)):
        if y[i] == cluster_id:
            y_relabel[i] = majority_label

correct = 0
total = len(y_checker)

for i in range(total):
    if y_relabel[i] == y_checker[i]:
        correct += 1

accuracy = correct / total
print("Accuracy:", accuracy)
