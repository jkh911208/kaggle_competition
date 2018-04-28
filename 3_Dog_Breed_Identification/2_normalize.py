
# coding: utf-8

# In[ ]:


import cv2
import os
import numpy as np
from multiprocessing import Pool
from random import shuffle


# In[ ]:


class_name = np.load("numpy/class_name.npy")
num_classes = len(class_name)

data_dir = "augmented/"
for name in class_name:
    data_list = os.listdir(data_dir + name)
    shuffle(data_list)
    shuffle(data_list)
    np.save("numpy/{}_list.npy".format(name), data_list)


# In[ ]:


target = 3000
each = 20

def normalize(i):
    x_train = []
    y_train = []
    temp_train = []
    for name in class_name:
        class_dir = data_dir + name + "/"
        #print(class_dir)
        file_list = np.load("numpy/" + name + "_list.npy")
        file_list = file_list[i*each:(i+1)*each]

        for file in file_list:
            file_dir = class_dir + file
            image = cv2.imread(file_dir) 
            image = image.astype('float32')/255
            temp_train.append([image, name])
        
#         print(len(temp_train))
        
    shuffle(temp_train)
    
    y_train = [j[1] for j in temp_train]
    
    # convert to one hot encoing 
    train_hot = np.zeros([len(y_train), num_classes]).astype("uint8")

    for k in range(len(y_train)):
        j = 0
        for name in class_name:
            if y_train[k] == name:
                break
            else:
                j += 1

        train_hot[k][j] = 1   

    x_train = np.array([j[0] for j in temp_train]).reshape(-1,150,150,3)
    y_train = train_hot
    
    np.save("numpy/x_train_{}.npy".format(i) ,x_train)
    np.save("numpy/y_train_{}.npy".format(i) ,y_train)
    
    #train_hot = []
    #x_train = []
    #y_train = []
    #temp_train = []


# In[ ]:


p = Pool()
p.map(normalize, [i for i in range(int(target/each))])

