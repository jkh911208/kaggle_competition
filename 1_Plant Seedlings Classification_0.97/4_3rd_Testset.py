
# coding: utf-8

# In[5]:


import os
import numpy as np
import cv2
from random import shuffle


# In[6]:


class_name = np.load("numpy/class_name.npy")
num_classes = len(class_name)


# In[8]:


raw_dir = "plants/raw/"
temp = []

for name in class_name:
    class_dir = raw_dir + name + "/"
    file_list = os.listdir(class_dir)
    shuffle(file_list)
    file_list = file_list[:50]
    
    for file in file_list:
        if not file.endswith(".png"):
            continue

        file_dir = class_dir + file
        image = cv2.imread(file_dir) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (150,150))
        image = image.astype('float32')/255
        temp.append([image, name])
        
y_test = [j[1] for j in temp]
train_hot = np.zeros([len(y_test), num_classes]).astype("uint8")

for k in range(len(y_test)):
    j = 0
    for name in class_name:
        if y_test[k] == name:
            break
        else:
            j += 1

    train_hot[k][j] = 1   

y_test = train_hot
x_test = np.array([j[0] for j in temp]).reshape(-1,150,150,3)

np.save("numpy/x_test.npy" ,x_test)
np.save("numpy/y_test.npy" ,y_test)

