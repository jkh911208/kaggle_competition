
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
from multiprocessing import Pool
import cv2


# In[2]:


label = pd.read_csv("labels.csv")


# In[5]:


label_unique = list(set(list(label["breed"])))


# In[10]:


class_name = label_unique
class_name = sorted(class_name)
print(class_name)

# In[9]:

if not os.path.exists("numpy"):
        os.mkdir("numpy")
np.save("numpy/class_name.npy", label_unique)


# In[14]:


for name in class_name:
    dir_path = "train/" + name + "/"
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


# In[15]:


file = list(label["id"])
label = list(label["breed"])


# In[17]:


for i in range(len(file)):
    file_name = file[i]
    label_name = label[i]
    
    original_path = "train/" + file_name + ".jpg"
    new_path = "train/" + label_name + "/" + file_name + ".jpg"
    if os.path.exists(original_path):
        os.rename(original_path, new_path)


# In[24]:


augmented_dir = "augmented/"

if not os.path.exists(augmented_dir):
        os.mkdir(augmented_dir)
        
for name in class_name:
    class_dir = augmented_dir + name + "/"
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)


# In[32]:


target = 3000
raw_dir = "train/"

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=False,
        horizontal_flip=True,
        fill_mode='constant',
        cval=0)

def augmentation(name):
    print("Start Augmenting " + name)
    class_dir = raw_dir + name +"/"
    augmented_class_dir = augmented_dir + name + "/"
    
    file_list = os.listdir(class_dir)    
    num_augment = ceil(target / len(file_list))
    
    for file_name in file_list:
        file_dir = class_dir + file_name
        image = cv2.imread(file_dir)
        
        if image is None or image.shape[2] != 3:
            continue
        
        height, width, channels = image.shape
        max_val = max(height, width)
        square= np.zeros((max_val,max_val,3), np.uint8)
        height_from, height_to = int((max_val-height)/2), int(max_val-(max_val-height)/2)
        width_from, width_to = int((max_val-width)/2), int(max_val-(max_val-width)/2)
        square[height_from:height_to, width_from:width_to] = image
        image = square
        image = cv2.resize(image,(150,150))
        
        x = np.array(image).reshape([1,150,150,3])
    
        for i in range(num_augment):
            for gen_image in datagen.flow(x, batch_size=1):
                gen_image = gen_image.reshape([150, 150, 3])
                cv2.imwrite('{}aug_{}_{}.png'.format(augmented_class_dir,file_name,i), gen_image)
                break

    augmented_list = os.listdir(augmented_class_dir)
    if len(augmented_list) < target:
        for i in range(target - len(augmented_list)-1):
            file_dir = augmented_class_dir + augmented_list[i]
            image = cv2.imread(file_dir)

            for gen_image in datagen.flow(x, batch_size=1):
                gen_image = gen_image.reshape([150, 150, 3])
                cv2.imwrite('{}aug_{}_{}.png'.format(augmented_class_dir,augmented_list[i],i), gen_image)
                break

    print("Finished Augmenting " + name)


# In[ ]:


p = Pool()
p.map(augmentation, [name for name in class_name])

