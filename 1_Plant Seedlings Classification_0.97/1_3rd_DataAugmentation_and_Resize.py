import os
import numpy as np
import cv2
from keras.preprocessing.image import ImageDataGenerator
from math import ceil
from multiprocessing import Pool

raw_dir = "plants/raw/"
file_list = os.listdir(raw_dir)
class_name = []
for name in file_list:
    if not name.startswith('.'):
        class_name.append(name)

np.save("numpy/class_name.npy", class_name)

augmented_dir = "plants/augmented_resized/"

if not os.path.exists(augmented_dir):
        os.mkdir(augmented_dir)
        
for name in class_name:
    class_dir = augmented_dir + name + "/"
    if not os.path.exists(class_dir):
        os.mkdir(class_dir)

target = 3000

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        vertical_flip=True,
        horizontal_flip=True,
        fill_mode='reflect')

def augmentation(name):
    print("Start Augmenting " + name)
    class_dir = raw_dir + name +"/"
    augmented_class_dir = augmented_dir + name + "/"
    
    file_list = os.listdir(class_dir)
    augmented_list = os.listdir(augmented_class_dir)
    
    num_augment = ceil(target / len(file_list)) + 1
    
    for file_name in file_list:
        if not file_name.endswith(".png"):
            continue

        file_dir = class_dir + file_name
        image = cv2.imread(file_dir)
        image = cv2.resize(image,(150,150))
        height, width, channel = image.shape
        
        if channel != 3:
            continue
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = np.array(image).reshape([1,height,width,channel])
    
        for i in range(num_augment):
            for gen_image in datagen.flow(x, batch_size=1):
                gen_image = gen_image.reshape([height, width, channel])
                RGB_image = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite('{}aug_{}_{}.png'.format(augmented_class_dir,file_name,i), RGB_image)
                break

    print("Finished Augmenting " + name)


# In[ ]:

p = Pool()
p.map(augmentation, [name for name in class_name])