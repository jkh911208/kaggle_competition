
# coding: utf-8

# In[1]:


import numpy as np
import os, cv2

from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications import *


# In[ ]:


dim = 150
epochs = 150
batch_size = 32 

base_model = InceptionResNetV2(input_shape=(dim, dim, 3), include_top=False, weights=None, pooling='max')
x = base_model.output
x = Dense(1024, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(120, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.Adam(), metrics=['accuracy'])
model.summary()


# In[ ]:


filepath = "weights-improvement-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=6, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks_list = [checkpoint]


# In[ ]:
print("debug_1")
temp = []
x_list = sorted(os.listdir("numpy/"))
for file in x_list:
	if file.startswith("x_train"):
		temp.append(file)

x_list = temp

for i in range(int(epochs/10)):
    for file in x_list:
        print("training on " + file)
        x_train = np.load("numpy/" + file)
        y_name = "y" + file[1:]
        y_train = np.load("numpy/" + y_name)

        model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=int(epochs/10),
                        verbose=1,
                         validation_split=0.2,
                        callbacks=callbacks_list)
    # Save the trained weights in to .h5 format
    model.save_weights("finished_{}.hdf5".format((i+1)*int(epochs/10)))