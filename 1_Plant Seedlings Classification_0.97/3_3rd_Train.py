
import numpy as np
import os, cv2
from tqdm import tqdm

from keras import optimizers
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.applications import *


# In[2]:


dim = 150
epochs = 100
batch_size = 32 

base_model = InceptionResNetV2(input_shape=(dim, dim, 3), include_top=False, weights=None, pooling='max')
x = base_model.output
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(12, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.Adam(), metrics=['accuracy'])
model.summary()

# Convert model into JSON Format
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# In[8]:


filepath = "weights-improvement-{val_acc:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

callbacks_list = [checkpoint, learning_rate_reduction]


# In[ ]:


x_list = os.listdir("numpy/")
temp = []
for file in x_list:
    if file.startswith("x_train"):
        temp.append(file)
x_list = temp
del temp

x_test = np.load("numpy/x_test.npy")
y_test = np.load("numpy/y_test.npy")

for i in range(int(epochs/10)):
    print("Epochs : ", i+1)
    for file in x_list:
        x_train = np.load("numpy/" + file)
        y_name = "y" + file[1:]
    #     print(y_name)
        y_train = np.load("numpy/" + y_name)

        model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=int(epochs/10),
                        verbose=1,
                        validation_data=(x_test,y_test),
                        callbacks=callbacks_list)

    model.save_weights("finished_{}.hdf5".format((i+1)*int(epochs/10)))