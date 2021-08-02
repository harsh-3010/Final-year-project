# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 11:40:16 2021

@author: Krishna
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.models import model_from_json

classifier=Sequential()
classifier.add(Convolution2D(64,3,3, input_shape = (28, 28, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Flatten())

classifier.add(Dense(output_dim = 256, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation='softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy' , metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset1/train',
                                                 target_size = (28, 28),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset1/test',
                                            target_size = (28, 28),
                                            batch_size = 32,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         samples_per_epoch = 60000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 10000)


model_json=classifier.to_json()
with open("classifier.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")


test_datagen_m = ImageDataGenerator(rescale = 1./255)
test_set_m=test_datagen_m.flow_from_directory('dataset1/m',
                                            target_size = (28, 28),
                                            batch_size = 32,
                                            class_mode = 'categorical')



predictions = classifier.predict_generator(test_set_m, steps=1, verbose=0)

test_imgs, test_labels = next(test_set_m)
test_labels = test_labels[:,0]

x=len(predictions)
for i in range(x):
    lst=[]
    for j in predictions[i]:
        lst.append(j)
    ind=lst.index(max(lst))
    print(ind)