# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 16:14:16 2021

@author: krish
"""

from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import cv2
import os

json_file = open('classifier.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model= model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


    if ret==True:
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        _,mask=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
        cv2.imshow('frame',gray)
        width=28
        height=28
        dimension=(width,height)
        resized=cv2.resize(mask,dimension,interpolation=cv2.INTER_AREA)
        cv2.imshow('img',resized)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break
x_test_m=resized[:,:]
(thresh, blackAndWhiteImage) = cv2.threshold(x_test_m, 127, 255, cv2.THRESH_BINARY)

#showPic = cv2.imwrite("E:\num\dataset1\m\q\filename.jpg",blackAndWhiteImage)

status = cv2.imwrite('dataset1/m/q/filename.jpg',blackAndWhiteImage)
 
print("Image written to file-system : ",status) 
 
cv2.imshow('Black white image', blackAndWhiteImage)
cv2.imshow('Original image',x_test_m)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()









test_datagen_m = ImageDataGenerator(rescale = 1./255)
test_set_m=test_datagen_m.flow_from_directory('dataset1/m',
                                            target_size = (28, 28),
                                            batch_size = 32,
                                            class_mode = 'categorical')

predictions = loaded_model.predict_generator(test_set_m, steps=1, verbose=0)

test_imgs, test_labels = next(test_set_m)
test_labels = test_labels[:,0]

x=len(predictions)
for i in range(x):
    lst=[]
    for j in predictions[i]:
        lst.append(j)
    ind=lst.index(max(lst))
    print(ind)
    
ind=str(ind)
    
 #text speaking   
from gtts import gTTS
from playsound import playsound
import random

# The text that you want to convert to audio 
r1 = random.randint(1,1000)
r2 = random.randint(1,1000)

randfile1 = str(r2)+"randomtext"+str(r1) +".mp3"

tts = gTTS(text=ind, lang='en')
tts.save(randfile1)
playsound(randfile1)

print(randfile1)
os.remove(randfile1)
