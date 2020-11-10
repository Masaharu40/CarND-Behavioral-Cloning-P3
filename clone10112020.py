from math import ceil
import os
import csv
import cv2
import numpy as np

samples =[]
lines =[]
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)

    for line in reader:
        lines.append(line)
        
from sklearn.model_selection import train_test_split

import sklearn
from sklearn.utils import shuffle

samples = lines[1:]
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+ batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[0]
                    filename = source_path.split('/')[-1]
                    current_path = './data/IMG/'+ filename
                    image =cv2.imread(current_path)
                    images.append(image)
                    measurement = float(batch_sample[3])
                    measurements.append(measurement)
    
            augmented_images, augmented_measurements = [] , []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1.0)
    
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield sklearn.utils.shuffle(X_train, y_train)

batch_size = 32

train_samples, validation_samples = train_test_split(samples, test_size=0.2)
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x /255.0 - 0.5, input_shape=(160,320,3))) 
model.add(Cropping2D(cropping=((60,25),(0,0))))
model.add(Convolution2D(24, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(36, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(48, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(100))
model.add(Dropout(0.5))
model.add(Dense(50))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
#model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=2)

from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, steps_per_epoch = ceil(len(train_samples)/batch_size),
                                     validation_data = validation_generator, 
                                     validation_steps = ceil(len(validation_samples)/batch_size),
                                     nb_epoch=1, verbose=1)
### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

model.save('model.h5')
