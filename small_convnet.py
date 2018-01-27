# coding:utf-8
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense
from keras.layers import Activation, Flatten, Dropout
from keras.layers.core import Activation
from keras.preprocessing.image import ImageDataGenerator

model = Sequential()
model.add(Conv2D(32,(3,3),input_shape=(150,150,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
          
model.add(Conv2D(32,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2,2)))
# the model so far output 3-D feature maps (h,w,features)

# 添加2层全连接层
model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# 准备数据并进行训练
batch_size = 16
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('data/train', target_size=(150,150), class_mode='binary')
validation_generator = test_datagen.flow_from_directory('data/validation',target_size=(150,150), class_mode='binary')

model.fit_generator(train_generator, steps_per_epoch=2000, epochs=2, validation_data=validation_generator , validation_steps=800, use_multiprocessing=True)
model.save_weights('first_try.h5')