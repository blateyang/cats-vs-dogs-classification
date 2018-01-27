# coding:utf-8
import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from vgg16 import VGG16
import vgg16

# dimensions of our images.
img_width, img_height = 150, 150

# path to the top classifier model weights file
top_model_weights_path = 'bottleneck_fc_model.h5'

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 2
batch_size = 16

# build the VGG16 network
input_tensor = Input(shape=(img_height,img_width,3)) # 当使用不包括top的VGG16时，要指定输入的shape，否则会报错
base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
print 'Model loaded.'

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256,activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1,activation='sigmoid'))

# 注意为了成功进行fine-tuning,必须从一个包括top classifier的完全训练的分类器开始
top_model.load_weights(top_model_weights_path)
# add the model on the top of the convolutional base
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# set the first 25 layers(up to the last conv block) to non-trainable(weights will not be updated)
for layer in model.layers[:15]:
    layer.trainable = False
# compile the model with a SGD/momentum optimizer and a very slow learning rate
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4,momentum=0.9),
              metrics=['accuracy'])

# prepare data augmentation configuration
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    target_size=(img_height,img_width),
                                                    batch_size=batch_size,
                                                    class_mode='binary')
test_generator = test_datagen.flow_from_directory(validation_data_dir,
                                                  target_size=(img_height,img_width),
                                                  batch_size=batch_size,
                                                  class_mode='binary')

# fine-tune the model
model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples//batch_size,
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=nb_validation_samples//batch_size)

model.save_weights('fine_tune_vgg16_model.h5')