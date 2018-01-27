# coding:utf-8
import keras 
from scipy import misc
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40, 
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest') # shear_range涉及的是shear mapping,包括水平和垂直shear,水平shear:y不变，x'=x+m*x,垂直shear同理

img = misc.imread('data/train/cats/cat.1.jpg')
x = img.reshape((1,)+img.shape)

i = 0
for batch in datagen.flow(x,batch_size=1,save_to_dir='preview',save_prefix='cat',save_format='jpg'):
    i +=1
    if i > 20:
        break
