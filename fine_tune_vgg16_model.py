# coding:utf-8
import keras
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from vgg16 import VGG16
from keras import regularizers
from keras.utils import plot_model
import matplotlib.pyplot as plt

# dimensions of our images.
img_width, img_height = 150, 150

# path to the top classifier model weights file
top_model_weights_path = 'bottleneck_fc_model.h5'

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

# build the VGG16 network
input_tensor = Input(shape=(img_height,img_width,3)) # 当使用不包括top的VGG16时，要指定输入的shape，否则会报错
base_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
print 'Model loaded.'

# build a classifier model to put on top of the convolutional model
top_model = Sequential()
top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
top_model.add(Dense(256,activation='relu'))
top_model.add(Dropout(0.5)) # change dropout value to 0.8 can further improve accuracy 
top_model.add(Dense(1,activation='sigmoid'))

# 注意为了成功进行fine-tuning,必须从一个包括top classifier的完全训练的分类器开始
top_model.load_weights(top_model_weights_path)
# add the model on the top of the convolutional base 
# 原来此处是model.add(top_model)，但由于model是函数式模型，不能像序贯模型那样直接用model.add()方法，因此此处用Model()构造函数将两个模型连接起来
model = Model(inputs=base_model.input, outputs=top_model(base_model.output))

# set the first 15 layers(up to the last conv block) to non-trainable(weights will not be updated)
# 个人认为此处并没有25层，应该只有15层（可以从vgg16.py文件中很容易数出来）
for layer in model.layers[:15]: # set the first 11 layers(fine tune conv4 and conv5 block can also further improve accuracy
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
history=model.fit_generator(train_generator,
                    steps_per_epoch=nb_train_samples//batch_size,
                    epochs=epochs,
                    validation_data=test_generator,
                    validation_steps=nb_validation_samples//batch_size)

model.save_weights('fine_tune_vgg16_model.h5')
<<<<<<< HEAD
# 绘制模型到图片
plot_model(model, to_file='total_model.png', show_shapes=True)
# 训练过程可视化
print(history.history.keys())
fig = plt.figure()
plt.subplot(121)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.subplot(122)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower left')
fig.savefig('finetune_conv4-5_dropout0.8_performance.png')
=======
>>>>>>> df401a45e2929fb38b9d53ad2f6b0431bf933d05
