import numpy as np
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Activation, Dropout, Flatten, Input, Dense, add, BatchNormalization
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.datasets import mnist


(x_train, y_train),(x_test,y_test) = mnist.load_data()

lbl_unique = len(np.unique(y_train))
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

image_size = x_train.shape[1]
input_shape = (image_size, image_size, 1)

x_train = np.reshape(x_train, [-1,image_size,image_size,1]) / 255
x_test = np.reshape(x_test, [-1,image_size,image_size,1]) / 255 

filters = 64
kernel_size = 3
batch_size = 200

save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cnn_resnet.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

model_checkpoint = ModelCheckpoint(filepath=filepath, monitor='val_acc', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(monitor = 'accuracy', patience = 10)
call_backs = [model_checkpoint,early_stopping]

#Model layers:
input_d = Input(shape = input_shape)
x = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', kernel_initializer='he_normal', activation = 'relu')(input_d)
x = BatchNormalization()(x)
x = Activation('relu')(x)

y = Conv2D(filters = filters, kernel_size = kernel_size, padding = 'same', kernel_initializer='he_normal', activation = 'relu')(x)
y = BatchNormalization()(y)
y = Activation('relu')(y)

x = add([x, y])

x = Flatten()(x)
output_D = Dense(lbl_unique, activation='softmax', kernel_initializer='he_normal')(x)

model = Model(input_d,output_D)
model.summary()
plot_model(model, to_file='mnist_CNN_ResNet.png', show_shapes=True)

model.compile(optimizer = 'adam', metrics = ['acc'], loss = 'categorical_crossentropy')
model.fit(x_train, y_train, batch_size=batch_size, verbose=1, epochs=10, validation_data=(x_test, y_test), shuffle=True, callbacks=call_backs)

_, acc = model.evaluate(x_test,y_test, batch_size = batch_size, verbose = 0)

print("\nTesting accuracy: %.1f%%" % (100.0 * acc))