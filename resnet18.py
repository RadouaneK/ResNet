import numpy as np
import keras
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.datasets import cifar10

def ID_block(X, f, filters, stage, block):
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2 = filters
    
    # Save the input value
    X_shortcut = X
    
    # First component
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F1, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2a')(X)
    
    # Second component
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)

    # Add shortcut value 
    X = Add()([X, X_shortcut])
    
    return X

def CONV_block(X, f, filters, stage, block, s = 2):
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2 = filters
    
    X_shortcut = X

    # First component  
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Conv2D(F1, (f, f), strides = (s,s), name = conv_name_base + '2a', padding='same')(X)
    
    # Second component
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(F2, (f, f), strides = (1,1), name = conv_name_base + '2b',padding='same')(X)

    X_shortcut = Conv2D(F2, (1, 1), strides = (s,s), name = conv_name_base + '1', padding='same')(X_shortcut)
    #X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Add shortcut value 
    X = Add()([X, X_shortcut])
    
    return X


def ResNet18(input_shape, classes = 10):

    X_input = Input(input_shape)

    # Stage 1
    X = Conv2D(32, (3, 3), strides = (1, 1), name = 'conv1', padding='same')(X_input)

    # Stage 2 
    X = CONV_block(X, f = 3, filters = [32, 32], stage = 2, block='a', s = 1)
    X = ID_block(X, f = 3, filters =[32, 32], stage=2, block='b')

    # Stage 3
    X = CONV_block(X, f = 3, filters = [64, 64], stage = 3, block='a', s = 2)
    X = ID_block(X, f = 3, filters =[64, 64], stage=3, block='b')

    # Stage 4
    X = CONV_block(X, f = 3, filters = [128, 128], stage = 4, block='a', s = 2)
    X = ID_block(X, f = 3, filters =[128, 128], stage=4, block='b')

    # Stage 5
    X = CONV_block(X, f = 3, filters = [256, 256], stage = 5, block='a', s = 2)
    X = ID_block(X, f = 3, filters =[256, 256], stage=5, block='b')

    X = BatchNormalization(axis = 3, name = 'bn_conv6')(X)
    X = Activation('relu')(X)

    X = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    model = Model(inputs = X_input, outputs = X, name='ResNet18')

    return model

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


model = ResNet18(input_shape = (32, 32, 3), classes = 10)
model.summary()

Tboard = keras.callbacks.TensorBoard(log_dir='./GraphResnet18', histogram_freq=0, write_graph=True,write_images = True)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,
              batch_size=32,
              epochs=1000,
              validation_data=(x_test, y_test),
              callbacks = [Tboard],
              shuffle=True)

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('ResNet18.h5')
