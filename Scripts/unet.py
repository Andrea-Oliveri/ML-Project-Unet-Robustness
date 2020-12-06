# -*- coding: utf-8 -*-
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dropout, AveragePooling2D, Conv2DTranspose, Concatenate


def unet(input_shape=(256, 256, 1), show_summary=True):
    inputs         = Input(input_shape)
    
    layers_block_1 = Conv2D(filters = 16, kernel_size = 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    layers_block_1 = Conv2D(filters = 16, kernel_size = 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(layers_block_1)
    layers_block_1 = Dropout(0.1)(layers_block_1)

    layers_block_2 = AveragePooling2D(pool_size=(2, 2))(layers_block_1)
    layers_block_2 = Conv2D(filters = 32, kernel_size = 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(layers_block_2)
    layers_block_2 = Conv2D(filters = 32, kernel_size = 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(layers_block_2)
    layers_block_2 = Dropout(0.1)(layers_block_2)

    layers_block_3 = AveragePooling2D(pool_size=(2, 2))(layers_block_2)
    layers_block_3 = Conv2D(filters = 64, kernel_size = 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(layers_block_3)
    layers_block_3 = Conv2D(filters = 64, kernel_size = 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(layers_block_3)
    layers_block_3 = Dropout(0.2)(layers_block_3)

    layers_block_4 = AveragePooling2D(pool_size=(2, 2))(layers_block_3)
    layers_block_4 = Conv2D(filters = 128, kernel_size = 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(layers_block_4)
    layers_block_4 = Conv2D(filters = 128, kernel_size = 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(layers_block_4)
    layers_block_4 = Dropout(0.2)(layers_block_4)

    layers_block_5 = Conv2DTranspose(filters = 64, kernel_size = 2, strides = (2, 2), padding = 'same')(layers_block_4)
    layers_block_5 = Concatenate()([layers_block_3, layers_block_5])
    layers_block_5 = Conv2D(filters = 64, kernel_size = 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(layers_block_5)
    layers_block_5 = Conv2D(filters = 64, kernel_size = 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(layers_block_5)
    layers_block_5 = Dropout(0.2)(layers_block_5)

    layers_block_6 = Conv2DTranspose(filters = 32, kernel_size = 2, strides = (2, 2), padding = 'same')(layers_block_5)
    layers_block_6 = Concatenate()([layers_block_2, layers_block_6])
    layers_block_6 = Conv2D(filters = 32, kernel_size = 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(layers_block_6)
    layers_block_6 = Conv2D(filters = 32, kernel_size = 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(layers_block_6)
    layers_block_6 = Dropout(0.1)(layers_block_6)
    
    layers_block_7 = Conv2DTranspose(filters = 16, kernel_size = 2, strides = (2, 2), padding = 'same')(layers_block_6)
    layers_block_7 = Concatenate()([layers_block_1, layers_block_7])
    layers_block_7 = Conv2D(filters = 16, kernel_size = 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(layers_block_7)
    layers_block_7 = Conv2D(filters = 16, kernel_size = 3, activation = 'elu', padding = 'same', kernel_initializer = 'he_normal')(layers_block_7)
    layers_block_7 = Dropout(0.1)(layers_block_7)
    
    outputs        = Conv2D(filters = 1, kernel_size = 1, activation = 'sigmoid', padding = 'same')(layers_block_7)

    model = Model(inputs = inputs, outputs = outputs)

    model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    if show_summary:
        model.summary()
    
    return model