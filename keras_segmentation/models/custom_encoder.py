import keras
from keras.models import *
from keras.layers import *
import tensorflow as tf
from .config import IMAGE_ORDERING

if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1

def BR(input):
    output = BatchNormalization()(input)
    output = PReLU()(output)
    return output

def CDilated(input, filters, kernel, stride, dilate):
    output = Conv2D(filters= filters, kernel_size = kernel, strides = stride, dilation_rate = dilate, padding = 'same', data_format = IMAGE_ORDERING)(input)
    return output


def get_custom_encoder_v1(input_height=512,  input_width=512):

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))

    levels = []
    x = Conv2D(filters = 32, kernel_size = 3, strides = 2, padding = 'same', data_format = IMAGE_ORDERING)(img_input)
    #print(f'Conv1_1: {x.shape}')
    x = CDilated(x, 32, 3, 1, 2)
    x = BR(x)
    #print(f'Conv1_2: {x.shape}')
    pool1 = AveragePooling2D(pool_size=(3,3), strides=2, padding='same')(img_input)
    #print(f'Pool1: {pool1.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, pool1])
    #print(f'Concat1: {x.shape}')
    levels.append(x)
    x = Conv2D(filters = 64, kernel_size = 3, strides = 2, padding = 'same', data_format = IMAGE_ORDERING)(x)
    #print(f'Conv2_1: {x.shape}')
    x = CDilated(x, 64, 3, 1, 4)
    x = BR(x)
    #print(f'Conv2_2: {x.shape}')
    pool2 = AveragePooling2D(pool_size=(3,3), strides=2, padding='same')(pool1)
    #print(f'Pool2: {pool2.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, pool2])
    #print(f'Concat2: {x.shape}')
    levels.append(x)
    x = Conv2D(filters = 128, kernel_size = 3, strides = 2, padding = 'same', data_format = IMAGE_ORDERING)(x)
    #print(f'Conv3_1: {x.shape}')
    x = CDilated(x, 128, 3, 1, 8)
    x = BR(x)
    #print(f'Conv3_2: {x.shape}')
    pool3 = AveragePooling2D(pool_size=(3,3), strides=2, padding='same')(pool2)
    #print(f'Pool3: {pool2.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, pool3])
    x = BR(x)
    #print(f'Concat3: {x.shape}')
    levels.append(x)
    x = Conv2D(filters = 256, kernel_size = 3, strides = 2, padding='same',data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'Conv4: {x.shape}')
    levels.append(x)
    return img_input, levels
