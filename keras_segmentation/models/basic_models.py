from keras.models import *
from keras.layers import *
import keras.backend as K
import keras
from keras import layers
from .config import IMAGE_ORDERING


def vanilla_encoder(input_height=224,  input_width=224):

    kernel = 3
    filter_size = 64
    pad = 1
    pool_size = 2

    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))

    x = img_input
    levels = []

    x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (Conv2D(filter_size, (kernel, kernel),
                data_format=IMAGE_ORDERING, padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
    x = (Conv2D(128, (kernel, kernel), data_format=IMAGE_ORDERING,
         padding='valid'))(x)
    x = (BatchNormalization())(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D((pool_size, pool_size), data_format=IMAGE_ORDERING))(x)
    levels.append(x)

    for _ in range(3):
        x = (ZeroPadding2D((pad, pad), data_format=IMAGE_ORDERING))(x)
        x = (Conv2D(256, (kernel, kernel),
                    data_format=IMAGE_ORDERING, padding='valid'))(x)
        x = (BatchNormalization())(x)
        x = (Activation('relu'))(x)
        x = (MaxPooling2D((pool_size, pool_size),
             data_format=IMAGE_ORDERING))(x)
        levels.append(x)

    return img_input, levels


# IMAGE_ORDERING = 'channels_last'
if IMAGE_ORDERING == 'channels_first':
    MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
    MERGE_AXIS = -1

def CDilated(input, filters, kernel, stride, dilate):
    output = Conv2D(filters= filters, kernel_size = kernel, strides = stride, dilation_rate = dilate, padding = 'same', data_format = IMAGE_ORDERING)(input)
    return output

def ESP_down(input, nOut):
    n = int(nOut/5)
    n1=nOut - 4*n
    c = Conv2D(filters = n, kernel_size = 3, strides = 2, padding = 'same', data_format = IMAGE_ORDERING)(input)
    #print(f'Reduce: {c.shape}')
    d1 = CDilated(c,n1,3,1,1)
    #print(f'd1: {d1.shape}')
    d2 = CDilated(c,n,3,1,2)
    #print(f'd2: {d2.shape}')
    d4 = CDilated(c,n,3,1,4)
    #print(f'd4: {d4.shape}')
    d8 = CDilated(c,n,3,1,8)
    #print(f'd8: {d8.shape}')
    d16 = CDilated(c,n,3,1,16)
    #print(f'd16: {d16.shape}')
    add1 = d2
    add2 = layers.add([add1, d4])
    #print(f'add2: {add2.shape}')
    add3 = layers.add([add2, d8])
    #print(f'add3: {add3.shape}')
    add4 = layers.add([add3, d16])
    #print(f'add4: {add4.shape}')
    combine = Concatenate(axis=MERGE_AXIS)([d1, add1, add2, add3, add4])
    #print(f'combine: {combine.shape}')
    output = BatchNormalization()(combine)
    output = PReLU()(output)
    return output

def ESP_alpha(input, nOut, add=True):
    n = int(nOut/5)
    n1=nOut - 4*n
    c = Conv2D(filters = n, kernel_size = 1, strides = 1, data_format = IMAGE_ORDERING)(input)
    #print(f'Reduce: {c.shape}')
    d1 = CDilated(c,n1,3,1,1)
    #print(f'd1: {d1.shape}')
    d2 = CDilated(c,n,3,1,2)
    #print(f'd2: {d2.shape}')
    d4 = CDilated(c,n,3,1,4)
    #print(f'd4: {d4.shape}')
    d8 = CDilated(c,n,3,1,8)
    #print(f'd8: {d8.shape}')
    d16 = CDilated(c,n,3,1,16)
    #print(f'd16: {d16.shape}')
    add1 = d2
    add2 = layers.add([add1, d4])
    #print(f'add2: {add2.shape}')
    add3 = layers.add([add2, d8])
    #print(f'add3: {add3.shape}')
    add4 = layers.add([add3, d16])
    #print(f'add4: {add4.shape}')
    combine = Concatenate(axis=MERGE_AXIS)([d1, add1, add2, add3, add4])
    #print(f'combine: {combine.shape}')
    if add == True:
        combine = layers.add([combine,input])
    #print(f'combine add: {combine.shape}')
    output = BatchNormalization()(combine)
    output = PReLU()(output)
    return output


def get_espnet_encoder(input_height=512, input_width=512):
    if IMAGE_ORDERING == 'channels_first':
        img_input = Input(shape=(3, input_height, input_width))
    elif IMAGE_ORDERING == 'channels_last':
        img_input = Input(shape=(input_height, input_width, 3))

    levels = []

    x = Conv2D(filters = 16, kernel_size = 3, strides = 2, padding='same',data_format = IMAGE_ORDERING)(img_input)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #print(f'Conv1: {x.shape}')
    pool1 = AveragePooling2D(pool_size=(3,3), strides=2, padding='same')(img_input)
    #print(f'Pool1: {pool1.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, pool1])
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #print(f'Concat1: {x.shape}')
    levels.append(x)
    #print('- ESP 1')
    x = ESP_down(x, 64)
    esp1 = x
    for i in range(0,2):
        #print(f'- ESP alpha {i+1}')
        x = ESP_alpha(x, 64, True)

    pool2 = AveragePooling2D(pool_size=(3,3), strides=2, padding='same')(pool1)
    #print(f'Pool2: {pool2.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x,esp1,pool2])
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #print(f'Concat2: {x.shape}')
    levels.append(x)
    #print('- ESP 2')
    x = ESP_down(x, 128)
    esp2 = x
    for i in range(0,8):
        #print(f'- ESP alpha {i+1}')
        x = ESP_alpha(x, 128, True)

    x = Concatenate(axis=MERGE_AXIS)([x, esp2])
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #print(f'Concat3: {x.shape}')
    x = Conv2D(filters = 2, kernel_size = 1, strides = 1, padding='same',data_format = IMAGE_ORDERING)(x)
    #print(f'Conv/Ending: {x.shape}')
    levels.append(x)
    levels.append(x)
    levels.append(x)
    return img_input, levels

