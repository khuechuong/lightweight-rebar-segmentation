from keras.models import *
from keras.layers import *
from keras import layers
from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model
from .custom_encoder import get_custom_encoder_v1


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

"""	ESP Module	"""
def esp(input):
  #input = Input(shape=(256, 256, 2))
  """Reduce -> Split -> Transform -> Merge"""
  c = Conv2D(filters = 2, kernel_size = 1, strides = 1, data_format = IMAGE_ORDERING)(input)
  #print(f'Reduce: {c.shape}')
  d1 = CDilated(c,2,3,1,1)
  #print(f'd1: {d1.shape}')
  d2 = CDilated(c,2,3,1,2)
  #print(f'd2: {d2.shape}')
  d4 = CDilated(c,2,3,1,4)
  #print(f'd4: {d4.shape}')
  d8 = CDilated(c,2,3,1,8)
  #print(f'd8: {d8.shape}')
  d16 = CDilated(c,2,3,1,16)
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
  #output = layers.add([combine,input]) 
  output = BR(combine)
  return output#Model(input, output)

"""
Custom mod with esp but has added a max pool and reduce to the HFF
"""
def custom_mod_v1(input):
  #input = Input(shape=(256, 256, 2))
  """Reduce -> Split -> Transform -> Merge"""
  c = Conv2D(filters = 2, kernel_size = 1, strides = 1, data_format = IMAGE_ORDERING)(input)
  #print(f'Reduce: {c.shape}')
  d1 = CDilated(c,2,3,1,1)
  #print(f'd1: {d1.shape}')
  d2 = CDilated(c,2,3,1,2)
  #print(f'd2: {d2.shape}')
  d4 = CDilated(c,2,3,1,4)
  #print(f'd4: {d4.shape}')
  d8 = CDilated(c,2,3,1,8)
  #print(f'd8: {d8.shape}')
  d16 = CDilated(c,2,3,1,16)
  #print(f'd16: {d16.shape}')
  max = MaxPool2D(pool_size = (3,3), strides=1, padding='same')(c)
  #print(f'max" {max.shape}')

  add1 = d1
  add2 = layers.add([add1, d2])
  #print(f'add2: {add2.shape}')
  add3 = layers.add([add2, d4])
  #print(f'add3: {add3.shape}')
  add4 = layers.add([add3, d8])
  #print(f'add4: {add4.shape}')
  add5 = layers.add([add3, d16])
  #print(f'add5: {add4.shape}')
  add6 = layers.add([add3, max])
  #print(f'add6: {add4.shape}')
  combine = Concatenate(axis=MERGE_AXIS)([c, add1, add2, add3, add4, add5, add6])
  #print(f'combine: {combine.shape}')
  #output = layers.add([combine,input]) 
  output = BR(combine)
  return output#Model(input, output)

def custom_mod_v2(input):
  #input = Input(shape=(256, 256, 2))
  """Reduce -> Split -> Transform -> Merge"""
  c = Conv2D(filters = 2, kernel_size = 1, strides = 1, data_format = IMAGE_ORDERING)(input)
  #print(f'Reduce: {c.shape}')
  d1 = CDilated(c,2,3,1,1)
  #print(f'd1: {d1.shape}')
  d2 = CDilated(c,2,3,1,2)
  #print(f'd2: {d2.shape}')
  d4 = CDilated(c,2,3,1,4)
  #print(f'd4: {d4.shape}')
  d8 = CDilated(c,2,3,1,8)
  #print(f'd8: {d8.shape}')
  max = MaxPool2D(pool_size = (3,3), strides=1, padding='same')(input)
  #print(f'max" {max.shape}')
  max_proj = Conv2D(filters = 2, kernel_size = 1, padding='same', data_format = IMAGE_ORDERING)(max)
  #print(f'max_proj: {max_proj.shape}')

  add1 = d1
  add2 = layers.add([add1, d2])
  #print(f'add2: {add2.shape}')
  add3 = layers.add([add2, d4])
  #print(f'add3: {add3.shape}')
  add4 = layers.add([add3, d8])
  #print(f'add4: {add4.shape}')
  add5 = layers.add([add3, max_proj])
  #print(f'add6: {add4.shape}')
  combine = Concatenate(axis=MERGE_AXIS)([c, add1, add2, add3, add4, add5])
  #print(f'combine: {combine.shape}')
  #output = layers.add([combine,input]) 
  output = BR(combine)
  return output#Model(input, output)

def custom_mod_v3(input):
#input = Input(shape=(256, 256, 2))
  """Reduce -> Split -> Transform -> Merge"""
  c = CDilated(input,2,(3,1),1,2)
  c = PReLU()(c)
  c = CDilated(c,2,(1,3),1,2)
  c = BR(c)
  #print(f'Reduce: {c.shape}')
  d1 = CDilated(c,2,3,1,1)
  #print(f'd1: {d1.shape}')
  d2 = CDilated(c,2,3,1,2)
  #print(f'd2: {d2.shape}')
  d4 = CDilated(c,2,3,1,4)
  #print(f'd4: {d4.shape}')
  d8 = CDilated(c,2,3,1,8)
  #print(f'd8: {d8.shape}')
  max = MaxPool2D(pool_size = (3,3), strides=1, padding='same')(input)
  #print(f'max" {max.shape}')
  max_proj = Conv2D(filters = 2, kernel_size = 1, padding='same', data_format = IMAGE_ORDERING)(max)
  #print(f'max_proj: {max_proj.shape}')

  add1 = d1
  add2 = layers.add([add1, d2])
  #print(f'add2: {add2.shape}')
  add3 = layers.add([add2, d4])
  #print(f'add3: {add3.shape}')
  add4 = layers.add([add3, d8])
  #print(f'add4: {add4.shape}')
  add5 = layers.add([add3, max_proj])
  #print(f'add6: {add4.shape}')
  combine = Concatenate(axis=MERGE_AXIS)([c, add1, add2, add3, add4, add5])
  #print(f'combine: {combine.shape}')
  #output = layers.add([combine,input]) 
  output = BR(combine)
  return output#Model(input, output)

"""
decoder no module
"""
def custom_decoder_v1(n_classes, encoder, input_height=512, input_width=512):
    img_input, levels = encoder(input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4] = levels

    x = f4
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(f4)
    x = BR(x)
    #print(f'DeConv1: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f3])
    x = BR(x)
    #print(f'Concat5: {x.shape}')
    """
    Put a module here
    """
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv2: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f2])
    x = BR(x)
    #print(f'Concat6: {x.shape}')
    """
    Put a module here
    """
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv3: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f1])
    x = BR(x)
    #print(f'Concat7: {x.shape}')
    """
    Put a module here
    """
    x = Conv2DTranspose(filters = n_classes, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv4: {x.shape}')

    model = get_segmentation_model(img_input, x)
    return model

"""
Decoder with ESP module
"""
def custom_decoder_v2(n_classes, encoder, input_height=512, input_width=512):
    img_input, levels = encoder(input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4] = levels

    x = f4
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(f4)
    x = BR(x)
    #print(f'DeConv1: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f3])
    x = BR(x)
    #print(f'Concat5: {x.shape}')
    """
    Put a module here
    """
    x = esp(x)
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv2: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f2])
    x = BR(x)
    #print(f'Concat6: {x.shape}')
    """
    Put a module here
    """
    x = esp(x)
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv3: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f1])
    x = BR(x)
    #print(f'Concat7: {x.shape}')
    """
    Put a module here
    """
    x = esp(x)
    x = Conv2DTranspose(filters = n_classes, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv4: {x.shape}')

    model = get_segmentation_model(img_input, x)
    return model

"""
Decoder with custom module
"""
def custom_decoder_v3(n_classes, encoder, input_height=512, input_width=512):
    img_input, levels = encoder(input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4] = levels

    x = f4
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(f4)
    x = BR(x)
    #print(f'DeConv1: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f3])
    x = BR(x)
    #print(f'Concat5: {x.shape}')
    """
    Put a module here
    """
    x = custom_mod_v1(x)
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv2: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f2])
    x = BR(x)
    #print(f'Concat6: {x.shape}')
    """
    Put a module here
    """
    x = custom_mod_v1(x)
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv3: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f1])
    x = BR(x)
    #print(f'Concat7: {x.shape}')
    """
    Put a module here
    """
    x = custom_mod_v1(x)
    x = Conv2DTranspose(filters = n_classes, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv4: {x.shape}')

    model = get_segmentation_model(img_input, x)
    return model

"""
Decoder with custom v2 module
"""
def custom_decoder_v4(n_classes, encoder, input_height=512, input_width=512):
    img_input, levels = encoder(input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4] = levels

    x = f4
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(f4)
    x = BR(x)
    #print(f'DeConv1: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f3])
    x = BR(x)
    #print(f'Concat5: {x.shape}')
    """
    Put a module here
    """
    x = custom_mod_v2(x)
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv2: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f2])
    x = BR(x)
    #print(f'Concat6: {x.shape}')
    """
    Put a module here
    """
    x = custom_mod_v2(x)
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv3: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f1])
    x = BR(x)
    #print(f'Concat7: {x.shape}')
    """
    Put a module here
    """
    x = custom_mod_v2(x)
    x = Conv2DTranspose(filters = n_classes, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv4: {x.shape}')

    model = get_segmentation_model(img_input, x)
    return model


"""
Decoder with custom v3 module
"""
def custom_decoder_v5(n_classes, encoder, input_height=512, input_width=512):
    img_input, levels = encoder(input_height=input_height, input_width=input_width)
    [f1, f2, f3, f4] = levels

    x = f4
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(f4)
    x = BR(x)
    #print(f'DeConv1: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f3])
    x = BR(x)
    #print(f'Concat5: {x.shape}')
    """
    Put a module here
    """
    x = custom_mod_v3(x)
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv2: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f2])
    x = BR(x)
    #print(f'Concat6: {x.shape}')
    """
    Put a module here
    """
    x = custom_mod_v3(x)
    x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv3: {x.shape}')
    x = Concatenate(axis=MERGE_AXIS)([x, f1])
    x = BR(x)
    #print(f'Concat7: {x.shape}')
    """
    Put a module here
    """
    x = custom_mod_v3(x)
    x = Conv2DTranspose(filters = n_classes, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
    x = BR(x)
    #print(f'DeConv4: {x.shape}')

    model = get_segmentation_model(img_input, x)
    return model


def customEv1Dv1(n_classes, input_height=512, input_width=512):

    model = custom_decoder_v1(n_classes, get_custom_encoder_v1,
                  input_height=input_height, input_width=input_width)
    model.model_name = "customEv1Dv1"
    return model

def customEv1Dv2(n_classes, input_height=512, input_width=512):

    model = custom_decoder_v2(n_classes, get_custom_encoder_v1,
                  input_height=input_height, input_width=input_width)
    model.model_name = "customEv1Dv2"
    return model

def customEv1Dv3(n_classes, input_height=512, input_width=512):

    model = custom_decoder_v3(n_classes, get_custom_encoder_v1,
                  input_height=input_height, input_width=input_width)
    model.model_name = "customEv1Dv3"
    return model

def customEv1Dv4(n_classes, input_height=512, input_width=512):

    model = custom_decoder_v4(n_classes, get_custom_encoder_v1,
                  input_height=input_height, input_width=input_width)
    model.model_name = "customEv1Dv4"
    return model

def customEv1Dv5(n_classes, input_height=512, input_width=512):

    model = custom_decoder_v5(n_classes, get_custom_encoder_v1,
                  input_height=input_height, input_width=input_width)
    model.model_name = "customEv1Dv5"
    return model

if __name__ == '__main__':
    pass
