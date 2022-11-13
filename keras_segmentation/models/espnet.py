from keras.models import *
from keras.layers import *
from keras import layers
from .config import IMAGE_ORDERING
from .model_utils import get_segmentation_model
from .xception import get_Xception_encoder
from .basic_models import get_espnet_encoder

if IMAGE_ORDERING == 'channels_first':
  MERGE_AXIS = 1
elif IMAGE_ORDERING == 'channels_last':
  MERGE_AXIS = -1


def CDilated(input, filters, kernel, stride, dilate):
  output = Conv2D(filters= filters, kernel_size = kernel, strides = stride, dilation_rate = dilate, padding = 'same', data_format = IMAGE_ORDERING)(input)
  return output

def esp(input):
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
  # output = layers.add([combine,input]) change combine channels to match input
  output = BatchNormalization()(combine)
  output = PReLU()(output)
  return output

def _espnet(n_classes, encoder, input_height=512, input_width=512):
  img_input, levels = encoder(input_height=input_height, input_width=input_width)
  [f1, f2, f3, f4, f5] = levels
  x = f3
  # print(f'Decoder input: {img_input.shape}')
  x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
  x = BatchNormalization()(x)
  x = PReLU()(x)
  # print(f'Deconv1: {x.shape}')
  skip1_conv = Conv2D(filters = 2, kernel_size = 1, strides=(1, 1), data_format = IMAGE_ORDERING)(f2)
  # print(f'Skip1 Conv: {skip1_conv.shape}')
  x = Concatenate(axis=MERGE_AXIS)([skip1_conv, x])
  # print(f'Concat1: {x.shape}')
  # print('- ESP Module starts')
  x = esp(x)
  # print('- ESP Module ends')
  x = Conv2DTranspose(filters = 2, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
  x = BatchNormalization()(x)
  x = PReLU()(x)
  # print(f'Deconv2: {x.shape}')
  skip2_conv = Conv2D(filters = 2, kernel_size = 1, strides=(1, 1), data_format = IMAGE_ORDERING)(f1)
  # print(f'Skip2 Conv: {skip2_conv.shape}')
  x = Concatenate(axis=MERGE_AXIS)([skip2_conv, x])
  # print(f'Concat2: {x.shape}')
  x = Conv2D(filters = 2, kernel_size = 3, strides=(1, 1), padding='same', data_format = IMAGE_ORDERING)(x)
  x = BatchNormalization()(x)
  x = PReLU()(x)
  # print(f'Conv: {x.shape}')
  x = Conv2DTranspose(filters = n_classes, kernel_size = 2, strides=(2, 2), data_format = IMAGE_ORDERING)(x)
  # print(f'Deconv/Ending: {x.shape}')
  model = get_segmentation_model(img_input, x)
  return model

def xception_espnet(n_classes, input_height = 512, input_width= 512, encoder_level=3):
  model = _espnet(n_classes, get_Xception_encoder, input_height= input_height, input_width = input_width)
  model.model_name = "xception_espnet"
  return model

def espnet(n_classes,input_height = 512, input_width= 512):
  model = _espnet(n_classes, get_espnet_encoder, input_height=input_height, input_width=input_width)
  model.model_name = "espnet"
  return model

if __name__ == '__main__':
  m = xception_espnet(2)
  from keras.utils import plot_model
  plot_model( m , show_shapes=True , to_file='model.png')
