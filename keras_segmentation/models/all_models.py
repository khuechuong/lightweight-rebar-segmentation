from . import pspnet
from . import segnet
from . import unet
from . import fcn
from . import espnet
from . import custom_decoder
from . import enet_d
model_from_name = {}

#FCN
model_from_name["fcn_8"] = fcn.fcn_8
model_from_name["fcn_32"] = fcn.fcn_32
model_from_name["fcn_8_vgg"] = fcn.fcn_8_vgg
model_from_name["fcn_32_vgg"] = fcn.fcn_32_vgg
model_from_name["fcn_8_resnet50"] = fcn.fcn_8_resnet50
model_from_name["fcn_32_resnet50"] = fcn.fcn_32_resnet50
model_from_name["fcn_8_mobilenet"] = fcn.fcn_8_mobilenet
model_from_name["fcn_32_mobilenet"] = fcn.fcn_32_mobilenet


#PSPNET
model_from_name["pspnet"] = pspnet.pspnet
model_from_name["vgg_pspnet"] = pspnet.vgg_pspnet
#####################################################
model_from_name["vgg19_pspnet"] = pspnet.vgg19_pspnet
model_from_name["xception_pspnet"] = pspnet.xception_pspnet
model_from_name["inception_v3_pspnet"] = pspnet.inception_v3_pspnet
#####################################################
model_from_name["resnet50_pspnet"] = pspnet.resnet50_pspnet

model_from_name["vgg_pspnet"] = pspnet.vgg_pspnet
model_from_name["resnet50_pspnet"] = pspnet.resnet50_pspnet

model_from_name["pspnet_50"] = pspnet.pspnet_50
model_from_name["pspnet_101"] = pspnet.pspnet_101
#model_from_name["mobilenet_pspnet"] = pspnet.mobilenet_pspnet

#UNET
model_from_name["unet_mini"] = unet.unet_mini
model_from_name["unet"] = unet.unet
model_from_name["vgg_unet"] = unet.vgg_unet
#####################################################
model_from_name["vgg19_unet"] = unet.vgg19_unet
model_from_name["xception_unet"] = unet.xception_unet
model_from_name["inception_v3_unet"] = unet.inception_v3_unet
#####################################################
model_from_name["resnet50_unet"] = unet.resnet50_unet
model_from_name["mobilenet_unet"] = unet.mobilenet_unet

#SEGNET
model_from_name["segnet"] = segnet.segnet
model_from_name["whole_segnet"] = segnet.big_segnet
model_from_name["vgg_segnet"] = segnet.vgg_segnet
#####################################################
model_from_name["vgg19_segnet"] = segnet.vgg19_segnet
model_from_name["xception_segnet"] = segnet.xception_segnet
model_from_name["inception_v3_segnet"] = segnet.inception_v3_segnet
#####################################################
model_from_name["resnet50_segnet"] = segnet.resnet50_segnet
model_from_name["mobilenet_segnet"] = segnet.mobilenet_segnet

#ESPNET
model_from_name["espnet"] = espnet.espnet
model_from_name["xception_espnet"] = espnet.xception_espnet

#Custom
model_from_name["customEv1Dv1"] = custom_decoder.customEv1Dv1
model_from_name["customEv1Dv2"] = custom_decoder.customEv1Dv2
model_from_name["customEv1Dv3"] = custom_decoder.customEv1Dv3
model_from_name["customEv1Dv4"] = custom_decoder.customEv1Dv4
model_from_name["customEv1Dv5"] = custom_decoder.customEv1Dv5

#ENET
model_from_name["enet"] = enet_d.enet
