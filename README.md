# Lightweight Rebar Segmentation

This code is based on [Image Segmentation Keras](https://github.com/divamgupta/image-segmentation-keras), but is heavily modified although **the instructions to run it should be the same**.

This network uses a small number of parameters, is time efficient, and has higher accuracy than other architecture.

| Methods       |  Accuracy (%)   | Mean IoU (%)  | Params | FPS|
| :-------------: |:-------------:  | :-----:|:-----:|:-----:|
| ESPnet        | 99.17           | $1600 |$1600 |$1600 |
| Segnet        | 99.35           |   $12 |$1600 |$1600 |
| PSPnet        | 98.57           |    $1 |$1600 |$1600 |
| ENet          | 99.40           |    $1 |$1600 |$1600 |
| Our approach  | **99.49**       |    $1 |$1600 |$1600 |
