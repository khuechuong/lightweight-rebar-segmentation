# Lightweight Rebar Segmentation

This code is based on [Image Segmentation Keras](https://github.com/divamgupta/image-segmentation-keras), but is heavily modified although **the instructions to run it should be the same**.

This network uses a small number of parameters, is time efficient, and has higher accuracy than other architecture.

| Methods       |  Accuracy (%)   | Mean IoU (%)  | Params | FPS|
| :-------------|:-------------:  | :-----:|:-----:|:-----:|
| ESPnet        | 99.17           | 79.39  |14,158,566 |$1600 |
| Segnet        | 99.35           |   81.40|20,861,480 |$1600 |
| PSPnet        | 98.57           |   69.44|25,563,348  |$1600 |
| ENet          | 99.40           |   84.30|363,206|$1600 |
| Our approach  | **99.49**       |   86.25|10,898,960 |$1600 |
