# ColorPipette

An interactive tool for paper *Image-Driven Harmonious Color Palette Generation for Diverse Information Visualization*.
Generate harmonious color palette for visualization from the input image.

## Prerequisites
The core code was mainly developed with python 3.6, PyTorch 1.10.0, CUDA 11, and Ubuntu 16.04.
The web application was developed with Node 12, Vue 6.14.16.

1. In the superpixel segmentation module, we make use of component connection method in [SSN](http://github.com/NVlabs/ssn_superpixels) to enforce the connectivity in superpixels. The code has been included in ```/src/flask/third_party/cython```. To compile it:
```
cd src/flask/third_party/cython/
python setup.py install --user
cd ../../../..
```

2. To start the web server based on flask, run app.py:
```
cd src/flask/
python app.py
cd ../..
```

3. We use [qarsar framework](https://quasar.dev/) to design the web application. Please follow [quasar-cli](https://quasar.dev/start/quasar-cli) to install it.

4. Start ColorPipette!
```
yarn
quasar dev
```

## Reference
Some of the code reference [Superpixel segmentation with fully convolutional networks](https://github.com/fuy34/superpixel_fcn) and [BASNet: Boundary-Aware Salient Object Detection](https://github.com/xuebinqin/BASNet).

## Enjoy ColorPipette!
You can also enjoy ColorPipette at [ColorPipette](http://47.243.22.82:8080).
