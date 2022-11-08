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



1. Run ./src/flask/app.py to build a web server based on flask.
2. In the superpixel segmentation module, we 

## Install the dependencies
```bash
yarn
```

### Start the app in development mode (hot-code reloading, error reporting, etc.)
```bash
quasar dev
```

### Lint the files
```bash
yarn run lint
```

### Build the app for production
```bash
quasar build
```
