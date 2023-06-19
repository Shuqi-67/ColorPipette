# ColorPipette

This is the interactive tool for the paper [Image-Driven Harmonious Color Palette Generation for Diverse Information Visualization (TVCG 2022)](https://ieeexplore.ieee.org/document/9969167), by Shuqi Liu, Mingtian Tao, Yifei Huang, Changbo Wang and [Chenhui Li*](http://chenhui.li/).

Generate color palette from vivid images for your visualization!

<img src="https://i.postimg.cc/X71zZt1S/image.png" width=550 height=245>

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

3. We use [quasar framework](https://quasar.dev/) to design the web application. Please follow [quasar-instructor](https://quasar.dev/start/quasar-cli) to install it.

4. Start ColorPipette!
```
yarn
yarn quasar dev
```

## Reference
Some of the code reference [Superpixel segmentation with fully convolutional networks](https://github.com/fuy34/superpixel_fcn) and [BASNet: Boundary-Aware Salient Object Detection](https://github.com/xuebinqin/BASNet).

## Citation
If this work is helpful for you, please cite
```
@article{liu2022image,
  title={Image-Driven Harmonious Color Palette Generation for Diverse Information Visualization},
  author={Liu, Shuqi and Tao, Mingtian and Huang, Yifei and Wang, Changbo and Li, Chenhui},
  journal={IEEE Transactions on Visualization \& Computer Graphics},
  number={01},
  pages={1--16},
  year={2022},
  publisher={IEEE Computer Society}
}
```

## Enjoy ColorPipette!
You can also enjoy ColorPipette at [ColorPipette](http://47.243.22.82:8080).
