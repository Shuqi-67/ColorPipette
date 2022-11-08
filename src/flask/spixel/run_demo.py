import argparse
import os
import torch.backends.cudnn as cudnn
from spixel import models
import torchvision.transforms as transforms
from spixel import flow_transforms
import imageio
import numpy as np
import cv2
from spixel.train_util import *
import torch.nn.functional as F

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))

parser = argparse.ArgumentParser(description='PyTorch SPixelNet inference on a folder of imgs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_dir', metavar='DIR', default='./imgs/inputs', help='path to images folder')
parser.add_argument('--data_suffix', default='jpg', help='suffix of the testing image')
parser.add_argument('--pretrained', metavar='PTH', help='path to pre-trained sal_model',
                    default='./pretrain_ckpt/SpixelNet_bsd_ckpt.tar')
parser.add_argument('--output', metavar='DIR', default='./imgs', help='path to output folder')

parser.add_argument('--downsize', default=16, type=float, help='superpixel grid cell, must be same as training setting')

parser.add_argument('-nw', '--num_threads', default=1, type=int, help='num_threads')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size')

args = parser.parse_args()

def label2rgb(label_field, image, kind='mix', bg_label=-1, bg_color=0):
    # std_list = list()
    out = np.zeros_like(image)
    labels = np.unique(label_field)
    bg = (labels == bg_label)
    if bg.any():
        labels = labels[labels != bg_label]
        mask = (label_field == bg_label).nonzero()
        out[mask] = bg_color
    for label in labels:
        mask = (label_field == label).nonzero()
        # std = np.std(image[mask])
        # std_list.append(std)
        if kind == 'avg':
            color = image[mask].mean(axis=0)
        elif kind == 'median':
            color = np.median(image[mask], axis=0)
        elif kind == 'mix':
            std = np.std(image[mask])
            if std < 20:
                color = image[mask].mean(axis=0)
            elif 20 < std < 40:
                mean = image[mask].mean(axis=0)
                median = np.median(image[mask], axis=0)
                color = 0.5 * mean + 0.5 * median
            elif 40 < std:
                color = np.median(image[mask],axis=0)
        out[mask] = color
    return out

def load_model(model_dir):
    # create sal_model
    network_data = torch.load(model_dir, map_location='cpu')
    # print("=> using pre-trained sal_model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](data=network_data)
    model.eval()
    args.arch = network_data['arch']
    cudnn.benchmark = True

    return model

@torch.no_grad()
def test_single(model, img_path):
    #settings

    downsize=16
    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    load_path = img_path
    imgId = os.path.basename(img_path)[:-4]

    # may get 4 channel (alpha channel) for some format
    img_ = imageio.imread(load_path)[:, :, :3]
    H, W, _ = img_.shape
    rate=H/W
    # resize image
    min_=256
    if min(H,W)>256:
      if H<W:
          H=min_
          W=H/rate
      else:
          W=min_
          H=W*rate

    H_, W_ = int(np.ceil(H / 16.) * 16), int(np.ceil(W / 16.) * 16)
    # img_=cv2.resize(img_,(160,160))
    img_ = cv2.resize(img_, (W_, H_))

    # get spixel id
    n_spixl_h = int(np.floor(H_ / downsize))
    n_spixl_w = int(np.floor(W_ / downsize))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    # 填充周围16*16范围
    spix_idx_tensor = np.repeat(
        np.repeat(spix_idx_tensor_, downsize, axis=1), downsize, axis=2)

    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float)

    n_spixel = int(n_spixl_h * n_spixl_w)

    img = cv2.resize(img_, (W_, H_), interpolation=cv2.INTER_CUBIC)
    img1 = input_transform(img)
    ori_img = input_transform(img_)

    # compute output
    output = model(img1.unsqueeze(0))

    # assign the spixel map
    curr_spixl_map = update_spixl_map(spixeIds, output)
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(H_, W_), mode='nearest').type(torch.int)

    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.unsqueeze(0).dtype).view(3, 1, 1)
    spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(),
                                                    n_spixels=n_spixel, b_enforce_connect=True)

    label = (spixel_label_map + 1).astype(int)

    rgb_ori_img = (ori_img + mean_values).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    adp_rgb = label2rgb(label, rgb_ori_img, kind='mix')

    return label,adp_rgb
