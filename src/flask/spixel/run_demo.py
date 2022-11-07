import argparse
import os
import torch.backends.cudnn as cudnn
from spixel import models
import torchvision.transforms as transforms
from spixel import flow_transforms
import imageio
from spixel.loss import *
import time
import random
from glob import glob
from PIL import Image
from skimage.color import rgb2hsv, rgb2lab\
  #, rgb2grey
import numpy as np
import cv2
from spixel.train_util import *
import torch.nn.functional as F

import matplotlib.pyplot as plt

# import sys
# sys.path.append('../cython')
# from connectivity import enforce_connectivity


'''
Infer from custom dataset:
author:Fengting Yang
last modification: Mar.5th 2020

usage:
1. set the ckpt path (--pretrained) and output
2. comment the output if do not need

results will be saved at the args.output

'''

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

random.seed(100)


@torch.no_grad()
def test(args, model, img_paths, save_path, idx):
    # Data loading code
    input_transform = transforms.Compose([
        flow_transforms.ArrayToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[255, 255, 255]),
        transforms.Normalize(mean=[0.411, 0.432, 0.45], std=[1, 1, 1])
    ])

    img_file = img_paths[idx]
    load_path = img_file
    imgId = os.path.basename(img_file)[:-4]

    # may get 4 channel (alpha channel) for some format
    img_ = imageio.imread(load_path)[:, :, :3]
    H, W, _ = img_.shape
    rate=H/W
    min_=160
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
    n_spixl_h = int(np.floor(H_ / args.downsize))
    n_spixl_w = int(np.floor(W_ / args.downsize))

    spix_values = np.int32(np.arange(0, n_spixl_w * n_spixl_h).reshape((n_spixl_h, n_spixl_w)))
    spix_idx_tensor_ = shift9pos(spix_values)

    # 填充周围16*16范围
    spix_idx_tensor = np.repeat(
        np.repeat(spix_idx_tensor_, args.downsize, axis=1), args.downsize, axis=2)

    spixeIds = torch.from_numpy(np.tile(spix_idx_tensor, (1, 1, 1, 1))).type(torch.float)

    n_spixel = int(n_spixl_h * n_spixl_w)

    img = cv2.resize(img_, (W_, H_), interpolation=cv2.INTER_CUBIC)
    img1 = input_transform(img)
    ori_img = input_transform(img_)

    # compute output
    tic = time.time()
    output = model(img1.unsqueeze(0))
    toc = time.time() - tic

    # assign the spixel map
    curr_spixl_map = update_spixl_map(spixeIds, output)
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(H_, W_), mode='nearest').type(torch.int)

    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.unsqueeze(0).dtype).view(3, 1, 1)
    spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(),
                                                    n_spixels=n_spixel, b_enforce_connect=True)

    # ************************ Save all result********************************************
    # save img, uncomment it if needed
    if not os.path.isdir(os.path.join(save_path, 'img')):
        os.makedirs(os.path.join(save_path, 'img'))
    spixl_save_name = os.path.join(save_path, 'img', imgId + '.jpg')
    img_save = (ori_img + mean_values).clamp(0, 1)
    imageio.imsave(spixl_save_name, img_save.detach().cpu().numpy().transpose(1, 2, 0))

    label = (spixel_label_map + 1).astype(int)

    # if not os.path.isdir(os.path.join(save_path, 'img')):
    #     os.makedirs(os.path.join(save_path, 'img'))
    # spixl_save_name = os.path.join(save_path, 'img', imgId + '.png')
    # img_save = ((ori_img + mean_values)*255)
    # spix_avg_color(img_save.detach().cpu().numpy().transpose(1, 2, 0),label,spixl_save_name)

    # save spixel viz
    if not os.path.isdir(os.path.join(save_path, 'spixel_viz')):
        os.makedirs(os.path.join(save_path, 'spixel_viz'))
    spixl_save_name = os.path.join(save_path, 'spixel_viz', imgId + '_sPixel.png')
    imageio.imsave(spixl_save_name, spixel_viz.transpose(1, 2, 0))

    # save the unique maps as csv, uncomment it if needed
    if not os.path.isdir(os.path.join(save_path, 'map_csv')):
        os.makedirs(os.path.join(save_path, 'map_csv'))
    output_path = os.path.join(save_path, 'map_csv', imgId + '.csv')
    # plus 1 to make it consistent with the toolkit format
    np.savetxt(output_path, label, fmt='%i', delimiter=",")

    # 保存处理颜色后的图像
    rgb_ori_img = (ori_img + mean_values).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    mix_rgb = label2rgb(label, rgb_ori_img, kind='mix')
    avg_rgb = label2rgb(label, rgb_ori_img, kind='avg')
    median_rgb = label2rgb(label, rgb_ori_img, kind='median')

    try:
        os.makedirs(os.path.join(save_path, 'results','avg','img'))
        os.makedirs(os.path.join(save_path, 'results', 'avg', 'csv'))
        os.makedirs(os.path.join(save_path, 'results', 'mix','img'))
        os.makedirs(os.path.join(save_path, 'results', 'mix','csv'))
        os.makedirs(os.path.join(save_path, 'results', 'median','img'))
        os.makedirs(os.path.join(save_path, 'results', 'median','csv'))
    except:
        pass

    mix_save_path = os.path.join(save_path, 'results','mix' )
    avg_save_path = os.path.join(save_path, 'results','avg')
    median_save_path = os.path.join(save_path, 'results','median')

    # 保存超像素结果
    imageio.imsave(os.path.join(mix_save_path,'img',imgId + '.png'), mix_rgb)
    imageio.imsave(os.path.join(avg_save_path,'img',imgId + '.png'), avg_rgb)
    imageio.imsave(os.path.join(median_save_path,'img',imgId + '.png'), median_rgb)
    # 保存csv颜色列表
    # 转lab

    mix_lab=rgb2lab(mix_rgb)
    avg_lab=rgb2lab(avg_rgb)
    median_lab=rgb2lab(median_rgb)
    np.savetxt(os.path.join(mix_save_path,'csv',imgId+'.csv'), np.unique(mix_lab.reshape(-1,3),axis=0), fmt='%i', delimiter=",")
    np.savetxt(os.path.join(mix_save_path,'csv',imgId+'.csv'), np.unique(avg_lab.reshape(-1,3),axis=0), fmt='%i', delimiter=",")
    np.savetxt(os.path.join(mix_save_path,'csv',imgId+'.csv'), np.unique(median_lab.reshape(-1,3),axis=0), fmt='%i', delimiter=",")

    if idx % 10 == 0:
        print("processing %d" % idx)

    return toc

# 用来自适应调整每个超像素颜色
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

def cal_saliency(label_field, image):
    out = np.zeros_like(image)
    labels = np.unique(label_field)

    for label in labels:
      mask = (label_field == label).nonzero()
      color = image[mask].mean()
      out[mask] = color
    return out

def main():
    global args, save_path
    data_dir = args.data_dir
    print("=> fetching img pairs in '{}'".format(data_dir))

    save_path = args.output
    print('=> will save everything to {}'.format(save_path))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # tst_lst = glob(args.data_dir + '/*.' + args.data_suffix)
    # tst_lst = glob(data_dir + '/*.' + '.jpg')
    # tst_lst.sort()

    data_dir = r'C:\Users\74555\Desktop\superpixel_fcn\demo\inputs'
    save_path = r'C:\Users\74555\Desktop\superpixel_fcn\demo'
    tst_lst = os.listdir(data_dir)
    tst_lst.sort()
    for i, file in enumerate(tst_lst):
        tst_lst[i] = os.path.join(data_dir, file)

    if len(tst_lst) == 0:
        print('Wrong data dir or suffix!')
        exit(1)

    print('{} samples found'.format(len(tst_lst)))

    # create sal_model
    network_data = torch.load(args.pretrained, map_location='cpu')
    # print("=> using pre-trained sal_model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](data=network_data)
    model.eval()
    args.arch = network_data['arch']
    cudnn.benchmark = True

    mean_time = 0
    for n in range(len(tst_lst)):
        time = test(args, model, tst_lst, save_path, n)
        mean_time += time
    print("avg_time per img: %.3f" % (mean_time / len(tst_lst)))


def load_model(model_dir):
    # create sal_model
    network_data = torch.load(model_dir, map_location='cpu')
    # print("=> using pre-trained sal_model '{}'".format(network_data['arch']))
    model = models.__dict__[network_data['arch']](data=network_data)
    model.eval()
    args.arch = network_data['arch']
    cudnn.benchmark = True

    return model


'''
输入图片路径
返回超像素图片的路径
'''
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
    tic = time.time()
    output = model(img1.unsqueeze(0))
    toc = time.time() - tic

    # assign the spixel map
    curr_spixl_map = update_spixl_map(spixeIds, output)
    ori_sz_spixel_map = F.interpolate(curr_spixl_map.type(torch.float), size=(H_, W_), mode='nearest').type(torch.int)

    mean_values = torch.tensor([0.411, 0.432, 0.45], dtype=img1.unsqueeze(0).dtype).view(3, 1, 1)
    spixel_viz, spixel_label_map = get_spixel_image((ori_img + mean_values).clamp(0, 1), ori_sz_spixel_map.squeeze(),
                                                    n_spixels=n_spixel, b_enforce_connect=True)

    # label instead csv
    label = (spixel_label_map + 1).astype(int)

    # 处理颜色
    rgb_ori_img = (ori_img + mean_values).mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    adp_rgb = label2rgb(label, rgb_ori_img, kind='mix')

    return label,adp_rgb


def lab2hex(lab: np.ndarray):
  lab = np.array(lab, np.uint8)
  lab = lab.reshape(1, -1, 3)
  # print(lab)
  rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
  rgb = rgb.reshape(-1, 3)

  res = []
  for c in rgb:
    res.append(rgb2hex(c))
  return res

def rgb2hex(rgb: np.ndarray):
  rgb = np.array(rgb, np.uint8)
  rgb = rgb.reshape(-1, 3)
  res=[]
  for c in rgb:
    strs = '#'
    for i in c:
      num = int(i)
      strs += str(hex(num))[-2:].replace('x', '0').upper()
    res.append(strs)
  return res

if __name__ == '__main__':
    main()
