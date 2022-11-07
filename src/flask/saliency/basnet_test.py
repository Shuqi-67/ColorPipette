import os
from skimage import io, transform
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import glob

from data_loader import RescaleT
from data_loader import CenterCrop
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from sal_model import BASNet
from scipy.special import expit
import cv2
import csv

def write_csv(sv_fp,  sv_list, mode = 'w'):
    with open(sv_fp, mode, encoding='utf-8') as w:
        writer = csv.writer(w, delimiter=",")
        writer.writerow(sv_list)

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def overlay_imp_on_img(img, imp, fname, colormap='jet'):
    cm = plt.get_cmap(colormap)
    img2 = np.array(img, dtype=np.uint8)
    imp2 = np.array(imp, dtype=np.uint8)
    imp3 = (cm(imp2)[:, :, :3] * 255).astype(np.uint8)
    img3 = Image.fromarray(img2)
    imp3 = Image.fromarray(imp3)
    im_alpha = Image.blend(img3, imp3, 0.5)
    im_alpha.save(fname)

def cut_image_saliency(img, saliency_threshold):
    left, right, top, bottom = 0, 0, 0, 0
    for i in range(0, img.shape[0]):
        for ix in range(0, img.shape[1]):
            if img[i][ix][0] > saliency_threshold * 255:
                bottom = i
                break
    for i in range(img.shape[0] - 1, -1, -1):
        for ix in range(0, img.shape[1]):
            if img[i][ix][0] > saliency_threshold * 255:
                top = i
                break
    for i in range(0, img.shape[1]):
        for ix in range(0, img.shape[0]):
            if img[ix][i][0] > saliency_threshold * 255:
                right = i
                break
    for i in range(img.shape[1] - 1, -1, -1):
        for ix in range(0, img.shape[0]):
            if img[ix][i][0] > saliency_threshold * 255:
                left = i
                break
    return left, right, top, bottom

# def save_saliency(res, image_name):
#     os.makedirs('/home/ecnu-9/Documents/lsq/VisColor/code/img_to_palette/img/all_data/saliency_out/map/', exist_ok=True)
#     f = open('/home/ecnu-9/Documents/lsq/VisColor/code/img_to_palette/img/all_data/saliency_out/' + 'map/' + image_name[0: -4] + '.csv', "w")
#     writer = csv.writer(f)
#     print(res.shape)
#     for i in range(len(res)):
#         temp = []
#         for j in range(len(res[0])):
#             temp.append(str(res[i][j]))
#         writer.writerow(temp)
#     f.close()

def get_output(image_path, pred):
    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    im = Image.fromarray(predict_np * 255).convert('RGB')
    image = io.imread(image_path)
    image = image[:, :, :3]
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    return imo


def save_output(image_path, pred, d_dir, d_dir_ovl, d_dir_crop, d_dir_bin_crop):
    image_name=os.path.basename(image_path)

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()

    # save_saliency(predict_np, image_name)

    im = Image.fromarray(predict_np*255).convert('RGB')
    # img_name = image_name.split("/")[-1]
    image = io.imread(image_path)
    image = image[:,:,:3]
    imo = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)
    #
    # cut image where saliency > 0.7 (rgb > 255 * 0.7)
    # img_array = np.array(imo)

    # saliency_threshold = 0.5
    # left, right, top, bottom = cut_image_saliency(img_array, saliency_threshold)
    #
    imo.save(os.path.join(d_dir, 'sal_' + image_name))

    # imo_crop = imo.crop((left, top, right, bottom))
    # imo_crop.save(os.path.join(d_dir_bin_crop, image_name[:-4] + "_bin_crop.jpg"))
    #
    img_ini = image
    # img_ini_crop = Image.fromarray(image).crop((left, top, right, bottom))
    # img_ini_crop.save(os.path.join(d_dir_crop, image_name[:-4] + "_crop.jpg"))
    predict_np_my = cv2.imread(os.path.join(d_dir,'sal_' + image_name))
    predict_np_my = predict_np_my[:, :, :1]
    predict_np_my = predict_np_my.squeeze()
    #
    # tmp_sum = np.sum(predict_np_my)
    # tmp_sum = tmp_sum / image.shape[1] / image.shape[0]
    # # sv_list = [img_name, tmp_sum]
    # # write_csv(sv_fp = logs_fp, sv_list = sv_list, mode = 'a')
    #
    fname = os.path.join(d_dir_ovl,image_name.replace('.png','_ovl.png'))
    overlay_imp_on_img(img_ini, predict_np_my, fname, colormap='jet')
    return 2

def main():
    # --------- 1. get image path and name ---------
    dt_type = 'gdi'
    base_dir = './test_data/'
    image_dir = 'test_data/sali_gdi/input/'
    prediction_dir = base_dir + 'sali_' + dt_type + '/'
    prediction_dir_ovl = prediction_dir
    prediction_dir_bin_crop = base_dir + 'sali_bin_crop_' + dt_type + '/'
    prediction_dir_crop = base_dir + 'sali_crop_' + dt_type + '/'
    model_dir = './saved_models/basnet_bsi_' + dt_type + '/basnet_best_train_' + dt_type + '.pth'
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(prediction_dir_ovl, exist_ok=True)
    os.makedirs(prediction_dir_bin_crop, exist_ok=True)
    os.makedirs(prediction_dir_crop, exist_ok=True)

    logs_fp = base_dir + 'sali_' + dt_type + '.csv'
    sv_list = ['img_name', 'sali_val']
    write_csv(sv_fp=logs_fp, sv_list=sv_list, mode='w')
    img_name_list = glob.glob(image_dir + '*')

    # --------- 2. dataloader ---------
    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)

    # --------- 3. sal_model define ---------
    print("...load BASNet...")
    net = BASNet(3, 1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    sali_sum = 0.0
    len_imgs = len(test_salobj_dataloader)
    print('len_imgs = ', len_imgs)

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:", img_name_list[i_test].split("/")[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7, d8 = net(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        tmp_sum = save_output(img_name_list[i_test], pred, prediction_dir, prediction_dir_ovl, prediction_dir_crop, prediction_dir_bin_crop)
        sali_sum += tmp_sum

        del d1, d2, d3, d4, d5, d6, d7, d8

    sv_list = ['avg_sali', sali_sum / len_imgs]
    write_csv(sv_fp=logs_fp, sv_list=sv_list, mode='a')

def load_model(model_dir):
    net = BASNet(3, 1)
    net.load_state_dict(torch.load(model_dir))
    if torch.cuda.is_available():
        net.cuda()
    net.eval()
    return net


def test_single(model, img_path):

    img_name = os.path.basename(img_path)

    # bin_save_path=os.path.join(save_path,'bin')
    # ovl_save_path=os.path.join(save_path,'ovl')
    # bin_crop_save_path = os.path.join(save_path, 'bin_crop')
    # d_dir_crop = os.path.join(save_path, 'crop')
    # if not os.path.exists(bin_save_path):
    #     os.mkdir(bin_save_path)
    # if not os.path.exists(ovl_save_path):
    #     os.mkdir(ovl_save_path)
    # if not os.path.exists(bin_crop_save_path):
    #     os.mkdir(bin_crop_save_path)
    # if not os.path.exists(d_dir_crop):
    #     os.mkdir(d_dir_crop)

    img_name_list = list({img_path})

    test_salobj_dataset = SalObjDataset(img_name_list=img_name_list, lbl_name_list=[],
                                        transform=transforms.Compose([RescaleT(256), ToTensorLab(flag=0)]))
    test_salobj_dataloader = DataLoader(test_salobj_dataset, batch_size=1, shuffle=False, num_workers=1)


    for i_test, data_test in enumerate(test_salobj_dataloader):
        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)

        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1, d2, d3, d4, d5, d6, d7, d8 = model(inputs_test)

        # normalization
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        # save results to test_results folder
        # save_output(img_path, pred, bin_save_path, ovl_save_path, d_dir_crop, bin_crop_save_path)
        imo = get_output(img_path, pred)

        del d1, d2, d3, d4, d5, d6, d7, d8

        return imo


if __name__ == '__main__':
    model_dir='./saved_models/basnet_best_train_gdi.pth'
    # model_dir = './saved_models/basnet.pth'
    model=load_model(model_dir)

    img_path = r'/home/ecnu-9/Documents/lsq/VisColor/code/cluster/img/'
    save_path = r'/home/ecnu-9/Documents/lsq/VisColor/code/cluster/vaa/'
    imgs = os.listdir(img_path)
    for img in imgs:
        test_single(model,img_path + img,save_path)


