import spixel.run_demo as spxiel
import saliency.basnet_test as saliency
import os
import cv2
import numpy as np
import tools

class Spixel:
    def __init__(self, sp_model_dir, sp_save_dir):
        self.sp_model_dir = sp_model_dir
        self.sp_save_dir = sp_save_dir
        self.sp_model = spxiel.load_model(sp_model_dir)

    def do_spixel(self, img_path):
        label, sp_img = spxiel.test_single(self.sp_model, img_path)
        return label, sp_img

class Saliency:
    def __init__(self, sal_model_dir, sal_save_dir):
        self.sal_model_dir = sal_model_dir
        self.sal_save_dir = sal_save_dir
        self.sal_model = saliency.load_model(sal_model_dir)

    def saliency_detect(self, img_path):
        sal_imo = saliency.test_single(self.sal_model, img_path)
        return sal_imo

    def do_crop(self, src_img_path,sal_img_path):
        img_name=os.path.basename(src_img_path)
        sal_path=os.path.join(self.sal_save_dir,'bin',img_name)
        save_path=os.path.join(self.sal_save_dir,'crop')
        if not os.path.exists(sal_path):
            os.mkdir(sal_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        idx = img_name.split('.')[0]
        src_img = cv2.imread(src_img_path)
        sal_img = cv2.imread(sal_img_path)
        sal_img = cv2.cvtColor(sal_img, cv2.COLOR_BGR2GRAY)
        or_sal_img=sal_img
        sal_img = sal_img / 255 * sal_img / 255

        save_img = np.zeros_like(sal_img)

        crop_name_list = []

        for i in {0.1, 0.3, 0.5, 0.8}:
            save_img[sal_img < i] = 0
            save_img[sal_img >= i] = 255
            # save_img[sal_img>=i]=math.sqrt(i)*255

            if not os.path.exists(os.path.join(save_path, 'bin_img')):
                os.mkdir(os.path.join(save_path, 'bin_img'))
            if not os.path.exists(os.path.join(save_path, 'src_img')):
                os.mkdir(os.path.join(save_path, 'src_img'))
            if not os.path.exists(os.path.join(save_path, 'sal_img')):
                os.mkdir(os.path.join(save_path, 'sal_img'))


            save_name = os.path.join(save_path, 'bin_img', idx + '_' + str(i) + '.png')
            cv2.imwrite(save_name, save_img)

            crop_area = cv2.imread(save_name, 0)
            crop_area = cv2.threshold(crop_area, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
            ret, labels = cv2.connectedComponents(crop_area)

            for k in range(ret):
                mask = (labels == k).nonzero()
                x1 = max(mask[0])
                x2 = min(mask[0])
                y1 = max(mask[1])
                y2 = min(mask[1])
                if x1 - x2 < 64 or y1 - y2 < 64:
                    continue
                if x1 - x2 > src_img.shape[0] - 64 and y1 - y2 > src_img.shape[1] - 64:
                    continue
                src_crop_img = src_img[x2:x1, y2:y1]
                sal_crop_img = or_sal_img[x2:x1, y2:y1]

                src_crop_name = save_name.replace('.png', '_' + str(k) + '.png').replace('bin_img', 'src_img')
                cv2.imwrite(src_crop_name, src_crop_img)
                sal_crop_name=src_crop_name.replace('src_img', 'sal_img')
                cv2.imwrite(sal_crop_name, sal_crop_img)
                crop_name_list.append(src_crop_name)
        return crop_name_list

def rgb2hex(rgb):
    strs = '#'
    for i in rgb:
        num = int(i)
        strs += str(hex(num))[-2:].replace('x', '0').upper()
    return strs
