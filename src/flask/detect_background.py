import cv2
import csv
import os
import math


def rgb2lab(rgb):
    r = rgb[0] / 255.0  # rgb range: 0 ~ 1
    g = rgb[1] / 255.0
    b = rgb[2] / 255.0

    # gamma 2.2
    if r > 0.04045:
        r = pow((r + 0.055) / 1.055, 2.4)
    else:
        r = r / 12.92

    if g > 0.04045:
        g = pow((g + 0.055) / 1.055, 2.4)
    else:
        g = g / 12.92

    if b > 0.04045:
        b = pow((b + 0.055) / 1.055, 2.4)
    else:
        b = b / 12.92

    # sRGB
    X = r * 0.436052025 + g * 0.385081593 + b * 0.143087414
    Y = r * 0.222491598 + g * 0.716886060 + b * 0.060621486
    Z = r * 0.013929122 + g * 0.097097002 + b * 0.714185470

    # XYZ range: 0~100
    X = X * 100.000
    Y = Y * 100.000
    Z = Z * 100.000

    # Reference White Point

    ref_X = 96.4221
    ref_Y = 100.000
    ref_Z = 82.5211

    X = X / ref_X
    Y = Y / ref_Y
    Z = Z / ref_Z

    # Lab
    if X > 0.008856:
        X = pow(X, 1 / 3.000)
    else:
        X = (7.787 * X) + (16 / 116.000)

    if Y > 0.008856:
        Y = pow(Y, 1 / 3.000)
    else:
        Y = (7.787 * Y) + (16 / 116.000)

    if Z > 0.008856:
        Z = pow(Z, 1 / 3.000)
    else:
        Z = (7.787 * Z) + (16 / 116.000)

    Lab_L = round((116.000 * Y) - 16.000, 2)
    Lab_a = round(500.000 * (X - Y), 2)
    Lab_b = round(200.000 * (Y - Z), 2)

    return [Lab_L, Lab_a, Lab_b]

def lab2lch(lab): # input lab range: [0, 255]
    l = lab[0]
    a = lab[1]
    b = lab[2]
    c = math.sqrt(a ** 2 + b ** 2)
    h = (math.atan2(b, a) * 180 / math.pi + 360) % 360  # h : 0 ~ 359
    # h = h * math.pi / 180
    return [l, c, h]

def lch2lab(lch): # output lab range: [0, 255]
    l = lch[0]
    c = lch[1]
    h = lch[2]
    h = h * math.pi / 180
    a = c * math.cos(h)
    b = c * math.sin(h)
    return [l, a, b]

def detect_background2(sal_bin, ori_img):
    histogram = [[[0 for i in range(16)] for j in range(16)] for k in range(16)]
    lch_histogram = [[[0 for i in range(16)] for j in range(16)] for k in range(16)]
    max_color_pixel = [[0, 0, 0], [0, 0, 0]]
    max_pixel_num = [0, 0] # max, sec
    for i in range(0, sal_bin.shape[0]):
        for j in range(0, sal_bin.shape[1]):
            if sal_bin[i][j][0] < 0.2 * 255:
                #rgb histogram
                # histogram[int(ori_img[i][j][2] / 16)][int(ori_img[i][j][1] / 16)][int(ori_img[i][j][0] / 16)] += 1

                #lch histogram
                lch = lab2lch(rgb2lab([ori_img[i][j][2], ori_img[i][j][1], ori_img[i][j][0]])) #BGR
                if int(lch[2] / 22.5) > 15 or int(lch[1] / 11.25) > 15 or int(lch[0] / 6.251) > 15:
                    print(lch, rgb2lab(ori_img[i][j]))
                lch_histogram[int(lch[0] / 6.251)][int(lch[1] / 11.25)][int(lch[2] / 22.5)] += 1
    # rgb histogram
    # for i in range(0, 16):
    #     for j in range(0, 16):
    #         for k in range(0, 16):
    #             if histogram[i][j][k] > max_pixel_num[0]:
    #                 max_color_pixel[1] = max_color_pixel[0]
    #                 max_pixel_num[1] = max_pixel_num[0]
    #                 max_color_pixel[0] = [i, j, k]
    #                 max_pixel_num[0] = histogram[i][j][k]
    #             elif histogram[i][j][k] > max_pixel_num[1]:
    #                 max_color_pixel[1] = [i, j, k]
    #                 max_pixel_num[1] = histogram[i][j][k]

    # lch histogram
    for i in range(0, 16):
        for j in range(0, 16):
            for k in range(0, 16):
                if lch_histogram[i][j][k] > max_pixel_num[0]:
                    max_color_pixel[1] = max_color_pixel[0]
                    max_pixel_num[1] = max_pixel_num[0]
                    max_color_pixel[0] = [i, j, k]
                    max_pixel_num[0] = lch_histogram[i][j][k]
                elif lch_histogram[i][j][k] > max_pixel_num[1]:
                    max_color_pixel[1] = [i, j, k]
                    max_pixel_num[1] = lch_histogram[i][j][k]

    # rgb histogram
    # return rgb2lab([max_color_pixel[0][0] * 16 + 8, max_color_pixel[0][1] * 16 + 8, max_color_pixel[0][2] * 16 + 8]), \
    #        rgb2lab([max_color_pixel[1][0] * 16 + 8, max_color_pixel[1][1] * 16 + 8, max_color_pixel[1][2] * 16 + 8]), \
    #        [max_color_pixel[0][0] * 16 + 8, max_color_pixel[0][1] * 16 + 8, max_color_pixel[0][2] * 16 + 8], \
    #        [max_color_pixel[1][0] * 16 + 8, max_color_pixel[1][1] * 16 + 8, max_color_pixel[1][2] * 16 + 8]

    # lch histogram
    return lch2lab([max_color_pixel[0][0] * 6.251 + 3.125, max_color_pixel[0][1] * 11.25 + 5.625, max_color_pixel[0][2] * 22.5 + 11.25]), \
           lch2lab([max_color_pixel[1][0] * 6.251 + 3.125, max_color_pixel[1][1] * 11.25 + 5.625, max_color_pixel[1][2] * 22.5 + 11.25]), \
           [max_color_pixel[0][0] * 16 + 8, max_color_pixel[0][1] * 16 + 8, max_color_pixel[0][2] * 16 + 8], \
           [max_color_pixel[1][0] * 16 + 8, max_color_pixel[1][1] * 16 + 8, max_color_pixel[1][2] * 16 + 8] # the last two return values should not be used

def detect_background1(sal_bin, ori_img):
    histogram = [[[0 for i in range(16)] for j in range(16)] for k in range(16)]  # 8 * 8 * 8
    max_color_pixel = [0, 0, 0]
    max_pixel_num = 0 # max, sec
    for i in range(0, sal_bin.shape[0]):
        for j in range(0, sal_bin.shape[1]):
            if sal_bin[i][j][0] < 0.2 * 255:
                histogram[int(ori_img[i][j][2] / 16)][int(ori_img[i][j][1] / 16)][int(ori_img[i][j][0] / 16)] += 1
    for i in range(0, 16):
        for j in range(0, 16):
            for k in range(0, 16):
                if histogram[i][j][k] > max_pixel_num:
                    max_color_pixel = [i, j, k]
                    max_pixel_num = histogram[i][j][k]
    return rgb2lab(max_color_pixel[0] * 16 + 8, max_color_pixel[1] * 16 + 8, max_color_pixel[2] * 16 + 8)  # return lab
    # return [max_color_pixel[0] * 16 + 8, max_color_pixel[1] * 16 + 8, max_color_pixel[2] * 16 + 8] # return rgb

if __name__ == '__main__':
    file_root = '/home/ecnu-9/Documents/lsq/VisColor/code/new_dataset/'
    fw_lab = open(file_root + 'background_lab.csv', 'w')
    csv_writer_lab = csv.writer(fw_lab)
    # fw_rgb = open(file_root + 'background_rgb.csv', 'w')
    # csv_writer_rgb = csv.writer(fw_rgb)

    img_dir = r'/home/ecnu-9/Documents/lsq/VisColor/code/new_dataset/imgs/'
    imgs = os.listdir(img_dir)
    for img_idx in imgs:
        bin_crop = cv2.imread(file_root + 'sal_img/sal_' + img_idx, 1)
        img_crop = cv2.imread(file_root + 'imgs/' + img_idx, 1)

        # detect_background2 return lab + rgb
        res_lab_rgb = detect_background2(bin_crop, img_crop)

        # rgb
        # res_write = [str(img_idx)]
        # for res_rgb in res_lab_rgb[2: 4]:
        #     res_write.append('[' + str(int(res_rgb[0])) + " " + str(int(res_rgb[1])) + " " + str(int(res_rgb[2])) + ']')
        # csv_writer_rgb.writerow(res_write)

        # lab
        res_write = [str(img_idx)]
        for res_lab in res_lab_rgb[0: 2]:
            res_lab[0] = res_lab[0] / 100 * 255
            res_lab[1] = res_lab[1] + 127.5
            res_lab[2] = res_lab[2] + 127.5
            res_write.append('[' + str(int(res_lab[0])) + " " + str(int(res_lab[1])) + " " + str(int(res_lab[2])) + ']')
        csv_writer_lab.writerow(res_write)

        print(img_idx)

        # detect_background1 return lab
        # res_labs = detect_background1(bin_crop, img_crop)
        # res_write = [str(img_idx)]
        # for res_lab in res_labs:
        #     res_lab[0] = res_lab[0] / 100 * 255
        #     res_lab[1] = res_lab[1] + 127.5
        #     res_lab[2] = res_lab[2] + 127.5
        #     res_write.append('[' + str(int(res_lab[0])) + " " + str(int(res_lab[1])) + " " + str(int(res_lab[2])) + ']')
        # csv_writer.writerow(res_write)
        # print(img_idx)

    fw_lab.close()