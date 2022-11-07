import cv2
import numpy as np


def spix_avg_color(img_arr,label_arr,output_path):
    # path=''
    # path2=''
    # img = cv2.imread(path)
    # label = cv2.imread(path2)
    img=img_arr
    label=label_arr
    h=img.shape[0]
    w=img.shape[1]
    save_img = np.zeros((h, w), dtype=np.uint16)

    sum_color=np.zeros(h*w)
    num_pix=np.zeros(h*w)
    num=0
    max=0

    for i in range(h):
        for j in range(w):
            num = label[i][j]
            if num>max:
                max=num
            num_pix[num]+=1
            sum_color+=img[i][j]

    for i in range(h):
        for j in range(w):
            id=label[i][j]
            save_img[i][j]=sum_color[id]/num_pix[id]

    cv2.imwrite((output_path+".png"), save_img)