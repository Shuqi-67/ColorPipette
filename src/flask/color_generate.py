import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tools import *

def generate_color_from_image(img,clusters=4):
    if (img is None):
        print('Invaild image path.')
        exit()
    # 获取初始图像的宽高比例，以便resize不改变图像比例
    ratio = img.shape[0] / img.shape[1]
    img_resize = cv2.resize(img, (640, int(640 * ratio)), interpolation=cv2.INTER_CUBIC)
    # 将Opencv中图像默认BGR转换为通用的RGB格式
    img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
    (height, width, channels) = img_rgb.shape
    # 将图像数据转换为需要进行Kmeans聚类的Data
    img_data = img_rgb.reshape(height * width, channels)

    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(img_data)

    colors = []
    for i in range(len(kmeans.cluster_centers_)):
        colors.append(RGB_to_Hex(kmeans.cluster_centers_[i]))
    # print(colors)
    return colors
