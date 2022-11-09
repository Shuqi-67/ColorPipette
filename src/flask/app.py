from flask import Flask, request,jsonify
from flask_cors import CORS
from scipy.cluster.vq import kmeans
from PIL import Image
import io,os
import time
import model_unit
import colorsys
import random
import math
import har_colors
import base64
import cv2
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.route('/get_pic', methods=['GET', 'POST'], endpoint='get_pic')
def gather():
    root = os.getcwd()
    src_img_root = os.path.join(root, "src_img")
    img_name = time.strftime("%Y%m%d%H%M%S", time.localtime())+'.png'
    src_img_path = os.path.join(src_img_root,img_name)
    file_obj = request.files['file']
    file_obj.save(src_img_path)
    print(src_img_path)

    global latest_img
    latest_img = src_img_path

    with open(src_img_path, 'rb') as f:
        a = f.read()
    img_stream = io.BytesIO(a)
    img = Image.open(img_stream)

    imgByteArr = io.BytesIO()
    img.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

@app.route('/get/get_sample', methods=['GET', 'POST'], endpoint='get_sample')
def gather_sample():
    img_name = request.args.get("sample_index")
    root = os.path.abspath('../..')
    src_img_root = os.path.join(root, "public", "samples")
    src_img_path = os.path.join(src_img_root, img_name)

    global latest_img
    latest_img = src_img_path

    with open(src_img_path, 'rb') as f:
      image = f.read()
    image_base64 = str(base64.b64encode(image), encoding='utf-8')

    return image_base64

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

def lab2lch(lab):  # input lab range: [0, 255]
  l = lab[0]
  a = lab[1]
  b = lab[2]
  c = math.sqrt(a ** 2 + b ** 2)
  h = (math.atan2(b, a) * 180 / math.pi + 360) % 360  # h : 0 ~ 359
  return [l, c, h]

def lch2lab(lch):  # output lab range: [0, 255]
  l = lch[0]
  c = lch[1]
  h = lch[2]
  h = h * math.pi / 180
  a = c * math.cos(h)
  b = c * math.sin(h)
  return [l, a, b]

def detect_background(sal_bin, ori_img):
  histogram = [[[0 for i in range(16)] for j in range(16)] for k in range(16)]
  lch_histogram = [[[0 for i in range(16)] for j in range(16)] for k in range(16)]
  max_color_pixel = [[0, 0, 0], [0, 0, 0]]
  max_pixel_num = [0, 0]  # max, sec
  for i in range(0, sal_bin.shape[0]):
    for j in range(0, sal_bin.shape[1]):
      if sal_bin[i][j][0] < 0.2 * 255:
        # lch histogram
        lch = lab2lch(rgb2lab([ori_img[i][j][2], ori_img[i][j][1], ori_img[i][j][0]]))  # BGR
        lch_histogram[int(lch[0] / 6.251)][int(lch[1] / 11.25)][int(lch[2] / 22.5)] += 1

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

  # lch histogram
  return lch2lab([max_color_pixel[0][0] * 6.251 + 3.125, max_color_pixel[0][1] * 11.25 + 5.625,
                  max_color_pixel[0][2] * 22.5 + 11.25]), \
         lch2lab([max_color_pixel[1][0] * 6.251 + 3.125, max_color_pixel[1][1] * 11.25 + 5.625,
                  max_color_pixel[1][2] * 22.5 + 11.25]), \
         [max_color_pixel[0][0] * 16 + 8, max_color_pixel[0][1] * 16 + 8, max_color_pixel[0][2] * 16 + 8], \
         [max_color_pixel[1][0] * 16 + 8, max_color_pixel[1][1] * 16 + 8,
          max_color_pixel[1][2] * 16 + 8]  # the last two return values should not be used

def lab_dis(c1, c2):
  return math.sqrt((int(c1[0]) - int(c2[0])) * (int(c1[0]) - int(c2[0])) + (int(c1[1]) - int(c2[1])) * (
    int(c1[1]) - int(c2[1])) + (int(c1[2]) - int(c2[2])) * (int(c1[2]) - int(c2[2])))

@app.route('/get/color_open', methods=['GET', 'POST'], endpoint='color_open')
def generate_color_open():
  number = request.args.get('number')
  number = int(number)
  bcg_flag = request.args.get("bcg_flag")

  bcg = ""
  if bcg_flag == 'true':
    bcg_flag = True
  elif bcg_flag == 'false':
    bcg_flag = False

  color_list = []

  global latest_img, Sal, Sp, Enc
  # 1. saliency detection & pixel segmentation
  sal_imo_pil = Sal.saliency_detect(latest_img)
  sal_imo_cv2 = cv2.cvtColor(np.asarray(sal_imo_pil), cv2.COLOR_RGB2BGR)

  label, sp_img_cv2 = Sp.do_spixel(latest_img)
  sp_img_cv2_lab = cv2.cvtColor(sp_img_cv2, cv2.COLOR_RGB2LAB)
  sal_imo_cv2 = cv2.resize(sal_imo_cv2, (sp_img_cv2.shape[1], sp_img_cv2.shape[0]))

  img_cv2_bgr = cv2.imread(latest_img)
  img_cv2_lab = cv2.cvtColor(img_cv2_bgr, cv2.COLOR_BGR2LAB)

  # 2. background color detection
  bcg_lab_lch = detect_background(sal_imo_cv2, img_cv2_bgr)
  bcg_lab = bcg_lab_lch[0:2]
  bcg_lch = bcg_lab_lch[2:4]

  # 3. candidate set
  all_color = {}
  all_sal = {}
  ave_sal = []
  for i in range(1, 1000):
    all_color[i] = []
    all_sal[i] = []

  i = 0
  for m in label:
    for j in range(len(m)):
      all_sal[m[j]].append(sal_imo_cv2[i][j][0])
      if len(all_color[m[j]]) == 0:
        all_color[m[j]] = [sp_img_cv2_lab[i][j][0], sp_img_cv2_lab[i][j][1], sp_img_cv2_lab[i][j][2]]
    i += 1

  for j in range(1, 1000):
    if len(all_color[j]) != 0:
      # idx, sal, pixel_num
      ave_sal.append([j, sum(all_sal[j]) / len(all_sal[j]), len(all_sal[j])])
    else:
      break
  ave_sal.sort(key=lambda student: student[1], reverse=True)

  palette = [bcg_lab[0], bcg_lab[1]]

  palette_sal = []

  for c in palette:
    min_dis = 99999
    color_sal = 0
    for j in range(0, len(ave_sal)):
      sp_c = all_color[ave_sal[j][0]]
      dis = lab_dis(sp_c, c)
      if dis < min_dis:
        min_dis = dis
        color_sal = ave_sal[j][1]
    palette_sal.append(color_sal)

  palette_lab, palette_sal = har_colors.get_har_colors(palette[0], palette[1], ave_sal, number, all_color, 'lab',
                                                     bcg_flag)

  palette_rgb = np.zeros((1, len(palette_lab), 3), np.uint8)
  for i in range(len(palette_lab)):
    palette_rgb[0][i] = palette_lab[i]
  palette_rgb = cv2.cvtColor(palette_rgb, cv2.COLOR_LAB2RGB)
  palette_hex = []
  for color_rgb in palette_rgb[0]:
      palette_hex.append(rgb2hex(color_rgb))

  if bcg_flag:
    bcg = palette_hex[number]
    palette_hex = palette_hex[:-1]

  return jsonify({
    'message': '',
    'data': {
      'color_list': palette_hex,
      'bcg' : bcg
    }})

def nets_init():
    # global settings
    global root_dir, src_img_root
    root_dir = os.path.dirname(os.path.abspath(__file__))  #./flask
    src_img_root = os.path.join(root_dir, 'src_img')

    global latest_img
    latest_img = ''

    sal_model_dir = os.path.join(root_dir, 'saliency', 'saved_models', 'basnet_best_train_gdi.pth')
    sal_save_dir = os.path.join(root_dir, 'res', 'sal')
    global Sal
    Sal = model_unit.Saliency(sal_model_dir, sal_save_dir)

    sp_model_dir = os.path.join(root_dir, 'spixel', 'pretrain_ckpt', 'SpixelNet_bsd_ckpt.tar')
    sp_save_dir = os.path.join(root_dir, 'res', 'sp')
    global Sp
    Sp = model_unit.Spixel(sp_model_dir, sp_save_dir)

def rgb2hex(rgb):
  strs = '#'
  for i in rgb:
    num = int(i)
    strs += str(hex(num))[-2:].replace('x', '0').upper()
  return strs

if __name__ == '__main__':

    nets_init()
    app.run()
