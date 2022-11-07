from math import pow, sqrt

param_13 = 1.0 / 3.0
param_16116 = 16.0 / 116.0

Xn = 0.950456
Yn = 1.0
Zn = 1.088754


def RGB2XYZ(r, g, b):
  x = 0.412453 * r + 0.357580 * g + 0.180423 * b
  y = 0.212671 * r + 0.715160 * g + 0.072169 * b
  z = 0.019334 * r + 0.119193 * g + 0.950227 * b
  return x, y, z


def XYZ2Lab(x, y, z):
  x /= 255 * Xn
  y /= 255 * Yn
  z /= 255 * Zn
  if y > 0.008856:
    fy = pow(y, param_13)
    l = 116.0 * fy - 16.0
  else:
    fy = 7.787 * y + param_16116
    l = 903.3 * fy

  if l < 0:
    l = 0.0

  if x > 0.008856:
    fx = pow(x, param_13)
  else:
    fx = 7.787 * x + param_16116

  if z > 0.008856:
    fz = pow(z, param_13)
  else:
    fz = 7.787 * z + param_16116

  a = 500.0 * (fx - fy)
  b = 200.0 * (fy - fz)

  return [round(l, 2), round(a, 2), round(b, 2)]


def RGB2Lab(r, g, b):
  x, y, z = RGB2XYZ(r, g, b)
  return XYZ2Lab(x, y, z)


def hex2rgb(hex_color):
  hex_color = hex_color[1:]
  hexcolor = int(hex_color, base=16) if isinstance(hex_color, str) else hex_color
  rgb = ((hexcolor >> 16) & 0xff, (hexcolor >> 8) & 0xff, hexcolor & 0xff)
  return rgb


def lab_dis(lab1, lab2):
  dis = (lab1[0] - lab2[0]) * (lab1[0] - lab2[0]) + (lab1[1] - lab2[1]) * (lab1[1] - lab2[1]) + (lab1[2] - lab2[2]) * (
      lab1[2] - lab2[2])
  return sqrt(dis)


def perm(data):
  if len(data) == 1:  # 和阶乘一样，需要有个结束条件
    return [data]
  r = []
  for i in range(len(data)):
    s = data[:i] + data[i + 1:]  # 去掉第i个元素，进行下一次的递归
    p = perm(s)
    for x in p:
      r.append(data[i:i + 1] + x)  # 一直进行累加
  return r


def group_lab_dis(colors1, colors2):
  dis = 0
  for i in range(0, len(colors1)):
    dis += lab_dis(colors1[i], colors2[i])
  return dis


def min_dis(aim_colors, colors_rgb):
  colors = [[] for i in range(0, len(colors_rgb))]

  full_permutation_rgb = perm(aim_colors)

  for i in range(0, len(aim_colors)):
    rgb = hex2rgb(aim_colors[i])
    lab = RGB2Lab(rgb[0], rgb[1], rgb[2])
    aim_colors[i] = lab

    rgb = hex2rgb(colors_rgb[i])
    lab = RGB2Lab(rgb[0], rgb[1], rgb[2])
    colors[i] = lab

  full_permutation = perm(aim_colors)
  min_layout = []
  min_dis = 9999999
  for i in range(0, len(full_permutation)):
    dis = group_lab_dis(full_permutation[i], colors)
    if min_dis > dis:
      min_dis = dis
      min_layout = full_permutation_rgb[i]

  print("min distance:", min_dis)
  for i in range(0, len(colors)):
    print(colors_rgb[i], min_layout[i])

'''
tar_hex是个list
tar_hex[i][0] 16进制编码
tar_hex[i][1] 权重
'''
def closest_color(src_hex, tar_hex):

  res=[]

  for i,sc in enumerate(src_hex):
    min=999999
    id=-1
    rgb1 = hex2rgb(sc)
    lab1 = RGB2Lab(rgb1[0], rgb1[1], rgb1[2])
    for k,tc in enumerate(tar_hex):
      rgb2 = hex2rgb(tc[0])
      lab2 = RGB2Lab(rgb2[0], rgb2[1], rgb2[2])
      dis=lab_dis(lab1,lab2)
      if dis<min:
        min=dis
        id=k
    if dis>200:
      res.append(tar_hex[id])
    else:
      res.append([sc,tar_hex[id][1]])
  print("closest_color res:")
  print(res)
  return res

if __name__ == '__main__':
  aim_colors = ["#0A2A41", "#1E485D", "#487698", "#80A9BE", "#CDD2C5", "#E8E9EA"]
  colors_rgb = ["#DDA0DD", "#DA70D6", "#BA55D3", "#9932CC", "#9400D3", "#8A2BE2", "#0A2A41", "#1E485D", "#487698",
                "#80A9BE", "#CDD2C5", "#E8E9EA"]
  closest = closest_color(aim_colors, colors_rgb)
