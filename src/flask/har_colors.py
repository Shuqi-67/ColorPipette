import math
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def not_ambiguous_to_bcg_colors(bcg, all_color): # idx, sal, pixel_num
    candidates = {}

    # B_dis of each 2 colors > 3
    SL = []
    SC = []
    X = []  # C
    Y = []  # L

    # bcg idx = 0
    X.append(bcg[1] / 255.0 * 100.0)
    Y.append(bcg[0] / 255.0 * 100.0)
    SL.append(1 + (0.015 * (Y[0] - 50) ** 2) / math.sqrt(20 + (Y[0] - 50) ** 2))
    SC.append(1 + 0.045 * X[0])

    for i in range(1, len(all_color) + 1):
        if len(all_color[i]) == 0:
            break
        X.append(all_color[i][1] / 255.0 * 100.0)
        Y.append(all_color[i][0] / 255.0 * 100.0)
    for i in range(1, len(all_color) + 1):
        # SL/SC: sigma  L/C: mu
        if len(all_color[i]) == 0:
            break
        SL.append(1 + (0.015 * (Y[i] - 50) ** 2) / math.sqrt(20 + (Y[i] - 50) ** 2))
        SC.append(1 + 0.045 * X[i])

    mu_bcg = np.array([Y[0], X[0]]).reshape(2, 1)
    for i in range(1, len(all_color) + 1):
        if len(all_color[i]) == 0:
            break
        if abs(all_color[i][2] - bcg[2]) >= 25:
            candidates[i] = all_color[i]
            continue
        mui = np.array([Y[i],X[i]]).reshape(2,1)
        sigma = np.array([[0.5 * (SC[i] + SC[0]), 0], [0, 0.5 * (SL[i] + SL[0])]])
        BD = 0.5 * np.log(np.linalg.det(sigma) / np.sqrt(
            np.linalg.det(np.array([[SC[i] * SC[0], 0], [0, SL[i] * SL[0]]])))) + 0.125 * \
             np.dot(np.dot(np.transpose(mui - mu_bcg), np.linalg.inv(sigma)), (mui - mu_bcg))[0][0]
        if BD >= 3:
            # plt.plot(X[i], Y[i], 'rp', label='point')
            candidates[i] = all_color[i]
    # plt.show()
    return candidates

def fit_chroma_lightness_line(lch_colors):
    # fit chorma_lightness line
    X = [] # C
    Y = [] # L
    w = []
    for i in range(len(lch_colors)):
        # SL/SC: sigma  L/C: mu
        X.append(lch_colors[i][1] / 255.0 * 100.0)
        Y.append(lch_colors[i][0] / 255.0 * 100.0)
        SL_temp = 1 + (0.015 * (Y[i] - 50) ** 2) / math.sqrt(20 + (Y[i] - 50) ** 2)
        SC_temp = 1 + 0.045 * X[i]
        w.append((SL_temp * SC_temp) ** (-2))
    avg_L = np.dot(np.array(Y), np.array(w)) / sum(w)
    avg_C = np.dot(np.array(X), np.array(w)) / sum(w)
    up = down = 0
    for i in range(len(lch_colors)):
        up += w[i] * (avg_L - Y[i]) * (avg_C - X[i])
        down += w[i] * ((avg_L - Y[i]) ** 2 - (avg_C - X[i]) ** 2)
    phi = np.arctan((-2 * up) / down) / 2
    r = avg_C * np.cos(phi) + avg_L * np.sin(phi)
    _Y = [(r - np.cos(phi) * x) / np.sin(phi) for x in X]

    for i in range(len(lch_colors)):
        MD = abs(X[i] * np.cos(phi) + Y[i] * np.sin(phi) - r)
        while MD > 15:
            ver_k = np.sin(phi) / np.cos(phi)
            if (r - Y[i] * np.sin(phi)) / np.cos(phi) < X[i]:
                X[i] -= 0.1
                Y[i] -= 0.1 * ver_k
            else:
                X[i] += 0.1
                Y[i] += 0.1 * ver_k
            MD = abs(X[i] * np.cos(phi) + Y[i] * np.sin(phi) - r)
        if X[i] < 0:
            X[i] = 0
        if Y[i] < 0:
            Y[i] = 0
    for i in range(len(lch_colors)):
        lch_colors[i][1] = int(X[i]) / 100 * 255
        lch_colors[i][0] = int(Y[i]) / 100 * 255

    return lch_colors

def select_from_candidates_not_ambiguous(candidates, ave_sal, num, max_sal_idx): # ave_sal: idx, sal, pixel_num,  sorted by sal
    # B_dis of each 2 colors > 3
    SL = []
    SC = []
    X = []  # C
    Y = []  # L

    # sal max color
    X.append(candidates[max_sal_idx][1] / 255.0 * 100.0)
    Y.append(candidates[max_sal_idx][0] / 255.0 * 100.0)
    SL.append(1 + (0.015 * (Y[0] - 50) ** 2) / math.sqrt(20 + (Y[0] - 50) ** 2))
    SC.append(1 + 0.045 * X[0])

    # bcg idx = 0
    X.append(candidates[0][1] / 255.0 * 100.0)
    Y.append(candidates[0][0] / 255.0 * 100.0)
    SL.append(1 + (0.015 * (Y[1] - 50) ** 2) / math.sqrt(20 + (Y[1] - 50) ** 2))
    SC.append(1 + 0.045 * X[1])

    i = 0
    ave_i = 1
    palette_lch_sal = [[candidates[max_sal_idx], 255], [candidates[0], 0]]  # not real 255, means max
    while ave_i < len(ave_sal) and len(palette_lch_sal) < num:
        if ave_sal[ave_i][0] in candidates.keys():
            lch_temp = candidates[ave_sal[ave_i][0]]
            ok_flag = True
            X_temp = lch_temp[1] / 255.0 * 100.0
            Y_temp = lch_temp[0] / 255.0 * 100.0
            SL_temp = 1 + (0.015 * (Y_temp - 50) ** 2) / math.sqrt(20 + (Y_temp - 50) ** 2)
            SC_temp = 1 + 0.045 * X_temp
            mu_temp = np.array([Y_temp, X_temp]).reshape(2, 1)
            for k in range(len(palette_lch_sal)):
                if abs(lch_temp[2] - palette_lch_sal[k][0][2]) < 5:
                    ok_flag = False
                    break
                else:
                    mui = np.array([Y[k], X[k]]).reshape(2, 1)
                    sigma = np.array([[0.5 * (SC[k] + SC_temp), 0], [0, 0.5 * (SL[k] + SC_temp)]])
                    BD = 0.5 * np.log(np.linalg.det(sigma) / np.sqrt(
                        np.linalg.det(np.array([[SC[k] * SC_temp, 0], [0, SL[k] * SL_temp]])))) + 0.125 * \
                         np.dot(np.dot(np.transpose(mui - mu_temp), np.linalg.inv(sigma)), (mui - mu_temp))[0][0]
                    if BD < 3:
                        ok_flag = False
            if ok_flag:
                i += 1
                X.append(X_temp)
                Y.append(Y_temp)
                SL.append(SL_temp)
                SC.append(SC_temp)
                palette_lch_sal.append([lch_temp, ave_sal[ave_i][1]])
        ave_i += 1

    ave_i = 1
    if num > len(palette_lch_sal):
        group_num = (len(candidates) - 1) // (num - len(palette_lch_sal))
        for i in range(0, num - len(palette_lch_sal)): # select 1 color
            j = i * group_num
            alternate = [[0, 0, 0], -999, 0.0, 0.0, 0.0, 0.0, 0.0] # [lch, BD_min, sal, X, Y, SL, SC]
            selected_flag = False
            while j < (i + 1) * group_num:
                if ave_sal[ave_i][0] in candidates.keys():
                    j += 1
                    lch_temp = candidates[ave_sal[ave_i][0]]
                    X_temp = lch_temp[1] / 255.0 * 100.0
                    Y_temp = lch_temp[0] / 255.0 * 100.0
                    SL_temp = 1 + (0.015 * (Y_temp - 50) ** 2) / math.sqrt(20 + (Y_temp - 50) ** 2)
                    SC_temp = 1 + 0.045 * X_temp
                    mu_temp = np.array([Y_temp, X_temp]).reshape(2, 1)
                    ok_flag = True
                    BD_min = 9999
                    for k in range(len(palette_lch_sal)):
                        if abs(lch_temp[2] - palette_lch_sal[k][0][2]) < 5:
                            mui = np.array([Y[k], X[k]]).reshape(2, 1)
                            sigma = np.array([[0.5 * (SC[k] + SC_temp), 0], [0, 0.5 * (SL[k] + SC_temp)]])
                            BD = 0.5 * np.log(np.linalg.det(sigma) / np.sqrt(
                                np.linalg.det(np.array([[SC[k] * SC_temp, 0], [0, SL[k] * SL_temp]])))) + 0.125 * \
                                 np.dot(np.dot(np.transpose(mui - mu_temp), np.linalg.inv(sigma)), (mui - mu_temp))[0][0]
                            if BD < 3:
                                ok_flag = False
                            if BD < BD_min:
                                BD_min = BD
                    if ok_flag:
                        selected_flag = True
                        alternate = [lch_temp, 0.0, ave_sal[ave_i][1], X_temp, Y_temp, SL_temp, SC_temp]
                        while j < (i + 1) * group_num:
                            if ave_sal[ave_i][0] in candidates.keys():
                                j += 1
                            ave_i += 1
                        break
                    else:
                        if BD_min > alternate[1]:
                            alternate = [lch_temp, BD_min, ave_sal[ave_i][1], X_temp, Y_temp, SL_temp, SC_temp]
                ave_i += 1
            X.append(alternate[3])
            Y.append(alternate[4])
            SL.append(alternate[5])
            SC.append(alternate[6])
            palette_lch_sal.append([alternate[0], alternate[2]])

    palette_lch_sal = sorted(palette_lch_sal, key = lambda x : -1 * x[1])
    palette_lch = []
    palette_sal = []
    for p in palette_lch_sal:
        palette_lch.append(p[0])
        palette_sal.append(p[1])
    return palette_lch, palette_sal

def lab2lch(lab): # input lab range: [0, 255]
    l = lab[0] / 255 * 100
    a = lab[1] - 128
    b = lab[2] - 128
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
    return [l / 100 * 255, a + 128, b + 128]

def adjust_hue(Msh, Munsat):
    if Msh[0] > Munsat:
        return Msh[2]
    else:
        hSpin = Msh[1]*math.sqrt(Munsat*Munsat - Msh[0]*Msh[0]) / Msh[0]*math.sin(Msh[1])
        if Msh[2] > -1 * math.pi/3:
            return Msh[2] + hSpin
        else:
            return Msh[2] - hSpin

def get_har_colors(bcg_lab1, bcg_lab2, ave_sal, num, all_color, type, bcg_flag):
    if type == 'lab':
        for i in range(1, len(all_color) + 1):
            if len(all_color[i]) == 0:
                break
            all_color[i] = lab2lch(all_color[i])
        bcg_lch1 = lab2lch([bcg_lab1[0] / 100 * 255, bcg_lab1[1] + 128, bcg_lab1[2] + 128])

    max_sal_idx = 0
    candidates = not_ambiguous_to_bcg_colors(bcg_lch1, all_color)
    candidates[0] = bcg_lch1
    for i in range(0, len(ave_sal)):
      if ave_sal[i][0] in candidates.keys():
        max_sal_idx = ave_sal[i][0]
        break
    if bcg_flag:
        num += 1

    if len(candidates) <= num:
        print("error")

    palette_lch, palette_sal = select_from_candidates_not_ambiguous(candidates, ave_sal, num, max_sal_idx)

    palette_lab = []
    for lch in palette_lch:
        palette_lab.append(lch2lab(lch))

    palette_lch = fit_chroma_lightness_line(palette_lch)

    palette_lab = []
    for lch in palette_lch:
        palette_lab.append(lch2lab(lch))

    return palette_lab, palette_sal

