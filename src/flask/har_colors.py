import math
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# harmonic h circle
# i: 18       V: 93.6     L: 18 + 79.2
# I: 18       T: 180      Y: 18 + 93.6
# X: 93.6

# def compute_error_for_line_given_points(b, w, X, Y):
#     totalError = 0
#     for i in range(len(X)):
#         x = X[i]
#         y = Y[i]
#         totalError += (y - (w * x + b)) ** 2
#     return totalError / float(len(X))
#
# def step_gradient(b_current, w_current, X, Y, learningRate):
#     b_gradient = 0
#     w_gradient = 0
#     N = float(len(X))
#     for i in range(len(X)):
#         x = X[i]
#         y = Y[i]
#         b_gradient += 2 * ((w_current * x) + b_current - y)
#         w_gradient += 2 * x * ((w_current * x) + b_current - y)
#     b_gradient = b_current / N
#     w_gradient = w_gradient / N
#     new_b = b_current - (learningRate * b_gradient)
#     new_w = w_current - (learningRate * w_gradient)
#     return [new_b, new_w]
#
# def gradient_descent_runner(X, Y, starting_b, starting_w, learning_rate, num_iterations):
#     b = starting_b
#     w = starting_w
#     for i in range(num_iterations):
#         b, w = step_gradient(b, w, X, Y, learning_rate)
#     return [b, w]

def find_Matsuda_hue_type(lch_colors):
    # fit h circle type
    # L T X
    # if all not suitable, save color num outside T/X/L

    best_outside_num = len(lch_colors)
    best_outside_lch_idx = []
    best_type = 'L'
    best_range = []

    # L
    h_range = [0, 18]
    best_range_L = []
    best_outside_num_L = len(lch_colors)
    best_outside_lch_idx_L = []
    for i in range(0, 359):
        outside_num = len(lch_colors)
        outside_lch_idx = []
        for j in range(len(lch_colors)):
            h = lch_colors[j][2]
            if ((h_range[0] <= h_range[1]) and ((((h_range[0] + 59.4) % 360 > (h_range[1] + 120.6) % 360) and (
                    h_range[0] <= h <= h_range[1] or (
                    (h_range[0] + 59.4) % 360 <= h <= 359 and 0 <= h <= (h_range[1] + 120.6) % 360))) or (
                                                        ((h_range[0] + 59.4) % 360 <= (h_range[1] + 120.6) % 360) and (
                                                        h_range[0] <= h <= h_range[1] or (
                                                        h_range[0] + 59.4) % 360 <= h <= (
                                                                h_range[1] + 120.6) % 360)))) or (
                    (h_range[0] > h_range[1]) and (
                    ((h_range[0] + 59.4) % 360 <= (h_range[1] + 120.6) % 360) and (
                    (h_range[0] <= h <= 359 and 0 <= h <= h_range[1]) or (
                    (h_range[0] + 59.4) % 360 <= h <= (h_range[1] + 120.6) % 360))) or (((h_range[0] + 59.4) % 360 > (
                    h_range[1] + 120.6) % 360) and ((h_range[0] <= h <= 359 and 0 <= h <= h_range[1]) or (
                    (h_range[0] + 59.4) % 360 <= h <= 359 and 0 <= h <= (h_range[1] + 120.6) % 360)))):
                outside_num -= 1
            else:
                outside_lch_idx.append(j)
        if outside_num == 0:
            return 'L', lch_colors
        if best_outside_num_L > outside_num:
            best_outside_num_L = outside_num
            best_outside_lch_idx_L = outside_lch_idx.copy()
            best_range_L = [h_range[0], h_range[1], (h_range[0] + 59.4) % 360, (h_range[1] + 120.6) % 360]
        h_range[0] = i
        h_range[1] = (18 + i) % 360
    best_outside_num = best_outside_num_L
    best_type = 'L'
    best_range = best_range_L.copy()
    best_outside_lch_idx = best_outside_lch_idx_L.copy()

    # T
    h_range = [0, 180]
    best_range_T = []
    best_outside_num_T = len(lch_colors)
    best_outside_lch_idx_T = []
    for i in range(0, 359):
        outside_num = len(lch_colors)
        outside_lch_idx = []
        for j in range(len(lch_colors)):
            h = lch_colors[j][2]
            if (h_range[0] <= h_range[1] and h_range[0] <= h <= h_range[1]) or \
                    (h_range[0] > h_range[1] and (h_range[0] <= h <= 359 and 0 <= h <= h_range[1])):
                outside_num -= 1
            else:
                outside_lch_idx.append(j)
        # 因为是从最显著颜色开始的，所以如果是落入范围颜色数量相同的情况下，优先保证显著性高的颜色不落入和谐范围外
        if outside_num == 0:
            return 'T', lch_colors
        if best_outside_num_T > outside_num:
            best_outside_num_T = outside_num
            best_outside_lch_idx_T = outside_lch_idx.copy()
            best_range_T = [h_range[0], h_range[1], h_range[0], h_range[1]]
        h_range[0] = i
        h_range[1] = (180 + i) % 360
    if best_outside_num > best_outside_num_T:
        best_outside_num = best_outside_num_T
        best_type = 'T'
        best_range = best_range_T.copy()
        best_outside_lch_idx = best_outside_lch_idx_T.copy()

    # X
    h_range = [0, 93.6]
    best_range_X = []
    best_outside_num_X = len(lch_colors)
    best_outside_lch_idx_X = []
    for i in range(0, 359):
        outside_num = len(lch_colors)
        outside_lch_idx = []
        for j in range(len(lch_colors)):
            h = lch_colors[j][2]
            if ((h_range[0] <= h_range[1]) and ((((h_range[0] + 180) % 360 > (h_range[1] + 180) % 360) and (
                h_range[0] <= h <= h_range[1] or (
                (h_range[0] + 180) % 360 <= h <= 359 and 0 <= h <= (h_range[1] + 180) % 360))) or (
                                                    ((h_range[0] + 180) % 360 <= (h_range[1] + 180) % 360) and (
                                                    h_range[0] <= h <= h_range[1] or (h_range[0] + 180) % 360 <= h <= (
                                                    h_range[1] + 180) % 360)))) or ((h_range[0] > h_range[1]) and (
                ((h_range[0] + 180) % 360 <= (h_range[1] + 180) % 360) and (
                (h_range[0] <= h <= 359 and 0 <= h <= h_range[1]) or (
                (h_range[0] + 180) % 360 <= h <= (h_range[1] + 180) % 360))) or (((h_range[0] + 180) % 360 > (
                h_range[1] + 180) % 360) and ((h_range[0] <= h <= 359 and 0 <= h <= h_range[1]) or (
                (h_range[0] + 180) % 360 <= h <= 359 and 0 <= h <= (h_range[1] + 180) % 360)))):
                outside_num -= 1
            else:
                outside_lch_idx.append(j)
        if outside_num == 0:
            return 'X', lch_colors
        if best_outside_num_X > outside_num:
            best_outside_num_X = outside_num
            best_outside_lch_idx_X = outside_lch_idx.copy()
            best_range_X = [h_range[0], h_range[1], (h_range[0] + 180) % 360, (h_range[1] + 180) % 360]
        h_range[0] = i
        h_range[1] = (93.6 + i) % 360
    if best_outside_num > best_outside_num_X:
        best_outside_num = best_outside_num_X
        best_type = 'X'
        best_range = best_range_X.copy()
        best_outside_lch_idx = best_outside_lch_idx_X.copy()

    # print(best_outside_num_L, best_outside_num_T, best_outside_num_X)
    for i in range(best_outside_num):
        h = lch_colors[best_outside_lch_idx[i]][2]
        closest_h = 0
        min_dis = 9999
        for j in range(4):
            if min_dis > min((h - best_range[j] + 360) % 360, (best_range[j] - h + 360) % 360):
                min_dis = min((h - best_range[j] + 360) % 360, (best_range[j] - h + 360) % 360)
                closest_h = best_range[j]
        # print(best_outside_lch_idx[i], end = ' ')
        # print(lch_colors[best_outside_lch_idx[i]], end = '->')
        lch_colors[best_outside_lch_idx[i]][2] = closest_h
        # print(lch_colors[best_outside_lch_idx[i]])

    return 'miss ' + str(best_outside_num) + ' ' + best_type, lch_colors

def not_ambiguous_colors(lch_colors):
    # B_dis of each 2 colors > 3
    SL = []
    SC = []
    X = []  # C
    Y = []  # L
    for i in range(len(lch_colors)):
        X.append(lch_colors[i][1] / 255.0 * 100.0)
        Y.append(lch_colors[i][0] / 255.0 * 100.0)
    for i in range(len(lch_colors)):
        # SL/SC: sigma  L/C: mu
        SL.append(1 + (0.015 * (Y[i] - 50) ** 2) / math.sqrt(20 + (Y[i] - 50) ** 2))
        SC.append(1 + 0.045 * X[i])

    # for i in range(len(lch_colors)):
    #     # generate randn points
    #     num, dim = 300, 2
    #     np.random.seed(i)
    #     xnd = np.random.randn(num, dim) # standard 2-dim normal distribution
    #     C = [[SC[i], 0], [0, SL[i]]]
    #     W = [X[i], Y[i]]
    #     Z = np.dot(xnd, C) + W
    #     # CLnd.append(Z / 100.0) # CL 2-dim normal distribution  /100 : 为了方便计算巴氏距离
    #     plt.scatter(Z[:, 0], Z[:,1])
    # plt.show()

    for i in range(1, len(lch_colors)):
        for j in range(0, i):
            mui = np.array([Y[i],X[i]]).reshape(2,1)
            muj = np.array([Y[j],X[j]]).reshape(2,1)
            sigma = np.array([[0.5 * (SC[i] + SC[j]), 0], [0, 0.5 * (SL[i] + SL[j])]])
            BD = 0.5 * np.log(np.linalg.det(sigma) / np.sqrt(
                np.linalg.det(np.array([[SC[i] * SC[j], 0], [0, SL[i] * SL[j]]])))) + 0.125 * \
                 np.dot(np.dot(np.transpose(mui - muj), np.linalg.inv(sigma)), (mui - muj))[0][0]
            # if BD < 3:
            #     print("change color amb: " + str(i))
            while BD < 3:
                # change color[i]
                if 0 <= X[i] - X[j] < 0.1:
                    k = (Y[i] - Y[j]) / 0.1
                elif -0.1 < X[i] - X[j] < 0:
                    k = (Y[i] - Y[j]) / -0.1
                else:
                    k = (Y[i] - Y[j]) / (X[i] - X[j])
                b = Y[i] - k * X[i]
                if X[i] - X[j] > 0:
                    X[i] += 0.1
                    Y[i] = X[i] * k + b
                else:
                    X[i] += 0.1
                    Y[i] = X[i] * k + b
                mui = np.array([Y[i], X[i]]).reshape(2, 1)
                muj = np.array([Y[j], X[j]]).reshape(2, 1)
                BD = 0.5 * np.log(np.linalg.det(sigma) / np.sqrt(
                    np.linalg.det(np.array([[SC[i] * SC[j], 0], [0, SL[i] * SL[j]]])))) + 0.125 * \
                     np.dot(np.dot(np.transpose(mui - muj), np.linalg.inv(sigma)), (mui - muj))[0][0]
    for i in range(len(lch_colors)):
        lch_colors[i][1] = int(X[i]) / 100 * 255
        lch_colors[i][0] = int(Y[i]) / 100 * 255
    return lch_colors

def not_ambiguous_to_bcg_colors(bcg, all_color): # idx, sal, pixel_num
    candidates = {}

    # plt.title("chroma-lightness", fontsize=20)
    # plt.xlabel("chroma", fontsize=14)
    # plt.ylabel("lightness", fontsize=14)
    # plt.gca().set_aspect(1)

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

    # plt.plot(X[0], Y[0], 'bp', label='point')

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

def fit_Matsuda_hue_type_colors(candidates, max_sal_idx):
    # L T X
    # return a type with max colors in it

    new_candidates = {}
    h_range = [0, 18]
    best_range = []
    best_type = 'L'

    # L
    for i in range(0, 359):
        h_range[0] = i
        h_range[1] = (18 + i) % 360
        bcg_out_flag = False
        new_candidates_L_temp = {}
        for j in candidates.keys():
            h = candidates[j][2]
            if ((h_range[0] <= h_range[1]) and ((((h_range[0] + 59.4) % 360 > (h_range[1] + 120.6) % 360) and (
                    h_range[0] <= h <= h_range[1] or (
                    (h_range[0] + 59.4) % 360 <= h <= 359 and 0 <= h <= (h_range[1] + 120.6) % 360))) or (
                                                        ((h_range[0] + 59.4) % 360 <= (h_range[1] + 120.6) % 360) and (
                                                        h_range[0] <= h <= h_range[1] or (
                                                        h_range[0] + 59.4) % 360 <= h <= (
                                                                h_range[1] + 120.6) % 360)))) or (
                    (h_range[0] > h_range[1]) and (
                    ((h_range[0] + 59.4) % 360 <= (h_range[1] + 120.6) % 360) and (
                    (h_range[0] <= h <= 359 and 0 <= h <= h_range[1]) or (
                    (h_range[0] + 59.4) % 360 <= h <= (h_range[1] + 120.6) % 360))) or (((h_range[0] + 59.4) % 360 > (
                    h_range[1] + 120.6) % 360) and ((h_range[0] <= h <= 359 and 0 <= h <= h_range[1]) or (
                    (h_range[0] + 59.4) % 360 <= h <= 359 and 0 <= h <= (h_range[1] + 120.6) % 360)))):
                new_candidates_L_temp[j] = candidates[j]
            elif j == max_sal_idx: # max sal color outside
                bcg_out_flag = True
                break
        if bcg_out_flag:
            h_range[0] = i
            h_range[1] = (18 + i) % 360
            continue
        if len(new_candidates_L_temp) > len(new_candidates):
            best_type = 'L'
            new_candidates = new_candidates_L_temp.copy()
            best_range = [h_range[0], h_range[1], (h_range[0] + 59.4) % 360, (h_range[1] + 120.6) % 360]

    # T
    h_range = [0, 180]
    for i in range(0, 359):
        h_range[0] = i
        h_range[1] = (180 + i) % 360
        bcg_out_flag = False
        new_candidates_T_temp = {}
        for j in candidates.keys():
            h = candidates[j][2]
            if (h_range[0] <= h_range[1] and h_range[0] <= h <= h_range[1]) or \
                    (h_range[0] > h_range[1] and (h_range[0] <= h <= 359 and 0 <= h <= h_range[1])):
                new_candidates_T_temp[j] = candidates[j]
            elif j == max_sal_idx:  # max sal color outside
                bcg_out_flag = True
                break
        if bcg_out_flag:
            continue
        if len(new_candidates_T_temp) > len(new_candidates):
            best_type = 'T'
            new_candidates = new_candidates_T_temp.copy()
            best_range = [h_range[0], h_range[1], h_range[0], h_range[1]]

    # X
    h_range = [0, 93.6]
    for i in range(0, 359):
        h_range[0] = i
        h_range[1] = (93.6 + i) % 360
        bcg_out_flag = False
        new_candidates_X_temp = {}
        for j in candidates.keys():
            h = candidates[j][2]
            if ((h_range[0] <= h_range[1]) and ((((h_range[0] + 180) % 360 > (h_range[1] + 180) % 360) and (
                h_range[0] <= h <= h_range[1] or (
                (h_range[0] + 180) % 360 <= h <= 359 and 0 <= h <= (h_range[1] + 180) % 360))) or (
                                                    ((h_range[0] + 180) % 360 <= (h_range[1] + 180) % 360) and (
                                                    h_range[0] <= h <= h_range[1] or (h_range[0] + 180) % 360 <= h <= (
                                                    h_range[1] + 180) % 360)))) or ((h_range[0] > h_range[1]) and (
                ((h_range[0] + 180) % 360 <= (h_range[1] + 180) % 360) and (
                (h_range[0] <= h <= 359 and 0 <= h <= h_range[1]) or (
                (h_range[0] + 180) % 360 <= h <= (h_range[1] + 180) % 360))) or (((h_range[0] + 180) % 360 > (
                h_range[1] + 180) % 360) and ((h_range[0] <= h <= 359 and 0 <= h <= h_range[1]) or (
                (h_range[0] + 180) % 360 <= h <= 359 and 0 <= h <= (h_range[1] + 180) % 360)))):
                new_candidates_X_temp[j] = candidates[j]
            elif j == max_sal_idx:  # max sal color outside
                bcg_out_flag = True
                break
        if bcg_out_flag:
            continue
        if len(new_candidates_X_temp) > len(new_candidates):
            best_type = 'X'
            new_candidates = new_candidates_X_temp.copy()
            best_range = [h_range[0], h_range[1], (h_range[0] + 180) % 360, (h_range[1] + 180) % 360]

    return best_type, new_candidates, best_range

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

    # plt.figure()
    # plt.title("chroma-lightness-ori", fontsize=20)
    # plt.xlabel("chroma", fontsize=14)
    # plt.ylabel("lightness", fontsize=14)
    # xlim = max(X) + 20
    # ylim = max(Y) + 20
    # plt.xlim(0, xlim)
    # plt.ylim(0, ylim)
    # plt.plot(X, Y, 'bp', label='point')
    # plt.plot(X, _Y, 'r', label='line')
    # plt.gca().set_aspect(1)
    #
    # plt.figure()
    # plt.title("chroma-lightness-changed", fontsize=20)
    # plt.xlabel("chroma", fontsize=14)
    # plt.ylabel("lightness", fontsize=14)
    # plt.xlim(0, xlim)
    # plt.ylim(0, ylim)
    # plt.plot(X, _Y, 'r', label='line')
    # plt.gca().set_aspect(1)

    for i in range(len(lch_colors)):
        MD = abs(X[i] * np.cos(phi) + Y[i] * np.sin(phi) - r)
        # if MD > 5:
        #     print("change color cl line:" + str(i))
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

    # plt.plot(X, Y, 'bp', label='point')
    # plt.show()

    return lch_colors

    # fisrt Ba dis then Ma dis

    # initial_b = random.random()
    # initial_w = random.random()
    # # print("error:", end=" ")
    # # print(compute_error_for_line_given_points(initial_b, initial_w, X, Y), end="->")
    # [b, w] = gradient_descent_runner(X, Y, initial_b, initial_w, learning_rate=0.0001,
    #                                  num_iterations=1000)
    # print(compute_error_for_line_given_points(b, w, X, Y))
    # # print(str(w) + "* x + " + str(b))
    # _Y = [(w * x + b) for x in X]
    # # draw chorma_lightness line
    # plt.title("chroma-lightness222", fontsize=20)
    # plt.xlabel("chroma", fontsize=14)
    # plt.ylabel("lightness", fontsize=14)
    # plt.xlim(0, max(X) + 20)
    # plt.ylim(0, max(Y) + 20)
    # plt.plot(X, Y, 'bp', label='point')
    # plt.plot(X, _Y, 'r', label='line')
    # plt.show()

    # z1 = np.polyfit(X, Y, deg = 1)
    # p1 = np.poly1d(z1)
    # print(p1)
    # initial_b = random.random()
    # initial_w = random.random()
    # print("error:", end=" ")
    # print(compute_error_for_line_given_points(initial_b, initial_w, X, Y), end="->")
    # [b, w] = gradient_descent_runner(X, Y, initial_b, initial_w, learning_rate=0.0001,
    #                                  num_iterations=1000)
    # print(compute_error_for_line_given_points(b, w, X, Y))
    # print(str(w) + "* x + " + str(b))
    # _Y = [(w * x + b) for x in X]
    # # draw chorma_lightness line
    # plt.title("chroma-lightness", fontsize=20)
    # plt.xlabel("chroma", fontsize=14)
    # plt.ylabel("lightness", fontsize=14)
    # plt.plot(X, Y, 'bp', label='point')
    # plt.plot(X, _Y, 'r', label='line')
    # plt.show()

def select_from_candidates_not_ambiguous(candidates, ave_sal, num, max_sal_idx): # ave_sal: idx, sal, pixel_num,  sorted by sal
    # B_dis of each 2 colors > 3
    SL = []
    SC = []
    X = []  # C
    Y = []  # L

    # palette_lch = [candidates['0']]
    # palette_sal = [0]
    # j = 0
    # ave_i = 0
    # while(j < num):
    #     if str(ave_sal[ave_i][0]) in candidates.keys():
    #         lch_temp = candidates[str(ave_sal[ave_i][0])]
    #         palette_lch.append(lch_temp)
    #         palette_sal.append('1')
    #         j += 1
    #     ave_i += 1
    # return palette_lch, palette_sal

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

    # 前一半颜色取显著性高且h不同的颜色
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

    # 后一半颜色每个candidates显著性阶段选一个颜色
    ave_i = 1
    if num > len(palette_lch_sal):
        group_num = (len(candidates) - 1) // (num - len(palette_lch_sal))
        for i in range(0, num - len(palette_lch_sal)): # select 1 color
            j = i * group_num
            # HD_min > 25 or (HD_min <= 30 and DB > 3)就满足条件，这种情况下看sal最大的颜色
            # 如果区间内没有一个颜色满足前面的情况，则不看sal，看BD_min最大的颜色
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
                        # print(BD_min, i, alternate)
                        if BD_min > alternate[1]:
                            alternate = [lch_temp, BD_min, ave_sal[ave_i][1], X_temp, Y_temp, SL_temp, SC_temp]
                ave_i += 1
            X.append(alternate[3])
            Y.append(alternate[4])
            SL.append(alternate[5])
            SC.append(alternate[6])
            palette_lch_sal.append([alternate[0], alternate[2]])


    # ave_i = 0
    # palette_lch = [candidates['0']]
    # palette_sal = [0]
    # palette_lch_alternate = [[[0, 0, 0], 0, 0] for i in range(6)]  # [lch, BD_sum, sal] sorted by BD_sum, len <= 6
    #
    #
    # while len(palette_lch) < num + 1 and ave_i < len(ave_sal):
    #     if str(ave_sal[ave_i][0]) in candidates.keys():
    #         lch_temp = candidates[str(ave_sal[ave_i][0])]
    #         X_temp = lch_temp[1] / 255.0 * 100.0
    #         Y_temp = lch_temp[0] / 255.0 * 100.0
    #         SL_temp = 1 + (0.015 * (Y_temp - 50) ** 2) / math.sqrt(20 + (Y_temp - 50) ** 2)
    #         SC_temp = 1 + 0.045 * X_temp
    #         mu_temp = np.array([Y_temp, X_temp]).reshape(2, 1)
    #         ok_flag = True
    #         BD_sum = 0
    #         for i in range(len(palette_lch)):
    #             mui = np.array([Y[i], X[i]]).reshape(2, 1)
    #             sigma = np.array([[0.5 * (SC[i] + SC_temp), 0], [0, 0.5 * (SL[i] + SC_temp)]])
    #             BD = 0.5 * np.log(np.linalg.det(sigma) / np.sqrt(
    #                 np.linalg.det(np.array([[SC[i] * SC_temp, 0], [0, SL[i] * SL_temp]])))) + 0.125 * \
    #                  np.dot(np.dot(np.transpose(mui - mu_temp), np.linalg.inv(sigma)), (mui - mu_temp))[0][0]
    #             if BD < 3:
    #                 ok_flag = False
    #             BD_sum += BD
    #         if ok_flag:
    #             X.append(X_temp)
    #             Y.append(Y_temp)
    #             SL.append(SL_temp)
    #             SC.append(SC_temp)
    #             palette_lch.append(lch_temp)
    #             palette_sal.append(ave_sal[ave_i][1])
    #         else:
    #             for j in range(len(palette_lch_alternate)):
    #                 BD_avg = BD_sum / len(palette_lch)
    #                 if BD_avg > palette_lch_alternate[j][1]:
    #                     for k in range(len(palette_lch_alternate) - 1, j, -1):
    #                         palette_lch_alternate[k] = palette_lch_alternate[k - 1]
    #                     palette_lch_alternate[j] = [lch_temp, BD_avg, ave_sal[ave_i][1]]
    #     ave_i += 1
    # alternate_idx = 0
    # while len(palette_lch) < num + 1:
    #     palette_lch.append(palette_lch_alternate[alternate_idx][0])
    #     palette_sal.append(palette_lch_alternate[alternate_idx][2])
    #     alternate_idx += 1
    palette_lch_sal = sorted(palette_lch_sal, key = lambda x : -1 * x[1])
    print(palette_lch_sal)
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

# get 2 colors from candidates for continuous colormap
def get_continuous_colors(ave_sal, all_color): # all_color is lab
    palette = [all_color[ave_sal[0][0]], all_color[ave_sal[1][0]]]
    palette_sal = [ave_sal[0][1], ave_sal[1][1]]
    max_lab_dis = 0
    palette[0] = [palette[0][0] / 255 * 100, palette[0][1] - 128, palette[0][2] - 128]
    ave_sal.sort(key=lambda student: student[2], reverse=True)
    for i in range(1, int(len(ave_sal) / 5)):
        color2 = all_color[ave_sal[i][0]]
        # trans to [0~100, -127~128, -127~128]
        color2 = [color2[0] / 255 * 100, color2[1] - 128, color2[2] - 128]
        lab_dis = (color2[0] - palette[0][0])*(color2[0] - palette[0][0]) + (color2[1] - palette[0][1])*(color2[1] - palette[0][1]) + (color2[2] - palette[0][2])*(color2[2] - palette[0][2])
        if max_lab_dis < lab_dis:
            palette[1] = color2
            max_lab_dis = lab_dis
            palette_sal[1] = ave_sal[i][1]
    palette[0] = [palette[0][0] / 100 * 255, palette[0][1] + 128, palette[0][2] + 128]
    palette[1] = [palette[1][0] / 100 * 255, palette[1][1] + 128, palette[1][2] + 128]
    return  palette, palette_sal

def adjust_hue(Msh, Munsat):
    if Msh[0] > Munsat:
        return Msh[2]
    else:
        hSpin = Msh[1]*math.sqrt(Munsat*Munsat - Msh[0]*Msh[0]) / Msh[0]*math.sin(Msh[1])
        if Msh[2] > -1 * math.pi/3:
            return Msh[2] + hSpin
        else:
            return Msh[2] - hSpin

def rad_diff(h0, h1):
    if h0 < 0:
        h0 += math.pi * 2
    if h1 < 0:
        h1 += math.pi * 2
    diff = abs(h0 - h1)
    if diff > math.pi:
        diff = math.pi * 2 - diff
    return diff


def interpolate_color(Msh0, Msh1, interp): # return lab
    if Msh0[1] > 0.05 and Msh1[1] > 0.05 and rad_diff(Msh0[2], Msh1[2]) > math.pi/3:
        Mmid = max(Msh1[0], Msh0[0], 88)
        if interp < 0.5:
            Msh1[0] = Mmid
            Msh1[1] = 0
            Msh1[2] = 0
            interp = 2 * interp
        else:
            Msh0[0] = Mmid
            Msh0[1] = 0
            Msh0[2] = 0
            interp = 2 * interp - 1
    if Msh0[1] < 0.05 and Msh1[1] > 0.05:
        Msh0[2] = adjust_hue(Msh1, Msh0[0])
    elif Msh1[1] < 0.05 and Msh0[1] > 0.05:
        Msh1[2] = adjust_hue(Msh0, Msh1[0])

    M_res = (1 - interp) * Msh0[0] + interp * Msh1[0]
    s_res = (1 - interp) * Msh0[1] + interp * Msh1[1]
    h_res = (1 - interp) * Msh0[2] + interp * Msh1[2]


    return [M_res * math.cos(s_res), M_res * math.sin(s_res) * math.cos(h_res), M_res * math.sin(s_res) * math.sin(h_res)]

def get_har_colors(bcg_lab1, bcg_lab2, ave_sal, num, all_color, type, bcg_flag):
    # about 400+ colors in all_color
    # ave_sal: idx, sal, pixel_num
    # num - the number of foreground colors

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

def change_to_har_colors(ori_color, type, img_name):
    color_num = len(ori_color)
    ori_lch = []

    if type == 'lab':
        for color in ori_color:
            ori_lch.append(lab2lch(color))

        typer, h_change_lch = find_Matsuda_hue_type(ori_lch)
        print(typer)
        h_change_lch = not_ambiguous_colors(h_change_lch)
        h_change_lch = fit_chroma_lightness_line(h_change_lch)

        # fit_chroma_lightness_line(h_change_lch)
        # not_ambiguous_colors(h_change_lch)

        h_change_lab = []
        for lch in h_change_lch:
            h_change_lab.append(lch2lab(lch))
