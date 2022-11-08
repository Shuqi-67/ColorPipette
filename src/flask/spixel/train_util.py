import torch
import torch.nn.functional as F
import numpy as np
from skimage.segmentation import mark_boundaries
import cv2
import sys
sys.path.append('./third_party/cython')
from connectivity import enforce_connectivity

#===================== pooling and upsampling feature ==========================================

def shift9pos(input, h_shift_unit=1,  w_shift_unit=1):
    # input should be padding as (c, 1+ height+1, 1+width+1)
    input_pd = np.pad(input, ((h_shift_unit, h_shift_unit), (w_shift_unit, w_shift_unit)), mode='edge')
    input_pd = np.expand_dims(input_pd, axis=0)

    # assign to ...
    top     = input_pd[:, :-2 * h_shift_unit,          w_shift_unit:-w_shift_unit]
    bottom  = input_pd[:, 2 * h_shift_unit:,           w_shift_unit:-w_shift_unit]
    left    = input_pd[:, h_shift_unit:-h_shift_unit,  :-2 * w_shift_unit]
    right   = input_pd[:, h_shift_unit:-h_shift_unit,  2 * w_shift_unit:]

    center = input_pd[:,h_shift_unit:-h_shift_unit,w_shift_unit:-w_shift_unit]

    bottom_right    = input_pd[:, 2 * h_shift_unit:,   2 * w_shift_unit:]
    bottom_left     = input_pd[:, 2 * h_shift_unit:,   :-2 * w_shift_unit]
    top_right       = input_pd[:, :-2 * h_shift_unit,  2 * w_shift_unit:]
    top_left        = input_pd[:, :-2 * h_shift_unit,  :-2 * w_shift_unit]

    shift_tensor = np.concatenate([     top_left,    top,      top_right,
                                        left,        center,      right,
                                        bottom_left, bottom,    bottom_right], axis=0)
    return shift_tensor

def update_spixl_map (spixl_map_idx_in, assig_map_in):
    assig_map = assig_map_in.clone()

    b,_,h,w = assig_map.shape
    _, _, id_h, id_w = spixl_map_idx_in.shape

    if (id_h == h) and (id_w == w):
        spixl_map_idx = spixl_map_idx_in
    else:
        spixl_map_idx = F.interpolate(spixl_map_idx_in, size=(h,w), mode='nearest')

    assig_max,_ = torch.max(assig_map, dim=1, keepdim= True)
    assignment_ = torch.where(assig_map == assig_max, torch.ones(assig_map.shape),torch.zeros(assig_map.shape))
    new_spixl_map_ = spixl_map_idx * assignment_ # winner take all
    new_spixl_map = torch.sum(new_spixl_map_,dim=1,keepdim=True).type(torch.int)

    return new_spixl_map

def get_spixel_image(given_img, spix_index, n_spixels = 600, b_enforce_connect = False):

    if not isinstance(given_img, np.ndarray):
        given_img_np_ = given_img.detach().cpu().numpy().transpose(1,2,0)
    else: # for cvt lab to rgb case
        given_img_np_ = given_img

    if not isinstance(spix_index, np.ndarray):
        spix_index_np = spix_index.detach().cpu().numpy().transpose(0,1)
    else:
        spix_index_np = spix_index


    h, w = spix_index_np.shape
    given_img_np = cv2.resize(given_img_np_, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

    if b_enforce_connect:
        spix_index_np = spix_index_np.astype(np.int64)
        segment_size = (given_img_np_.shape[0] * given_img_np_.shape[1]) / (int(n_spixels) * 1.0)
        min_size = int(0.06 * segment_size)
        max_size =  int(3 * segment_size)
        spix_index_np = enforce_connectivity(spix_index_np[None, :, :], min_size, max_size)[0]
    cur_max = np.max(given_img_np)
    spixel_bd_image = mark_boundaries(given_img_np/cur_max, spix_index_np.astype(int), color = (0,1,1)) #cyna
    return (cur_max*spixel_bd_image).astype(np.float32).transpose(2,0,1), spix_index_np #
