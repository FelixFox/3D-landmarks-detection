#!/usr/bin/env python3
# coding: utf-8
__author__ = 'cleardusk'

import numpy as np
from math import sqrt
import scipy.io as sio
import matplotlib.pyplot as plt
from .ddfa_utils import reconstruct_vertex


def get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos:]


def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res


def calc_roi_box(pts):
    bbox = [min(pts[0, :]), min(pts[1, :]), max(pts[0, :]), max(pts[1, :])]
    center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
    radius = max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 2
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]

    llength = sqrt((bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2)
    center_x = (bbox[2] + bbox[0]) / 2
    center_y = (bbox[3] + bbox[1]) / 2

    roi_box = [0] * 4
    roi_box[0] = center_x - llength / 2
    roi_box[1] = center_y - llength / 2
    roi_box[2] = roi_box[0] + llength
    roi_box[3] = roi_box[1] + llength

    return roi_box


def dump_key_points_to_ply(key_points,  wfp):
    header = """ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    element face {}
    property list uchar int vertex_indices
    end_header"""

    n_key_points = key_points.shape[1]
    n_face = 1
    header = header.format(n_key_points, n_face)
    with open(wfp, 'w') as f:
        f.write(header + '\n')
        for i in range(n_key_points):
            x, y, z = key_points[:,i]
            f.write('{:.4f} {:.4f} {:.4f}\n'.format(x, y, z))
        f.write('{} '.format(n_key_points))
        for i in range(n_key_points):
            f.write('{} '.format(i))

    print('Dump tp {}'.format(wfp))

    

def dump_to_ply(vertex, tri, wfp):
    header = """ply
    format ascii 1.0
    element vertex {}
    property float x
    property float y
    property float z
    element face {}
    property list uchar int vertex_indices
    end_header"""

    n_vertex = vertex.shape[1]
    n_face = tri.shape[1]
    header = header.format(n_vertex, n_face)

    with open(wfp, 'w') as f:
        f.write(header + '\n')
        for i in range(n_vertex):
            x, y, z = vertex[:, i]
            f.write('{:.4f} {:.4f} {:.4f}\n'.format(x, y, z))
        for i in range(n_face):
            idx1, idx2, idx3 = tri[:, i]
            f.write('3 {} {} {}\n'.format(idx1 - 1, idx2 - 1, idx3 - 1))
    print('Dump tp {}'.format(wfp))


def dump_vertex(vertex, wfp):
    sio.savemat(wfp, {'vertex': vertex})
    print('Dump tp {}'.format(wfp))


def _predict_vertices(param, roi_box, dense):
    vertex = reconstruct_vertex(param, dense=dense)
    sx, sy, ex, ey = roi_box
    scale_x = (ex - sx) / 120
    scale_y = (ey - sy) / 120
    vertex[0, :] = vertex[0, :] * scale_x + sx
    vertex[1, :] = vertex[1, :] * scale_y + sy

    s = (scale_x + scale_y) / 2
    vertex[2, :] *= s

    return vertex


def predict_68pts(param, roi_box):
    return _predict_vertices(param, roi_box, dense=False)


def predict_dense(param, roi_box):
    return _predict_vertices(param, roi_box, dense=True)


def draw_landmarks(img, pts, style='fancy', wfp=None, show_flg=True, **kwargs):
    """Draw landmarks using matpliotlib"""
    plt.figure(figsize=(12, 8))
  
    for i in range(len(pts)):
        alpha = 0.8
        markersize = 4
        lw = 1.5
        color = kwargs.get('color', 'w')
        markeredgecolor = kwargs.get('markeredgecolor', 'black')

        nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

        # close eyes and mouths
        plot_close = lambda i1, i2: plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                 color=color, lw=lw, alpha=alpha - 0.1)
        plot_close(41, 36)
        plot_close(47, 42)
        plot_close(59, 48)
        plot_close(67, 60)

        for ind in range(len(nums) - 1):
            l, r = nums[ind], nums[ind + 1]
            plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

            plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                        color=color,
                        markeredgecolor=markeredgecolor, alpha=alpha)

    plt.axis('on')
    plt.tight_layout()
    return plt
    
def main():
    pass

if __name__ == '__main__':
    main()