import cv2
import numpy as np
import torch
from torchsample.transforms import Pad
import torch.nn.functional as F

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def im_to_torch(img):
    # img = np.transpose(img, (2, 0, 1))
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def im_to_numpy(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0)) # H*W*C
    return img


def color_normalize(x, mean):
    if x.size(0) == 1:
        x = x.repeat(3, 1, 1)
    normalized_mean = mean / 255
    for t, m in zip(x, normalized_mean):
        t.sub_(m)
    return x

def generate_heatmap(heatmap, x, y, sigma):
    heatmap[x][y] = 1
    heatmap = cv2.GaussianBlur(heatmap, sigma, 0)
    am = np.amax(heatmap)
    heatmap /= am/255
    return heatmap

def data_resize(img, cfg, points=None):
    img = np.transpose(img,(2,0,1))
    img_H = img.shape[1]
    img_W = img.shape[2]
    input_H = int(max(img_H, img_H*cfg.input_H/cfg.input_W))
    input_W = int(input_H * cfg.input_W / cfg.input_H)
    newDim = torch.IntTensor((img.shape[0], input_H, input_W))

    newimg = torch.FloatTensor(img,)
    newimg = Pad(newDim)(newimg)

    v_img = torch.unsqueeze(newimg, 0)
    newimg = F.upsample(v_img, size=(cfg.input_H, cfg.input_W),
                        mode='bilinear', align_corners=True).data[0]

    if input_H > img_H:
        top_pad_len = (input_H - img_H)/2
        if points is not None:
            points[:,1] += top_pad_len
    else:
        left_pad_len = (input_W - img_W)/2
        if points is not None:
            points[:,0] += left_pad_len
    h_scale = cfg.input_H/input_H
    w_scale = cfg.input_W/input_W
    if points is not None:
        points[:,0] *= w_scale
        points[:,1] *= h_scale
        return newimg, points
    return newimg

def hp2points(hp, cfg, img_h, img_w):
    """

    :param hp:np.ndarray
    :param cfg:
    :param img_h:
    :param img_w:
    :return:
    """
    points = np.zeros((17, 2))
    for p in range(17):
        sp = hp[p]
        if p==10:
            import time
            time.sleep(1)
        sp /= np.amax(sp)
        border = 10
        dr = np.zeros((cfg.output_H + 2*border, cfg.output_W + 2*border))
        dr[border:-border, border:-border] = sp.copy()
        dr = cv2.GaussianBlur(dr, (21, 21), 0)
        lb = dr.argmax()
        y, x = np.unravel_index(lb, dr.shape)
        dr[y, x] = 0
        lb = dr.argmax()
        py, px = np.unravel_index(lb, dr.shape)
        y -= border
        x -= border
        py -= border + y
        px -= border + x
        ln = (px ** 2 + py ** 2) ** 0.5
        delta = 0.25
        if ln > 1e-3:
            x += delta * px / ln
            y += delta * py / ln
        x = max(0, min(x, cfg.output_W - 1))
        y = max(0, min(y, cfg.output_H - 1))

        pad_h = int(max(img_h, img_h*cfg.input_H/cfg.input_W))
        pad_w = int(pad_h * cfg.input_W / cfg.input_H)
        resx = int(x * (pad_w / cfg.output_W) - (pad_w - img_w)/2)
        resy = int(y * (pad_h / cfg.output_H) - (pad_h - img_h)/2)
        points[p] = np.array([resx, resy])
    return points
