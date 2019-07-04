import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import random
from collections import OrderedDict

from utils.Config import cfg
from utils.loss import JointsMSELoss
from utils.evaluate import *
from utils.imutils import *
from models.Simple_baseline import get_pose_net
from utils.vis import vis

def vis_img(path, cfg):
    os.environ['CUDA_VISIBLE_DEVICE'] = '3'
    if not os.path.exists(path):
        return ValueError("the img path is not exits")

    model = get_pose_net(cfg)
    model = model.cuda()


    ckpt = torch.load(os.path.join(cfg.checkpoint, cfg.model_path))['state_dict']

    #之前训练是多卡训练的，参数会带上“module.”的前缀，这里需要除掉
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        name = k[7:]
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    img = cv2.imread(path)

    img_h, img_w = img.shape[0], img.shape[1]
    img = data_resize(img, cfg)
    img = im_to_torch(img)

    img[:, :, 0].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
    img[:, :, 0].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
    img[:, :, 0].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

    img = color_normalize(img, cfg.pixel_means)
    img = np.expand_dims(img,0)
    img = torch.autograd.Variable(torch.from_numpy(img).cuda())

    output = model(img).detach().cpu().numpy()
    output = output[0]
    points = hp2points(output, cfg, img_h, img_w)

    vis(path, points)
    return output, points


if __name__ == '__main__':
    matplotlib.use('TKAgg')
    output, points = vis_img('data/images/s_11_act_16_subact_02_ca_04/'
                             's_11_act_16_subact_02_ca_04_001111.jpg', cfg)
    print(output)
    print(points)