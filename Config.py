import argparse
import numpy as np
from easydict import EasyDict as edict

class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser()

        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--lr', type=float, default=5e-4)
        parser.add_argument('--lr_factor', type=float, default=0.1)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--checkpoint', type=str, default="checkpoint")
        parser.add_argument('--epoch', type=int, default=100)
        parser.add_argument('--model', type=str, default='simple_baseline_v2')
        parser.add_argument('--use_target_weight', action='store_true')
        parser.add_argument('--log_freq', type=int, default=100)
        parser.add_argument('--model_path', type=str, default="")
        parser.add_argument("--ckpt", type=str)

        parser.add_argument('--input_H', type=int, default=256, help="the height of input data")
        parser.add_argument('--input_W', type=int, default=192, help="the width of input data")
        parser.add_argument('--output_H', type=int, default=64, help='the height of output data')
        parser.add_argument('--output_W', type=int, default=48, help='the width of output data')
        parser.add_argument('--rot_factor', type=int, default=45,
                            help='one of data augmentation setting')
        parser.add_argument('--num_class', type=int, default=17, help='the numbers of joints')
        parser.add_argument('--crop_aug', type=bool, default=False,
                            help='whether use the crop augmentation')
        parser.add_argument('--flip_aug', type=bool, default=False,
                            help='whether use the flip augmentation')
        parser.add_argument('--rotate_aug', type=bool, default=False,
                            help='whether use the rotate augmentation')
        parser.add_argument('--flip', type=bool, default=True,
                            help='whether use the flip when predict')

        # ======================= simple baseline ===========================
        parser.add_argument('--res', type=int, default=50)
        parser.add_argument('--deconv_with_bias', type=bool, default=False)
        parser.add_argument('--num_deconv_layers', type=int, default=3)
        parser.add_argument('--num_deconv_filters', type=str, default='256, 256, 256')
        parser.add_argument('--num_deconv_kernels', type=str, default='4,4,4')
        parser.add_argument('--final_conv_kernel', type=int, default=1)

        parser.add_argument('--gpus', type=str, default='0,1,2,3')
        parser.add_argument('--load_pickle', action='store_true')

        self.symmetry_h36m = [(1,4),(2,5),(3,6),(11,14),(12,15),(13,16)]
        self.scale_factor = (0,7, 1.35)
        self.pixel_means = np.array([122.7717, 115.9465, 102.9801])
        self.lr_step = [90, 110]
        self.gk = (7,7)

        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()

        # ======================== param for COCO ===============================
        self.OUTPUT_DIR = ''

        self.DATASET = edict()
        self.DATASET.SELECT_DATA = False
        self.DATASET.DATA_FORMAT = 'jpg'
        self.DATASET.SCALE_FACTOR = 0.25
        self.DATASET.ROT_FACTOR = 30
        self.DATASET.FLIP = True

        self.TEST = edict()
        self.TEST.NMS_THRE = 1.0
        self.TEST.IMAGE_THRE = 0.0
        self.TEST.OKS_THRE = 0.5
        self.TEST.IN_VIS_THRE = 0.0
        self.TEST.COCO_BOX_FILE = ''
        self.TEST.USE_GT_BBOX = True
        self.TEST.COCO_BBOX_FILE = ''


        self.MODEL = edict()
        self.MODEL.IMAGE_SIZE = [256, 256]
        self.MODEL.EXTRA = edict()
        self.MODEL.EXTRA.TARGET_TYPE = 'gaussian'
        self.MODEL.EXTRA.HEATMAP_SIZE = [64, 64]
        self.MODEL.EXTRA.SIGMA = 2

        for key, value in self.args.__dict__.items():
            exec ('self.%s = self.args.%s' % (key, key))

cfg = Config()