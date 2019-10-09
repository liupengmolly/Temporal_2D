import os
import numpy as np
import json
import random
import math
import cv2
import time
import pickle
import scipy.io as sio
import skimage
import skimage.transform
from torchvision.datasets import ImageFolder
import torch.utils.data as data

from Config import cfg
from lib.imutils import *

import torch

#需要保证数据能一个序列（即连续的视频）的处理，这样才能保证LSTM序列处理的正确性
#但是现在这里先实现简单的CPN，所以统一处理，不考虑序列
class H36m(data.Dataset):
    def __init__(self, cfg, img_path, train=True):
        self.img_folder = img_path
        self.is_train = train
        self.inp_res = (cfg.input_H, cfg.input_W)
        self.out_res = (cfg.output_H, cfg.output_W)
        self.num_joints = 17
        self.cfg = cfg
        self.pixel_means = self.cfg.pixel_means
        if self.is_train:
            self.scale_factor = cfg.scale_factor
            self.rot_factor = cfg.rot_factor
            self.symmetry = cfg.symmetry

        imgs_dataset = ImageFolder(self.img_folder)
        self.img_classes= imgs_dataset.classes
        self.imgs = imgs_dataset.imgs
        self.classes2idx = imgs_dataset.class_to_idx

        self.annot_imgs_paths = []
        self.annot_Y2ds = []
        self.annot_Y3ds = []
        self.annot_bboxs = []
        self.annot_num_imgs = []

        if not cfg.load_pickle:
            if self.is_train:
                subjects = ['s_01',
                            's_05',
                            's_06',
                            's_07',
                            's_08',
                            's_11'
                            ]
            else:
                #为了快速评测，暂时只用subjects 9
                subjects = ['s_09']

            # =============
            # 提取对应的annot, 并获取对应的imgs
            for root, dirs, files in os.walk(self.img_folder, topdown=False):
                sub_root = root.split("/")[-1]
                sub_root_subj = sub_root[:4]
                if sub_root_subj in subjects:
                    if cfg.test_seq:
                        #测试一条完整序列的准确率，只取一个子文件夹中的序列
                        if sub_root != cfg.test_seq:
                            continue
                    tmp_annot = sio.loadmat(os.path.join(root,'matlab_meta'))

                    self.annot_imgs_paths.append(self.pad(tmp_annot['images_path'],5))
                    self.annot_Y2ds.append(self.pad(tmp_annot['Y2d'],5))
                    self.annot_Y3ds.append(self.pad(tmp_annot['Y3d_mono'],5))
                    self.annot_bboxs.append(self.pad(tmp_annot['bbox'],5))
                    num_images = tmp_annot['num_images']
                    num_images[[0]] = 5 * math.ceil(num_images[[0]] / 5)
                    self.annot_num_imgs.append(num_images)

            self.annot_imgs_path = np.transpose(np.concatenate(self.annot_imgs_paths, axis=-1))
            self.annot_Y2d = np.transpose(np.concatenate(self.annot_Y2ds, axis=-1))
            self.annot_Y3d = np.transpose(np.concatenate(self.annot_Y3ds, axis=-1))
            self.annot_bbox = np.transpose(np.concatenate(self.annot_bboxs, axis=-1))
            self.annot_num_img = np.transpose(np.concatenate(self.annot_num_imgs, axis=-1))

        self.sub_root_nums = self.annot_num_img.shape[0]
        #由于数据集是每五帧取一帧，所以需要除以5，这点在对标注数据的处理也要注意
        self.imgs_num = math.ceil(np.sum(self.annot_num_img,axis=0)[0]/5)
        if self.is_train:
            self.img_classes = self.img_classes[:self.sub_root_nums]
            self.imgs = self.imgs[:self.imgs_num]
        else:
            self.img_classes = self.img_classes[-self.sub_root_nums:]
            self.imgs = self.imgs[-self.imgs_num:]

    def pad(self, annot, k):
        # 保证每个子文件夹中的图片数目为5的倍数,这样在多个子文件夹合并后,以5为步长去图片时,能保证取到有效的图片
        if annot.shape[1] % k !=0:
            pad_len = k - annot.shape[1]%k
            pad = np.concatenate([annot[:, -1:] for i in range(pad_len)], axis=1)
            annot = np.concatenate([annot, pad], axis=1)
        return annot

    def augmentation(self, img, label):
        """
        由于这只是对一个数据样本的梳理，返回也只能是一个数据样本，所以会有选择性的增强,
        这里在被选择的增强方式上使用随机概率决定
        :param img:
        :param label:
        :return:
        """
        height, width = img.shape[0], img.shape[1]
        center = (width/2., height/2.)
        n = label.shape[0]

        # ======================= crop_aug ==============================
        # 缩小检测的图片，导致图片只截取中间的局部，训练出的模型或许能预测出最终的超出边界的情况，
        # 同时由于超出边界的不可预测性，也可能导致模型预测难以收敛
        if self.cfg.crop_aug and random.uniform(0,1)>0.7:
            affrat = random.uniform(self.scale_factor[0], self.scale_factor[1])
            halfl_w = min(width-center[0], (width-center[0])/1.25*affrat)
            halfl_h = min(height - center[1], (height-center[1])/1.25*affrat)
            img = skimage.transform.resize(
                img[int(center[1]-halfl_h):int(center[1]+halfl_h+1),
                    int(center[0]-halfl_w):int(center[0]+halfl_w+1)],
                (height, width)
            )

            for i in range(n):
                label[i][0] = (label[i][0] - center[0]) / halfl_w * (width - center[0]) + center[0]
                label[i][1] = (label[i][1] - center[1]) / halfl_h * (height - center[1]) + center[1]
        # =================================================================

        # ============================== flip aug ===============================
        if self.cfg.flip_aug and random.uniform(0,1)>0.7:
            img = cv2.flip(img, 1)
            cod = []
            allc = []
            for i in range(n):
                x, y = label[i][0], label[i][1]
                if x >= 0:
                    x = width - 1 - x
                cod.append((x, y))
            # **** the joint index depends on the dataset ****
            for (q, w) in self.symmetry:
                cod[q], cod[w] = cod[w], cod[q]
            for i in range(n):
                allc.append(cod[i][0])
                allc.append(cod[i][1])
            label = np.array(allc).reshape(n, 2)
        # ==========================================================================

        # =========================== rotate aug ===================================
        if self.cfg.rotate_aug and random.uniform(0,1)>0.7:
            angle = random.uniform(0, self.rot_factor)
            if random.randint(0, 1):
                angle *= -1
            rotMat = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotMat, (width, height))

            allc = []
            for i in range(n):
                x, y = label[i][0], label[i][1]
                coor = np.array([x, y])
                if x >= 0 and y >= 0:
                    R = rotMat[:, : 2]
                    W = np.array([rotMat[0][2], rotMat[1][2]])
                    coor = np.dot(R, coor) + W
                allc.append(int(coor[0]))
                allc.append(int(coor[1]))
            label = np.array(allc).reshape(n, 2).astype(np.int)
        # ==========================================================================
        return img, label

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        annot_index = index*5
        meta = {}
        img_path = self.annot_imgs_path[annot_index][0][0]
        if not isinstance(img_path, str):
            return ValueError('the img path is not correct: {}'.format(img_path  ))
        img_dir1, img_name = img_path.split("/")[-2:]
        #todo 测试修改路径
        img_path = os.path.join('/home/chenyuhang/Temporal_2D/data/images',img_dir1, img_name)
        img = cv2.imread(img_path)
        meta['img_h'], meta['img_w'] = img.shape[0], img.shape[1]

        points = self.annot_Y2d[annot_index].reshape(17,2)

        if self.is_train:
            img, points = self.augmentation(img, points)
            img, points = data_resize(img, cfg, points)
            img = im_to_torch(img)

            #color dithering
            img[:, :, 0].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[:, :, 1].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[:, :, 2].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        else:
            img, points = data_resize(img, cfg, points)
            img = im_to_torch(img)

        # output size is 1/4 input size
        points[:,:2] /= 4
        pts = torch.Tensor(points)

        img = color_normalize(img ,self.pixel_means)

        # if self.is_train:
        target = np.zeros((self.num_joints, self.out_res[0], self.out_res[1]))
        for i in range(self.num_joints):
            # 过滤掉不在图片内的关节点
            if pts[i][0]<cfg.output_W and pts[i][1]<cfg.output_H:
                pt = pts[i].numpy()
                x = int(min(round(pt[1]), cfg.output_H-1))
                y = int(min(round(pt[0]), cfg.output_W-1))
                target[i] = generate_heatmap(target[i], x, y, self.cfg.gk)
        target = torch.FloatTensor(target)

        if self.is_train:
            return img, target
        else:
            return img, target

class MultiH36m(H36m):
    def __init__(self, cfg, img_path, train=True, num=5, start_idx = 0):
        super(MultiH36m, self).__init__(cfg, img_path, train)
        # 丢入lstm的图片数
        if train:
            self.interval = 5
        else:
            self.interval = 1
        self.num  = num
        self.start_idx = start_idx

    def __getitem__(self, index):
        item_index =  index * self.interval + self.start_idx
        inputs = []
        targets = []
        #每隔三张取一个序列
        img_path_start = self.annot_imgs_path[item_index*5][0][0]
        img_dir1, img_name1 = img_path_start.split("/")[-2:]
        img_path_end = self.annot_imgs_path[(item_index+self.num-1)*5][0][0]
        img_dir2, img_name2 = img_path_end.split("/")[-2:]

        if img_dir1 != img_dir2:
            inputs, targets = self.padding(item_index)
        else:
            for i in range(self.num):
                input, target = H36m.__getitem__(self, item_index + i)
                inputs.append(input.unsqueeze(0))
                targets.append(target.unsqueeze(0))
        # 如果第一张图片的文件夹和最后一张图片的文件夹不同

        # input_concated = inputs[0]
        # target_concated = targets[0]
        # for i in range(len(inputs)):
        #     input_concated = torch.cat([input_concated,inputs[i]],0)
        #     target_concated = np.concatenate((target_concated, targets[i]), axis=0)
        # input_concated, target_concated = torch.cat(inputs,dim=0), torch.cat(targets, dim=0)
        inputs = torch.cat(inputs, dim=0)
        targets = torch.cat(targets, dim=0)
        return inputs, targets

    def __len__(self):
        return int(int(super().__len__() - self.num + 1)/self.interval) - self.start_idx

    def padding(self, index):
        inputs = []
        targets = []
        flag = False
        padding_index=0
        for i in range(self.num-1, 0, -1):
            img_path_start = self.annot_imgs_path[(index+i)*5][0][0]
            img_dir1, img_name1 = img_path_start.split("/")[-2:]
            img_path_end = self.annot_imgs_path[(index+i-1)*5][0][0]
            img_dir2, img_name2 = img_path_end.split("/")[-2:]
            if flag:
                input, target = H36m.__getitem__(self, padding_index)
                inputs.append(input.unsqueeze(0))
                targets.append(target.unsqueeze(0))
                continue

            if img_dir2 == img_dir1:
                input, target = H36m.__getitem__(self, index+i)
                inputs.append(input.unsqueeze(0))
                targets.append(target.unsqueeze(0))
            else:
                input, target = H36m.__getitem__(self, index+i)
                inputs.append(input.unsqueeze(0))
                targets.append(target.unsqueeze(0))
                # 如果在边界处的话， 把边界后面的那个作为用来 padding的
                padding_index = index+i
                flag = True

        inputs.append(inputs[-1])
        targets.append(targets[-1])

        inputs.reverse()
        targets.reverse()
        return inputs, targets

if __name__ == '__main__':
    '''
    dataset = H36m(cfg,'../data/images',True)
    multiset = MultiH36m(cfg, '../data/images', True)
    print(dataset.__len__())
    item = multiset.__getitem__(1)
    print()
    '''
    '''
    dataset = MultiH36m(cfg, '/home/chenyuhang/Temporal_2D/data/images', True)
    print(dataset.__len__())
    '''


    train_loader = torch.utils.data.DataLoader(
        MultiH36m(cfg, '/home/chenyuhang/Temporal_2D/data/images', True),
        batch_size=2,
        shuffle=True,
        num_workers=1,
        pin_memory=True
    )
    print('ha?')
    #for bidx, (input, target) in enumerate(train_loader):
        #print(bidx)
    




