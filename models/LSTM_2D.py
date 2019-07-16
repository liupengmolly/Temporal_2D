import torch
import torch.nn as nn
import torch.nn.functional as F

BN_MOMENTUM = 0.1

class ConvNet1(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ConvNet1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=128, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.ReLU(inplace=False)
        #self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=9, stride=1, padding=4, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=9, stride=1, padding=4, bias=False)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.conv5 = nn.Conv2d(32, 512, kernel_size=9, stride=1, padding=4, bias=False)
        self.drop5 = nn.Dropout(p=0.5, inplace=True)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.drop6 = nn.Dropout(p=0.5, inplace=True)
        self.conv7 = nn.Conv2d(512, out_channels=outplanes, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn1 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        self.bn2 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        self.bn3 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        self.bn4 = nn.BatchNorm2d(32, momentum=BN_MOMENTUM)
        self.bn5 = nn.BatchNorm2d(512, momentum=BN_MOMENTUM)
        self.bn6 = nn.BatchNorm2d(512, momentum=BN_MOMENTUM)
        self.bn7 = nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM)

    def forward(self, x):
        # residual
        #residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.pool3(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        out = self.conv5(out)
        out = self.bn5(out)
        out = self.relu(out)
        out = self.drop5(out)

        out = self.conv6(out)
        out = self.bn6(out)
        out = self.relu(out)
        out = self.drop6(out)

        out = self.conv7(out)
        out = self.bn7(out)

        return out

class ConvNet2(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ConvNet2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=128, kernel_size=9, stride=1, padding=4, bias=False)
        self.relu = nn.ReLU(inplace=False)
        #self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=9, stride=1, padding=4, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=9, stride=1, padding=4, bias=False)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, out_channels=outplanes, kernel_size=5, stride=1, padding=2, bias=False)

        self.bn1 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        self.bn2 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        self.bn3 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        self.bn4 = nn.BatchNorm2d(outplanes, momentum=BN_MOMENTUM)
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.pool2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.pool3(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)

        return out

class ConvNet3(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ConvNet3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inplanes, out_channels=128, kernel_size=11, stride=1, padding=5, bias=False)
        self.relu = nn.ReLU(inplace=False)
        self.bn1 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=11, stride=1, padding=5, bias=False)
        self.bn2 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=11, stride=1, padding=5, bias=False)
        self.bn3 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(128, momentum=BN_MOMENTUM)
        self.conv5 = nn.Conv2d(128, out_channels=outplanes, kernel_size=1,padding=0, bias=False)
        #self.bn5 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)

    def forward(self, x):
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        #out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        #out = self.bn4(out)
        out = self.relu(out)

        out = self.conv5(out)

        return out

class Conv_LSTM(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Conv_LSTM, self).__init__()
        '''
        self.conv_ix = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1, bias=True)
        self.conv_ih = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)

        self.conv_fx = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1, bias=True)
        self.conv_fh = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)

        self.conv_ox = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1, bias=True)
        self.conv_oh = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)

        self.conv_gx = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1, bias=True)
        self.conv_gh = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)
        '''
        # 感觉不需要下面的初始化
        self.init_gx = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        self.init_ix = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        self.init_ox = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)


    def forward(self, x, cell_t_1, hide_t_1):

        gx = self.conv_gx(x)
        gh = self.conv_gh(hide_t_1)
        g_sum = gx + gh
        gt = torch.tanh(g_sum)

        ox = self.conv_ox(x)
        oh = self.conv_oh(hide_t_1)
        o_sum = ox + oh
        ot = torch.sigmoid(o_sum)

        ix = self.conv_ix(x)
        ih = self.conv_ih(hide_t_1)
        i_sum = ix + ih
        it = torch.sigmoid(i_sum)

        fx = self.conv_fx(x)  # output: 48 * 45 * 45
        fh = self.conv_fh(hide_t_1)  # output: 48 * 45 * 45
        f_sum = fx + fh
        ft = torch.sigmoid(f_sum)

        cell_t = ft * cell_t_1 + it * gt
        hide_t = ot * torch.tanh(cell_t)
        return cell_t, hide_t
    
    def lstm_init(self, x):
        gx = self.init_gx(x)
        ix = self.init_ix(x)
        ox = self.init_ox(x)

        gx = torch.tanh(gx)
        ix = torch.sigmoid(ix)
        ox = torch.sigmoid(ox)

        cell1 = torch.tanh(gx * ix)
        hide_1 = ox * cell1
        return cell1, hide_1


    '''
    def forward(self, x):
        gx = self.init_gx(x)
        ix = self.init_ix(x)
        ox = self.init_ox(x)

        gx = torch.tanh(gx)
        ix = torch.sigmoid(ix)
        ox = torch.sigmoid(ox)

        cell1 = torch.tanh(gx * ix)
        hide_1 = ox * cell1
        return cell1, hide_1
    '''

class LSTM_2D(nn.Module):
    """
    heatmap的输出顺序：
    "keypoints": {
        0: "nose",
        1: "left_eye",
        2: "right_eye",
        3: "left_ear",
        4: "right_ear",
        5: "left_shoulder",
        6: "right_shoulder",
        7: "left_elbow",
        8: "right_elbow",
        9: "left_wrist",
        10: "right_wrist",
        11: "left_hip",
        12: "right_hip",
        13: "left_knee",
        14: "right_knee",
        15: "left_ankle",
        16: "right_ankle",
        17: "Pelv",
        18: "Thrx",
        19: "Neck",
        20: "Head"
    },
    """
    def __init__(self, type, inplanes=3, outclass=21, stage_num=1):
        super(LSTM_2D, self).__init__()
        coco_kps = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        mpii_kps = [16,14,12,11,13,15,17,18,19,20,10,8,6,5,7,9]
        h36m_kps = [17,12,14,16,11,13,15,18,19,0,20,5,7,9,6,8,10]

        if type=='coco':
            self.kps = coco_kps
        elif type=='mpii':
            self.kps = mpii_kps
        elif type=='h36m':
            self.kps = h36m_kps
        else:
            raise ValueError("the type {} not in ['coco','mpii','h36m']".format(type))

        #common kps包括人的四肢对应的关节点，还可以有nose(mpii由head与neck得来)
        #之所以引入nose是因为nose信息由当前四肢关节点并不能得到，其他的如hip，neck可以由四肢关节点估计，强行引入反而引入了不确定性
        self.type = type
        self.common_kps = [5,6,7,8,9,10,11,12,13,14,15,16]

        self.convnet1 = ConvNet1(inplanes,outclass)
        self.convnet2 = ConvNet2(inplanes, 32)
        # 这边要是原图4分之一的话 下面这种也可以 不知道哪种效果好一点
        #conv_lstm 的 input 上述两个output(avgpool)
        self.conv_lstm = Conv_LSTM(32+len(self.common_kps)+1, 48)
        self.convnet3 = ConvNet3(48, outclass)
        self.stage_num = stage_num

    def get_common_heatmap(self, heatmap):
        common_heatmap = torch.ones((
            heatmap.size(0),
            len(self.common_kps)+1,
            heatmap.size(2),
            heatmap.size(3)
        ), dtype=torch.float32)

        common_heatmap[:, :len(self.common_kps)] = heatmap[:,self.common_kps]
        if self.type == 'coco' or self.type == 'h36m':
            common_heatmap[:, -1] = heatmap[:, 0]
        elif self.type == 'mpii':
            common_heatmap[:, -1] = (heatmap[:, 19] + heatmap[:, 20])/2

        common_heatmap = common_heatmap.cuda()

        return common_heatmap


    def init(self, image):
        initial_heatmap = self.convnet1(image)

        common_heatmap = self.get_common_heatmap(initial_heatmap)

        features = self.convnet2(image)

        x = torch.cat([common_heatmap, features], dim=1)
        cell, hide = self.conv_lstm.lstm_init(x)
        heatmap = self.convnet3(hide)

        initial_heatmap =  initial_heatmap[:, self.kps]
        heatmap = heatmap[:, self.kps]
        return initial_heatmap, heatmap, cell, hide

    def stage(self, image, prev_heatmap, cell_t_1, hide_t_1):
        features = self.convnet2(image)

        common_heatmap = self.get_common_heatmap(prev_heatmap)

        x = torch.cat([common_heatmap, features],  dim=1)
        cell_t, hide_t = self.conv_lstm(x, cell_t_1,hide_t_1)
        heatmap = self.convnet3(hide_t)
        #TODO 这边还需要根据热图的尺寸再加个卷积修改一下尺寸

        heatmap = heatmap[:, self.kps]
        return heatmap, cell_t, hide_t

    def forward(self, images, heatmap=None, hide=None, cell=None, initial=True):
        heatmaps = []
        #这边把一个网络的图片堆在了第2维上
        #第一维是网络训练时的样本数量
        #第二个是一次训练需要的样本数 0～3是RGB
        image = images[:,0:3,:,:]
        initial_heatmap, heatmap, cell, hide = self.init(image=image)
        heatmaps.append(initial_heatmap)
        heatmaps.append(heatmap)

        for i in range(1, self.stage_num):
            image = images[:,(3*i):(3*i+3),:,:]
            # image = images
            heatmap, cell, hide = self.stage(image, heatmap, cell, hide)
            heatmaps.append(heatmap)

        #暂时只出第一张热图 todo
        # 这边返回的值一定要是能算网络最后输出的值 之前输出的是heatmaps[0] 一直是convnet1的值
        # 所以反向求导的时候一直只能求导convnet1 后面全是None
        return heatmaps


        # ===这边的concat操作需要再确定=====
        # X = x_feature+heatmap+center_map

        #=================================



    '''
    def tss(self, x):
        x = self.convnet1(x)
        t, x = self.conv_lstm(x)
        x = self.convnet3(x)

        return x,1,2
    def forward(self, x, image=None, center=None):
        #c,_,_ = self.tss(x)
        #return c
        x1,x2,_,_= self.init(x,x)
        return x2
    '''


class lstm_complete(nn.Module):
    def __init__(self, outclass=17, T=1):
        super(lstm_complete, self).__init__()
        self.outclass = outclass
        self.T = T

        # conv_net1
        self.conv1_convnet1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)  # 3 * 368 * 368
        #self.pool1_convnet1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_convnet1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool2_convnet1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_convnet1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.pool3_convnet1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_convnet1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5_convnet1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6_convnet1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7_convnet1 = nn.Conv2d(512, self.outclass, kernel_size=1)  # 512 * 45 * 45

        # conv_net2
        self.conv1_convnet2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)  # 3 * 368 * 368
        #self.pool1_convnet2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2_convnet2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)  # 128 * 184 * 184
        self.pool2_convnet2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv3_convnet2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)  # 128 * 92 * 92
        self.pool3_convnet2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv4_convnet2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)  # 32 * 45 * 45

        # conv_net3
        self.Mconv1_convnet3 = nn.Conv2d(48, 128, kernel_size=11, padding=5)
        self.Mconv2_convnet3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3_convnet3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4_convnet3 = nn.Conv2d(128, 128, kernel_size=1, padding=0)
        self.Mconv5_convnet3 = nn.Conv2d(128, self.outclass, kernel_size=1, padding=0)

        # lstm
        self.conv_ix_lstm = nn.Conv2d(32  + self.outclass, 48, kernel_size=3, padding=1, bias=True)
        self.conv_ih_lstm = nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False)

        self.conv_fx_lstm = nn.Conv2d(32  + self.outclass, 48, kernel_size=3, padding=1, bias=True)
        self.conv_fh_lstm = nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False)

        self.conv_ox_lstm = nn.Conv2d(32  + self.outclass, 48, kernel_size=3, padding=1, bias=True)
        self.conv_oh_lstm = nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False)

        self.conv_gx_lstm = nn.Conv2d(32  + self.outclass, 48, kernel_size=3, padding=1, bias=True)
        self.conv_gh_lstm = nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False)

        # initial lstm
        self.conv_gx_lstm0 = nn.Conv2d(32  + self.outclass, 48, kernel_size=3, padding=1)
        self.conv_ix_lstm0 = nn.Conv2d(32  + self.outclass, 48, kernel_size=3, padding=1)
        self.conv_ox_lstm0 = nn.Conv2d(32  + self.outclass, 48, kernel_size=3, padding=1)

    def convnet1(self, image):
        '''
        :param image: 3 * 368 * 368
        :return: initial_heatmap out_class * 45 * 45
        '''
        #x = self.pool1_convnet1(F.relu(self.conv1_convnet1(image)))  # output 128 * 184 * 184
        x = F.relu(self.conv1_convnet1(image))
        x = self.pool2_convnet1(F.relu(self.conv2_convnet1(x)))  # output 128 * 92 * 92
        x = self.pool3_convnet1(F.relu(self.conv3_convnet1(x)))  # output 128 * 45 * 45
        x = F.relu(self.conv4_convnet1(x))  # output 32 * 45 * 45
        x = F.relu(self.conv5_convnet1(x))  # output 512 * 45 * 45
        x = F.relu(self.conv6_convnet1(x))  # output 512 * 45 * 45
        initial_heatmap = self.conv7_convnet1(x)  # output (class + 1) * 45 * 45
        return initial_heatmap

    def convnet2(self, image):
        '''
        :param image: 3 * 368 * 368
        :return: Fs(.) features 32 * 45 * 45
        '''
        #x = self.pool1_convnet2(F.relu(self.conv1_convnet2(image)))  # output 128 * 184 * 184
        x = F.relu(self.conv1_convnet2(image))
        x = self.pool2_convnet2(F.relu(self.conv2_convnet2(x)))  # output 128 * 92 * 92
        x = self.pool3_convnet2(F.relu(self.conv3_convnet2(x)))  # output 128 * 45 * 45
        x = F.relu(self.conv4_convnet2(x))  # output 32 * 45 * 45
        return x  # output 32 * 45 * 45

    def convnet3(self, hide_t):
        """
        :param h_t: 48 * 45 * 45
        :return: heatmap   out_class * 45 * 45
        """
        x = F.relu(self.Mconv1_convnet3(hide_t))  # output 128 * 45 * 45
        x = F.relu(self.Mconv2_convnet3(x))  # output 128 * 45 * 45
        x = F.relu(self.Mconv3_convnet3(x))  # output 128 * 45 * 45
        x = F.relu(self.Mconv4_convnet3(x))  # output 128 * 45 * 45
        x = self.Mconv5_convnet3(x)  # output (class+1) * 45 * 45
        return x  # heatmap (class+1) * 45 * 45

    def lstm(self, heatmap, features, hide_t_1, cell_t_1):
        '''
        :param heatmap:     (class+1) * 45 * 45
        :param features:    32 * 45 * 45
        :param hide_t_1:    48 * 45 * 45
        :param cell_t_1:    48 * 45 * 45
        :return:
        hide_t:    48 * 45 * 45
        cell_t:    48 * 45 * 45
        '''
        xt = torch.cat([heatmap, features], dim=1)  # (32+ class+1 +1 ) * 45 * 45

        gx = self.conv_gx_lstm(xt)  # output: 48 * 45 * 45
        gh = self.conv_gh_lstm(hide_t_1)  # output: 48 * 45 * 45
        g_sum = gx + gh
        gt = F.tanh(g_sum)

        ox = self.conv_ox_lstm(xt)  # output: 48 * 45 * 45
        oh = self.conv_oh_lstm(hide_t_1)  # output: 48 * 45 * 45
        o_sum = ox + oh
        ot = F.sigmoid(o_sum)

        ix = self.conv_ix_lstm(xt)  # output: 48 * 45 * 45
        ih = self.conv_ih_lstm(hide_t_1)  # output: 48 * 45 * 45
        i_sum = ix + ih
        it = F.sigmoid(i_sum)

        fx = self.conv_fx_lstm(xt)  # output: 48 * 45 * 45
        fh = self.conv_fh_lstm(hide_t_1)  # output: 48 * 45 * 45
        f_sum = fx + fh
        ft = F.sigmoid(f_sum)

        cell_t = ft * cell_t_1 + it * gt
        hide_t = ot * F.tanh(cell_t)

        return cell_t, hide_t

    def lstm0(self, x):
        gx = self.conv_gx_lstm0(x)
        ix = self.conv_ix_lstm0(x)
        ox = self.conv_ox_lstm0(x)

        gx = F.tanh(gx)
        ix = F.sigmoid(ix)
        ox = F.sigmoid(ox)

        cell1 = F.tanh(gx * ix)
        hide_1 = ox * cell1
        return cell1, hide_1

    def stage2(self, image, cmap, heatmap, cell_t_1, hide_t_1):
        '''
        :param image:               3 * 368 * 368
        :param cmap: gaussian       1 * 368 * 368
        :param heatmap:             out_class * 45 * 45
        :param cell_t_1:            48 * 45 * 45
        :param hide_t_1:            48 * 45 * 45
        :return:
        new_heatmap:                out_class * 45 * 45
        cell_t:                     48 * 45 * 45
        hide_t:                     48 * 45 * 45
        '''
        features = self.convnet2(image)
        cell_t, hide_t = self.lstm(heatmap, features, hide_t_1, cell_t_1)
        new_heat_map = self.convnet3(hide_t)
        return new_heat_map, cell_t, hide_t

    def stage1(self, image, cmap):
        '''
        :param image:                3 * 368 * 368
        :param cmap:                 1 * 368 * 368
        :return:
        heatmap:                     out_class * 45 * 45
        cell_t:                      48 * 45 * 45
        hide_t:                      48 * 45 * 45
        '''
        initial_heatmap = self.convnet1(image)
        features = self.convnet2(image)

        x = torch.cat([initial_heatmap, features], dim=1)
        cell1, hide1 = self.lstm0(x)
        heatmap = self.convnet3(hide1)
        return initial_heatmap, heatmap, cell1, hide1

    def forward(self, images):
        '''
        :param images:      Tensor      (T * 3) * w(368) * h(368)
        :return:
        heatmaps            list        (T + 1)* out_class * 45 * 45  includes the initial heatmap
        '''
        image = images[:, 0:3, :, :]

        heat_maps = []
        initial_heatmap, heatmap, cell, hide = self.stage1(image, image)  # initial heat map

        heat_maps.append(initial_heatmap)  # for initial loss
        heat_maps.append(heatmap)
        #
        for i in range(1, self.T):
            image = images[:, (3 * i):(3 * i + 3), :, :]
            heatmap, cell, hide = self.stage2(image, image, heatmap, cell, hide)
            heat_maps.append(heatmap)
        return heat_maps[0]


def get_model(type):
    model = LSTM_2D(type)
    return model

def get_lstm():
    model = lstm_complete()
    return model

# 总结
# pytorch的loss、优化是在外面考虑的，定义网络的时候不用管，只需要输出计算loss需要的值即可
#
