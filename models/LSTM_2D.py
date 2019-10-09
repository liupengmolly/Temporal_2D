import torch
import torch.nn as nn
import torch.nn.functional as F
from Config import cfg
from models.Simple_baseline import get_pose_net
from models.modules.aspp import build_aspp

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


class Conv_LSTM_backup(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Conv_LSTM_backup, self).__init__()

        self.init_ix = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        # self.init_ix_bn = nn.BatchNorm2d(outplanes)
        self.conv_ih = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)
        # self.conv_ih_bn = nn.BatchNorm2d(outplanes)
        # self.bias_i = nn.Parameter(torch.ones((outplanes,1,1)))

        self.init_fx = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        # self.init_fx_bn = nn.BatchNorm2d(outplanes)
        self.conv_fh = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)
        # self.conv_fh_bn = nn.BatchNorm2d(outplanes)
        # self.bias_f = nn.Parameter(torch.ones((outplanes,1,1)))

        self.init_ox = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        # self.init_ox_bn = nn.BatchNorm2d(outplanes)
        self.conv_oh = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)
        # self.conv_oh_bn = nn.BatchNorm2d(outplanes)
        # self.bias_o = nn.Parameter(torch.ones((outplanes,1,1)))

        self.init_gx = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        # self.init_gx_bn = nn.BatchNorm2d(outplanes)
        self.conv_gh = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)
        # self.conv_gh_bn = nn.BatchNorm2d(outplanes)
        # self.bias_g = nn.Parameter(torch.ones((outplanes,1,1)))

        '''
        # 感觉不需要下面的初始化
        self.init_gx = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        self.init_ix = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        self.init_ox = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        '''

    def forward(self, x, cell_t_1, hide_t_1):

        gx = self.init_gx(x)
        gh = self.conv_gh(hide_t_1)
        g_sum = gx + gh
        gt = torch.tanh(g_sum)

        ox = self.init_ox(x)
        oh = self.conv_oh(hide_t_1)
        o_sum = ox + oh
        ot = torch.sigmoid(o_sum)

        ix = self.init_ix(x)
        ih = self.conv_ih(hide_t_1)
        i_sum = ix + ih
        it = torch.sigmoid(i_sum)

        fx = self.init_fx(x)
        fh = self.conv_fh(hide_t_1)
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


class Advanced_Conv_LSTM(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Advanced_Conv_LSTM, self).__init__()

        self.init_ix = nn.Sequential(
            nn.Conv2d(inplanes, 32, kernel_size=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, outplanes, kernel_size=1, padding=0)
        )
        self.conv_ih = nn.Sequential(
            nn.Conv2d(outplanes, 32, kernel_size=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, outplanes, kernel_size=1, padding=0)
        )
        self.bias_i = nn.Parameter(torch.ones((outplanes,1,1)))

        self.init_fx = nn.Sequential(
            nn.Conv2d(inplanes, 32, kernel_size=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, outplanes, kernel_size=1, padding=0)
        )
        self.conv_fh = nn.Sequential(
            nn.Conv2d(outplanes, 32, kernel_size=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, outplanes, kernel_size=1, padding=0)
        )
        self.bias_f = nn.Parameter(torch.ones((outplanes,1,1)))

        self.init_ox = nn.Sequential(
            nn.Conv2d(inplanes, 32, kernel_size=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, outplanes, kernel_size=1, padding=0)
        )
        self.conv_oh = nn.Sequential(
            nn.Conv2d(outplanes, 32, kernel_size=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, outplanes, kernel_size=1, padding=0)
        )
        self.bias_o = nn.Parameter(torch.ones((outplanes,1,1)))

        self.init_gx = nn.Sequential(
            nn.Conv2d(inplanes, 32, kernel_size=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, outplanes, kernel_size=1, padding=0)
        )
        self.conv_gh = nn.Sequential(
            nn.Conv2d(outplanes, 32, kernel_size=1, padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, outplanes, kernel_size=1, padding=0)
        )
        self.bias_g = nn.Parameter(torch.ones((outplanes,1,1)))

    def forward(self, x, cell_t_1, hide_t_1):

        gx = self.init_gx(x)
        gh = self.conv_gh(hide_t_1)
        g_sum = gx + gh + self.bias_g
        gt = torch.tanh(g_sum)

        ox = self.init_ox(x)
        oh = self.conv_oh(hide_t_1)
        o_sum = ox + oh + self.bias_o
        ot = torch.sigmoid(o_sum)

        ix = self.init_ix(x)
        ih = self.conv_ih(hide_t_1)
        i_sum = ix + ih + self.bias_i
        it = torch.sigmoid(i_sum)

        fx = self.init_fx(x)
        fh = self.conv_fh(hide_t_1)
        f_sum = fx + fh + self.bias_f
        ft = torch.sigmoid(f_sum)

        cell_t = ft * cell_t_1 + it * gt
        hide_t = ot * torch.tanh(cell_t)
        return cell_t, hide_t

    def lstm_init(self, x):
        gx = self.init_gx(x)
        ix = self.init_ix(x)
        ox = self.init_ox(x)

        gx = torch.tanh(gx + self.bias_g)
        ix = torch.sigmoid(ix + self.bias_i)
        ox = torch.sigmoid(ox + self.bias_o)

        cell1 = torch.tanh(gx * ix)
        hide_1 = ox * cell1
        return cell1, hide_1

class Conv_LSTM(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(Conv_LSTM, self).__init__()

        self.init_ix = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        # self.init_ix_bn = nn.BatchNorm2d(outplanes)
        self.conv_ih = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)
        # self.conv_ih_bn = nn.BatchNorm2d(outplanes)
        self.bias_i = nn.Parameter(torch.ones((outplanes,1,1)))

        self.init_fx = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        # self.init_fx_bn = nn.BatchNorm2d(outplanes)
        self.conv_fh = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)
        # self.conv_fh_bn = nn.BatchNorm2d(outplanes)
        self.bias_f = nn.Parameter(torch.ones((outplanes,1,1)))

        self.init_ox = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        # self.init_ox_bn = nn.BatchNorm2d(outplanes)
        self.conv_oh = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)
        # self.conv_oh_bn = nn.BatchNorm2d(outplanes)
        self.bias_o = nn.Parameter(torch.ones((outplanes,1,1)))

        self.init_gx = nn.Conv2d(inplanes, outplanes, kernel_size=3, padding=1)
        # self.init_gx_bn = nn.BatchNorm2d(outplanes)
        self.conv_gh = nn.Conv2d(outplanes, outplanes, kernel_size=3, padding=1, bias=False)
        # self.conv_gh_bn = nn.BatchNorm2d(outplanes)
        self.bias_g = nn.Parameter(torch.ones((outplanes,1,1)))

    def forward(self, x, cell_t_1, hide_t_1):

        gx = self.init_gx(x)
        gh = self.conv_gh(hide_t_1)
        g_sum = gx + gh + self.bias_g
        gt = torch.tanh(g_sum)

        ox = self.init_ox(x)
        oh = self.conv_oh(hide_t_1)
        o_sum = ox + oh + self.bias_o
        ot = torch.sigmoid(o_sum)

        ix = self.init_ix(x)
        ih = self.conv_ih(hide_t_1)
        i_sum = ix + ih + self.bias_i
        it = torch.sigmoid(i_sum)

        fx = self.init_fx(x)
        fh = self.conv_fh(hide_t_1)
        f_sum = fx + fh + self.bias_f
        ft = torch.sigmoid(f_sum)

        cell_t = ft * cell_t_1 + it * gt
        hide_t = ot * torch.tanh(cell_t)
        return cell_t, hide_t
    
    def lstm_init(self, x):
        gx = self.init_gx(x)
        ix = self.init_ix(x)
        ox = self.init_ox(x)

        gx = torch.tanh(gx + self.bias_g)
        ix = torch.sigmoid(ix + self.bias_i)
        ox = torch.sigmoid(ox + self.bias_o)

        cell1 = torch.tanh(gx * ix)
        hide_1 = ox * cell1
        return cell1, hide_1


class LSTM_2D_backup(nn.Module):

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

        super(LSTM_2D_backup, self).__init__()
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

        self.convnet1 = get_pose_net(cfg, 50)
        self.convnet2 = get_pose_net(cfg, 34, 32)
        # 这边要是原图4分之一的话 下面这种也可以 不知道哪种效果好一点
        #conv_lstm 的 input 上述两个output(avgpool)
        self.conv_lstm = Conv_LSTM_backup(32+len(self.common_kps), 48)

        self.convnet3 = build_aspp(48, outclass, False)
        # self.convnet3 = ConvNet3(48, outclass)

        self.stage_num = stage_num

    def get_common_heatmap(self, heatmap):
        common_heatmap = torch.ones((
            heatmap.size(0),
            len(self.common_kps),
            heatmap.size(2),
            heatmap.size(3)
        ), dtype=torch.float32)


        common_heatmap = heatmap[:,self.common_kps]
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
        common_heatmap = self.get_common_heatmap(heatmap)
        heatmap = heatmap[:, self.kps]
        return initial_heatmap, common_heatmap,heatmap, cell, hide

    def stage(self, image, prev_heatmap, cell_t_1, hide_t_1):
        features = self.convnet2(image)
        #common_heatmap = self.get_common_heatmap(prev_heatmap)

        x = torch.cat([prev_heatmap, features],  dim=1)
        cell_t, hide_t = self.conv_lstm(x, cell_t_1,hide_t_1)
        heatmap = self.convnet3(hide_t)
        #TODO 这边还需要根据热图的尺寸再加个卷积修改一下尺寸

        common_heatmap = self.get_common_heatmap(heatmap)
        heatmap = heatmap[:, self.kps]
        return common_heatmap,heatmap, cell_t, hide_t

    def forward(self, images, initial=True, cell=None, hide=None):
        heatmaps = []
        #这边把一个网络的图片堆在了第2维上
        #第一维是网络训练时的样本数量
        #第二个是一次训练需要的样本数 0～3是RGB
        if len(images.size())>4:
            image = images[:,0,:,:,:]
        else:
            image = images
        initial_heatmap, common_heatmap, heatmap, cell, hide = self.init(image=image)
        heatmaps.append(initial_heatmap)
        heatmaps.append(heatmap)

        for i in range(1, self.stage_num):
            image = images[:,i,:,:,:]

            # image = images
            common_heatmap, heatmap, cell, hide = self.stage(image, common_heatmap, cell, hide)
            heatmaps.append(heatmap)
        return heatmaps, cell, hide

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
    def __init__(self, type, outclass=21, stage_num=1):

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

        self.convnet1 = get_pose_net(cfg, 50, 32)
        # self.convnet2 = get_pose_net(cfg, 34, 32)

        #conv_lstm 的 input 上述两个output(avgpool)
        # self.conv_lstm = Conv_LSTM(32, 64)
        self.conv_lstm = Advanced_Conv_LSTM(32, 64)

        self.convnet3 = build_aspp(64, outclass)

        self.stage_num = stage_num

    def get_common_heatmap(self, heatmap):
        common_heatmap = torch.ones((
            heatmap.size(0),
            len(self.common_kps),
            heatmap.size(2),
            heatmap.size(3)
        ), dtype=torch.float32)


        common_heatmap = heatmap[:,self.common_kps]
        common_heatmap = common_heatmap.cuda()

        return common_heatmap

    def init(self, image):
        features = self.convnet1(image)

        cell, hide = self.conv_lstm.lstm_init(features)
        heatmap = self.convnet3(hide)

        heatmap =  heatmap[:, self.kps]
        return heatmap, cell, hide

    def stage(self, image, cell_t_1, hide_t_1):
        features = self.convnet1(image)

        cell_t, hide_t = self.conv_lstm(features, cell_t_1,hide_t_1)
        heatmap = self.convnet3(hide_t)

        heatmap = heatmap[:, self.kps]
        return heatmap, cell_t, hide_t

    def forward(self, images, initial=True, cell=None, hide=None):
        heatmaps = []
        #这边把一个网络的图片堆在了第2维上
        #第一维是网络训练时的样本数量
        #第二个是一次训练需要的样本数 0～3是RGB
        start_idx = 0
        if initial:
            image = images[:,0,:,:,:]
            heatmap, cell, hide = self.init(image=image)
            heatmaps.append(heatmap)
            start_idx = 1

        for i in range(start_idx, images.size(1)):
            image = images[:,i,:,:,:]

            # image = images
            heatmap, cell, hide = self.stage(image, cell, hide)
            heatmaps.append(heatmap)
        return heatmaps, cell, hide

def get_model(type, stages=1):
    model = LSTM_2D(type,stage_num=stages)
    return model

def get_old_model(type, stages=1):
    model = LSTM_2D_backup(type, stage_num=stages)
    return model

