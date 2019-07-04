import torch
import torch.nn as nn

class ConvNet1(nn.Module):
    pass

class ConvNet2(nn.Module):
    pass

class ConvNet3(nn.Module):
    pass

class Conv_LSTM(nn.Module):
    # 需要参考别人的代码
    pass

class LSTM_2D(nn.Module):
    def __init__(self, ):
        self.convnet1 = ConvNet1()
        self.convnet2 = ConvNet2()
        self.convnet3 = ConvNet3()
        self.conv_lstm = Conv_LSTM()

    def forward(self, x, heatmap, center_map, h_last=None, c_last=None, initial=False):
        if initial:
            x_initial_feature = self.convnet1(x)
        x_feature = self.convnet2(x)

        x_feature_filtered = x_feature.mul(center_map)

        # ===这边的concat操作需要再确定=====
        X = x_feature+heatmap+center_map

        #=================================




# 总结
# pytorch的loss、优化是在外面考虑的，定义网络的时候不用管，只需要输出计算loss需要的值即可
#
