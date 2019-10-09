import os
import matplotlib.pyplot as plt
import cv2
import random
import logging
from collections import OrderedDict

from Config import cfg
from lib.loss import MultiJointsMSELoss
from lib.evaluate import *
from lib.imutils import *
from lib.data.H36m import H36m
from models.Simple_baseline import get_pose_net
from models.LSTM_2D import get_model,get_old_model

def get_img(path):
    if not os.path.exists(path):
        return ValueError("the img path is not exits")

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
    img = img.unsqueeze(0)
    return img, img_h, img_w

def load_model(model_name, old=False):
    if old:
        model = get_old_model(cfg.type)
    else:
        model = get_model(cfg.type)

    model = model.cuda()

    ckpt = torch.load(os.path.join('/home/liupeng/workspace/Temporal_2D',
                                   cfg.checkpoint, model_name))['state_dict']
    #之前训练是多卡训练的，参数会带上“module.”的前缀，这里需要除掉
    model.load_state_dict(ckpt)
    model.eval()

    return model

def validate(cfg, loader, model, criterion):
    loss_sum = AverageMeter()
    acc_05 = AverageMeter()
    acc_025 = AverageMeter()

    initial = True
    c, h = None, None
    with torch.no_grad():
        for bidx, (inputs, targets) in enumerate(loader):
            inputs = inputs.cuda()
            if bidx>0:
                initial = False
            inputs = inputs.unsqueeze(1)
            score_maps, c, h = model(inputs, initial, c, h)

            targets = targets.unsqueeze(1).float().cuda(non_blocking=True)

            loss = criterion(score_maps, targets)

            loss_sum.update(loss, 1)

            _, avg_acc_05, cnt, _ = accuracy(score_maps[0].detach().cpu().numpy(),
                                             targets[:, 0].detach().cpu().numpy(), thr=0.5)
            _, avg_acc_025, cnt, _ = accuracy(score_maps[0].detach().cpu().numpy(),
                                              targets[:, 0].detach().cpu().numpy(), thr=0.25)
            acc_05.update(avg_acc_05, cnt)
            acc_025.update(avg_acc_025, cnt)

            if bidx % cfg.log_freq == 0 or bidx+1==len(loader):
                msg = 'Test: [{0}/{1}] ' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\n' \
                      'Acc05 {acc_05.val:.3f} ({acc_05.avg:.3f}) ' \
                      'Acc025 {acc_025.val:.3f} ({acc_025.avg:.3f}) ' \
                    .format(bidx, len(loader), loss=loss_sum, acc_05 = acc_05, acc_025=acc_025)
                logging.info(msg)

def get_seq_accuray(model_name):
    logging.basicConfig(
        filename="/home/liupeng/workspace/Temporal_2D/log/log_{}".format(cfg.model),
        filemode="a+",
        format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
        level=logging.INFO
    )
    if 'seqlstm' in model_name:
        model = load_model(model_name, False)
    else:
        model = load_model(model_name, True)

    valid_loader = torch.utils.data.DataLoader(
        H36m(cfg, 'data/images', False),
        batch_size = 1,
        shuffle = False,
        num_workers = 1,
        pin_memory = True
    )

    criterion = MultiJointsMSELoss(use_target_weight=cfg.use_target_weight, temporal=1).cuda()
    validate(cfg, valid_loader, model, criterion)



if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    model_name = 'seqlstm_advanced_epoch_16'
    # model_name = 'seqlstm_spconv3_restore_epoch_15'
    # model_name = 'lstm_restore4_h36m_epoch_88'
    get_seq_accuray(model_name)



