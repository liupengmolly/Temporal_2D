import os
import time
import tqdm
import torch
import cv2
import torch.backends.cudnn as cudnn
import torch.utils.data
import logging
import torchvision.transforms as transforms
import torch.optim as optim
from collections import OrderedDict


from Config import cfg
from lib.loss import JointsMSELoss, MultiJointsMSELoss
from lib.data.H36m import H36m, MultiH36m
from lib.evaluate import *
from lib.imutils import *

from models.Simple_baseline import get_pose_net
from models.LSTM_2D import get_model

def train(cfg, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # loss0 = AverageMeter()
    # loss1 = AverageMeter()
    # loss2 = AverageMeter()
    # loss3 = AverageMeter()
    # loss4 = AverageMeter()
    # loss5 = AverageMeter()
    loss_sum = AverageMeter()
    acc0 = AverageMeter()
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    acc4 = AverageMeter()
    # acc5 = AverageMeter()

    model.train()

    begin = time.time()
    for bidx, (input, target) in enumerate(train_loader):
        #对应shuffle为false,保证把训练数据巡练完
        # if bidx>4:
        #     break
        data_time.update(time.time() - begin)

        input = input.cuda()
        output, _, _ = model(input)

        if isinstance(target, np.ndarray):
            target = target.astype(np.float32)
        else:
            target = target.float()
        target = target.cuda(non_blocking=True)

        loss = criterion(output, target)
        # loss = sum(losses)/len(losses)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),1)
        optimizer.step()

        # loss0.update(losses[0], input.size(0))
        # loss1.update(losses[1], input.size(0))
        # loss2.update(losses[2], input.size(0))
        # loss3.update(losses[3], input.size(0))
        # loss4.update(losses[4], input.size(0))
        # loss5.update(losses[5], input.size(0))
        loss_sum.update(loss, input.size(0))

        _, avg_acc0, cnt, _ = accuracy(output[0].detach().cpu().numpy(),
                                       target[:,0].detach().cpu().numpy(), thr=cfg.ac_thr)
        _, avg_acc1, cnt, _ = accuracy(output[1].detach().cpu().numpy(),
                                       target[:,1].detach().cpu().numpy(), thr=cfg.ac_thr)
        _, avg_acc2, cnt, _ = accuracy(output[2].detach().cpu().numpy(),
                                       target[:,2].detach().cpu().numpy(), thr=cfg.ac_thr)
        _, avg_acc3, cnt, _ = accuracy(output[3].detach().cpu().numpy(),
                                       target[:,3].detach().cpu().numpy(), thr=cfg.ac_thr)
        _, avg_acc4, cnt, _ = accuracy(output[4].detach().cpu().numpy(),
                                       target[:,4].detach().cpu().numpy(), thr=cfg.ac_thr)
        # _, avg_acc5, cnt, _ = accuracy(output[5].detach().cpu().numpy(), target[:,4].detach().cpu().numpy())
        acc0.update(avg_acc0, cnt)
        acc1.update(avg_acc1, cnt)
        acc2.update(avg_acc2, cnt)
        acc3.update(avg_acc3, cnt)
        acc4.update(avg_acc4, cnt)
        # acc5.update(avg_acc5, cnt)

        batch_time.update(time.time()-begin)
        begin = time.time()

        if bidx % cfg.log_freq == 0 or bidx+1 == len(train_loader):
            msg = 'Epoch: [{0}][{1}/{2}] ' \
                  'loss {loss_sum.val:.5f} ({loss_sum.avg:.5f})\n' \
                  'Ac0 {acc0.val:.3f} ({acc0.avg:.3f}) ' \
                  'Ac1 {acc1.val:.3f} ({acc1.avg:.3f}) ' \
                  'Ac2 {acc2.val:.3f} ({acc2.avg:.3f}) ' \
                  'Ac3 {acc3.val:.3f} ({acc3.avg:.3f}) ' \
                  'Ac4 {acc4.val:.3f} ({acc4.avg:.3f}) ' \
                .format(epoch, bidx, len(train_loader), loss_sum=loss_sum,
                        # loss0=loss0,loss1=loss1,loss2=loss2,loss3=loss3,loss4=loss4,
                        acc0=acc0,acc1=acc1,acc2=acc2,acc3=acc3,acc4=acc4)
            logging.info(msg)

        # if (bidx+1) % 1001 == 0:
        #     break

def validate(cfg, valid_loader, model, criterion):
    batch_time = AverageMeter()
    loss_sum = AverageMeter()
    # loss0 = AverageMeter()
    # loss1 = AverageMeter()
    # loss2 = AverageMeter()
    # loss3 = AverageMeter()
    # loss4 = AverageMeter()
    acc0 = AverageMeter()
    acc1 = AverageMeter()
    acc2 = AverageMeter()
    acc3 = AverageMeter()
    acc4 = AverageMeter()

    model.eval()
    with torch.no_grad():
        begin = time.time()
        for bidx, (inputs, targets) in enumerate(valid_loader):
            inputs = inputs.cuda()
            score_maps, _, _ = model(inputs)

            if isinstance(targets, np.ndarray):
                targets = targets.astype(np.float32)
            else:
                targets = targets.float()
            targets = targets.cuda(non_blocking=True)

            loss = criterion(score_maps, targets)
            # loss = sum(losses)/len(losses)

            num_images = inputs.size(0)
            # loss0.update(losses[0], inputs.size(0))
            # loss1.update(losses[1], inputs.size(0))
            # loss2.update(losses[2], inputs.size(0))
            # loss3.update(losses[3], inputs.size(0))
            # loss4.update(losses[4], inputs.size(0))
            loss_sum.update(loss, num_images)

            _, avg_acc0, cnt, _ = accuracy(score_maps[0].detach().cpu().numpy(),
                                           targets[:, 0].detach().cpu().numpy(), thr=cfg.ac_thr)
            _, avg_acc1, cnt, _ = accuracy(score_maps[1].detach().cpu().numpy(),
                                           targets[:, 1].detach().cpu().numpy(), thr=cfg.ac_thr)
            _, avg_acc2, cnt, _ = accuracy(score_maps[2].detach().cpu().numpy(),
                                           targets[:, 2].detach().cpu().numpy(), thr=cfg.ac_thr)
            _, avg_acc3, cnt, _ = accuracy(score_maps[3].detach().cpu().numpy(),
                                           targets[:, 3].detach().cpu().numpy(), thr=cfg.ac_thr)
            _, avg_acc4, cnt, _ = accuracy(score_maps[4].detach().cpu().numpy(),
                                           targets[:, 4].detach().cpu().numpy(), thr=cfg.ac_thr)
            # _, avg_acc5, cnt, _ = accuracy(output[5].detach().cpu().numpy(), target[:,4].detach().cpu().numpy())
            acc0.update(avg_acc0, cnt)
            acc1.update(avg_acc1, cnt)
            acc2.update(avg_acc2, cnt)
            acc3.update(avg_acc3, cnt)
            acc4.update(avg_acc4, cnt)

            batch_time.update(time.time() - begin)
            begin = time.time()

            if bidx % cfg.log_freq == 0 or bidx+1==len(valid_loader):
                msg = 'Test: [{0}/{1}] ' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\n' \
                      'Acc0 {acc0.val:.3f} ({acc0.avg:.3f}) ' \
                      'Acc1 {acc1.val:.3f} ({acc1.avg:.3f}) ' \
                      'Acc2 {acc2.val:.3f} ({acc2.avg:.3f}) ' \
                      'Acc3 {acc3.val:.3f} ({acc3.avg:.3f}) ' \
                      'Acc4 {acc4.val:.3f} ({acc4.avg:.3f}) ' \
                    .format(bidx, len(valid_loader), batch_time=batch_time, loss=loss_sum,
                    # loss0=loss0,loss1=loss1,loss2=loss2,loss3=loss3,loss4=loss4,
                    acc0=acc0, acc1=acc1, acc2=acc2, acc3=acc3, acc4=acc4)
                logging.info(msg)

    return acc4.avg, loss_sum.avg

def main(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus
    if not os.path.isdir(cfg.checkpoint):
        os.makedirs(cfg.checkpoint)

    model = get_model('h36m',stages=cfg.stages)

    best_loss = 100.0
    best_acc = 0.0
    prev_epoch = 0
    if cfg.ckpt:
        ckpt = torch.load(cfg.ckpt)
        # best_acc = ckpt['acc']
        # prev_epoch = ckpt['epoch']
        pretrained = ckpt['state_dict']
        # pretrained = {k: v for k, v in pretrained.items() if k in model_dict}
        # model_dict.update(pretrained)
        model.load_state_dict(pretrained)
    model = torch.nn.DataParallel(model).cuda()

    if cfg.freeze_params:
        for layer in ['convnet1','convnet2']:
                for param in eval('model.module.{}.parameters()'.format(layer)):
                    param.requires_grad = False

    cudnn.benchmark = True

    # 多图loss
    criterion = MultiJointsMSELoss(use_target_weight=cfg.use_target_weight,temporal=cfg.stages)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    valid_loader = torch.utils.data.DataLoader(
        MultiH36m(cfg,'data/images',False, cfg.stages),
        batch_size = cfg.batch_size * 8,
        shuffle = False,
        num_workers = 4,
        pin_memory = True
    )

    for epoch in range(prev_epoch, cfg.epoch):

        train_loader = torch.utils.data.DataLoader(
            MultiH36m(cfg, 'data/images', True, cfg.stages, epoch%5),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        train(cfg, train_loader, model, criterion, optimizer , epoch)

        acc, loss = validate(cfg, valid_loader, model, criterion)

        if acc > best_acc or loss < best_loss:
            best_loss = loss
            best_acc = acc
            torch.save({
                'epoch': epoch + 1,
                'model': cfg.model,
                'state_dict': model.module.state_dict(),
                'acc': acc,
                'optimizer': optimizer.state_dict()
            }, 'checkpoint/{}_epoch_{}'.format(cfg.model, epoch))
        torch.cuda.empty_cache()
        cfg.lr *= 0.95

if __name__ == '__main__':
    logging.basicConfig(
        filename="/home/liupeng/workspace/Temporal_2D/log/log_{}".format(cfg.model),
        filemode="a+",
        format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
        level=logging.INFO
    )
    print(cfg.model, cfg.epoch)
    main(cfg)