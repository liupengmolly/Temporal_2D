import os
import torch.backends.cudnn as cudnn
import torch.utils.data
import logging
import torch.optim as optim
import torchvision.transforms as transforms
from collections import OrderedDict

from Config import cfg
from lib.loss import JointsMSELoss
from lib.data.H36m import H36m
from lib.evaluate import *
from lib.imutils import *
from lib.data.coco import COCODataset
from lib.data.mpii import MPIIDataset
from models.Simple_baseline import get_pose_net
from lib.data.H36m import H36m

import numpy as np


def train(cfg, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    begin = time.time()
    for bidx, (input, target, target_weight, meta) in enumerate(train_loader):
        data_time.update(time.time() - begin)

        input = input.unsqueeze(1)
        target = target.unsqueeze(1)
        inputs = input
        targets = target

        # 多张图片训练单模型
        for i in range(1, cfg.stages):
            inputs = torch.cat((inputs,input), 1)
            targets = torch.cat((targets,target), 1)
        inputs = inputs.cuda()
        outputs, _, _ = model(inputs)
        if isinstance(target, np.ndarray):
            target = target.astype(np.float32)
        else:
            target = target.float()
        target = target.cuda(non_blocking = True)
        target_weight = target_weight.cuda(non_blocking=True)

        loss = 0.0
        target = target.squeeze(1)
        for hp in outputs:
            loss += criterion(hp, target, target_weight)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),1)
        optimizer.step()

        losses.update(loss.item(), input.size(0))
        _, avg_acc, cnt, pred = accuracy(hp.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        batch_time.update(time.time()-begin)
        begin = time.time()

        if bidx % cfg.log_freq == 0 or bidx+1 == len(train_loader):
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, bidx, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=losses, acc=acc)
            logging.info(msg)

def validate(cfg, valid_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()
    with torch.no_grad():
        begin = time.time()
        for bidx, (input, target, target_weight, meta) in enumerate(valid_loader):
            input = input.unsqueeze(1)
            target = target.unsqueeze(1)
            inputs = input
            targets = target

            # 多张图片训练单模型
            for i in range(1, cfg.stages):
                inputs = torch.cat((inputs, input), 1)
                targets = torch.cat((targets, target), 1)
            inputs = inputs.cuda()

            input_var = torch.autograd.Variable(inputs.cuda())

            outputs, _, _ = model(input_var)
            score_maps = [outputs[i].data.cpu().numpy() for i in range(len(outputs))]

            #翻转
            cfg.flip = False
            if cfg.flip:
                flip_inputs = inputs.clone()
                for i, finp in enumerate(flip_inputs):
                    finp = im_to_numpy(finp)
                    finp = cv2.flip(finp, 1)
                    finp = np.transpose(finp, (2, 0, 1))
                    finp = im_to_torch(finp)
                    flip_inputs[i] = finp
                flip_input_var = torch.autograd.Variable(flip_inputs.cuda())

                flip_outputs, _, _ = model(flip_input_var)
                flip_score_maps = [flip_outputs[i].data.cpu().numpy()
                                   for i in range(len(flip_outputs))]
                for i, flip_score_map in enumerate(flip_score_maps):
                    for j, fscore in enumerate(flip_score_map):
                        fscore = fscore.transpose((1, 2, 0))
                        fscore = cv2.flip(fscore, 1)
                        fscore = list(fscore.transpose((2, 0, 1)))
                        for (q, w) in cfg.symmetry:
                            fscore[q], fscore[w] = fscore[w], fscore[q]
                        fscore = np.array(fscore)
                        score_maps[i][j] += fscore
                        score_maps[i][j] /= 2

            if isinstance(target, np.ndarray):
                target = target.astype(np.float32)
            else:
                target = target.float()

            target = target.squeeze(1)
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = 0.0
            for i, score_map in enumerate(score_maps):
                score_map = torch.from_numpy(score_map).cuda(non_blocking=True)
                loss += criterion(score_map, target, target_weight)

            num_images = inputs.size(1)
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(score_map.cpu().numpy(),
                                             target.cpu().numpy())
            acc.update(avg_acc, cnt)

            batch_time.update(time.time() - begin)
            begin = time.time()

            if bidx % cfg.log_freq == 0 or bidx+1 == len(valid_loader):
                msg = 'Test[Epoch{0}]: [{1}/{2}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, bidx, len(valid_loader), batch_time=batch_time,
                    loss=losses, acc=acc)
                logging.info(msg)
    return acc.avg

def h36m_train(cfg, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    begin = time.time()
    for bidx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - begin)
        input = input.cuda()
        outputs, _, _ = model(input)
        if isinstance(target, np.ndarray):
            target = target.astype(np.float32)
        else:
            target = target.float()
        target = target.cuda(non_blocking = True)

        loss = 0.0
        for hp in outputs:
            loss += criterion(hp, target)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(),1)
        optimizer.step()

        losses.update(loss.item(), input.size(0))
        _, avg_acc, cnt, pred = accuracy(hp.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        batch_time.update(time.time()-begin)
        begin = time.time()

        if bidx % cfg.log_freq == 0 or bidx+1 == len(train_loader):
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, bidx, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=losses, acc=acc)
            logging.info(msg)

def h36m_validate(cfg, valid_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.eval()

    with torch.no_grad():
        begin = time.time()
        for bidx, (inputs, target) in enumerate(valid_loader):
            input_var = torch.autograd.Variable(inputs.cuda())

            outputs, _, _ = model(input_var)
            score_maps = [outputs[i].data.cpu().numpy() for i in range(len(outputs))]

            #翻转
            if cfg.flip:
                flip_inputs = inputs.clone()
                for i, finp in enumerate(flip_inputs):
                    finp = im_to_numpy(finp)
                    finp = cv2.flip(finp, 1)
                    finp = np.transpose(finp, (2, 0, 1))
                    finp = im_to_torch(finp)
                    flip_inputs[i] = finp
                flip_input_var = torch.autograd.Variable(flip_inputs.cuda())

                flip_outputs, _, _ = model(flip_input_var)
                flip_score_maps = [flip_outputs[i].data.cpu().numpy()
                                   for i in range(len(flip_outputs))]
                for i, flip_score_map in enumerate(flip_score_maps):
                    for j, fscore in enumerate(flip_score_map):
                        fscore = fscore.transpose((1, 2, 0))
                        fscore = cv2.flip(fscore, 1)
                        fscore = list(fscore.transpose((2, 0, 1)))
                        for (q, w) in cfg.symmetry:
                            fscore[q], fscore[w] = fscore[w], fscore[q]
                        fscore = np.array(fscore)
                        score_maps[i][j] += fscore
                        score_maps[i][j] /= 2

            if isinstance(target, np.ndarray):
                target = target.astype(np.float32)
            else:
                target = target.float()
            target = target.cuda(non_blocking=True)

            loss = 0.0
            for i, score_map in enumerate(score_maps):
                score_map = torch.from_numpy(score_map).cuda(non_blocking=True)
                loss += criterion(score_map, target)

            num_images = inputs.size(0)
            losses.update(loss.item(), num_images)

            score_map = torch.from_numpy(score_maps[0]).cuda(non_blocking=True)
            _, avg_acc, cnt, pred = accuracy(score_map.cpu().numpy(),
                                             target.cpu().numpy())
            acc.update(avg_acc, cnt)

            batch_time.update(time.time() - begin)
            begin = time.time()

            if bidx % cfg.log_freq == 0 or bidx+1 == len(valid_loader):
                msg = 'Test[Epoch{0}]: [{1}/{2}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    epoch, bidx, len(valid_loader), batch_time=batch_time,
                    loss=losses, acc=acc)
                logging.info(msg)

    return acc.avg

def main(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus
    if not os.path.isdir(cfg.checkpoint):
        os.makedirs(cfg.checkpoint)

    from models.LSTM_2D import get_model
    model = get_model(cfg.type,stages=cfg.stages)

    prev_epoch = 0
    best_acc = 0.0
    if cfg.ckpt:
        model_state_dict = model.state_dict()
        ckpt = torch.load(cfg.ckpt)
        prev_epoch = ckpt['epoch']
        # best_acc = ckpt['acc']
        # cfg.lr = ckpt['lr']
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            # k = k[7:]
            if 'convnet3' in k or 'conv_lstm' in k:
                continue
            new_state_dict[k] = v
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)

    conv3_params = list(map(id, model.convnet3.parameters()))
    conv_lstm_params = list(map(id, model.conv_lstm.parameters()))
    base_params = filter(lambda p: id(p) not in conv3_params + conv_lstm_params,
                         model.parameters())
    params = [
        {'params': base_params, 'lr':cfg.lr},
        {'params': model.conv_lstm.parameters(), 'lr':cfg.lr},
        {'params': model.convnet3.parameters(), 'lr': cfg.lr}
    ]

    if len(cfg.gpus)>1:
        model = torch.nn.DataParallel(model)

    model = model.cuda()
    cudnn.benchmark = True

    criterion = JointsMSELoss(use_target_weight=cfg.use_target_weight).cuda()


    optimizer = optim.Adam(params, lr=cfg.lr)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.lr_step, cfg.lr_factor
    )

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if cfg.type=='h36m':
        train_loader = torch.utils.data.DataLoader(
            H36m(cfg, 'data/images', True),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True
        )
        valid_loader = torch.utils.data.DataLoader(
            H36m(cfg, 'data/images', False),
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )

    else:
        if cfg.type == 'coco':
            Dataset = COCODataset
            Data_root = 'data/COCO'
            train_dir = 'train2017'
            val_dir = 'val2017'
        elif cfg.type == 'mpii':
            Dataset = MPIIDataset
            Data_root = 'data/MPII'
            train_dir = 'train'
            val_dir = 'valid'
        else:
            return ValueError("the coco_mpii_train scripy not support the training of "
                              "current data type {}".format(cfg.type))

        train_dataset = Dataset(
            cfg,
            Data_root,
            train_dir,
            True,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )
        val_dataset = Dataset(
            cfg,
            Data_root,
            val_dir,
            False,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
        )

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = cfg.batch_size,
            shuffle = True,
            num_workers = 8,
            pin_memory = True
        )
        valid_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = cfg.batch_size*4,
            shuffle = False,
            num_workers = 8,
            pin_memory = True
        )

    # 重新训练单模型多stages prev_epoch直接设为0
    for epoch in range(prev_epoch, cfg.epoch):
        lr_scheduler.step()

        if cfg.type == 'h36m':
            h36m_train(cfg, train_loader, model, criterion, optimizer, epoch)
            acc = h36m_validate(cfg, valid_loader, model, criterion, epoch)
        else:
            train(cfg, train_loader, model, criterion, optimizer, epoch)
            acc = validate(cfg, valid_loader, model, criterion, epoch)

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'epoch': epoch + 1,
                'model': cfg.model,
                'state_dict': model.module.state_dict() if len(cfg.gpus)>1 else model.state_dict(),
                'acc': acc,
                'optimizer': optimizer.state_dict(),
                'lr': cfg.lr
            }, 'checkpoint/{}_epoch_{}'.format(cfg.model, epoch))


if __name__ == '__main__':
    logging.basicConfig(
        filename="/home/liupeng/workspace/Temporal_2D/log/log_{}".format(cfg.model),
        filemode="a+",
        format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
        level=logging.INFO
    )
    main(cfg)
