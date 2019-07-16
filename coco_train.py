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
from models.Simple_baseline import get_pose_net


def train(cfg, train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    model.train()

    begin = time.time()
    for bidx, (input, target, target_weight,meta) in enumerate(train_loader):
        data_time.update(time.time() - begin)
        input = input.cuda()
        outputs = model(input)
        outputs = [output[:,:17] for output in outputs]
        if isinstance(target, np.ndarray):
            target = target.astype(np.float32)
        else:
            target = target.float()
        target = target.cuda(non_blocking = True)
        target_weight = target_weight.cuda(non_blocking=True)

        loss = 0.0
        for hp in outputs:
            loss += criterion(hp, target, target_weight)

        # output = output.detach().cpu().numpy()
        # target = target.detach().cpu().numpy()
        # target_weight = target_weight.detach().cpu().numpy()

        optimizer.zero_grad()
        loss.backward()
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
        for bidx, (inputs, target, target_weight, meta) in enumerate(valid_loader):
            input_var = torch.autograd.Variable(inputs.cuda())

            outputs = model(input_var)
            outputs = [output[:,:17] for output in outputs]
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

                flip_outputs = model(flip_input_var)
                flip_outputs = [flip_output[:, :17] for flip_output in flip_outputs]
                flip_score_maps = [flip_outputs[i].data.cpu().numpy()
                                   for i in range(len(flip_outputs))]
                for i, flip_score_map in enumerate(flip_score_maps):
                    for j, fscore in enumerate(flip_score_map):
                        fscore = fscore.transpose((1, 2, 0))
                        fscore = cv2.flip(fscore, 1)
                        fscore = list(fscore.transpose((2, 0, 1)))
                        for (q, w) in cfg.symmetry_h36m:
                            fscore[q], fscore[w] = fscore[w], fscore[q]
                        fscore = np.array(fscore)
                        score_maps[i][j] += fscore
                        score_maps[i][j] /= 2

            if isinstance(target, np.ndarray):
                target = target.astype(np.float32)
            else:
                target = target.float()
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)

            loss = 0.0
            for i, score_map in enumerate(score_maps):
                score_map = torch.from_numpy(score_map).cuda(non_blocking=True)
                loss += criterion(score_map, target, target_weight)

            num_images = inputs.size(0)
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

    #         for b in range(inputs.size(0)):
    #
    #             single_map = score_map[b]
    #             r0 = single_map.copy()
    #             r0 /= 255
    #             r0 += 0.5
    #             v_score = np.zeros(17)
    #
    #             for p in range(17):
    #                 single_map[p] /= np.amax(single_map[p])
    #                 # border的作用暂时不理解
    #                 border = 10
    #                 dr = np.zeros((cfg.output_H + 2*border, cfg.output_W + 2*border))
    #                 dr[border:-border, border:-border] = single_map[p].copy()
    #                 dr = cv2.GaussianBlur(dr, (21, 21), 0)
    #                 lb = dr.argmax()
    #                 y, x = np.unravel_index(lb, dr.shape)
    #                 dr[y, x] = 0
    #                 lb = dr.argmax()
    #                 py, px = np.unravel_index(lb, dr.shape)
    #                 y -= border
    #                 x -= border
    #                 py -= border + y
    #                 px -= border + x
    #                 ln = (px ** 2 + py ** 2) ** 0.5
    #                 delta = 0.25
    #                 if ln > 1e-3:
    #                     x += delta * px / ln
    #                     y += delta * py / ln
    #                 x = max(0, min(x, cfg.output_W - 1))
    #                 y = max(0, min(y, cfg.output_H - 1))
    #                 img_h = meta['img_h'].numpy()[b]
    #                 img_w = meta['img_w'].numpy()[b]
    #                 pad_h = max(img_h, img_h*cfg.input_H/cfg.input_W)
    #                 pad_w = pad_h * cfg.input_W / cfg.input_H
    #                 resx = int(x * (pad_w / cfg.output_W) - (pad_w - img_w)/2)
    #                 resy = int(y * (pad_h / cfg.output_H) - (pad_h - img_h)/2)
    #                 v_score[p] = float(r0[p, int(round(y) + 1e-10), int(round(x) + 1e-10)])
    #                 batch_kp_result[b,p,:] = np.array([resx, resy])
    #
    #             batch_score_result[b,0] = v_score.mean()
    #
    #         full_kp_result.append(batch_kp_result)
    #         full_score_result.append(batch_score_result)
    #
    # h36m_2d_evaluate(full_kp_result, full_score_result, full_gt)
    return acc.avg

def main(cfg):
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpus
    if not os.path.isdir(cfg.checkpoint):
        os.makedirs(cfg.checkpoint)

    # model = get_pose_net(cfg)
    from models.LSTM_2D import get_model
    model = get_model('coco')

    prev_epoch = 0
    best_acc = 0.0
    if cfg.ckpt:
        ckpt = torch.load(cfg.ckpt)
        prev_epoch = ckpt['epoch']
        best_acc = ckpt['acc']
        new_state_dict = OrderedDict()
        for k, v in ckpt['state_dict'].items():
            # name = k[7:]
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)

    if len(cfg.gpus)>1:
        model = torch.nn.DataParallel(model)

    model = model.cuda()
    cudnn.benchmark = True

    criterion = JointsMSELoss(use_target_weight=cfg.use_target_weight).cuda()

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.lr_step, cfg.lr_factor
    )

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = COCODataset(
        cfg,
        'data/COCO',
        'train2017',
        True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    val_dataset = COCODataset(
        cfg,
        'data/COCO',
        'val2017',
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
        batch_size = cfg.batch_size,
        shuffle = False,
        num_workers = 8,
        pin_memory = True
    )

    for epoch in range(prev_epoch, cfg.epoch):
        lr_scheduler.step()

        train(cfg, train_loader, model, criterion, optimizer, epoch)

        acc = validate(cfg, valid_loader, model, criterion, epoch)

        if acc > best_acc:
            best_acc = acc
            torch.save({
                'epoch': epoch + 1,
                'model': cfg.model,
                'state_dict': model.module.state_dict() if len(cfg.gpus)>1 else model.state_dict(),
                'acc': acc,
                'optimizer': optimizer.state_dict()
            }, 'checkpoint/{}_epoch_{}'.format(cfg.model, epoch))


if __name__ == '__main__':
    logging.basicConfig(
        filename="/home/liupeng/workspace/Temporal_2D/log/log_{}".format(cfg.model),
        filemode="w",
        format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
        level=logging.INFO
    )
    main(cfg)
