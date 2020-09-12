import numpy as np
import os
import torch.nn as nn
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn import functional as F

from config import ecssd_path, hkuis_path, pascals_path, sod_path, dutomron_path, davis_path, fbms_path
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from utils.utils_mine import MaxMinNormalization
from models.MGA.mga_model import MGA_Network
from models.BASNet.BASNet import BASNet
from models.R3Net.R3Net import R3Net
from models.DSS.DSSNet import build_model
from models.CPD.CPD_ResNet_models import CPD_ResNet
from models.RAS.RAS import RAS
from models.PiCANet.network import Unet
from models.PoolNet.poolnet import build_model_poolnet
from models.R2Net.r2net import build_model_r2net
from models.F3Net.net import F3Net

from train_distill import train_BASNet, train_CPD, train_DSSNet, train_F3Net, train_PoolNet, train_R2Net, train_R3Net, train_RAS
from torch import optim
from module.morphology import Erosion2d

import joint_transforms
from config import msra10k_path, video_train_path, datasets_root, video_seq_gt_path, video_seq_path
from datasets import ImageFolder, VideoImageFolder, VideoFSImageFolder, VideoFirstImageFolder
from matplotlib import pyplot as plt
import time

torch.manual_seed(2018)

# set which gpu to use
torch.cuda.set_device(0)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = './ckpt'
exp_name = 'VideoSaliency_2020-08-14 10:40:23'

args = {
    'model': 'F3Net',
    'online_train': True,
    'snapshot': '10000',  # your snapshot filename (exclude extension name)
    'crf_refine': False,  # whether to use crf to refine results
    'save_results': True,  # whether to save the resulting masks
    'input_size': (380, 380),
    'batch_size': 1,
    'start': 0
}

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
to_pil = transforms.ToPILImage()

to_test = {'davis': os.path.join(davis_path, 'davis_test2')}
gt_root = os.path.join(davis_path, 'GT')
imgs_path = os.path.join(davis_path, 'davis_test2_5f.txt')

# to_test = {'FBMS': os.path.join(fbms_path, 'FBMS_Testset')}
# gt_root = os.path.join(fbms_path, 'GT')
# imgs_path = os.path.join(fbms_path, 'FBMS_seq_file_5f.txt')

def fix_parameters(parameters):
    for name, parameter in parameters:
        if name.find('linearp') >= 0 or name.find('linearr') >= 0 or name.find('decoder') >= 0:
            print(name, 'is not fixed')
            # parameter.requires_grad = False
        else:
            print(name, 'is fixed')
            parameter.requires_grad = False

def train_online(net, seq_name='breakdance'):
    online_args = {
        'iter_num': 100,
        'train_batch_size': 1,
        'lr': 1e-10,
        'lr_decay': 0.95,
        'weight_decay': 5e-4,
        'momentum': 0.95,
    }

    joint_transform = joint_transforms.Compose([
        joint_transforms.ImageResize(380),
        # joint_transforms.RandomCrop(473),
        # joint_transforms.RandomHorizontallyFlip(),
        # joint_transforms.RandomRotate(10)
    ])
    target_transform = transforms.ToTensor()
    # train_set = VideoFSImageFolder(to_test['davis'], seq_name, use_first=True, joint_transform=joint_transform, transform=img_transform)
    train_set = VideoFirstImageFolder(to_test['davis'], gt_root, seq_name, joint_transform=joint_transform,
                                   transform=img_transform, target_transform=target_transform)
    online_train_loader = DataLoader(train_set, batch_size=online_args['train_batch_size'], num_workers=1, shuffle=False)

    # criterion = nn.MSELoss().cuda()
    criterion = nn.BCEWithLogitsLoss().cuda()
    erosion = Erosion2d(1, 1, 5, soft_max=False).cuda()
    net.train()
    net.cuda()
    # fix_parameters(net.named_parameters())

    optimizer = optim.SGD([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * online_args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
         'lr': online_args['lr'], 'weight_decay': online_args['weight_decay']}
    ], momentum=online_args['momentum'])

    for curr_iter in range(0, online_args['iter_num']):
        total_loss_record, loss0_record, loss1_record = AvgMeter(), AvgMeter(), AvgMeter()
        loss2_record = AvgMeter()

        for i, data in enumerate(online_train_loader):
            optimizer.param_groups[0]['lr'] = 2 * online_args['lr'] * (1 - float(curr_iter) / online_args['iter_num']
                                                                ) ** online_args['lr_decay']
            optimizer.param_groups[1]['lr'] = online_args['lr'] * (1 - float(curr_iter) / online_args['iter_num']
                                                            ) ** online_args['lr_decay']
            inputs, labels = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()

            optimizer.zero_grad()
            if args['model'] == 'BASNet':
                total_loss, loss0, loss1, loss2 = train_BASNet(net, inputs, criterion, erosion, labels)
            elif args['model'] == 'R3Net':
                total_loss, loss0, loss1, loss2 = train_R3Net(net, inputs, criterion, erosion, labels)
            elif args['model'] == 'DSSNet':
                total_loss, loss0, loss1, loss2 = train_DSSNet(net, inputs, criterion, erosion, labels)
            elif args['model'] == 'CPD':
                total_loss, loss0, loss1, loss2 = train_CPD(net, inputs, criterion, erosion, labels)
            elif args['model'] == 'RAS':
                total_loss, loss0, loss1, loss2 = train_RAS(net, inputs, criterion, erosion, labels)
            elif args['model'] == 'PoolNet':
                total_loss, loss0, loss1, loss2 = train_PoolNet(net, inputs, criterion, erosion, labels)
            elif args['model'] == 'F3Net':
                total_loss, loss0, loss1, loss2 = train_F3Net(net, inputs, criterion, erosion, labels)
            elif args['model'] == 'R2Net':
                total_loss, loss0, loss1, loss2 = train_R2Net(net, inputs, criterion, erosion, labels)
            total_loss.backward()
            optimizer.step()

            total_loss_record.update(total_loss.data, batch_size)
            loss0_record.update(loss0.data, batch_size)
            loss1_record.update(loss1.data, batch_size)
            loss2_record.update(loss2.data, batch_size)
            # loss3_record.update(loss3.data, batch_size)
            # loss4_record.update(loss4.data, batch_size)

            log = '[iter %d], [total loss %.5f], [loss0 %.8f], [loss1 %.8f], [loss2 %.8f], [lr %.13f]' % \
                  (curr_iter, total_loss_record.avg, loss0_record.avg, loss1_record.avg, loss2_record.avg,
                   optimizer.param_groups[1]['lr'])
            print(log)

    print('taking snapshot ...')
    torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, str(args['snapshot']) + '_' + seq_name + '_online.pth'))
    # torch.save(optimizer.state_dict(),
    #            os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))

    return net

def train_R2Net(net, inputs_s, criterion, erosion, labels):
    global_pre, outputs0, outputs1, outputs2, outputs3, outputs4 = net(inputs_s)
    global_pre2, outputs02, outputs12, outputs22, outputs32, outputs42 = net(labels)
    global_loss = criterion(F.sigmoid(global_pre), erosion(F.sigmoid(global_pre2)))
    loss0 = criterion(outputs0, erosion(F.sigmoid(outputs02)))
    loss1 = criterion(outputs1, erosion(F.sigmoid(outputs12)))
    loss2 = criterion(outputs2, erosion(F.sigmoid(outputs22)))
    loss3 = criterion(outputs3, erosion(F.sigmoid(outputs32)))
    loss4 = criterion(outputs4, erosion(F.sigmoid(outputs42)))
    # loss4 = criterion(F.sigmoid(global_pre), F.sigmoid(global_pre2))
    # loss5 = criterion(out5r, labels)
    # loss7 = criterion(outputs7, labels)

    total_loss = global_loss + loss0 + loss1 + loss2 + loss3 + loss4

    return total_loss, loss0, loss1, loss2

def train_F3Net(student, inputs_s, criterion, erosion, labels):
    out1u, out2u, out2r, out3r, out4r, out5r = student(inputs_s)
    # out1u2, out2u2, out2r2, out3r2, out4r2, out5r2 = student(labels)
    # loss0 = criterion(out1u, torch.where(F.sigmoid(out2u) > 0.25, torch.ones(out1u2.size()).cuda(),
    #                                      torch.zeros(out1u2.size()).cuda()))

    loss0 = criterion(out1u, labels)
    loss1 = criterion(out2u, labels)
    loss2 = criterion(out2r, labels)
    loss3 = criterion(out3r, labels)
    loss4 = criterion(out4r, labels)
    loss5 = criterion(out5r, labels)
    # loss7 = criterion(outputs7, labels)

    total_loss = (loss0 + loss1) / 2 + loss2 / 2 + loss3 / 4 + loss4 / 8 + loss5 / 16

    return total_loss, loss0, loss1, loss2

def train_PoolNet(student, inputs_s, criterion, erosion, labels):
    outputs0 = student(inputs_s)
    outputs02 = student(labels)

    loss0 = criterion(outputs0, erosion(F.sigmoid(outputs02)))
    # loss1 = criterion(outputs1, labels)
    # loss2 = criterion(outputs0, labels)
    # loss3 = criterion(outputs1, labels)
    # loss4 = criterion(outputs0, labels)
    # loss1 = criterion(outputs1, labels)
    # loss7 = criterion(outputs7, labels)

    total_loss = loss0

    return total_loss, loss0, loss0, loss0

def train_RAS(student, inputs_s, criterion, erosion, labels):
    outputs0, outputs1, outputs2, outputs3, outputs4 = student(inputs_s)
    outputs02, outputs12, outputs22, outputs32, outputs42 = student(labels)

    loss0 = criterion(outputs0, erosion(F.sigmoid(outputs02)))
    loss1 = criterion(outputs1, erosion(F.sigmoid(outputs12)))
    loss2 = criterion(outputs2, erosion(F.sigmoid(outputs22)))
    loss3 = criterion(outputs3, erosion(F.sigmoid(outputs32)))
    loss4 = criterion(outputs4, erosion(F.sigmoid(outputs42)))
    # loss1 = criterion(outputs1, labels)
    # loss7 = criterion(outputs7, labels)

    total_loss = loss0 + loss1 + loss2 + loss3 + loss4

    return total_loss, loss0, loss1, loss2

def train_CPD(student, inputs_s, criterion, erosion, labels):
    outputs0, outputs1 = student(inputs_s)
    outputs02, outputs12 = student(labels)

    loss0 = criterion(outputs0, erosion(F.sigmoid(outputs02)))
    loss1 = criterion(outputs1, erosion(F.sigmoid(outputs12)))

    # loss7 = criterion(outputs7, labels)

    total_loss = loss0 + loss1

    return total_loss, loss0, loss1, loss1

def train_DSSNet(student, inputs_s, criterion, erosion, labels):
    outputs = student(inputs_s)

    loss0 = criterion(outputs[0], labels)
    loss1 = criterion(outputs[1], labels)
    loss2 = criterion(outputs[2], labels)
    loss3 = criterion(outputs[3], labels)
    loss4 = criterion(outputs[4], labels)
    loss5 = criterion(outputs[5], labels)
    loss6 = criterion(outputs[6], labels)
    # loss7 = criterion(outputs7, labels)

    # a = outputs[0].data.cpu().numpy()
    # plt.imshow(a[0, 0, :, :])
    # plt.show()

    loss_hard = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6


    total_loss = loss_hard

    return total_loss, loss0, loss1, loss2

def train_R3Net(student, inputs_s, criterion, erosion, labels):
    outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6 = student(inputs_s)
    outputs02, outputs12, outputs22, outputs32, outputs42, outputs52, outputs62 = student(labels)

    loss0 = criterion(outputs0, erosion(F.sigmoid(outputs02)))
    loss1 = criterion(outputs1, erosion(F.sigmoid(outputs12)))
    loss2 = criterion(outputs2, erosion(F.sigmoid(outputs22)))
    loss3 = criterion(outputs3, erosion(F.sigmoid(outputs32)))
    loss4 = criterion(outputs4, erosion(F.sigmoid(outputs42)))
    loss5 = criterion(outputs5, erosion(F.sigmoid(outputs52)))
    loss6 = criterion(outputs6, erosion(F.sigmoid(outputs62)))
    # loss7 = criterion(outputs7, labels)

    total_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    return total_loss, loss0, loss1, loss2

def train_BASNet(student, inputs_s, criterion, erosion, labels):
    outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7 = student(inputs_s)
    outputs02, outputs12, outputs22, outputs32, outputs42, outputs52, outputs62, outputs72 = student(labels)

    loss0 = criterion(outputs0, erosion(F.sigmoid(outputs02)))
    loss1 = criterion(outputs1, erosion(F.sigmoid(outputs12)))
    loss2 = criterion(outputs2, erosion(F.sigmoid(outputs22)))
    loss3 = criterion(outputs3, erosion(F.sigmoid(outputs32)))
    loss4 = criterion(outputs4, erosion(F.sigmoid(outputs42)))
    loss5 = criterion(outputs5, erosion(F.sigmoid(outputs52)))
    loss6 = criterion(outputs6, erosion(F.sigmoid(outputs62)))
    loss7 = criterion(outputs7, erosion(F.sigmoid(outputs72)))

    total_loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

    return total_loss, loss0, loss1, loss2

def main():
    if args['model'] == 'BASNet':
        net = BASNet(3, 1)
    elif args['model'] == 'R3Net':
        net = R3Net()
    elif args['model'] == 'DSSNet':
        net = build_model()
    elif args['model'] == 'CPD':
        net = CPD_ResNet()
    elif args['model'] == 'RAS':
        net = RAS()
    elif args['model'] == 'PiCANet':
        net = Unet()
    elif args['model'] == 'PoolNet':
        net = build_model_poolnet(base_model_cfg='resnet')
    elif args['model'] == 'R2Net':
        net = build_model_r2net(base_model_cfg='resnet')
    elif args['model'] == 'F3Net':
        net = F3Net(cfg=None)

    print ('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), map_location='cuda:0'))
    # net = train_online(net)
    results = {}

    for name, root in to_test.items():

        precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
        mae_record = AvgMeter()

        if args['save_results']:
            check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'] + '_online')))

        folders = os.listdir(root)
        folders.sort()
        folders = ['bmx-trees']
        for folder in folders:
            if args['online_train']:
                net = train_online(net, seq_name=folder)
                net.load_state_dict(
                    torch.load(os.path.join(ckpt_path, exp_name, str(args['snapshot']) + '_' + folder + '_online.pth'), map_location='cuda:0'))
            with torch.no_grad():

                net.eval()
                net.cuda()
                imgs = os.listdir(os.path.join(root, folder))
                imgs.sort()
                for i in range(args['start'], len(imgs)):
                    print(imgs[i])
                    start = time.time()

                    img = Image.open(os.path.join(root, folder, imgs[i])).convert('RGB')
                    shape = img.size
                    img = img.resize(args['input_size'])
                    img_var = Variable(img_transform(img).unsqueeze(0), volatile=True).cuda()

                    if args['model'] == 'BASNet':
                        prediction, _, _, _, _, _, _, _ = net(img_var)
                        prediction = torch.sigmoid(prediction)
                    elif args['model'] == 'R3Net':
                        prediction = net(img_var)
                    elif args['model'] == 'DSSNet':
                        select = [1, 2, 3, 6]
                        prediction = net(img_var)
                        prediction = torch.mean(torch.cat([torch.sigmoid(prediction[i]) for i in select], dim=1), dim=1,
                                                keepdim=True)
                    elif args['model'] == 'CPD':
                        _, prediction = net(img_var)
                        prediction = torch.sigmoid(prediction)
                    elif args['model'] == 'RAS':
                        prediction, _, _, _, _ = net(img_var)
                        prediction = torch.sigmoid(prediction)
                    elif args['model'] == 'PoolNet':
                        prediction = net(img_var)
                        prediction = torch.sigmoid(prediction)
                    elif args['model'] == 'F3Net':
                        _, prediction, _, _, _, _ = net(img_var)
                        prediction = torch.sigmoid(prediction)
                    elif args['model'] == 'R2Net':
                        _, _, _, _, _, prediction = net(img_var)
                        prediction = torch.sigmoid(prediction)
                    end = time.time()
                    print('running time:', (end - start))

                    if args['crf_refine']:
                        prediction = crf_refine(np.array(img), prediction)

                    precision = to_pil(prediction.data.squeeze(0).cpu())
                    precision = precision.resize(shape)
                    prediction = np.array(precision)
                    prediction = prediction.astype('float')
                    prediction = MaxMinNormalization(prediction, prediction.max(), prediction.min()) * 255.0
                    prediction = prediction.astype('uint8')

                    gt = np.array(Image.open(os.path.join(gt_root, folder, imgs[i][:-4] + '.png')).convert('L'))
                    precision, recall, mae = cal_precision_recall_mae(prediction, gt)
                    for pidx, pdata in enumerate(zip(precision, recall)):
                        p, r = pdata
                        precision_record[pidx].update(p)
                        recall_record[pidx].update(r)
                    mae_record.update(mae)

                    if args['save_results']:
                        # folder, sub_name = os.path.split(imgs[i])
                        save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'] + '_online'),
                                                 folder)
                        if not os.path.exists(save_path):
                            os.makedirs(save_path)
                        Image.fromarray(prediction).save(os.path.join(save_path, imgs[i]))

        fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                [rrecord.avg for rrecord in recall_record])

        results[name] = {'fmeasure': fmeasure, 'mae': mae_record.avg}

    print ('test results:')
    print (results)


if __name__ == '__main__':
    main()

# bmx-trees
# {'davis': {'fmeasure': 0.6116564504502642, 'mae': 0.02240475388347857}}
# {'davis': {'fmeasure': 0.5172097270228496, 'mae': 0.06305300100984769}}
# {'davis': {'fmeasure': 0.4967318785455502, 'mae': 0.07540526049988441}}
# {'davis': {'fmeasure': 0.496028162803402, 'mae': 0.0758443760893737}}