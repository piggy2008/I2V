import datetime
import os

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn import functional as F
from matplotlib import pyplot as plt

import joint_transforms
from config import msra10k_path, video_train_path, datasets_root, video_seq_gt_path, video_seq_path
from datasets import ImageFolder, VideoImageFolder, VideoSequenceFolder, VideoImage2Folder, ImageFlowFolder, ImageFlow2Folder
from misc import AvgMeter, check_mkdir, CriterionKL3, CriterionKL, CriterionPairWise
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
from torch.backends import cudnn
import time
from utils.utils_mine import load_part_of_model, load_part_of_model2

import random

cudnn.benchmark = True
<<<<<<< HEAD
device_id = 1
device_id2 = 2
=======
device_id = 0
device_id2 = 1
>>>>>>> 7d05a275ea1202d69ad1d1bff257b30e322c60e3
torch.manual_seed(2019)
# torch.cuda.set_device(device_id)


time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
ckpt_path = './ckpt2'
exp_name = 'VideoSaliency' + '_' + time_str

args = {
    'model': 'F3Net',
    'motion': '',
    'prior': False,
    'se_layer': False,
    'dilation': False,
    'distillation': False,
    'L2': False,
    'KL': False,
    'iter_num': 20000,
    'iter_save': 4000,
    'iter_start_seq': 0,
    'train_batch_size': 12,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.95,
    'snapshot': '',
    # 'pretrain': os.path.join(ckpt_path, 'VideoSaliency_2020-07-24 15:18:51', '100000.pth'),
    'pretrain': '',
    'mga_model_path': 'pretrained/MGA_trained.pth',
    # 'imgs_file': 'Pre-train/pretrain_all_seq_DUT_DAFB2_DAVSOD.txt',
    'imgs_file': 'Pre-train/pretrain_all_seq_DAFB2_DAVSOD_flow.txt',
    # 'imgs_file': 'video_saliency/train_all_DAFB2_DAVSOD_5f.txt',
    # 'train_loader': 'video_image'
    'train_loader': 'flow_image',
    # 'train_loader': 'video_sequence'
    'image_size': 430,
    'crop_size': 380
}

imgs_file = os.path.join(datasets_root, args['imgs_file'])
# imgs_file = os.path.join(datasets_root, 'video_saliency/train_all_DAFB3_seq_5f.txt')

joint_transform = joint_transforms.Compose([
    joint_transforms.ImageResize(args['image_size']),
    joint_transforms.RandomCrop(args['crop_size']),
    # joint_transforms.ColorJitter(hue=[-0.1, 0.1], saturation=0.05),
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.RandomRotate(10)
])

# joint_transform = joint_transforms.Compose([
#     joint_transforms.ImageResize(290),
#     joint_transforms.RandomCrop(256),
#     joint_transforms.RandomHorizontallyFlip(),
#     joint_transforms.RandomRotate(10)
# ])

# joint_seq_transform = joint_transforms.Compose([
#     joint_transforms.ImageResize(520),
#     joint_transforms.RandomCrop(473)
# ])

input_size = (473, 473)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()

# train_set = ImageFolder(msra10k_path, joint_transform, img_transform, target_transform)
if args['train_loader'] == 'video_sequence':
    train_set = VideoSequenceFolder(video_seq_path, video_seq_gt_path, imgs_file, joint_transform, img_transform, target_transform)
elif args['train_loader'] == 'video_image':
    train_set = VideoImageFolder(video_train_path, imgs_file, joint_transform, img_transform, target_transform)
elif args['train_loader'] == 'flow_image':
    train_set = ImageFlowFolder(video_train_path, imgs_file, joint_transform, img_transform, target_transform)
elif args['train_loader'] == 'flow_image2':
    train_set = ImageFlow2Folder(video_train_path, imgs_file, video_seq_path + '/DAFB2', video_seq_gt_path + '/DAFB2',
                                 joint_transform, (args['crop_size'], args['crop_size']), img_transform, target_transform)
else:
    train_set = VideoImage2Folder(video_train_path, imgs_file, video_seq_path + '/DAFB2', video_seq_gt_path + '/DAFB2',
                                  joint_transform, None, input_size, img_transform, target_transform)

train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)

criterion = nn.BCEWithLogitsLoss()
<<<<<<< HEAD
# erosion = Erosion2d(1, 1, 5, soft_max=False).cuda()
=======

>>>>>>> 7d05a275ea1202d69ad1d1bff257b30e322c60e3
if args['L2']:
    criterion_l2 = nn.MSELoss().cuda()
    # criterion_pair = CriterionPairWise(scale=0.5).cuda()
if args['KL']:
    criterion_kl = CriterionKL3().cuda()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')

total_loss_record, loss0_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

def fix_parameters(parameters):
    for name, parameter in parameters:
        if name.find('linearp') >= 0 or name.find('linearr') >= 0 or name.find('decoder') >= 0:
            print(name, 'is not fixed')

        else:
            print(name, 'is fixed')
            parameter.requires_grad = False

def load_MGA(mga_model, model_path, device_id=2):
    pretrain_weights = torch.load(model_path, map_location='cuda:' + str(device_id))
    pretrain_keys = list(pretrain_weights.keys())
    pretrain_keys = [key for key in pretrain_keys if not key.endswith('num_batches_tracked')]
    net_keys = list(mga_model.state_dict().keys())

    for key in net_keys:
        # key_ = 'module.' + key
        key_ = key
        if key_ in pretrain_keys:
            assert (mga_model.state_dict()[key].size() == pretrain_weights[key_].size())
            mga_model.state_dict()[key].copy_(pretrain_weights[key_])
        else:
            print('missing key: ', key_)
    print('loaded pre-trained weights.')
    return mga_model


def main():
    teacher = MGA_Network(nInputChannels=3, n_classes=1, os=16,
                      img_backbone_type='resnet101', flow_backbone_type='resnet34')
    teacher = load_MGA(teacher, args['mga_model_path'])
    teacher.eval()
    teacher.cuda(device_id)

    if args['model'] == 'BASNet':
        student = BASNet(3, 1).cuda(device_id2).train()
    elif args['model'] == 'R3Net':
        student = R3Net().cuda(device_id2).train()
    elif args['model'] == 'DSSNet':
        student = build_model().cuda(device_id2).train()
    elif args['model'] == 'CPD':
        student = CPD_ResNet().cuda(device_id2).train()
    elif args['model'] == 'RAS':
        student = RAS().cuda(device_id2).train()
    elif args['model'] == 'PiCANet':
        student = Unet().cuda(device_id2).train()
    elif args['model'] == 'PoolNet':
        student = build_model_poolnet(base_model_cfg='resnet').cuda(device_id2).train()
    elif args['model'] == 'R2Net':
        student = build_model_r2net(base_model_cfg='resnet').cuda(device_id2).train()
    elif args['model'] == 'F3Net':
        student = F3Net(cfg=None).cuda(device_id2).train()

    # fix_parameters(net.named_parameters())
    optimizer = optim.SGD([
        {'params': [param for name, param in student.named_parameters() if name[-4:] == 'bias'],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in student.named_parameters() if name[-4:] != 'bias'],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ], momentum=args['momentum'])

    if len(args['snapshot']) > 0:
        print('training resumes from ' + args['snapshot'])
        student.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    if args['model'] == 'BASNet':
        print('pretrain model loading')
        student = load_part_of_model(student, 'pretrained/basnet.pth', device_id=device_id)
    elif args['model'] == 'R3Net':
        print('pretrain model loading')
        student = load_part_of_model(student, 'pretrained/R3Net.pth', device_id=device_id)
    elif args['model'] == 'DSSNet':
        print('pretrain model loading')
        student = load_part_of_model(student, 'pretrained/DSS.pth', device_id=device_id)
    elif args['model'] == 'CPD':
        print('pretrain model loading')
        student = load_part_of_model(student, 'pretrained/CPD-R.pth', device_id=device_id)
    elif args['model'] == 'RAS':
        print('pretrain model loading')
        student = load_part_of_model(student, 'pretrained/RAS.v1.pth', device_id=device_id)
    elif args['model'] == 'PiCANet':
        print('pretrain model loading')
        student = load_part_of_model(student, 'pretrained/PiCANet.ckpt', device_id=device_id)
    elif args['model'] == 'PoolNet':
        print('pretrain model loading')
        student = load_part_of_model(student, 'pretrained/poolnet.pth', device_id=device_id)
    elif args['model'] == 'R2Net':
        print('pretrain model loading')
        student = load_part_of_model2(student, 'pretrained/R2Net.pth', device_id=device_id)
    elif args['model'] == 'F3Net':
        print('pretrain model loading')
        student = load_part_of_model(student, 'pretrained/F3Net', device_id=device_id)


    if len(args['pretrain']) > 0:
        print('pretrain model from ' + args['pretrain'])
        student = load_part_of_model(student, args['pretrain'], device_id=device_id)
        # fix_parameters(student.named_parameters())

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(student, teacher, optimizer)


def train(student, teacher, optimizer):
    curr_iter = args['last_iter']
    while True:

        # loss3_record = AvgMeter()

        for i, data in enumerate(train_loader):

            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']
                                                            ) ** args['lr_decay']
            #
            # inputs, flows, labels, pre_img, pre_lab, cur_img, cur_lab, next_img, next_lab = data
            inputs, flows, labels = data
            if curr_iter < args['iter_start_seq']:
                train_single(student, teacher, inputs, flows, labels, optimizer, curr_iter)

            else:
                if curr_iter % 2 == 0:
                    # train_seg(student, pre_img, pre_lab, cur_img, cur_lab, next_img, next_lab, optimizer, curr_iter)
                    train_single(student, teacher, inputs, flows, labels, optimizer, curr_iter)
                else:
                    # train_seg(student, pre_img, pre_lab, cur_img, cur_lab, next_img, next_lab, optimizer, curr_iter)
                    train_single(student, teacher, inputs, flows, labels, optimizer, curr_iter, need_prior=args['prior'])
            # train_single(student, teacher, inputs, flows, labels, optimizer, curr_iter)


            curr_iter += 1

            if curr_iter % args['iter_save'] == 0:
                print('taking snapshot ...')
                torch.save(student.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))

            if curr_iter == args['iter_num']:
                torch.save(student.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
                torch.save(optimizer.state_dict(),
                           os.path.join(ckpt_path, exp_name, '%d_optim.pth' % curr_iter))
                return

def train_seg(student, pre_img, pre_lab, cur_img, cur_lab, next_img, next_lab, optimizer, curr_iter):
    pre_img = Variable(pre_img).cuda(device_id2)
    pre_lab = Variable(pre_lab).cuda(device_id2)
    cur_img = Variable(cur_img).cuda(device_id2)
    cur_lab = Variable(cur_lab).cuda(device_id2)
    next_img = Variable(next_img).cuda(device_id2)
    next_lab = Variable(next_lab).cuda(device_id2)

    optimizer.zero_grad()
    if args['model'] == 'F3Net':
        total_loss, loss0, loss1, loss2 = train_F3Net_seg(student, pre_img, pre_lab, cur_img, cur_lab, next_img, next_lab)

    total_loss.backward()
    optimizer.step()

    print_log(total_loss, loss0, loss1, loss2, args['train_batch_size'], curr_iter, optimizer, type='seg')

    return

def train_single(student, teacher, inputs, flows, labels, optimizer, curr_iter, need_prior=False):
    if args['distillation']:
        inputs_t = Variable(inputs).cuda(device_id)
        flows_t = Variable(flows).cuda(device_id)

        prediction, _, _, _, _ = teacher(inputs_t, flows_t)
        prediction = prediction.to(device_id2)

    inputs_s = Variable(inputs).cuda(device_id2)
    # flows_s = Variable(flows).cuda(device_id2)
    labels = Variable(labels).cuda(device_id2)
    # prediction = torch.nn.Sigmoid()(prediction)
    # prediction = prediction.data.cpu().numpy()
    # prediction = F.upsample(prediction, size=(), mode='bilinear', align_corners=True)
    # from matplotlib import pyplot as plt
    # plt.subplot(1, 2, 1)
    # plt.imshow(prediction[0, 0, :, :])
    # plt.show()
    optimizer.zero_grad()
    if args['model']  == 'BASNet':
        total_loss, loss0, loss1, loss2 = train_BASNet(student, inputs_s, None, labels, need_prior)
    elif args['model']  == 'R3Net':
        total_loss, loss0, loss1, loss2 = train_R3Net(student, inputs_s, prediction, labels, need_prior)
    elif args['model'] == 'DSSNet':
        total_loss, loss0, loss1, loss2 = train_DSSNet(student, inputs_s, prediction, labels)
    elif args['model'] == 'CPD':
        total_loss, loss0, loss1, loss2 = train_CPD(student, inputs_s, prediction, labels, need_prior)
    elif args['model'] == 'RAS':
        total_loss, loss0, loss1, loss2 = train_RAS(student, inputs_s, None, labels, need_prior)
    elif args['model'] == 'PoolNet':
        total_loss, loss0, loss1, loss2 = train_PoolNet(student, inputs_s, None, labels, need_prior)
    elif args['model'] == 'F3Net':
        total_loss, loss0, loss1, loss2 = train_F3Net(student, inputs_s, None, labels, need_prior)
    elif args['model'] == 'R2Net':
        total_loss, loss0, loss1, loss2 = train_R2Net(student, inputs_s, None, labels, need_prior)

    total_loss.backward()
    optimizer.step()

    print_log(total_loss, loss0, loss1, loss2, args['train_batch_size'], curr_iter, optimizer)

    return

def train_R2Net(student, inputs_s, prediction, labels, need_prior=False):
    if need_prior:
        global_pre, outputs0, outputs1, outputs2, outputs3, outputs4 = student(inputs_s, prediction)
    else:
        global_pre, outputs0, outputs1, outputs2, outputs3, outputs4 = student(inputs_s)

    global_loss = criterion(global_pre, labels)
    loss0 = criterion(outputs0, labels)
    loss1 = criterion(outputs1, labels)
    loss2 = criterion(outputs2, labels)
    loss3 = criterion(outputs3, labels)
    loss4 = criterion(outputs4, labels)
    # loss5 = criterion(out5r, labels)
    # loss7 = criterion(outputs7, labels)

    loss_hard = global_loss + loss0 + loss1 + loss2 + loss3 + loss4

    if args['distillation'] and prediction is not None:
        global_loss2 = criterion(global_pre, F.sigmoid(prediction))
        loss02 = criterion(outputs0, F.sigmoid(prediction))
        loss12 = criterion(outputs1, F.sigmoid(prediction))
        loss22 = criterion(outputs2, F.sigmoid(prediction))
        loss32 = criterion(outputs3, F.sigmoid(prediction))
        loss42 = criterion(outputs4, F.sigmoid(prediction))

        loss_soft = global_loss2 + loss02 + loss12 + loss22 + loss32 + loss42
        total_loss = loss_hard + 0.5 * loss_soft
    else:
        total_loss = loss_hard

    return total_loss, loss_hard, loss1, loss0

def train_F3Net(student, inputs_s, prediction, labels, need_prior=False):
    if need_prior:
        out1u, out2u, out2r, out3r, out4r, out5r = student(inputs_s, prediction)
    else:
        out1u, out2u, out2r, out3r, out4r, out5r = student(inputs_s)

    loss0 = criterion(out1u, labels)
    loss1 = criterion(out2u, labels)
    loss2 = criterion(out2r, labels)
    loss3 = criterion(out3r, labels)
    loss4 = criterion(out4r, labels)
    loss5 = criterion(out5r, labels)
    # loss7 = criterion(outputs7, labels)

    loss_hard = (loss0 + loss1) / 2 + loss2 / 2 + loss3 / 4 + loss4 / 8 + loss5 / 16

    if args['distillation'] and prediction is not None:
        loss02 = criterion(out1u, F.sigmoid(prediction))
        loss12 = criterion(out2u, F.sigmoid(prediction))
        loss22 = criterion(out2r, F.sigmoid(prediction))
        loss32 = criterion(out3r, F.sigmoid(prediction))
        loss42 = criterion(out4r, F.sigmoid(prediction))
        loss52 = criterion(out5r, F.sigmoid(prediction))

        loss_soft = (loss02 + loss12) / 2 + loss22 / 2 + loss32 / 4 + loss42 / 8 + loss52 / 16
        total_loss = loss_hard + 0.9 * loss_soft
    else:
        total_loss = loss_hard

    return total_loss, loss_hard, loss0, loss0

def train_F3Net_seg(student, pre_img, pre_lab, cur_img, cur_lab, next_img, next_lab):
    student.eval()
    out1u_pre, out2u_pre, out2r_pre, out3r_pre, out4r_pre, out5r_pre = student(pre_img)
    out1u_next, out2u_next, out2r_next, out3r_next, out4r_next, out5r_next = student(next_img)

    student.train()
    out1u_cur, out2u_cur, out2r_cur, out3r_cur, out4r_cur, out5r_cur = student(cur_img)


    loss0_cur = criterion(out1u_cur, cur_lab)
    loss1_cur= criterion(out2u_cur, cur_lab)
    loss2_cur = criterion(out2r_cur, cur_lab)
    loss3_cur = criterion(out3r_cur, cur_lab)
    loss4_cur = criterion(out4r_cur, cur_lab)
    loss5_cur= criterion(out5r_cur, cur_lab)

    loss_hard_cur = (loss0_cur + loss1_cur) / 2 + loss2_cur / 2 + loss3_cur / 4 + loss4_cur / 8 + loss5_cur / 16

    loss_hard = loss_hard_cur

    loss02 = criterion_l2(F.sigmoid(out1u_cur), pre_lab)/2 + criterion_l2(F.sigmoid(out1u_cur), next_lab)/2
    loss12 = criterion_l2(F.sigmoid(out2u_cur), pre_lab)/2 + criterion_l2(F.sigmoid(out2u_cur), next_lab)/2
    # loss22 = criterion_l2(F.sigmoid(out2r_cur), F.sigmoid(out2u_pre))/2 + criterion_l2(F.sigmoid(out2r_cur), F.sigmoid(out2u_next))/2
    # loss32 = criterion_l2(F.sigmoid(out3r_cur), F.sigmoid(out2u_pre))/2 + criterion_l2(F.sigmoid(out3r_cur), F.sigmoid(out2u_next))/2
    # loss42 = criterion_l2(F.sigmoid(out4r_cur), F.sigmoid(out2u_pre))/2 + criterion_l2(F.sigmoid(out4r_cur), F.sigmoid(out2u_next))/2
    # loss52 = criterion_l2(F.sigmoid(out5r_cur), F.sigmoid(out2u_pre))/2 + criterion_l2(F.sigmoid(out5r_cur), F.sigmoid(out2u_next))/2

    loss_soft = (loss02 + loss12) / 2
    total_loss = loss_hard + 0.01 * loss_soft


    return total_loss, loss_hard, loss_soft, loss0_cur

def train_PoolNet(student, inputs_s, prediction, labels, need_prior):
    if need_prior:
        outputs0 = student(inputs_s, prediction)
    else:
        outputs0 = student(inputs_s)

    loss0 = criterion(outputs0, labels)
    # loss1 = criterion(outputs1, labels)
    # loss2 = criterion(outputs0, labels)
    # loss3 = criterion(outputs1, labels)
    # loss4 = criterion(outputs0, labels)
    # loss1 = criterion(outputs1, labels)
    # loss7 = criterion(outputs7, labels)

    loss_hard = loss0

    if args['distillation'] and prediction is not None:
        loss02 = criterion(outputs0, F.sigmoid(prediction))
        # loss12 = criterion(outputs1, F.sigmoid(prediction))
        # loss22 = criterion(outputs2, F.sigmoid(prediction))
        # loss32 = criterion(outputs3, F.sigmoid(prediction))
        # loss42 = criterion(outputs4, F.sigmoid(prediction))

        loss_soft = loss02
        total_loss = loss_hard + 0.5 * loss_soft
    else:
        total_loss = loss_hard

    return total_loss, loss0, loss0, loss0

def train_RAS(student, inputs_s, prediction, labels, need_prior=False):
    if need_prior:
        outputs0, outputs1, outputs2, outputs3, outputs4 = student(inputs_s, prediction)
    else:
        outputs0, outputs1, outputs2, outputs3, outputs4 = student(inputs_s)

    loss0 = criterion(outputs0, labels)
    loss1 = criterion(outputs1, labels)
    loss2 = criterion(outputs2, labels)
    loss3 = criterion(outputs3, labels)
    loss4 = criterion(outputs4, labels)
    # loss1 = criterion(outputs1, labels)
    # loss7 = criterion(outputs7, labels)

    loss_hard = loss0 + loss1 + loss2 + loss3 + loss4

    if args['distillation'] and prediction is not None:
        loss02 = criterion(outputs0, F.sigmoid(prediction))
        loss12 = criterion(outputs1, F.sigmoid(prediction))
        loss22 = criterion(outputs2, F.sigmoid(prediction))
        loss32 = criterion(outputs3, F.sigmoid(prediction))
        loss42 = criterion(outputs4, F.sigmoid(prediction))

        loss_soft = loss02 + loss12 + loss22 + loss32 + loss42
        total_loss = loss_hard + 0.5 * loss_soft
    else:
        total_loss = loss_hard

    return total_loss, loss_hard, loss0, loss0

def train_CPD(student, inputs_s, prediction, labels, need_prior):
    if need_prior:
        outputs0, outputs1 = student(inputs_s, prediction)
    else:
        outputs0, outputs1 = student(inputs_s)

    loss0 = criterion(outputs0, labels)
    loss1 = criterion(outputs1, labels)

    # loss7 = criterion(outputs7, labels)

    loss_hard = loss0 + loss1

    if args['distillation'] and prediction is not None:
        loss02 = criterion(outputs0, F.sigmoid(prediction))
        loss12 = criterion(outputs1, F.sigmoid(prediction))

        # loss72 = criterion(outputs7, F.sigmoid(prediction))

        loss_soft = loss02 + loss12
        total_loss = loss_hard + 0.5 * loss_soft
    else:
        total_loss = loss_hard

    return total_loss, loss_hard, loss1, loss0

def train_DSSNet(student, inputs_s, prediction, labels):
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

    if args['distillation'] and prediction is not None:
        loss02 = criterion(outputs[0], F.sigmoid(prediction))
        loss12 = criterion(outputs[1], F.sigmoid(prediction))
        loss22 = criterion(outputs[2], F.sigmoid(prediction))
        loss32 = criterion(outputs[3], F.sigmoid(prediction))
        loss42 = criterion(outputs[4], F.sigmoid(prediction))
        loss52 = criterion(outputs[5], F.sigmoid(prediction))
        loss62 = criterion(outputs[6], F.sigmoid(prediction))
        # loss72 = criterion(outputs7, F.sigmoid(prediction))

        loss_soft = loss02 + loss12 + loss22 + loss32 + loss42 + loss52 + loss62
        total_loss = loss_hard + 0.5 * loss_soft
    else:
        total_loss = loss_hard

    return total_loss, loss0, loss1, loss2

def train_R3Net(student, inputs_s, prediction, labels, need_prior):
    if need_prior:
        outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6 = student(inputs_s, prediction)
    else:
        outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6 = student(inputs_s)

    loss0 = criterion(outputs0, labels)
    loss1 = criterion(outputs1, labels)
    loss2 = criterion(outputs2, labels)
    loss3 = criterion(outputs3, labels)
    loss4 = criterion(outputs4, labels)
    loss5 = criterion(outputs5, labels)
    loss6 = criterion(outputs6, labels)
    # loss7 = criterion(outputs7, labels)

    loss_hard = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

    if args['distillation'] and prediction is not None:
        loss02 = criterion(outputs0, F.sigmoid(prediction))
        loss12 = criterion(outputs1, F.sigmoid(prediction))
        loss22 = criterion(outputs2, F.sigmoid(prediction))
        loss32 = criterion(outputs3, F.sigmoid(prediction))
        loss42 = criterion(outputs4, F.sigmoid(prediction))
        loss52 = criterion(outputs5, F.sigmoid(prediction))
        loss62 = criterion(outputs6, F.sigmoid(prediction))
        # loss72 = criterion(outputs7, F.sigmoid(prediction))

        loss_soft = loss02 + loss12 + loss22 + loss32 + loss42 + loss52 + loss62
        total_loss = loss_hard + 0.5 * loss_soft
    else:
        total_loss = loss_hard

    return total_loss, loss_hard, loss_soft, loss6

def train_BASNet(student, inputs_s, prediction, labels, need_prior=False):
    if need_prior:
        outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7 = student(inputs_s, prediction)
    else:
        outputs0, outputs1, outputs2, outputs3, outputs4, outputs5, outputs6, outputs7 = student(inputs_s)

    loss0 = criterion(outputs0, labels)
    loss1 = criterion(outputs1, labels)
    loss2 = criterion(outputs2, labels)
    loss3 = criterion(outputs3, labels)
    loss4 = criterion(outputs4, labels)
    loss5 = criterion(outputs5, labels)
    loss6 = criterion(outputs6, labels)
    loss7 = criterion(outputs7, labels)

    loss_hard = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7

    if args['distillation'] and prediction is not None:
        loss02 = criterion(outputs0, F.sigmoid(prediction))
        loss12 = criterion(outputs1, F.sigmoid(prediction))
        loss22 = criterion(outputs2, F.sigmoid(prediction))
        loss32 = criterion(outputs3, F.sigmoid(prediction))
        loss42 = criterion(outputs4, F.sigmoid(prediction))
        loss52 = criterion(outputs5, F.sigmoid(prediction))
        loss62 = criterion(outputs6, F.sigmoid(prediction))
        loss72 = criterion(outputs7, F.sigmoid(prediction))

        loss_soft = loss02 + loss12 + loss22 + loss32 + loss42 + loss52 + loss62 + loss72
        total_loss = loss_hard + 0.5 * loss_soft
    else:
        total_loss = loss_hard

    return total_loss, loss0, loss1, loss2

def print_log(total_loss, loss0, loss1, loss2, batch_size, curr_iter, optimizer, type='normal'):
    total_loss_record.update(total_loss.data, batch_size)
    loss0_record.update(loss0.data, batch_size)
    loss1_record.update(loss1.data, batch_size)
    loss2_record.update(loss2.data, batch_size)
    # loss3_record.update(loss3.data, batch_size)
    # loss4_record.update(loss4.data, batch_size)
    log = '[iter %d][%s], [total loss %.5f], [loss0 %.5f], [loss1 %.5f], [loss2 %.5f] ' \
          '[lr %.13f]' % \
          (curr_iter, type, total_loss_record.avg, loss0_record.avg, loss1_record.avg, loss2_record.avg,
           optimizer.param_groups[1]['lr'])
    print(log)
    open(log_path, 'a').write(log + '\n')

if __name__ == '__main__':
    main()

