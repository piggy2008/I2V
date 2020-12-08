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
from misc import AvgMeter, check_mkdir, CriterionKL3, CriterionKL, CriterionPairWise, CriterionStructure
from models.MGA.mga_model import MGA_Network
from models.Ensemble import Ensemble
from torch.backends import cudnn
import time
from utils.utils_mine import load_part_of_model, load_part_of_model2, load_MGA

import random

cudnn.benchmark = True
device_id = 1
torch.manual_seed(2019)
# torch.cuda.set_device(device_id)


time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
ckpt_path = './ckpt2'
exp_name = 'VideoSaliency' + '_' + time_str

args = {
    'distillation': True,
    'L2': False,
    'KL': False,
    'structure': True,
    'iter_num': 10000,
    'iter_save': 2000,
    'iter_start_seq': 0,
    'train_batch_size': 5,
    'last_iter': 0,
    'lr': 1e-3,
    'lr_decay': 0.9,
    'weight_decay': 5e-4,
    'momentum': 0.925,
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

if args['L2']:
    criterion_l2 = nn.MSELoss().cuda()
    # criterion_pair = CriterionPairWise(scale=0.5).cuda()
if args['KL']:
    criterion_kl = CriterionKL3().cuda()

if args['structure']:
    criterion_str = CriterionStructure().cuda()

log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')

total_loss_record, loss0_record, loss1_record, loss2_record = AvgMeter(), AvgMeter(), AvgMeter(), AvgMeter()

def fix_parameters(parameters):
    for name, parameter in parameters:
        if name.find('linearp') >= 0 or name.find('linearr') >= 0 or name.find('decoder') >= 0:
            print(name, 'is not fixed')

        else:
            print(name, 'is fixed')
            parameter.requires_grad = False

def main():
    teacher = MGA_Network(nInputChannels=3, n_classes=1, os=16,
                      img_backbone_type='resnet101', flow_backbone_type='resnet34')
    teacher = load_MGA(teacher, args['mga_model_path'], device_id=device_id)
    teacher.eval()
    teacher.cuda(device_id)

    student = Ensemble(device_id).cuda(device_id).train()

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
            train_single(student, teacher, inputs, flows, labels, optimizer, curr_iter)


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

def train_single(student, teacher, inputs, flows, labels, optimizer, curr_iter):
    inputs = Variable(inputs).cuda(device_id)
    flows = Variable(flows).cuda(device_id)
    labels = Variable(labels).cuda(device_id)
    if args['distillation']:
        prediction, _, _, _, _ = teacher(inputs, flows)

    optimizer.zero_grad()
    outputs_a, outputs_b, outputs_c = student(inputs)
    a_out1u, a_out2u, a_out2r, a_out3r, a_out4r, a_out5r = outputs_a # F3Net
    b_outputs0, b_outputs1 = outputs_b # CPD
    c_outputs0, c_outputs1, c_outputs2, c_outputs3, c_outputs4 = outputs_c # RAS

    loss0_a = criterion_str(a_out1u, labels)
    loss1_a = criterion_str(a_out2u, labels)
    loss2_a = criterion_str(a_out2r, labels)
    loss3_a = criterion_str(a_out3r, labels)
    loss4_a = criterion_str(a_out4r, labels)
    loss5_a = criterion_str(a_out5r, labels)
    loss_hard_a = (loss0_a + loss1_a) / 2 + loss2_a / 2 + loss3_a / 4 + loss4_a / 8 + loss5_a / 16

    loss0_b = criterion_str(b_outputs0, labels)
    loss1_b = criterion_str(b_outputs1, labels)
    loss_hard_b = loss0_b + loss1_b

    loss0_c = criterion_str(c_outputs0, labels)
    loss1_c = criterion_str(c_outputs1, labels)
    loss2_c = criterion_str(c_outputs2, labels)
    loss3_c = criterion_str(c_outputs3, labels)
    loss4_c = criterion_str(c_outputs4, labels)
    loss_hard_c = loss0_c + loss1_c + loss2_c + loss3_c + loss4_c

    # ensemble
    loss_en_hard = criterion_str(a_out2u + b_outputs1 + c_outputs0, labels)

    if args['distillation']:
        loss0_a = criterion_str(a_out1u, F.sigmoid(prediction))
        loss1_a = criterion_str(a_out2u, F.sigmoid(prediction))
        loss2_a = criterion_str(a_out2r, F.sigmoid(prediction))
        loss3_a = criterion_str(a_out3r, F.sigmoid(prediction))
        loss4_a = criterion_str(a_out4r, F.sigmoid(prediction))
        loss5_a = criterion_str(a_out5r, F.sigmoid(prediction))
        loss_soft_a = (loss0_a + loss1_a) / 2 + loss2_a / 2 + loss3_a / 4 + loss4_a / 8 + loss5_a / 16

        loss0_b = criterion_str(b_outputs0, F.sigmoid(prediction))
        loss1_b = criterion_str(b_outputs1, F.sigmoid(prediction))
        loss_soft_b = loss0_b + loss1_b

        loss0_c = criterion_str(c_outputs0, F.sigmoid(prediction))
        loss1_c = criterion_str(c_outputs1, F.sigmoid(prediction))
        loss2_c = criterion_str(c_outputs2, F.sigmoid(prediction))
        loss3_c = criterion_str(c_outputs3, F.sigmoid(prediction))
        loss4_c = criterion_str(c_outputs4, F.sigmoid(prediction))
        loss_soft_c = loss0_c + loss1_c + loss2_c + loss3_c + loss4_c

        loss_en_soft = criterion_str(a_out2u + b_outputs1 + c_outputs0, F.sigmoid(prediction))

    loss_hard = loss_hard_a + loss_hard_b + loss_hard_c
    if args['distillation']:
        loss_soft = loss_soft_a + loss_soft_b + loss_soft_c
        total_loss = loss_hard + loss_soft + loss_en_hard + loss_en_soft
    else:
        total_loss = loss_hard + loss_en_hard
    total_loss.backward()
    optimizer.step()

    print_log(total_loss, loss_hard_a, loss_hard_b, loss_hard_c, args['train_batch_size'], curr_iter, optimizer)

    return

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
