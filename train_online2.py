# Package Includes
from __future__ import division

import os
import socket
import timeit
from datetime import datetime
import numpy as np
from PIL import Image

# PyTorch includes
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

# Custom includes
from dataloader import davis_2016 as db
from dataloader import custom_transforms as tr

import scipy.misc as sm
from models.F3Net.net import F3Net

from dataloader.helpers import *


# Setting of parameters
if 'SEQ_NAME' not in os.environ.keys():
    seq_name = 'blackswan'
else:
    seq_name = str(os.environ['SEQ_NAME'])

torch.cuda.set_device(0)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = './ckpt'
exp_name = 'VideoSaliency_2020-08-14 10:40:23'

args = {
    'model': 'F3Net',
    'online_train': True,
    'snapshot': '20000',  # your snapshot filename (exclude extension name)
    'crf_refine': False,  # whether to use crf to refine results
    'save_results': True,  # whether to save the resulting masks
    'input_size': (380, 380),
    'batch_size': 1,
    'start': 0
}

db_root_dir = '/home/ty/data/davis'
save_dir = os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, 'davis', args['snapshot'] + '_online'))

if not os.path.exists(save_dir):
    os.makedirs(os.path.join(save_dir))


vis_net = 0  # Visualize the network?
vis_res = 0  # Visualize the results?
nAveGrad = 0  # Average the gradient every nAveGrad iterations
nEpochs = 2000 * nAveGrad  # Number of epochs for training
snapshot = nEpochs  # Store a model every snapshot epochs
parentEpoch = 240

# Parameters in p are used for the name of the model
p = {
    'trainBatch': 1,  # Number of Images in each mini-batch
    }
seed = 0

parentModelName = 'parent'
# Select which GPU, -1 if CPU
gpu_id = 0
device = torch.device("cuda:"+str(gpu_id) if torch.cuda.is_available() else "cpu")

# Network definition
net = F3Net(cfg=None)
# net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), map_location='cuda:0'))
net.load_state_dict(
    torch.load(os.path.join(ckpt_path, exp_name, str(args['snapshot']) + '_' + seq_name + '_online.pth'),
               map_location='cuda:0'))

# Logging into Tensorboard

net.to(device)  # PyTorch 0.4.0 style
criterion = nn.BCEWithLogitsLoss().cuda()
# Visualize the network

# Use the following optimizer
lr = 1e-8
wd = 0.0002
optimizer = optim.SGD([
    {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
     'lr': 2 * lr},
    {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
     'lr': lr, 'weight_decay': wd}
], momentum=0.9)

# Preparation of the data loaders
# Define augmentation transformations as a composition
composed_transforms = transforms.Compose([tr.RandomHorizontalFlip(),
                                          # tr.ScaleNRotate(rots=(-30, 30), scales=(.75, 1.25)),
                                          tr.ToTensor()])
# Training dataset and its iterator
db_train = db.DAVIS2016(train=True, db_root_dir=db_root_dir, transform=composed_transforms, seq_name=seq_name)
trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=1)

# Testing dataset and its iterator
db_test = db.DAVIS2016(train=False, db_root_dir=db_root_dir, transform=tr.ToTensor(), seq_name=seq_name)
testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)


num_img_tr = len(trainloader)
num_img_ts = len(testloader)
loss_tr = []
aveGrad = 0

print("Start of Online Training, sequence: " + seq_name)
start_time = timeit.default_timer()
# Main Training and Testing Loop
for epoch in range(0, nEpochs):
    # One training epoch
    running_loss_tr = 0
    np.random.seed(seed + epoch)
    for ii, sample_batched in enumerate(trainloader):

        inputs, gts = sample_batched['image'], sample_batched['gt']

        # Forward-Backward of the mini-batch
        inputs.requires_grad_()
        inputs, gts = inputs.to(device), gts.to(device)

        out1u, out2u, out2r, out3r, out4r, out5r = net.forward(inputs)

        # Compute the fuse loss
        loss0 = criterion(out1u, gts)
        loss1 = criterion(out2u, gts)
        loss2 = criterion(out2r, gts)
        loss3 = criterion(out3r, gts)
        loss4 = criterion(out4r, gts)
        loss5 = criterion(out5r, gts)
        # loss7 = criterion(outputs7, labels)

        total_loss = (loss0 + loss1) / 2 + loss2 / 2 + loss3 / 4 + loss4 / 8 + loss5 / 16
        running_loss_tr += total_loss
        # Print stuff
        if epoch % (nEpochs//20) == (nEpochs//20 - 1):
            running_loss_tr /= num_img_tr
            loss_tr.append(running_loss_tr)

            print('[Epoch: %d, numImages: %5d]' % (epoch+1, ii + 1))
            print('Loss: %f' % running_loss_tr)


        # Backward the averaged gradient
        total_loss /= nAveGrad
        total_loss.backward()
        aveGrad += 1

        # Update the weights once in nAveGrad forward passes
        if aveGrad % nAveGrad == 0:

            optimizer.step()
            optimizer.zero_grad()
            aveGrad = 0

    # Save the model
    if (epoch % snapshot) == snapshot - 1 and epoch != 0:
        print('taking snapshot ...')
        torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, str(args['snapshot']) + '_' + seq_name + '_online.pth'))

stop_time = timeit.default_timer()
print('Online training time: ' + str(stop_time - start_time))


# Testing Phase
if vis_res:
    import matplotlib.pyplot as plt
    plt.close("all")
    plt.ion()
    f, ax_arr = plt.subplots(1, 3)

save_dir_res = os.path.join(save_dir, 'Results', seq_name)
if not os.path.exists(save_dir_res):
    os.makedirs(save_dir_res)


print('Testing Network')
with torch.no_grad():  # PyTorch 0.4.0 style
    # Main Testing Loop
    for ii, sample_batched in enumerate(testloader):

        img, gt, fname = sample_batched['image'], sample_batched['gt'], sample_batched['fname']

        # Forward of the mini-batch
        inputs, gts = img.to(device), gt.to(device)

        outputs = net.forward(inputs)

        for jj in range(int(inputs.size()[0])):
            pred = np.transpose(outputs[1].cpu().data.numpy()[jj, :, :, :], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)
            pred = pred * 255.0
            pred = pred.astype('uint8')

            # Save the result, attention to the index jj
            save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, 'davis', args['snapshot'] + '_online'))
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            Image.fromarray(pred).save(os.path.join(save_path, fname[jj] + '.png'))

            if vis_res:
                img_ = np.transpose(img.numpy()[jj, :, :, :], (1, 2, 0))
                gt_ = np.transpose(gt.numpy()[jj, :, :, :], (1, 2, 0))
                gt_ = np.squeeze(gt)
                # Plot the particular example
                ax_arr[0].cla()
                ax_arr[1].cla()
                ax_arr[2].cla()
                ax_arr[0].set_title('Input Image')
                ax_arr[1].set_title('Ground Truth')
                ax_arr[2].set_title('Detection')
                ax_arr[0].imshow(im_normalize(img_))
                ax_arr[1].imshow(gt_)
                ax_arr[2].imshow(im_normalize(pred))
                plt.pause(0.001)
