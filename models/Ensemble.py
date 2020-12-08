import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.BASNet.BASNet import BASNet
from models.R3Net.R3Net import R3Net
from models.DSS.DSSNet import build_model
from models.CPD.CPD_ResNet_models import CPD_ResNet
from models.RAS.RAS import RAS
from models.PiCANet.network import Unet
from models.PoolNet.poolnet import build_model_poolnet
from models.R2Net.r2net import build_model_r2net
from models.F3Net.net import F3Net
from utils.utils_mine import load_part_of_model, load_part_of_model2

class Ensemble(nn.Module):
    def __init__(self, device, pretrained=True):
        super(Ensemble, self).__init__()
        self.student_a = F3Net(cfg=None)
        self.student_b = CPD_ResNet()
        # self.student_c = RAS()
        if pretrained:
            self.student_a = load_part_of_model(self.student_a, 'pretrained/F3Net', device_id=device)
            self.student_b = load_part_of_model(self.student_b, 'pretrained/CPD-R.pth', device_id=device)
            # self.student_c = load_part_of_model(self.student_c, 'pretrained/RAS.v1.pth', device_id=device)

    def forward(self, x):
        # a_out1u, a_out2u, a_out2r, a_out3r, a_out4r, a_out5r = self.student_a(x)
        # b_outputs0, b_outputs1 = self.student_b(x)
        # c_outputs0, c_outputs1, c_outputs2, c_outputs3, c_outputs4 = self.student_c(x)
        return self.student_a(x), self.student_b(x)
