import numpy as np
import os
import torch
import torch.nn as nn
import sys
# import pydensecrf.densecrf as dcrf
import torch.nn.functional as F
torch_ver = torch.__version__[:3]

eps = sys.float_info.epsilon

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CriterionDSN(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''
    def __init__(self, dsn_weight=0.4):
        super(CriterionDSN, self).__init__()
        self.dsn_weight = dsn_weight

        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, preds, target):
        h, w = target.size(2), target.size(3)

        # print(preds[0].size())
        # print(target.size())

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[0], size=(h, w), mode='bilinear')
        loss1 = self.criterion(scale_pred, target)

        if torch_ver == '0.4':
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        else:
            scale_pred = F.upsample(input=preds[1], size=(h, w), mode='bilinear')
        loss2 = self.criterion(scale_pred, target)
        return self.dsn_weight*loss1 + loss2

class CriterionKL(nn.Module):
    def __init__(self):
        super(CriterionKL, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, preds, target):
        assert preds.size() == target.size()

        n, c, w, h = preds.size()
        softmax_preds = F.softmax(target.permute(0, 2, 3, 1).contiguous().view(-1, c), dim=1)
        loss = (torch.sum(-softmax_preds * self.log_softmax(preds.permute(0, 2, 3, 1).contiguous().view(-1, c)))) / w / h

        return loss

class CriterionKL2(nn.Module):
    def __init__(self):
        super(CriterionKL2, self).__init__()

    def forward(self, preds, target):
        assert preds.size() == target.size()

        b, c, w, h = preds.size()
        preds = F.softmax(preds.view(b, -1), dim=1)
        target = F.softmax(target.view(b, -1), dim=1)
        loss = (preds * (preds / target).log()).sum() / b

        return loss

class CriterionStructure(nn.Module):
    def __init__(self):
        super(CriterionStructure, self).__init__()
        self.gamma = 2

    def forward(self, pred, target):
        assert pred.size() == target.size()

        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduce='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        ##### focal loss #####
        # p_t = torch.exp(-wbce)
        # f_loss = (1 - p_t) ** self.gamma * wbce

        pred = torch.sigmoid(pred)
        inter = ((pred * target) * weit).sum(dim=(2, 3))
        union = ((pred + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)
        return (wbce + wiou).mean()

def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)

class CriterionKL3(nn.Module):
    def __init__(self):
        super(CriterionKL3, self).__init__()

    def KLD(self, input, target):
        input = input / torch.sum(input)
        target = target / torch.sum(target)
        eps = sys.float_info.epsilon
        return torch.sum(target * torch.log(eps + torch.div(target, (input + eps))))

    def forward(self, input, target):
        assert input.size() == target.size()

        return _pointwise_loss(lambda a, b:self.KLD(a,b), input, target)

class CriterionPairWise(nn.Module):
    def __init__(self, scale):
        super(CriterionPairWise, self).__init__()
        self.scale = scale

    def L2(self, inputs):
        return (((inputs ** 2).sum(dim=1)) ** 0.5).reshape(inputs.shape[0], 1, inputs.shape[2], inputs.shape[3]) + 1e-8

    def similarity(self, inputs):
        inputs = inputs.float()
        tmp = self.L2(inputs).detach()
        inputs = inputs / tmp
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1], -1)
        return torch.einsum('icm, icn->imn', [inputs, inputs])

    def sim_dis_compute(self, preds, targets):
        sim_err = ((self.similarity(targets) - self.similarity(preds)) ** 2) / ((targets.size(-1) * targets.size(-2)) ** 2) / targets.size(0)
        sim_dis = sim_err.sum()
        return sim_dis

    def forward(self, preds, targets):
        total_w, total_h = preds.shape[2], preds.shape[3]
        patch_w, patch_h = int(total_w * self.scale), int(total_h * self.scale)
        max_pooling = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True)
        loss = self.sim_dis_compute(max_pooling(preds), max_pooling(targets))
        return loss

class CriterionDice(nn.Module):
    def __init__(self):
        super(CriterionDice, self).__init__()

    def forward(self, pred, target):
        n = target.size(0)
        smooth = 1
        pred = F.sigmoid(pred)
        pred_flat = pred.view(n, -1)
        target_flat = target.view(n, -1)

        intersection = pred_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (pred_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / n

        return loss



def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def s_measure(gt, sm):
    """
    This fucntion computes the structural similarity (S-Measure) between the saliency map and the ground truth
    article: https://www.crcv.ucf.edu/papers/iccv17/1164.pdf
    original code [Matlab]: https://github.com/DengPingFan/S-measure
    parameters
    ----------
    gt : numpy.ndarray
        The path to the ground truth directory
    sm : numpy.ndarray
        The path to the predicted saliency map directory
    Returns
    -------
    value : float
        The calculated S-masure
    """
    gt_mean = np.mean(gt)

    if gt_mean == 0:  # if the GT is completely black
        sm_mean = np.mean(sm)
        measure = 1.0 - sm_mean  # only calculate the area of intersection
    elif gt_mean == 1:  # if the GT is completely white
        sm_mean = np.mean(sm)
        measure = sm_mean.copy()  # only calcualte the area of intersection
    else:
        alpha = 0.5
        measure = alpha * s_object(sm, gt) + (1 - alpha) * s_region(sm, gt)
        if measure < 0:
            measure = 0

    return measure


def ssim(gt, sm):
    gt = gt.astype(np.float32)

    height, width = sm.shape
    num_pixels = width * height

    # Compute the mean of SM,GT
    sm_mean = np.mean(sm)
    gt_mean = np.mean(gt)

    # Compute the variance of SM,GT
    sigma_x2 = np.sum(np.sum((sm - sm_mean) ** 2)) / (num_pixels - 1 + eps)
    sigma_y2 = np.sum(np.sum((gt - gt_mean) ** 2)) / (num_pixels - 1 + eps)

    # Compute the covariance
    sigma_xy = np.sum(np.sum((sm - sm_mean) * (gt - gt_mean))) / (num_pixels - 1 + eps)

    alpha = 4 * sm_mean * gt_mean * sigma_xy
    beta = (sm_mean ** 2 + gt_mean ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        ssim_value = alpha / (beta + eps)
    elif alpha == 0 and beta == 0:
        ssim_value = 1.0
    else:
        ssim_value = 0

    return ssim_value


def divide_sm(sm, x, y):
    # copy the 4 regions
    lt = sm[:y, :x]
    rt = sm[:y, x:]
    lb = sm[y:, :x]
    rb = sm[y:, x:]

    return lt, rt, lb, rb


def divide_gt(gt, x, y):
    height, width = gt.shape
    area = width * height

    # copy the 4 regions
    lt = gt[:y, :x]
    rt = gt[:y, x:]
    lb = gt[y:, :x]
    rb = gt[y:, x:]

    # The different weight (each block proportional to the GT foreground region).
    w1 = (x * y) / area
    w2 = ((width - x) * y) / area
    w3 = (x * (height - y)) / area
    w4 = 1.0 - w1 - w2 - w3

    return lt, rt, lb, rb, w1, w2, w3, w4


def centroid(gt):
    # col
    rows, cols = gt.shape

    if np.sum(gt) == 0:
        x = np.round(cols / 2)
        y = np.round(rows / 2)
    else:
        total = np.sum(gt)
        i = np.arange(cols).reshape(1, cols) + 1
        j = np.arange(rows).reshape(rows, 1) + 1

        x = int(np.round(np.sum(np.sum(gt, 0, keepdims=True) * i) / total))
        y = int(np.round(np.sum(np.sum(gt, 1, keepdims=True) * j) / total))

    return x, y


def s_region(gt, sm):
    x, y = centroid(gt)
    gt_1, gt_2, gt_3, gt_4, w1, w2, w3, w4 = divide_gt(gt, x, y)

    sm_1, sm_2, sm_3, sm_4 = divide_sm(sm, x, y)

    q1 = ssim(sm_1, gt_1)
    q2 = ssim(sm_2, gt_2)
    q3 = ssim(sm_3, gt_3)
    q4 = ssim(sm_4, gt_4)

    region_value = w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4

    return region_value


def object(gt, sm):
    x = np.mean(sm[gt == 1])
    # compute the standard deviations of the foreground or background in sm
    sigma_x = np.std(sm[gt == 1])
    score = 2.0 * x / (x ** 2 + 1.0 + sigma_x + eps)
    return score


def s_object(gt, sm):
    # compute the similarity of the foreground in the object level

    sm_fg = sm.copy()
    sm_fg[gt == 0] = 0
    o_fg = object(sm_fg, gt)

    # compute the similarity of the background
    sm_bg = 1.0 - sm.copy()
    sm_bg[gt == 1] = 0
    o_bg = object(sm_bg, gt == 0)

    u = np.mean(gt)
    object_value = u * o_fg + (1 - u) * o_bg
    return object_value

def cal_precision_recall_mae(prediction, gt):
    # input should be np array with data type uint8
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    eps = 1e-4

    prediction = prediction / 255.
    gt = gt / 255.

    mae = np.mean(np.abs(prediction - gt))

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt)

    precision, recall = [], []
    # calculating precision and recall at 255 different binarizing thresholds
    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1

        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)

        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))

    return precision, recall, mae

def cal_precision_recall_mae_smeasure(prediction, gt):
    # input should be np array with data type uint8
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape

    eps = 1e-4

    prediction = prediction / 255.
    gt = gt / 255.

    mae = np.mean(np.abs(prediction - gt))

    smeasure = s_measure(gt, prediction)

    hard_gt = np.zeros(prediction.shape)
    hard_gt[gt > 0.5] = 1
    t = np.sum(hard_gt)

    precision, recall = [], []
    # calculating precision and recall at 255 different binarizing thresholds
    for threshold in range(256):
        threshold = threshold / 255.

        hard_prediction = np.zeros(prediction.shape)
        hard_prediction[prediction > threshold] = 1

        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)

        precision.append((tp + eps) / (p + eps))
        recall.append((tp + eps) / (t + eps))

    return precision, recall, mae, smeasure


def cal_fmeasure(precision, recall):
    assert len(precision) == 256
    assert len(recall) == 256
    beta_square = 0.3
    max_fmeasure = max([(1 + beta_square) * p * r / (beta_square * p + r) for p, r in zip(precision, recall)])

    return max_fmeasure


# codes of this function are borrowed from https://github.com/Andrew-Qibin/dss_crf
# def crf_refine(img, annos):
#     def _sigmoid(x):
#         return 1 / (1 + np.exp(-x))
#
#     assert img.dtype == np.uint8
#     assert annos.dtype == np.uint8
#     assert img.shape[:2] == annos.shape
#
#     # img and annos should be np array with data type uint8
#
#     EPSILON = 1e-8
#
#     M = 2  # salient or not
#     tau = 1.05
#     # Setup the CRF model
#     d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)
#
#     anno_norm = annos / 255.
#
#     n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
#     p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))
#
#     U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
#     U[0, :] = n_energy.flatten()
#     U[1, :] = p_energy.flatten()
#
#     d.setUnaryEnergy(U)
#
#     d.addPairwiseGaussian(sxy=3, compat=3)
#     d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)
#
#     # Do the inference
#     infer = np.array(d.inference(1)).astype('float32')
#     res = infer[1, :]
#
#     res = res * 255
#     res = res.reshape(img.shape[:2])
#     return res.astype('uint8')

if __name__ == '__main__':
    pixel_wise_loss = CriterionKL3()
    pair_wise_loss = CriterionPairWise(scale=0.5)
    preds = torch.rand([2, 1, 10, 10])
    # print(torch.sum(F.softmax(preds, dim=1)))
    targets = torch.rand([2, 1, 10, 10])
    # loss = pixel_wise_loss(F.sigmoid(preds), F.sigmoid(preds))
    loss = F.kl_div(preds, preds)
    # loss2 = pair_wise_loss(preds, targets)
    print(pixel_wise_loss(preds, targets))
    # print(loss2)
