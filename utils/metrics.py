# GlaS metrics, translated from the official Matlab code:
# https://warwick.ac.uk/fac/sci/dcs/research/tia/glascontest/evaluation/evaluation_v6.zip
#

import numpy as np
import torch

from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from medpy.metric.binary import jc, dc, hd, hd95, recall, specificity, precision

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)

    return iou


def hd95_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = (intersection + smooth) / (union + smooth)
    dice = (2 * iou) / (iou + 1)

    try:
        hd95_ = hd95(output_, target_)
    except:
        hd95_ = 0

    return hd95_

def ObjectHausdorff(S=None, G=None):
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    totalAreaS = (S > 0).sum()
    totalAreaG = (G > 0).sum()
    listLabelS = np.unique(S)
    listLabelS = np.delete(listLabelS, np.where(listLabelS == 0))
    listLabelG = np.unique(G)
    listLabelG = np.delete(listLabelG, np.where(listLabelG == 0))

    temp1 = 0
    for iLabelS in range(len(listLabelS)):
        Si = S == listLabelS[iLabelS]
        intersectlist = G[Si]
        if intersectlist.any():
            indexGi = stats.mode(intersectlist).mode
            Gi = G == indexGi
        else:
            tempDist = np.zeros((len(listLabelG), 1))
            for iLabelG in range(len(listLabelG)):
                Gi = G == listLabelG[iLabelG]
                tempDist[iLabelG] = Hausdorff(Gi, Si)
            minIdx = np.argmin(tempDist)
            Gi = G == listLabelG[minIdx]
        omegai = Si.sum() / totalAreaS
        temp1 = temp1 + omegai * Hausdorff(Gi, Si)

    temp2 = 0
    for iLabelG in range(len(listLabelG)):
        tildeGi = G == listLabelG[iLabelG]
        intersectlist = S[tildeGi]
        if intersectlist.any():
            indextildeSi = stats.mode(intersectlist).mode
            tildeSi = S == indextildeSi
        else:
            tempDist = np.zeros((len(listLabelS), 1))
            for iLabelS in range(len(listLabelS)):
                tildeSi = S == listLabelS[iLabelS]
                tempDist[iLabelS] = Hausdorff(tildeGi, tildeSi)
            minIdx = np.argmin(tempDist)
            tildeSi = S == listLabelS[minIdx]
        tildeOmegai = tildeGi.sum() / totalAreaG
        temp2 = temp2 + tildeOmegai * Hausdorff(tildeGi, tildeSi)

    objHausdorff = (temp1 + temp2) / 2
    return objHausdorff


def Hausdorff(S=None, G=None, *args, **kwargs):
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    listS = np.unique(S)
    listS = np.delete(listS, np.where(listS == 0))
    listG = np.unique(G)
    listG = np.delete(listG, np.where(listG == 0))

    numS = len(listS)
    numG = len(listG)
    if numS == 0 and numG == 0:
        hausdorffDistance = 0
        return hausdorffDistance
    else:
        if numS == 0 or numG == 0:
            hausdorffDistance = np.Inf
            return hausdorffDistance

    y = np.where(S > 0)
    x = np.where(G > 0)

    x = np.vstack((x[0], x[1])).transpose()
    y = np.vstack((y[0], y[1])).transpose()

    nbrs = NearestNeighbors(n_neighbors=1).fit(x)
    distances, indices = nbrs.kneighbors(y)
    dist1 = np.max(distances)

    nbrs = NearestNeighbors(n_neighbors=1).fit(y)
    distances, indices = nbrs.kneighbors(x)
    dist2 = np.max(distances)

    hausdorffDistance = np.max((dist1, dist2))
    return hausdorffDistance


def F1score(S=None, G=None):
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    listS = np.unique(S)
    listS = np.delete(listS, np.where(listS == 0))
    numS = len(listS)
    listG = np.unique(G)
    listG = np.delete(listG, np.where(listG == 0))
    numG = len(listG)

    if numS == 0 and numG == 0:
        return 1
    elif numS == 0 or numG == 0:
        return 0

    tempMat = np.zeros((numS, 3))
    tempMat[:, 0] = listS
    for iSegmentedObj in range(numS):
        intersectGTObjs = G[S == tempMat[iSegmentedObj, 0]]
        if intersectGTObjs.any():
            intersectGTObjs_flat = np.delete(
                intersectGTObjs.flatten(), np.where(intersectGTObjs.flatten() == 0)
            )

            if len(intersectGTObjs_flat) == 0:
                maxGTi = 0
            else:
                maxGTi = stats.mode(intersectGTObjs_flat).mode
            tempMat[iSegmentedObj, 1] = maxGTi

    for iSegmentedObj in range(numS):
        if tempMat[iSegmentedObj, 1] != 0:
            SegObj = S == tempMat[iSegmentedObj, 0]
            GTObj = G == tempMat[iSegmentedObj, 1]
            overlap = np.logical_and(SegObj, GTObj)
            areaOverlap = overlap.sum()
            areaGTObj = GTObj.sum()
            if areaOverlap / areaGTObj > 0.5:
                tempMat[iSegmentedObj, 2] = 1

    TP = (tempMat[:, 2] == 1).sum()
    FP = (tempMat[:, 2] == 0).sum()
    FN = numG - TP

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    if precision + recall == 0:
        return 0

    score = (2 * precision * recall) / (precision + recall)

    return score


def ObjectDice(S, G):
    S = np.array(S).astype(np.uint8)
    G = np.array(G).astype(np.uint8)

    totalAreaG = (G > 0).sum()
    listLabelS = np.unique(S)
    listLabelS = np.delete(listLabelS, np.where(listLabelS == 0))
    numS = len(listLabelS)
    listLabelG = np.unique(G)
    listLabelG = np.delete(listLabelG, np.where(listLabelG == 0))
    numG = len(listLabelG)

    if numS == 0 and numG == 0:
        return 1
    elif numS == 0 or numG == 0:
        return 0

    temp1 = 0
    totalAreaS = (S > 0).sum()
    for iLabelS in range(len(listLabelS)):
        Si = S == listLabelS[iLabelS]
        intersectlist = G[Si]
        if intersectlist.any():
            indexG1 = stats.mode(intersectlist).mode
            Gi = G == indexG1
        else:
            Gi = np.zeros(G.shape)

        omegai = Si.sum() / totalAreaS
        temp1 += omegai * Dice(Gi, Si)

    temp2 = 0
    totalAreaG = (G > 0).sum()
    for iLabelG in range(len(listLabelG)):
        tildeGi = G == listLabelG[iLabelG]
        intersectlist = S[tildeGi]
        if intersectlist.any():
            indextildeSi = stats.mode(intersectlist).mode
            tildeSi = S == indextildeSi  # np logical and?
        else:
            tildeSi = np.zeros(S.shape)

        tildeOmegai = tildeGi.sum() / totalAreaG
        temp2 += tildeOmegai * Dice(tildeGi, tildeSi)

    return (temp1 + temp2) / 2


def Dice(A, B):
    intersection = np.logical_and(A, B)
    return 2.0 * intersection.sum() / (A.sum() + B.sum())

# def Dice(A, B):
#     intersection = torch.sum(A * B)
    # print("intersection", intersect
#     union = torch.sum(A) + torch.sum(B)
#     dice = (2. * intersection + 1e-6) / (union + 1e-6)
#     return dice.item()

def pixelwise_acc(pred: torch.Tensor, lbl: torch.Tensor) -> float:
    pixel_num = pred.shape[0] * pred.shape[1] * pred.shape[2] * pred.shape[3]
    correct = (pred == lbl).sum().item()
    return correct/pixel_num


# Code copied from https://github.com/javiribera/locating-objects-without-bboxes/blob/master/object-locator/losses.py 
def averaged_hausdorff_distance(set1, set2, max_ahd=np.inf):
    """
    Compute the Averaged Hausdorff Distance function
    between two unordered sets of points (the function is symmetric).
    Batches are not supported, so squeeze your inputs first!
    :param set1: Array/list where each row/element is an N-dimensional point.
    :param set2: Array/list where each row/element is an N-dimensional point.
    :param max_ahd: Maximum AHD possible to return if any set is empty. Default: inf.
    :return: The Averaged Hausdorff Distance between set1 and set2.
    """

    if len(set1) == 0 or len(set2) == 0:
        return max_ahd

    set1 = np.array(set1)
    set2 = np.array(set2)

    assert set1.ndim == 2, 'got %s' % set1.ndim
    assert set2.ndim == 2, 'got %s' % set2.ndim

    assert set1.shape[1] == set2.shape[1], \
        'The points in both sets must have the same number of dimensions, got %s and %s.'\
        % (set2.shape[1], set2.shape[1])

    d2_matrix = pairwise_distances(set1, set2, metric='euclidean')

    res = np.average(np.min(d2_matrix, axis=0)) + \
        np.average(np.min(d2_matrix, axis=1))

    return res

# Copied from https://github.com/Project-MONAI/research-contributions/blob/main/UNETR/BTCV/utils/utils.py
def compute_dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)
